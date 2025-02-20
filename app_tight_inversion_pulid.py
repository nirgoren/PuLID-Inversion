import time

import gradio as gr
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, rf_denoise, rf_inversion, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long


def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip_model = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model, version=args.version)

    # function to encode an image into latents
    def encode_image_to_latents(self, img, opts):
        """
        Opposite of decode: Takes a PIL image and encodes it into latents (x).
        """
        t0 = time.perf_counter()
        img = Image.fromarray(img)
        img = img.resize((opts.width, opts.height), resample=Image.LANCZOS)

        # Convert image to torch.Tensor and scale to [-1, 1]
        # Image is in [0, 255] → scale to [0,1] → then map to [-1,1].
        x = np.array(img).astype(np.float32)
        x = torch.from_numpy(x)  # shape: (H, W, C)
        x = (x / 127.5) - 1.0    # now in [-1, 1]
        x = rearrange(x, "h w c -> 1 c h w")  # shape: (1, C, H, W)

        # Move encoder to device if you are offloading
        if self.offload:
            self.ae.encoder.to(self.device)

        x = x.to(self.device, dtype=torch.bfloat16)

        # 2) Encode with autocast
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.encode_no_sampling(x)

        # 3) Offload if needed
        if self.offload:
            self.ae.encoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"Encoded in {t1 - t0:.2f} seconds.")

        return x

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        id_image = None,
        width: int = 512,
        height: int = 512,
        num_steps: int = 20,
        start_step: int = 0,
        guidance: float = 4.0,
        seed: int = -1,
        id_weight: float = 1.0,
        neg_prompt: str = "",
        true_cfg: float = 1.0,
        timestep_to_start_cfg: int = 1,
        max_sequence_length: int = 128,
        gamma: float = 0.5,
        eta: float = 0.7,
        s: float = 0,
        tau: float = 5,
        perform_inversion: bool = True,
        perform_reconstruction: bool = False,
        perform_editing: bool = True,
        inversion_true_cfg: float = 1.0,
    ):
        """
        Core function that performs the image generation.
        """
        self.t5.max_length = max_sequence_length

        # If seed == -1, random
        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        torch.manual_seed(opts.seed)

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()

        print(f"Generating prompt: '{opts.prompt}' (seed={opts.seed})...")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-6


        # 1) Prepare input noise
        noise = get_noise(
            num_samples=1,
            height=opts.height,
            width=opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        bs, c, h, w = noise.shape
        noise = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if noise.shape[0] == 1 and bs > 1:
            noise = repeat(noise, "1 ... -> bs ...", bs=bs)
        # encode
        x = self.encode_image_to_latents(id_image, opts)
        
        timesteps = get_schedule(opts.num_steps, x.shape[-1] * x.shape[-2] // 4, shift=False)

        # 2) Prepare text embeddings
        if self.offload:
            self.t5 = self.t5.to(self.device)
            self.clip_model = self.clip_model.to(self.device)

        inp = prepare(t5=self.t5, clip=self.clip_model, img=x, prompt=opts.prompt)
        inp_inversion = prepare(t5=self.t5, clip=self.clip_model, img=x, prompt="")
        inp_neg = None
        if use_true_cfg:
            inp_neg = prepare(t5=self.t5, clip=self.clip_model, img=x, prompt=neg_prompt)

        # Offload text encoders, load ID detection to GPU
        if self.offload:
            self.t5 = self.t5.cpu()
            self.clip_model = self.clip_model.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        # 3) ID Embeddings (optional)
        id_embeddings = None
        uncond_id_embeddings = None
        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # Offload ID pipeline, load main FLUX model to GPU
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()

            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        y_0 = inp["img"].clone().detach()

        inverted = None
        if perform_inversion:
            inverted = rf_inversion(
                self.model,
                **inp_inversion,
                timesteps=timesteps,
                guidance=opts.guidance,
                id=id_embeddings,
                id_weight=id_weight,
                start_step=start_step,
                uncond_id=uncond_id_embeddings,
                true_cfg=inversion_true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=inp_neg["txt"] if use_true_cfg else None,
                neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
                neg_vec=inp_neg["vec"] if use_true_cfg else None,
                aggressive_offload=self.aggressive_offload,
                y_1=noise,
                gamma=gamma
            )

            img = inverted
        else:
            img = noise
        inp["img"] = img
        inp_inversion["img"] = img

        recon = None
        if perform_reconstruction:
            recon = rf_denoise(
                self.model,
                **inp_inversion,
                timesteps=timesteps,
                guidance=opts.guidance,
                id=id_embeddings,
                id_weight=id_weight,
                start_step=start_step,
                uncond_id=uncond_id_embeddings,
                true_cfg=inversion_true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=inp_neg["txt"] if use_true_cfg else None,
                neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
                neg_vec=inp_neg["vec"] if use_true_cfg else None,
                aggressive_offload=self.aggressive_offload,
                y_0=y_0,
                eta=eta,
                s=s,
                tau=tau,
            )

        edited = None
        if perform_editing:
            edited = rf_denoise(
                self.model,
                **inp,
                timesteps=timesteps,
                guidance=opts.guidance,
                id=id_embeddings,
                id_weight=id_weight,
                start_step=start_step,
                uncond_id=uncond_id_embeddings,
                true_cfg=true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=inp_neg["txt"] if use_true_cfg else None,
                neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
                neg_vec=inp_neg["vec"] if use_true_cfg else None,
                aggressive_offload=self.aggressive_offload,
                y_0=y_0,
                eta=eta,
                s=s,
                tau=tau,
            )

        # Offload flux model, load auto-decoder
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # 5) Decode latents
        if edited is not None:
            edited = unpack(edited.float(), opts.height, opts.width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                edited = self.ae.decode(edited)

        if inverted is not None:
            inverted = unpack(inverted.float(), opts.height, opts.width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                inverted = self.ae.decode(inverted)
        
        if recon is not None:
            recon = unpack(recon.float(), opts.height, opts.width)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                recon = self.ae.decode(recon)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.2f} seconds.")

        # Convert to PIL
        if edited is not None:
            edited = edited.clamp(-1, 1)
            edited = rearrange(edited[0], "c h w -> h w c")
            edited = Image.fromarray((127.5 * (edited + 1.0)).cpu().byte().numpy())

        if inverted is not None:
            inverted = inverted.clamp(-1, 1)
            inverted = rearrange(inverted[0], "c h w -> h w c")
            inverted = Image.fromarray((127.5 * (inverted + 1.0)).cpu().byte().numpy())
        
        if recon is not None:
            recon = recon.clamp(-1, 1)
            recon = rearrange(recon[0], "c h w -> h w c")
            recon = Image.fromarray((127.5 * (recon + 1.0)).cpu().byte().numpy())

        return edited, str(opts.seed), self.pulid_model.debug_img_list

# <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2404.16022' target='_blank'>PuLID: Pure and Lightning ID Customization via Contrastive Alignment</a> | Codes: <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>GitHub</a></p>
_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">Tight Inversion for Portrait Editing with FLUX</h1>
</div>

❗️❗️❗️**Tips:**

'''  # noqa E501

_CITE_ = r"""
"""  # noqa E501


def create_demo(args, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                offload: bool = False, aggressive_offload: bool = False):
    generator = FluxGenerator(model_name, device, offload, aggressive_offload, args)

    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="portrait, color, cinematic")
                id_image = gr.Image(label="ID Image")
                id_weight = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="id weight")

                width = gr.Slider(256, 1536, 1024, step=16, label="Width")
                height = gr.Slider(256, 1536, 1024, step=16, label="Height")
                num_steps = gr.Slider(1, 28, 28, step=1, label="Number of steps")
                start_step = gr.Slider(0, 10, 0, step=1, label="timestep to start inserting ID")
                guidance = gr.Slider(1.0, 10.0, 4.0, step=0.1, label="Guidance")
                seed = gr.Textbox(-1, label="Seed (-1 for random)")
                max_sequence_length = gr.Slider(128, 512, 128, step=128,
                                                label="max_sequence_length for prompt (T5), small will be faster")

                with gr.Accordion("Advanced Options (True CFG, true_cfg_scale=1 means use fake CFG, >1 means use true CFG", open=False):    # noqa E501
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="")
                    true_cfg = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="true CFG scale")
                    timestep_to_start_cfg = gr.Slider(0, 20, 1, step=1, label="timestep to start cfg", visible=args.dev)

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Textbox(label="Used Seed")
                intermediate_output = gr.Gallery(label='Output', elem_id="gallery", visible=args.dev)
                gr.Markdown(_CITE_)

        with gr.Row(), gr.Column():
                gr.Markdown("## Examples")
                example_inps = [
                    [
                        'a portrait of an alien',
                        'example_inputs/unsplash/alexander-jawfox-dNVjtsFA0p4-unsplash.jpg',
                        0, 4.0, 42, 3.5
                    ],
                    [
                        'a portrait of a clown',
                        'example_inputs/unsplash/lhon-karwan-11tbHtK5STE-unsplash.jpg',
                        0, 4.0, 42, 3.5
                    ],
                    [
                        'a portrait of a wizard',
                        'example_inputs/unsplash/arad-adiban-r--05n7pQ3g-unsplash.jpg',
                        0, 4.0, 42, 3.5
                    ],
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, start_step, guidance, seed, true_cfg])

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[prompt, id_image, width, height, num_steps, start_step, guidance, seed, id_weight, neg_prompt,
                    true_cfg, timestep_to_start_cfg, max_sequence_length],
            outputs=[output_image, seed_output, intermediate_output],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument('--version', type=str, default='v0.9.1', help='version of the model', choices=['v0.9.0', 'v0.9.1'])
    parser.add_argument("--name", type=str, default="flux-dev", choices=list('flux-dev'),
                        help="currently only support flux-dev")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, for 24G GPUs")
    parser.add_argument("--fp8", action="store_true", help="use flux-dev-fp8 model")
    parser.add_argument("--onnx_provider", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="set onnx_provider to cpu (default gpu) can help reduce RAM usage, and when combined with"
                             "fp8 option, the peak RAM is under 15GB")
    parser.add_argument("--port", type=int, default=8080, help="Port to use")
    parser.add_argument("--dev", action='store_true', help="Development mode")
    parser.add_argument("--pretrained_model", type=str, help='for development')
    args = parser.parse_args()

    if args.aggressive_offload:
        args.offload = True

    demo = create_demo(args, args.name, args.device, args.offload, args.aggressive_offload)
    demo.launch(server_name='0.0.0.0', server_port=args.port)
