import argparse
from pathlib import Path
import time
import numpy as np
import yaml
import torch
from einops import rearrange, repeat
from PIL import Image
import os

# -------------------------------------------------------------------
# The following imports must match your environment or local modules:
# If you have local modules named similarly to the original snippet,
# adjust as appropriate:
# -------------------------------------------------------------------
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, rf_denoise, rf_inversion
from flux.image_utils import find_and_plot_images
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


def get_models(model_name: str, device: torch.device, offload: bool, fp8: bool,
               onnx_provider: str = "gpu"):
    """
    Loads T5, CLIP, FLUX flow-model, and autoencoder for the given 'model_name'.
    """
    t5 = load_t5(device, max_length=128)
    clip_model = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(model_name, device="cpu" if offload else device)
    else:
        model = load_flow_model(model_name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(model_name, device="cpu" if offload else device)

    # Also create the PuLID pipeline for ID embedding
    pulid_model = PuLIDPipeline(model, device="cpu" if offload else device,
                                weight_dtype=torch.bfloat16,
                                onnx_provider=onnx_provider)
    return model, ae, t5, clip_model, pulid_model


class FluxGenerator:
    def __init__(
        self,
        model_name: str,
        device: str,
        offload: bool,
        aggressive_offload: bool,
        fp8: bool,
        onnx_provider: str,
        pulid_pretrained_path: str,
        pulid_version: str,
    ):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name

        # Load models
        model, ae, t5, clip_model, pulid_model = get_models(
            model_name=model_name,
            device=self.device,
            offload=offload,
            fp8=fp8,
            onnx_provider=onnx_provider,
        )

        self.model = model
        self.ae = ae
        self.t5 = t5
        self.clip_model = clip_model
        self.pulid_model = pulid_model

        # Move some parts to GPU for face/ID detection if offload is enabled
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = \
                self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")

        self.pulid_model.load_pretrain(pulid_pretrained_path, version=pulid_version)

    # function to encode an image into latents
    def encode_image_to_latents(self, img_path, opts):
        """
        Opposite of decode: Takes a PIL image and encodes it into latents (x).
        """
        t0 = time.perf_counter()

        # 1) Load and preprocess PIL image
        img = Image.open(img_path).convert("RGB")
        # Resize if necessary, or use opts.height / opts.width if you want a fixed size:
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
            # x = self.ae.encode_no_sampling(x)
            x = self.ae.encode(x)
        x = x.to(torch.bfloat16)

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
        id_image_path: str = None,
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
        image_path: str = None,
        gamma: float = 0.5,
        eta: float = 0.9,
        s: float = 0,
        tau: float = 6,
        perform_inversion: bool = True,
        perform_reconstruction: bool = True,
        perform_editing: bool = True,
        inversion_true_cfg: float = 1.0,
    ):
        """
        Core function that performs the image generation.
        """
        self.t5.max_length = max_sequence_length

        # If seed == -1, random
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
        x = self.encode_image_to_latents(image_path, opts)
        
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
        if id_image_path is not None and os.path.exists(id_image_path):
            id_image = Image.open(id_image_path).convert("RGB")
            id_image = np.array(id_image)
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(
                id_image, cal_uncond=use_true_cfg
            )

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

        return edited, inverted, recon, opts.seed


def main():
    parser = argparse.ArgumentParser("Script to run PuLID+FLUX generation from YAML configs.")
    parser.add_argument("--run_yaml", required=True, help="Path to YAML with environment/settings.")
    parser.add_argument("--data_yaml", required=True, help="Path to YAML with prompt & ID image info.")
    parser.add_argument("--output_path", default="results", help="Path to save the generated image.")
    args = parser.parse_args()

    # 1) Load environment parameters
    with open(args.run_yaml, "r") as f:
        run_cfg = yaml.safe_load(f)

    # 2) Load job parameters
    with open(args.data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Unpack environment config
    version = run_cfg.get("version", "v0.9.1")
    model_name = run_cfg.get("model_name", "flux-dev")
    device = run_cfg.get("device", "cuda")
    offload = run_cfg.get("offload", False)
    aggressive_offload = run_cfg.get("aggressive_offload", False)
    fp8 = run_cfg.get("fp8", False)
    onnx_provider = run_cfg.get("onnx_provider", "gpu")
    pretrained_model = run_cfg.get("pretrained_model", None)
    t5_max_length = run_cfg.get("t5_max_length", 128)

    num_steps = run_cfg.get("num_steps", 20)
    guidance = run_cfg.get("guidance", 4.0)
    start_step = run_cfg.get("start_step", 0)
    id_weight = run_cfg.get("id_weight", 1.0)
    height = run_cfg.get("height", 512)
    width = run_cfg.get("width", 512)
    true_cfg = run_cfg.get("true_cfg", 1.0)
    timestep_to_start_cfg = run_cfg.get("timestep_to_start_cfg", 1)
    seed = run_cfg.get("seed", -1)
    neg_prompt = run_cfg.get("neg_prompt", "")
    gamma = run_cfg.get("gamma", 0.5)
    eta = run_cfg.get("eta", 0.9)
    s = run_cfg.get("s", 0)
    tau = run_cfg.get("tau", 25)
    use_ipa = run_cfg.get("use_ipa", True)
    perform_inversion = run_cfg.get("perform_inversion", True)
    save_inversion = run_cfg.get("save_inversion", False)
    perform_reconstruction = run_cfg.get("perform_reconstruction", True)
    perform_editing = run_cfg.get("perform_editing", True)

    # Unpack job config
    prompt = data_cfg["prompt"]
    id_image_path = data_cfg.get("id_image", None)
    image_path = id_image_path

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # Save config
    with open(output_path / "data_config.yaml", "w") as f:
        yaml.dump(data_cfg, f)
        print(f"Data config saved to: {output_path / 'data_config.yaml'}")
    with open(output_path / "run_config.yaml", "w") as f:
        yaml.dump(run_cfg, f)
        print(f"Run config saved to: {output_path / 'run_config.yaml'}")
    # Save original image
    if image_path is not None:
        img = Image.open(image_path)
        img.save(output_path / "original.jpg")
        print(f"Original image saved to: {output_path / 'original.jpg'}")

    # 3) Create generator
    generator = FluxGenerator(
        model_name=model_name,
        device=device,
        offload=offload,
        aggressive_offload=aggressive_offload,
        fp8=fp8,
        onnx_provider=onnx_provider,
        pulid_pretrained_path=pretrained_model,
        pulid_version=version,
    )

    # 4) Generate image
    generated_img, inverted, recon, used_seed = generator.generate_image(
        prompt=prompt,
        id_image_path=id_image_path if use_ipa else None,
        width=width,
        height=height,
        num_steps=num_steps,
        start_step=start_step,
        guidance=guidance,
        seed=seed,
        id_weight=id_weight,
        neg_prompt=neg_prompt,
        true_cfg=true_cfg,
        timestep_to_start_cfg=timestep_to_start_cfg,
        max_sequence_length=t5_max_length,
        image_path=image_path,
        gamma=gamma,
        eta=eta,
        s=s,
        tau=tau,
        perform_inversion=perform_inversion,
        perform_reconstruction=perform_reconstruction,
        perform_editing=perform_editing,
    )

    # 5) Save output
    prompt = prompt.replace(" ", "_")
    if generated_img is not None:
        generated_img.save(output_path / f'{prompt}.jpg')
        print(f"Generated image saved to: {output_path / f'{prompt}.jpg'}")
    if inverted is not None and save_inversion:
        inverted.save(output_path / "inverted.jpg")
        print(f"Inverted image saved to: {output_path / 'inverted.jpg'}")
    if recon is not None:
        recon.save(output_path / "reconstruction.jpg")
        print(f"Reconstructed image saved to: {output_path / 'reconstruction.jpg'}")
    print(f"Used seed: {used_seed}")
    find_and_plot_images(output_path, output_path / "results.jpg", recursive=False)


if __name__ == "__main__":
    main()