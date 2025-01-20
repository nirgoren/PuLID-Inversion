import argparse
import time
import numpy as np
import yaml
import torch
from einops import rearrange
from PIL import Image
import os

# -------------------------------------------------------------------
# The following imports must match your environment or local modules:
# If you have local modules named similarly to the original snippet,
# adjust as appropriate:
# -------------------------------------------------------------------
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
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

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()

        print(f"Generating prompt: '{opts.prompt}' (seed={opts.seed})...")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-6

        # 1) Prepare input noise
        x = get_noise(
            num_samples=1,
            height=opts.height,
            width=opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(opts.num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        # 2) Prepare text embeddings
        if self.offload:
            self.t5 = self.t5.to(self.device)
            self.clip_model = self.clip_model.to(self.device)

        inp = prepare(t5=self.t5, clip=self.clip_model, img=x, prompt=opts.prompt)
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
            from PIL import Image
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

        # 4) Denoise
        x = denoise(
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
        )

        # Offload flux model, load auto-decoder
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # 5) Decode latents
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.2f} seconds.")

        # Convert to PIL
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, opts.seed


def main():
    parser = argparse.ArgumentParser("Script to run PuLID+FLUX generation from YAML configs.")
    parser.add_argument("--env_yaml", required=True, help="Path to YAML with environment/settings.")
    parser.add_argument("--job_yaml", required=True, help="Path to YAML with prompt & ID image info.")
    args = parser.parse_args()

    # 1) Load environment parameters
    with open(args.env_yaml, "r") as f:
        env_cfg = yaml.safe_load(f)

    # 2) Load job parameters
    with open(args.job_yaml, "r") as f:
        job_cfg = yaml.safe_load(f)

    # Unpack environment config
    version = env_cfg.get("version", "v0.9.1")
    model_name = env_cfg.get("model_name", "flux-dev")
    device = env_cfg.get("device", "cuda")
    offload = env_cfg.get("offload", False)
    aggressive_offload = env_cfg.get("aggressive_offload", False)
    fp8 = env_cfg.get("fp8", False)
    onnx_provider = env_cfg.get("onnx_provider", "gpu")
    pretrained_model = env_cfg.get("pretrained_model", None)
    t5_max_length = env_cfg.get("t5_max_length", 128)

    num_steps = env_cfg.get("num_steps", 20)
    guidance = env_cfg.get("guidance", 4.0)
    start_step = env_cfg.get("start_step", 0)
    id_weight = env_cfg.get("id_weight", 1.0)
    height = env_cfg.get("height", 512)
    width = env_cfg.get("width", 512)
    true_cfg = env_cfg.get("true_cfg", 1.0)
    timestep_to_start_cfg = env_cfg.get("timestep_to_start_cfg", 1)
    seed = env_cfg.get("seed", -1)
    neg_prompt = env_cfg.get("neg_prompt", "")

    # Unpack job config
    prompt = job_cfg["prompt"]
    id_image_path = job_cfg.get("id_image", None)
    output_path = job_cfg.get("output_path", "output.jpg")

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
    generated_img, used_seed = generator.generate_image(
        prompt=prompt,
        id_image_path=id_image_path,
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
    )

    # 5) Save output
    generated_img.save(output_path)
    print(f"Image saved to: {output_path}")
    print(f"Used seed: {used_seed}")


if __name__ == "__main__":
    main()