import json
import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str
    ae_path: str
    repo_id: str
    repo_flow: str
    repo_ae: str


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path='models/flux1-dev.safetensors',
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path='models/ae.safetensors',
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        not os.path.exists(ckpt_path)
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow, local_dir='models')

# Initialize the model on the 'meta' device, which doesn't allocate real memory
    with torch.device('meta'):
        model = Flux(configs[name].params)
    model = model.to_empty(device=device)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # Load the state dictionary directly to the desired device
        sd = load_sft(ckpt_path, device=str(device))
        # Load the state dictionary into the model
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print_load_warning(missing, unexpected)
    model.to(torch.bfloat16)
    return model

# from XLabs-AI https://github.com/XLabs-AI/x-flux/blob/1f8ef54972105ad9062be69fe6b7f841bce02a08/src/flux/util.py#L330
def load_flow_model_quintized(name: str, device: str = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = 'models/flux-dev-fp8.safetensors'
    if (
        not os.path.exists(ckpt_path)
        and hf_download
    ):
        print("Downloading model")
        ckpt_path = hf_hub_download("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
        print("Model downloaded to", ckpt_path)
    json_path = hf_hub_download("XLabs-AI/flux-dev-fp8", 'flux_dev_quantization_map.json')

    model = Flux(configs[name].params).to(torch.bfloat16)
def load_flow_model_quintized(
    name: str,
    device: str = "cuda",
    hf_download: bool = True,
    cache_path: str = None,
):
    """
    Loads (or downloads) a FLUX-fp8 checkpoint, performs quantization once,
    and caches the quantized model to disk. Future calls load from cache.
    
    :param name: model name key in configs (e.g. "flux-dev-fp8")
    :param device: Torch device string ("cuda" or "cpu")
    :param hf_download: Whether to download from HF if local ckpt is missing
    :param cache_path: Filepath for cached quantized model
    :return: A quantized FLUX model on the specified device.
    """
    if cache_path is None:
        cache_path = os.path.join(os.path.expanduser("~"), ".cache/flux_dev_fp8_quantized_model.pth")



    # 1) Check if we already have a cached, quantized model
    if os.path.exists(cache_path):
        print(f"Loading cached quantized model from '{cache_path}'...")
        model = torch.load(cache_path, map_location=device)
        return model.to(device)

    # 2) If no cache, build and quantize for the first time.
    print("No cached model found. Initializing + quantizing from scratch.")

    # (A) Download or specify checkpoint paths
    ckpt_path = "models/flux-dev-fp8.safetensors"
    if not os.path.exists(ckpt_path) and hf_download:
        print("Downloading model checkpoint from HF...")
        ckpt_path = hf_hub_download("XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors")
        print("Model downloaded to:", ckpt_path)

    json_path = hf_hub_download("XLabs-AI/flux-dev-fp8", "flux_dev_quantization_map.json")

    # (B) Build the unquantized model
    print("Initializing model in bfloat16...")
    model = Flux(configs[name].params).to(torch.bfloat16)

    # (C) Load the unquantized weights
    print("Loading unquantized checkpoint to CPU...")
    sd = load_sft(ckpt_path, device="cpu")  # CPU load

    # (D) Load quantization map
    with open(json_path, "r") as f:
        quantization_map = json.load(f)

    # (E) Quantize
    print("Starting quantization process...")
    from optimum.quanto import requantize
    requantize(model, sd, quantization_map, device=device)
    print("Quantization complete.")

    # (F) Cache the fully quantized model to disk
    print(f"Saving the quantized model to '{cache_path}'...")
    torch.save(model, cache_path)
    print("Model saved. Future runs will load from cache.")

    return model.to(device)


def load_t5(device: str = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str = "cuda", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        not os.path.exists(ckpt_path)
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae, local_dir='models')

    # Loading the autoencoder
    print("Init AE")
    with torch.device(device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False)
        print_load_warning(missing, unexpected)
    return ae
