from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional


class ModelSettings(BaseSettings):
    """Model name and revision on huggingface.co"""
    repo_id: str = Field(None, description="Model name on huggingface.co", required=True)
    revision: str = Field("main", description="Model revision on huggingface.co", required=True)


class ConfigSettings(BaseSettings):
    """Settings class to hold configuration values. See https://github.com/turboderp/exllama/blob/master/model_init.py args for details."""
    # Do not need to set these with env variables
    directory: Optional[str] = Field(default=None, env=False, description="Directory to find model/tokenizer/config files in. Set by snapshot_download.")
    # Optional to specify.
    model: Optional[str] = Field(None, description="Model filename in directory. Only specify if exists multiple .safetensors files.", required=True)
    tokenizer: str = Field("tokenizer.model", description="Tokenizer filename in directory", required=True)
    config: str = Field("config.json", description="Model config filename in directory", required=True)
    
    # Settable by user environment variables
    gpu_split: Optional[str] = Field(None, description="Comma-separated list of VRAM (in GB) to use per GPU device for model layers")
    length: int = Field(2048, description="Maximum sequence length")
    compress_pos_emb: float = Field(1.0, description="Compression factor for positional embeddings")
    alpha: float = Field(1.0, description="Alpha for context size extension via embedding extension")
    theta: Optional[float] = Field(None, description="Theta (base) for RoPE embeddings")
    
    gpu_peer_fix: bool = Field(False, description="Prevent direct copies of data between GPUs")

    flash_attn: Optional[str] = Field(None, description="Use Flash Attention with specified input length")
    
    matmul_recons_thd: int = Field(8, description="Number of rows at which to use reconstruction and cuBLAS for quant matmul")
    fused_mlp_thd: int = Field(2, description="Maximum number of rows for which to use fused MLP")
    sdp_thd: int = Field(8, description="Number of rows at which to switch to scaled_dot_product_attention")
    matmul_fused_remap: bool = Field(False, description="Fuse column remapping in Q4 matmul kernel")
    no_fused_attn: bool = Field(False, description="Disable fused attention")

    rmsnorm_no_half2: bool = Field(False, description="Don't use half2 in RMS norm kernel")
    rope_no_half2: bool = Field(False, description="Don't use half2 in RoPE kernel")
    matmul_no_half2: bool = Field(False, description="Don't use half2 in Q4 matmul kernel")
    silu_no_half2: bool = Field(False, description="Don't use half2 in SiLU kernel")
    no_half2: bool = Field(False, description="Disable half2 in all kernels")
    force_half2: bool = Field(False, description="Force enable half2 even if unsupported")

    concurrent_streams: bool = Field(False, description="Use concurrent CUDA streams")
    
    affinity: Optional[str] = Field(None, description="Sets processor core affinity")


class DefaultExLlamaAltGeneratorSamplingSettings(BaseSettings):
    """Sampling Settings. See https://github.com/turboderp/exllama/blob/master/alt_generator.py for details."""
    
    # Sampling Settings
    temperature: float = Field(0.95, description="Temperature for sampling")
    top_k: int = Field(40, description="Consider the most probable top_k samples, 0 to disable top_k sampling")
    top_p: float = Field(0.65, description="Consider tokens up to a cumulative probability of top_p, 0.0 to disable top_p sampling")
    min_p: float = Field(0.0, description="Do not consider tokens with probability less than this")
    typical: float = Field(0.0, description="Locally typical sampling threshold, 0.0 to disable typical sampling")

    token_repetition_penalty_max: float = Field(1.15, description="Repetition penalty for most recent tokens")
    token_repetition_penalty_sustain: int = Field(-1, description="Number of most recent tokens to apply penalty for, -1 to apply to whole context")
    token_repetition_penalty_decay: int = Field(0, description="Gradually decrease penalty over this many tokens")

    disallowed_tokens: Optional[List[int]] = Field(None, description="List of tokens to inhibit, e.g. tokenizer.eos_token_id")
    

class DefaultExLlamaAltGeneratorStoppingSettings(BaseSettings):
    """Stopping Settings. See https://github.com/turboderp/exllama/blob/master/alt_generator.py for details."""
    stop_strings: List[str] = Field([], description="List of stop strings")
    stop_tokens: List[int] = Field([], description="List of stop tokens")
    held_text: str = Field("", description="Held text")
    max_stop_tokens: int = Field(2, description="Maximum number of stop tokens")
    remaining_tokens: int = Field(0, description="Remaining tokens for generation")
