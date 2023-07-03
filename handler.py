from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import logging
from typing import Generator
import runpod
from huggingface_hub import snapshot_download
from copy import copy

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

def load_model():
    global generator, default_settings

    if not generator:
        model_directory = snapshot_download(repo_id=os.environ["MODEL_REPO"])
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator
        config = ExLlamaConfig(model_config_path)               # create config from config.json
        config.model_path = model_path                          # supply path to model weights file

        model = ExLlama(config)                                 # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)                             # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
        default_settings = {
            k: getattr(generator.settings, k) for k in dir(generator.settings) if k[:2] != '__'
        }
    return generator, default_settings

generator = None
default_settings = None
prompt_prefix = os.getenv("PROMPT_PREFIX", "")
prompt_suffix = os.getenv("PROMPT_SUFFIX", "")

def inference(event) -> str:
    logging.info(event)
    job_input = event["input"]
    prompt: str = prompt_prefix + job_input.pop("prompt") + prompt_suffix
    max_new_tokens = job_input.pop("max_new_tokens", 50)

    generator, default_settings = load_model()

    settings = copy(default_settings)
    settings.update(job_input)
    for key, value in settings.items():
        setattr(generator.settings, key, value)

    output = generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
    return output[len(prompt):]

runpod.serverless.start({"handler": inference})
