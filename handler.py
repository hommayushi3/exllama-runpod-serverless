from model import ExLlama, ExLlamaCache, ExLlamaConfig
from lora import ExLlamaLora
from tokenizer import ExLlamaTokenizer
from alt_generator import ExLlamaAltGenerator
import torch
import model_init
from typing import Dict, Union, Generator, Any, List
from huggingface_hub import snapshot_download
from settings import ConfigSettings, DefaultExLlamaAltGeneratorSamplingSettings, \
                     DefaultExLlamaAltGeneratorStoppingSettings, ModelSettings
import runpod
import logging
import warnings
import inspect

logging.basicConfig(level=logging.INFO)


config: ExLlamaConfig                                                  # Config for the model, loaded from config.json
model: ExLlama                                                         # Model, initialized with ExLlamaConfig
cache: ExLlamaCache                                                    # Cache for generation, tied to model
generator: ExLlamaAltGenerator = None                                  # Generator
tokenizer: ExLlamaTokenizer                                            # Tokenizer
model_settings: ModelSettings                                          # Model settings
args: ConfigSettings                                                   # Configuration settings
default_sampling_settings: DefaultExLlamaAltGeneratorSamplingSettings  # Default sampling settings for generator


# To validate arguments to inference function
BEGIN_STREAM_ARGS = {arg for arg in inspect.getfullargspec(ExLlamaAltGenerator.begin_stream).args if arg not in ["self", "gen_settings"]}
GENERATE_ARGS = {arg for arg in inspect.getfullargspec(ExLlamaAltGenerator.generate).args if arg not in ["self", "gen_settings"]}
SETTINGS_ARGS = {arg for arg in inspect.getfullargspec(ExLlamaAltGenerator.Settings).args if arg not in ["self"]}


def load_model():
    """Based on init_args from https://github.com/turboderp/exllama/blob/master/example_alt_generator.py, but removes LoRA."""
    global model, cache, config, generator, tokenizer, default_sampling_settings

    # Global initialization
    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()

    model_settings = ModelSettings()
    args = ConfigSettings()
    args.directory = snapshot_download(repo_id=model_settings.repo_id, revision=model_settings.revision)
    model_init.post_parse(args)
    model_init.get_model_files(args)

    print_opts = []
    model_init.print_options(args, print_opts)

    # Model globals
    model_init.set_globals(args)

    # Instantiate model and generator
    config = model_init.make_config(args)

    model = ExLlama(config)
    cache = ExLlamaCache(model)
    tokenizer = ExLlamaTokenizer(args.tokenizer)

    model_init.print_stats(model)

    # Generator
    generator = ExLlamaAltGenerator(model, tokenizer, cache)
    default_sampling_settings = DefaultExLlamaAltGeneratorSamplingSettings()
    default_stopping_settings = DefaultExLlamaAltGeneratorStoppingSettings()
    for key, value in default_stopping_settings.dict().items():
        try:
            setattr(generator.settings, key, value)
        except AttributeError:
            warnings.warn(f"Could not set {key} to {value} in generator settings.")

    for key, value in default_stopping_settings.dict().items():
        try:
            setattr(generator, key, value)
        except AttributeError:
            warnings.warn(f"Could not set {key} to {value} in generator settings.")


def validate_arguments(kwargs: Dict[str, Any], expected_args: List[str], function_name: str):
    """Validate arguments passed to function."""
    for key in list(kwargs.keys()):
        if key not in expected_args:
            warnings.warn(f"Unknown argument {key} for {function_name}. Ignoring.")
            kwargs.pop(key)


def inference(event) -> Union[str, Generator[str, None, None]]:
    logging.info(event)
    job_input = event["input"]
    if generator is None:
        load_model()
    if not job_input:
        raise ValueError("No input provided.")
    if "prompt" not in job_input:
        raise ValueError("No prompt provided.")

    sampling_params = job_input.get("sampling_params", {})
    validate_arguments(sampling_params, SETTINGS_ARGS, "sampling_params")
    gen_settings = ExLlamaAltGenerator.Settings(**sampling_params)
    if job_input.get("stream", False):
        validate_arguments(job_input, BEGIN_STREAM_ARGS, "begin_stream")
        output = generator.begin_stream(
            gen_settings = gen_settings,
            **job_input
        )
        while True:
            chunk, eos = output.stream()
            yield chunk
            if eos: break
    else:  # batched prompts
        validate_arguments(job_input, GENERATE_ARGS, "generate")
        output = generator.generate(
            gen_settings = gen_settings,
            **job_input
        )
        yield output


runpod.serverless.start({"handler": inference, "return_aggregate_stream": True})
