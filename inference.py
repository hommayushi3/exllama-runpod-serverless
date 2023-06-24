from config import repo_name, model_name, model_basename, max_new_tokens, token_repetition_penalty_max, temperature, top_p, top_k, typical
from huggingface_hub import snapshot_download
import logging, os, glob
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

class Predictor:
    def setup(self):
        # Download model
        model_directory = f"/data/{model_name}"
        snapshot_download(repo_id=repo_name, local_dir=model_directory)
        
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        
        config = ExLlamaConfig(model_config_path)               # create config from config.json
        config.model_path = model_path                          # supply path to model weights file
        
        
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading tokenizer...")
        
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
        
        print("Loading model...")
        
        self.model = ExLlama(config)                                 # create ExLlama instance and load the weights
        
        print("Creating cache...")
        self.cache = ExLlamaCache(self.model)                             # create cache for inference
        
        print("Creating generator...")
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)   # create generator
        # Configure generator
        self.generator.disallow_tokens([self.tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = token_repetition_penalty_max
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.typical = typical
        
    def predict(self, prompt):
        
        return self.generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
