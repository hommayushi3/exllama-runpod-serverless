from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM
from config import model_dir, model_basename
import logging

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        
        print("Loading model...")
        self.model = AutoGPTQForCausalLM.from_quantized(f"{model_dir}/", model_basename=model_basename, device="cuda:0", use_safetensors=True, use_triton=True)
        
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        
    def predict(self, context, prompt):
        
        return self.pipeline(f"{context}{prompt}")[0]["generated_text"]
