from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM
from config import model_dir
import logging

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        
        print("Loading model...")
        self.model = AutoGPTQForCausalLM.from_quantized(f"{model_dir}/", model_basename="Guanaco-7B-GPTQ-4bit-128g.no-act-order", device="cuda:0", use_safetensors=True, use_triton=True)
        
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        
    def predict(self, context, prompt):
        
        return pipeline(f"{context}{prompt}")[0]["generated_text"]
