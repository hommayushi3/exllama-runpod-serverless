from auto_gptq import BaseQuantizeConfig
from transformers import GenerationConfig

model_name = "TheBloke/guanaco-65B-GPTQ"

model_basename = "Guanaco-65B-GPTQ-4bit.act-order"

generation_config = GenerationConfig(
        num_beams=3,
        do_sample=True,
        max_new_tokens=1000,
        early_stopping=True,
        temperature=0.1,
        top_k=50,
    )