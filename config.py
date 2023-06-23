from auto_gptq import BaseQuantizeConfig

model_dir = "guanaco-7B-GPTQ"

quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=1,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )