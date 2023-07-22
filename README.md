# exllama-runpod-serverless

> [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) [GPTQ](https://arxiv.org/abs/2210.17323) models with fast [ExLlama](https://github.com/turboderp/exllama) inference on [RunPod Serverless GPUs](https://www.runpod.io/serverless-gpu)

## Summary
This Docker image runs a Llama model on a serverless RunPod instance using the optimized [turboderp's exllama repo](https://github.com/turboderp/exllama).

## Set Up
1. Create a RunPod account and navigate to the [RunPod Serverless Console](https://www.runpod.io/console/serverless).
2. Navigate to `My Templates` and click on the `New Template` button.
3. Enter in the following fields and click on the `Save Template` button:

    | Template Field | Value |
    | --- | --- |
    | Template Name | `exllama-runpod-serverless` |
    | Container Image | `hommayushi3/exllama-runpod-serverless:latest` |
    | Container Disk | A size large enough to store your libraries + your desired model in 4bit. |

    - Container Disk Size Guide:
        | Model Parameters | Storage & VRAM |
        | --- | --- |
        | 7B | 6GB |
        | 13B | 9GB |
        | 33B | 19GB |
        | 65B | 35GB |

    - Environment Variables:

        | Environment Variable | Example Value |
        | --- | --- |
        | (Required) `MODEL_REPO` | `TheBloke/airoboros-7B-gpt4-1.4-GPTQ` or any other repo for GPTQ Llama model. See https://huggingface.co/models?other=llama&sort=trending&search=thebloke+gptq for other models. Must have `.safetensors` file(s). |
        | (Optional) `PROMPT_PREFIX` | `"USER: "` |
        | (Optional) `PROMPT_SUFFIX` | `"ASSISTANT: "` |

4. Now click on `My Endpoints` and click on the `New Endpoint` button.
5. Fill in the following fields and click on the `Create` button:
    | Endpoint Field | Value |
    | --- | --- |
    | Endpoint Name | `exllama-runpod-serverless` |
    | Select Template | `exllama-runpod-serverless` |
    | Min Provisioned Workers | `0` |
    | Max Workers | `1` |
    | Idle Timeout | `5` seconds |
    | FlashBoot | Checked/Enabled |
    | GPU Type(s) | Use the `Container Disk` section of step 3 to determine the smallest GPU that can load the entire 4 bit model. In our example's case, use 16 GB GPU. |

## Inference Usage
See the `predict.py` file for an example. For convenience we also copy the code below.

```py
import os
import requests
from time import sleep
import logging

endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
URI = f"https://api.runpod.ai/v2/{endpoint_id}/run"


def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 500,
        'temperature': 0.3,
        'top_k': 50,
        'top_p': 0.7,
        'repetition_penalty': 1.2,
        'batch_size': 8,
        'stop': ['</s>']
    }

    response = requests.post(URI, json=dict(input=request), headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    })

    if response.status_code == 200:
        data = response.json()
        task_id = data.get('id')
        return stream_output(task_id)


def stream_output(task_id):
    try:
        url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
        }

        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'COMPLETED':
                    return data['output']
            elif response.status_code >= 400:
                logging.error(response.json())
            # Sleep for 3 seconds between each request
            sleep(2)
    except Exception as e:
        print(e)
    

def cancel_task(task_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }
    response = requests.get(url, headers=headers)
    return response


if __name__ == '__main__':
    prompt = """Given the following clinical notes, what tests, diagnoses, and recommendations should the I give? Provide your answer as a detailed report with labeled sections "Diagnostic Tests", "Possible Diagnoses", and "Patient Recommendations".

17-year-old male, has come to the student health clinic complaining of heart pounding. Mr. Cleveland's mother has given verbal consent for a history, physical examination, and treatment
-began 2-3 months ago,sudden,intermittent for 2 days(lasting 3-4 min),worsening,non-allev/aggrav
-associated with dispnea on exersion and rest,stressed out about school
-reports fe feels like his heart is jumping out of his chest
-ros:denies chest pain,dyaphoresis,wt loss,chills,fever,nausea,vomiting,pedal edeam
-pmh:non,meds :aderol (from a friend),nkda
-fh:father had MI recently,mother has thyroid dz
-sh:non-smoker,mariguana 5-6 months ago,3 beers on the weekend, basketball at school
-sh:no std,no other significant medical conditions."""
    print(run(prompt))
```

Run the above code using the following command in terminal with the runpoint endpoint id assigned to your endpoint in step 5.
```bash
RUNPOD_AI_API_KEY='**************' RUNPOD_ENDPOINT_ID='*******' python predict.py
```
You can generate the API key [here](https://www.runpod.io/console/serverless/user/settings) under API Keys.
