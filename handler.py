''' infer.py for runpod worker '''

import os
import inference

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schema import INPUT_SCHEMA


MODEL = inference.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    '''
    job_input = job['input']
    print(f"Job: {job_input}")
    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    print(f"Validated: {validated_input}")
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    result = MODEL.predict(
        context=validated_input["context"],
        prompt=validated_input["prompt"]
    )
    print(f"Result: {result}")

    job_output = {
            "result": result
        }

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


runpod.serverless.start({"handler": run})