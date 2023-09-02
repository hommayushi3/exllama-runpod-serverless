import os
import requests
from time import sleep
import argparse
import sys
from typing import List, Dict
import time


class ChatPromptFormat:
    def __init__(self, system_prefix: str="", user_prefix: str="USER: ", 
                 assistant_prefix: str="ASSISTANT: ", message_separator: str="\n"):
        self.system_prefix = system_prefix
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.message_separator = message_separator

    def format(self, messages: [str, List[Dict[str, str]]]):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        prompt = ""
        for message in messages:
            prefix = ""
            if message['role'].lower() in ['system', 'system_prompt']:
                prefix = self.system_prefix
            elif message['role'].lower() in ['user', 'human']:
                prefix = self.user_prefix
            elif message['role'].lower() in ['assistant', 'gpt', 'bot']:
                prefix = self.assistant_prefix
            else:
                raise ValueError(f"Unknown role {message['role']}")
            prompt += prefix + message['content'] + self.message_separator
        prompt += self.assistant_prefix
        return prompt


class RunpodClient:
    def __init__(self, endpoint_id: str, system_prefix: str="", user_prefix: str="USER: ", 
                 assistant_prefix: str="ASSISTANT: ", message_separator: str="\n"):
        self.endpoint_id = endpoint_id
        self.run_uri = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.stream_uri = f"https://api.runpod.ai/v2/{endpoint_id}/stream/{{}}"
        self.headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
        }
        self.chat_prompt_format = ChatPromptFormat(system_prefix, user_prefix, assistant_prefix, message_separator)

    def chat(self, messages, sampling_params={}, stream=False):
        prompt = self.chat_prompt_format.format(messages)
        self.run(prompt, sampling_params, stream)
    
    def completion(self, prompt, sampling_params={}, stream=False):
        self.run(prompt, sampling_params, stream)
    
    def run(self, prompt, sampling_params={}, stream=False, **kwargs):
        input = {
            "prompt": prompt,
            "stream": stream,
            "sampling_params": sampling_params
        }
        input.update(kwargs)
        print(input)
        json_input = {"input": input}
        response = requests.post(self.run_uri, json=json_input, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('id')
            for _ in self.stream_output(task_id, stream=stream):
                pass

    def stream_output(self, task_id, stream=False):
        previous_output = ''
        while True:
            response = requests.get(self.stream_uri.format(task_id), headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                print(data)
                if len(data['stream']) > 0:
                    new_output = data['stream'][0]['output']

                    if stream:
                        sys.stdout.write(new_output)
                        sys.stdout.flush()
                    previous_output = new_output
                
                if data.get('status') == 'COMPLETED':
                    if not stream:
                        return previous_output
                    break
            elif response.status_code >= 400:
                ValueError(f"Error in stream output: {response}")
            # Sleep for 0.1 seconds between each request
            sleep(0.2 if stream else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runpod AI CLI')
    parser.add_argument('-s', '--stream', action='store_true', help='Stream output')

    client = RunpodClient(
        endpoint_id=os.environ['RUNPOD_ENDPOINT_ID'],
        system_prefix="### System Prompt\n",
        user_prefix="### User Message\n",
        assistant_prefix="### Assistant\n",
        message_separator="\n\n"
    )
    
    args = parser.parse_args()
    start = time.time()

    prompt = "What is the capital of New York?"
    print(client.chat(messages=prompt, sampling_params={"temperature": 0.01}, stream=args.stream))
    print("Time taken: ", time.time() - start, " seconds")
