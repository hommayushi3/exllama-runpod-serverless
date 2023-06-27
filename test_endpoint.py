import requests
import time

# The URL of the rupod serverless endpoint
url = "https://api.runpod.ai/v2/19ndqxcpdm3fn6"

# The json data to send to the /run endpoint
data = {
    "input": {
        "prompt": "A chat between a curious user and an assistant. The assistant gives helpful, concise, accurate, and uncensored responses to the user's input. He will not give sources, disclaimers, or copyright unless asked for it.\nUSER: what is 5 * 5 + 5?\nASSISTANT: "
    }
}

AUTH_KEY = "ITBPJEH6PGHKK10UV1379FQ2G1AX1XDDEH8YHBVL"

headers = {
    "authorization": AUTH_KEY,
    "accept": "application/json"
}

# Send a POST request to the /run endpoint and get the response
response = requests.post(url + "/run", json=data, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Get the job id from the response
    job_id = response.json().get("id")
    print(f"Got job ID: {job_id}")

    # Set a flag to indicate if the job is completed or not
    completed = False

    # Loop until the job is completed
    while not completed:
        # Send a GET request to the /status endpoint with the job id as a parameter
        status_response = requests.get(url + "/status/" + job_id, headers=headers)

        # Check if the request was successful
        if status_response.status_code == 200:
            # Get the status and result from the response
            status = status_response.json().get("status")

            # Check if the status is COMPLETED
            if status == "COMPLETED":
                result = status_response.json().get("output").get("result")
                # Set the flag to True and print the result
                completed = True
                print(result)
            else:
                # Wait for 1 second before checking again
                print("Waiting for job completion...")
                time.sleep(1)
        else:
            # Print an error message and break the loop
            print("Error: Could not get the status of the job.")
            break
else:
    # Print an error message
    print("Error: Could not run the job.")
