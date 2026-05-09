import requests
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer YOUR_API_KEY_HERE"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.status_code, response.text

print(query({"inputs": "test"}))
