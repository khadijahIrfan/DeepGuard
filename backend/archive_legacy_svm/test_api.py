import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "text": "Scientists discovered a new vaccine that shows promising results."
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)