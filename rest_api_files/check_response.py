import requests

url = 'http://127.0.0.1:5000/chatbot'
data = {'prompt': 'Hey, can you help me with something?', 'memory': None}
response = requests.post(url, json=data)

print(response.json())