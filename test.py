import requests

response = requests.post("http://127.0.0.1:5000/predict", files= {'file': open('../DL/1.png', 'rb')})

print(response.text)