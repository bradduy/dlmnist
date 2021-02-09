import requests

response = requests.post("http://0.0.0.0:5000/predict", files= {'file': open('./DL/1.png', 'rb')})

print(response.text)