import requests
url = 'https://dlmnist.herokuapp.com/'
response = requests.post(url + "predict", files= {'file': open('./DL/1.png', 'rb')})

print(response.text)