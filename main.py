import requests
import json
import sys
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
url = "https://api.donovan.scale.com/v1/chat"

user_request = "What is the world 'plata' in english"

payload = {
    "modelType": "COHERE_CHAT",
    "text": user_request,
    "workspace": "atrjw4p7vy9zi9p7kov0a40v",
    "thread": "thread"
}

payload2 = {
    "modelType": "SCALE_VICUNA_13B",
    "text": user_request,
    "workspace": "atrjw4p7vy9zi9p7kov0a40v",
    "thread": "thread"
}

payload3 = {
    "modelType": "SCALE_FALCON_40B_INSTRUCT",
    "text": user_request,
    "workspace": "atrjw4p7vy9zi9p7kov0a40v",
    "thread": "thread"
}

payload4 = {
    "modelType": "SCALE_MPT_7B_CHAT",
    "text": user_request,
    "workspace": "atrjw4p7vy9zi9p7kov0a40v",
    "thread": "thread"
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Basic bGl2ZV84NzdjZmYzZmMxZmQ0ZGMwOWFiYTZjNDViNGFlMTUwNTo="
}

#file_list = ['gpt-3.5-out.json','gpt-4-out.json']
#model_list = ['OPENAI_GPT3.5-TURBO', 'OPENAI_GPT4']


def write_outfile(filename, input):
  original_stdout = sys.stdout
  with open(filename, 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(input)
    sys.stdout = original_stdout
  
if __name__ == "__main__":

  response = requests.post(url, json=payload, headers=headers)
  response2 = requests.post(url, json=payload2, headers=headers)
  response3 = requests.post(url, json=payload3, headers=headers)
  response4 = requests.post(url, json=payload4, headers=headers)


  print("=========Dumping Request into a JSON FILE============\n")


  write_outfile('COHERE_CHAT.json', response.text)
  write_outfile('SCALE_VICUNA_13B.json', response2.text)
  write_outfile('FALCON-OUT.json', response3.text)
  write_outfile('MPT-OUT.json', response4.text)

  print(response.text)

  print("\n")

  print(response2.text)

  print("=========SUCCESSFULY DUMPED REQUEST============")

  