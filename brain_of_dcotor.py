import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)


import  base64

image_path= "acne.jpeg"
image_file=open(image_path,"rb")
encoded_image=base64.b64encode(image_file.read()).decode('utf-8')

from groq import Groq

client=Groq(api_key=GROQ_API_KEY)
query="is there something wrong with my face"
model= "llama-3.2-90b-vision-preview"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            },
        ],
    }]

chat_completion=client.chat.completions.create(
    messages=messages,
    model=model
)

# print(chat_completion)
print(chat_completion.choices[0].message.content)