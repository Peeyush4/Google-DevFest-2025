import chainlit as cl
from litellm import completion
from dotenv import load_dotenv
import os
load_dotenv()


# MODEL = "ollama_chat/llama3"
MODEL = "huggingface/meta-llama/Llama-3.3-70B-Instruct"
API_KEY = os.getenv("API_KEY")


@cl.on_message
async def on_message(message: cl.Message) -> None:
  """
  Receive a message from the user, send it to the LLM, and send a generated reply
  message back to the user.

  Args:
      message (cl.Message): The message from the user

  Returns:
      None
  """

  msg = cl.Message(content="")
  await msg.send()

  response = completion(
    model=MODEL,
    messages=[
      {
        "content": message.content,
        "role": "user"
      }
    ],
    stream=True,
    api_key=API_KEY,
  )

  for chunk in response:
    if token := chunk.choices[0].delta.content:
      await msg.stream_token(token)

  await msg.update()