import chainlit as cl
from litellm import completion
from chainlit.input_widget import Select, Slider
from dotenv import load_dotenv
import os
load_dotenv()


# LLAMA_MODEL = "ollama_chat/llama3"
LLAMA_MODEL = "huggingface/meta-llama/Llama-3.3-70B-Instruct"
DEEPSEEK_MODEL = "huggingface/together/deepseek-ai/DeepSeek-R1"
API_KEY = os.getenv("API_KEY")

@cl.on_chat_start
async def on_chat_start():
    """
    Let the user choose between various model options.

    Returns:
        None
    """
    # Set default settings
    default_settings = {
        "Model": LLAMA_MODEL,
        "Streaming": True,
        "Temperature": 1
    }
    cl.user_session.set("chat_history", [])
    cl.user_session.set("settings", default_settings)
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Choose between Llama3 and DeepSeek-R1",
                values=[LLAMA_MODEL, DEEPSEEK_MODEL],
                initial_value=LLAMA_MODEL,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """ Update the user session with the new settings """
    cl.user_session.set("settings", settings)


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

    settings = cl.user_session.get("settings")
    chat_history = cl.user_session.get("chat_history")

    chat_history.append(
        {
        "role": "user",
        "content": message.content,
        }
    )

    msg = cl.Message(content="")
    await msg.send()

    response = completion(
        model=f"{settings['Model']}",
        messages=chat_history,
        stream=True,
        api_key=API_KEY,
    )

    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            await msg.stream_token(token)

    chat_history.append({"role": "assistant", "content": msg.content})

    await msg.update()