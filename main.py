import textbase
from textbase.message import Message
from textbase import models
import json
import os
from typing import List, Tuple, Union, Dict
import requests
from textbase.trail import Trail

models.OpenAI.api_key = os.getenv("OPENAI_KEY")

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = """
You are an AI bike trail expert. You need to help users plan their bike trail journeys.
The user will ask you about the available bike trails in an area.
If the user asks anything not related to bike trails, you must tell the user to ask anything relavant to bike trails.
If you do not know the answer to the user's question, politely say that you don't know and ask the user if he or she would want help in any other bike trail information.

In the beginning, introduce yourself by telling that you can help find the best bike trails in the world and urge the user to ask anything abpout bike trails.
"""




@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history,
        model="gpt-3.5-turbo",
    )

    return bot_response, state
