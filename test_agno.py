from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat

load_dotenv()

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant.",
    markdown=True,
)
agent.print_response("Say hello and confirm you are working!", stream=True)
