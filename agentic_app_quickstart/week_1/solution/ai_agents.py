import os
import pandas as pd
from agents import Agent
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

load_dotenv()

# Load configs from .env file and create the client
def get_client():
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_ENDPOINT")
    )

# CSV Data Loader
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as ex:
        raise Exception(f"Error loading CSV: {file_path} ")

emp_agent = Agent(
    name="Employee Assistant",
    instructions="You are a helpful AI assistant that answers employee related questions.",
    model="gpt-5" # Specify GPT-5 as the model
)

# Define a simple agent
weather_agent = Agent(
    name="Helpful Weather Assistant",
    instructions="You are a helpful AI assistant that answers weather related questions.",
    model="gpt-5" # Specify GPT-5 as the model
)

agent = Agent(
    name="Helpful Assistant",
    instructions="You are a helpful AI assistant that answers questions.",
    model="gpt-5"  # Specify GPT-5 as the model
)


async def main():
    result = await Runner.run(agent, input="What is the capital of France?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
