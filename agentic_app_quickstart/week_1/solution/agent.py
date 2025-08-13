import os
from agents import Agent, Runner, set_tracing_disabled, function_tool
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import pandas as pd
import asyncio

load_dotenv()
set_tracing_disabled(True)

# In-memory storage for loaded dataframe
loaded_df = None

def get_client():
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_ENDPOINT")
    )

def get_model():
    model = OpenAIChatCompletionsModel(
        model = "gpt-4.1",
        openai_client=get_client()
    )
    return model

# CSV Data Loader
@function_tool()
def load_csv(file_path):
    """
    A tool function that reads the csv file and set the global variable.
    """
    try:
        df = pd.read_csv(file_path)
        loaded_df = df
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as ex:
        raise Exception(f"Error loading CSV: {file_path} ")

@function_tool
def calculate_column_average(column_name: str) -> float:
    """
    A tool function that returns the computes the average or mean of a column data.

    Returns:
        float: Average value of a numeric column
    """
    global loaded_df
    if loaded_df is None:
        raise ValueError("No CSV loaded yet.")
    if column_name not in loaded_df.columns:
        raise ValueError("Column name not found in the loaded dataframe from CSV")
    if not pd.api.types.is_numeric_dtype(loaded_df[column_name]):
        raise ValueError(f"Column {column_name} is not numeric")
    return loaded_df[column_name].mean()

agent = Agent(
    name="Helpful CSV Assistant",
    instructions="You are a helpful AI assistant that answers questions from the loaded csv.",
    model=get_model(),  # Specify GPT-5 as the model,
    tools=[load_csv, calculate_column_average]
)


async def process():
    """
    Main function that runs the agent conversation.

    The Runner.run() method:
    - Takes a starting agent and user input
    - Handles the conversation flow between user and agent
    - Returns the agent's final response
    """
    # Run the agent with user input asking for a language
    user_query = input("Enter your query: ")
    result = await Runner.run(starting_agent=agent, input=user_query)
    # Print the agent's final response
    print(result.final_output)



