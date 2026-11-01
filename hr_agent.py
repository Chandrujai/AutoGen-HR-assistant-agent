
import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# --------------------------------
# 1. Load API key from local .env
# --------------------------------

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in the local .env file!"
    )


# --------------------------------
# 2. Create async main function
# --------------------------------

async def main():

    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key
    )

    try:

        # Create AssistantAgent
        hr_assistant = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant."
        )

        # Ask the agent for a job description
        task = (
            "Write a 5-line job description for a "
            "Junior Python Developer."
        )

        # Run the agent and display the response
        await Console(
            hr_assistant.run_stream(task=task)
        )

    finally:

        # Close model client
        await model_client.close()


# --------------------------------
# 3. Run async application
# --------------------------------

if __name__ == "__main__":
    asyncio.run(main())
