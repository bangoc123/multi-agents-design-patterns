import asyncio
import random
from typing import Any
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from agents import Agent, AgentHooks, RunContextWrapper, Runner, Tool, function_tool


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    # You can add validation here if needed
    required_vars = ['OPENAI_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")


class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


###


@function_tool
def random_number(max: int) -> int:
    """
    Generate a random number up to the provided maximum.
    """
    return random.randint(0, max)


@function_tool
def multiply_by_two(x: int) -> int:
    """Simple multiplication by two."""
    return x * 2


@function_tool
def is_even(x: int) -> bool:
    """Check if a number is even."""
    return x % 2 == 0


class FinalResult(BaseModel):
    original_number: int
    multiplied_number: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="""
    You have received an even number from the Find Even Agent.
    1. Take the even number provided.
    2. Multiply it by 2 using the multiply_by_two tool.
    3. Return both the original even number and the multiplied result.
    """,
    tools=[multiply_by_two],
    output_type=FinalResult,
    hooks=CustomAgentHooks(display_name="Multiply Agent"),
)

find_even_agent = Agent(
    name="Find Even Agent",
    instructions="""
    Your task is to find an even number and then hand it off to the Multiply Agent.
    
    1. Generate a random number using the random_number tool.
    2. Check if the number is even using the is_even tool.
    3. If the number is odd, generate another random number and repeat until you find an even number.
    4. Once you find an even number, hand off to the Multiply Agent.
    
    Important: Be efficient and don't waste turns. If you get an odd number, immediately try again without unnecessary reasoning.
    """,
    tools=[random_number, is_even],
    output_type=FinalResult,
    handoffs=[multiply_agent],
    hooks=CustomAgentHooks(display_name="Find Even Agent"),
)


async def main() -> None:
    load_env()  # Load environment variables
    user_input = input("Enter a max number: ")
    try:
        result = await Runner.run(
            find_even_agent,
            input=f"Generate random numbers between 0 and {user_input} until you find an even one, then multiply it by 2.",
            max_turns=20,  # Increased max turns
        )
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Try running again with a different input number.")
    finally:
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())