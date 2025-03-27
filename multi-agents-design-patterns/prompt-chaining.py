import asyncio
from typing import Any, Optional
from pydantic import BaseModel

from agents import Agent, AgentHooks, RunContextWrapper, Runner, Tool, function_tool
from dotenv import load_dotenv

load_dotenv()

class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}")

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}")


class MarketingCopy(BaseModel):
    headline: str
    body: str
    call_to_action: str


class TranslatedCopy(BaseModel):
    original: MarketingCopy
    translated_headline: str
    translated_body: str
    translated_call_to_action: str


class ValidationResult(BaseModel):
    is_valid: bool
    feedback: str


@function_tool
def validate_marketing_copy(copy: MarketingCopy) -> ValidationResult:
    """
    Validate marketing copy against common criteria.
    """
    feedback = []
    is_valid = True

    # Check headline length
    if len(copy.headline) > 60:
        feedback.append("Headline is too long (should be under 60 characters)")
        is_valid = False

    # Check body length
    if len(copy.body) < 100:
        feedback.append("Body is too short (should be at least 100 characters)")
        is_valid = False

    # Check call to action
    if not any(cta in copy.call_to_action.lower() for cta in ["buy", "learn", "try", "get", "start"]):
        feedback.append("Call to action should include action words like 'buy', 'learn', 'try', etc.")
        is_valid = False

    return ValidationResult(
        is_valid=is_valid,
        feedback="\n".join(feedback) if feedback else "All checks passed!"
    )


@function_tool
def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to target language.
    """
    # This is a mock translation function
    # In a real implementation, you would use a translation API
    translations = {
        "es": {
            "Buy now": "Comprar ahora",
            "Learn more": "Más información",
            "Try it free": "Pruébalo gratis"
        }
    }
    return translations.get(target_language, {}).get(text, f"[Translated to {target_language}] {text}")


# Create the agents for each step in the chain
validation_agent = Agent(
    name="Validation Agent",
    instructions="Validate the marketing copy against common criteria and provide feedback.",
    tools=[validate_marketing_copy],
    output_type=ValidationResult,
    hooks=CustomAgentHooks(display_name="Validation Agent"),
)

translation_agent = Agent(
    name="Translation Agent",
    instructions="Translate the validated marketing copy to the target language.",
    tools=[translate_text],
    output_type=TranslatedCopy,
    hooks=CustomAgentHooks(display_name="Translation Agent"),
)

marketing_agent = Agent(
    name="Marketing Agent",
    instructions="Create compelling marketing copy with a headline, body, and call to action.",
    output_type=MarketingCopy,
    handoffs=[validation_agent],
    hooks=CustomAgentHooks(display_name="Marketing Agent"),
)


async def main() -> None:
    # Example workflow: Create marketing copy -> Validate -> Translate
    product_description = input("Enter product description: ")
    target_language = input("Enter target language (e.g., 'es' for Spanish): ")

    # Start the chain with the marketing agent
    result = await Runner.run(
        marketing_agent,
        input=f"Create marketing copy for: {product_description}",
    )

    # Check validation result
    validation_result = await Runner.run(
        validation_agent,
        input=f"Validate this marketing copy: {result}",
    )

    # Access the final_output property of RunResult
    if not validation_result.final_output.is_valid:
        print(f"Validation failed:\n{validation_result.final_output.feedback}")
        return

    # If validation passes, proceed with translation
    translation_result = await Runner.run(
        translation_agent,
        input=f"Translate this marketing copy to {target_language}: {result}",
    )

    print("\nFinal Results:")
    print(f"Original Headline: {translation_result.final_output.original.headline}")
    print(f"Translated Headline: {translation_result.final_output.translated_headline}")
    print(f"\nOriginal Body: {translation_result.final_output.original.body}")
    print(f"Translated Body: {translation_result.final_output.translated_body}")
    print(f"\nOriginal CTA: {translation_result.final_output.original.call_to_action}")
    print(f"Translated CTA: {translation_result.final_output.translated_call_to_action}")


if __name__ == "__main__":
    asyncio.run(main()) 