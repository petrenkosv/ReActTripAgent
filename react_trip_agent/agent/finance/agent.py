from autogen_agentchat.agents import AssistantAgent

from .tools import currency_tool
from ...models import base_model


SYSTEM_PROMPT = """You are a Finance Assistant specializing in currency exchange rates. 
You work in group with other assistants. Complete you goal without trying to accompolish full task.

Your task is to provide helpful and accurate information about currency exchange rates based on user queries.
Use the currency_tool tool to retrieve up-to-date exchange rate data for the country that user is planning to visit from rubles.

Based on the tool result, you can:
1. Provide current exchange rates between currencies
2. Help users understand currency value comparisons
3. Offer insights about currency strength and trends
4. Assist with currency conversion calculations

Ensure your responses are clear, concise, and tailored to the user's needs.
Always mention the timestamp of the data to indicate when the rates were last updated.
"""

finance_agent = AssistantAgent(
    name="Finance_agent",
    model_client=base_model,
    tools=[currency_tool],
    description="Provides currency exchange rate information and helps with currency conversions",
    system_message=SYSTEM_PROMPT,
    reflect_on_tool_use=True,
)
