from autogen_agentchat.agents import AssistantAgent

from ...models import base_model

summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=base_model,
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE. Respond only in Russian.",
)
