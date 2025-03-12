from autogen_ext.models.openai import OpenAIChatCompletionClient


model_info = {
    "vision": False,
    "function_calling": True,
    "json_output": False,
    "family": "Anthropic",
}

base_model = OpenAIChatCompletionClient(
    model="openai.gpt-4o",
    base_url="https://llm.glowbyteconsulting.com/api/",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjgyNjJkM2MxLWI0NDktNDhmYi1iNjk1LWI1ZmNiNDBiMjM5ZiIsImV4cCI6MTc0MjI3ODc1N30.LGPZCl98O783Pgv7lfQI_TmolVCpXjFr9mVWiTZivkU",
    model_info=model_info,
)
