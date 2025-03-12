from autogen_ext.models.openai import OpenAIChatCompletionClient
from .config import GB_AI_MODEL, GB_AI_BASE_URL, GB_AI_API_KEY


model_info = {
    "vision": False,
    "function_calling": True,
    "json_output": False,
    "family": "Anthropic",
}

base_model = OpenAIChatCompletionClient(
    model=GB_AI_MODEL,
    base_url=GB_AI_BASE_URL,
    api_key=GB_AI_API_KEY,
    model_info=model_info,
)
