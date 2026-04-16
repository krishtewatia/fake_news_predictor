"""
LLM Client Module
-----------------
Provides a reusable interface to OpenRouter API for text generation.
Uses the OpenAI-compatible endpoint with google/gemini-2.0-flash-001 model.
"""

from openai import OpenAI
from config import OPEN_ROUTER_KEY

# Initialize OpenRouter client (OpenAI-compatible API)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_KEY,
)

# Model to use (free/cheap model on OpenRouter)
MODEL_NAME = "google/gemini-2.0-flash-001"


def generate_text(prompt: str) -> str:
    """
    Generate text using OpenRouter API.

    Args:
        prompt: The text prompt to send to the model.

    Returns:
        Generated text response, or empty string on failure.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLMClient] Error generating text: {e}")
        return ""
