from openai import OpenAI
from loguru import logger

class LLMClient:
    def __init__(self, api_key, base_url, model="qwen3-8b"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call_llm(self, messages, temperature=0.9, max_tokens=400, seed=123):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"enable_thinking": False},
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            choice = response.choices[0]
            content = getattr(getattr(choice, "message", None), "content", None)
            if content:
                return content.strip()
            else:
                logger.error("LLM Client: Response does not contain a message/content.")
                return ""
        except Exception as e:
            logger.error(f"LLM Client: Exception during OpenAI call: {e}")
            return ""
