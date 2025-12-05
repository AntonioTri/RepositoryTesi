from typing import Optional
import dspy
import requests

# 🧠 Try importing backends (for older DSPy versions)
try:
    from dspy.backends import OpenAI, Anthropic, Cohere
except ImportError:
    # ✅ New DSPy versions don't expose these
    OpenAI = None
    Anthropic = None
    Cohere = None


# ======================
# Custom Local Provider
# ======================

class LocalOllama:
    """Minimal DSPy-compatible wrapper for Ollama local models"""

    def __init__(self, model: str, api_base: str = "http://127.0.0.1:11434/api/generate",
                 temperature: float = 0.7, max_tokens: int = 4000, **kwargs):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        """Send a prompt to the local Ollama API and return the response text"""
        try:
            response = requests.post(
                self.api_base,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=600
            )
            response.raise_for_status()

            # Ollama streams responses line by line — handle both streaming and static JSON
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        data = line[len(b"data:"):].decode("utf-8").strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = requests.utils.json.loads(data)
                            text += chunk.get("response", "")
                        except Exception:
                            continue
                return text.strip()
            else:
                data = response.json()
                return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama local model request failed: {str(e)}")

# ======================
# LM Manager
# ======================

class LMManager:
    """Manages Language Model initialization and configuration for different providers"""
    
    SUPPORTED_PROVIDERS = {}

    # Aggiungi solo i provider effettivamente disponibili
    if OpenAI:
        SUPPORTED_PROVIDERS['openai'] = OpenAI
    if Anthropic:
        SUPPORTED_PROVIDERS['anthropic'] = Anthropic
    if Cohere:
        SUPPORTED_PROVIDERS['cohere'] = Cohere

    # Sempre disponibile
    SUPPORTED_PROVIDERS['local'] = LocalOllama


    @classmethod
    def get_lm(cls, 
               provider: str, 
               model_name: str, 
               api_key: Optional[str] = None, 
               api_base: Optional[str] = None,
               temperature: float = 0.7,
               max_tokens: int = 4000,
               **kwargs):
        """
        Initialize and return appropriate language model based on provider.
        """
        provider = provider.lower()
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers are: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )

        lm_class = cls.SUPPORTED_PROVIDERS[provider]
        lm_args = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if provider == 'openai' and api_base:
            lm_args["api_base"] = api_base
            lm_args["api_key"] = api_key
        elif provider == 'local':
            lm_args["api_base"] = api_base or "http://127.0.0.1:11434/api/generate"

        try:
            return lm_class(**lm_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {provider} LM: {str(e)}")

    @staticmethod
    def configure_dspy(lm) -> None:
        """Configure DSPy with the given language model"""
        dspy.configure(lm=lm)
