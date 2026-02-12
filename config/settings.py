"""
Configuration settings for the AI Agent with Ollama.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the Ollama OpenAI-compatible API"
    )
    
    api_key: str = Field(
        default="ollama",
        description="API key for Ollama (required for cloud-routed models)"
    )
    
    model_name: str = Field(
        default="qwen3-coder:480b-cloud",
        description="Model name to use with Ollama"
    )
    
    # Generation settings
    max_tokens: int = Field(default=2048, description="Max tokens for generation")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_iterations: int = Field(default=5, description="Max ReAct iterations")
    max_tool_rounds: int = Field(default=25, description="Max tool call rounds per chat turn")
    
    # File Storage
    upload_dir: str = Field(default="./uploads", description="Directory for uploaded files")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=3000, description="API port")
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"


settings = Settings()
os.makedirs(settings.upload_dir, exist_ok=True)
