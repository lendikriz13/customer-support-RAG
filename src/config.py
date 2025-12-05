"""
Configuration management for AI Support Agent
Loads environment variables and provides validation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This looks for .env in the project root directory
load_dotenv()

class Config:
    """
    Configuration class that loads and validates all settings.

    Why a class? Keeps all config in one place and makes it easy to validate.
    Alternative would be scattered os.getenv() calls everywhere (messy!).
    """

    # === API Keys ===
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # === Document Processing ===
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # === Model Configuration ===
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    # === RAG Settings ===
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "3"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

    # === Debug Mode ===
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # === File Paths ===
    # Get project root directory (parent of src/)
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    SAMPLE_DOCS_DIR: Path = DATA_DIR / "sample_docs"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"

    @classmethod
    def validate(cls) -> None:
        """
        Validate that all required configuration is present.

        Why validate? Fail fast with helpful errors instead of mysterious crashes later.
        Better to see "Missing OPENAI_API_KEY" than "NoneType has no attribute 'split'"
        """
        errors = []

        # Check API keys exist and look valid
        if not cls.OPENAI_API_KEY or not cls.OPENAI_API_KEY.startswith("sk-"):
            errors.append("OPENAI_API_KEY is missing or invalid (should start with 'sk-')")

        if not cls.ANTHROPIC_API_KEY or not cls.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            errors.append("ANTHROPIC_API_KEY is missing or invalid (should start with 'sk-ant-')")

        # Check numeric values are reasonable
        if cls.CHUNK_SIZE < 100 or cls.CHUNK_SIZE > 5000:
            errors.append(f"CHUNK_SIZE={cls.CHUNK_SIZE} is unreasonable (use 500-1500)")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append(f"CHUNK_OVERLAP={cls.CHUNK_OVERLAP} must be less than CHUNK_SIZE={cls.CHUNK_SIZE}")

        if cls.RETRIEVAL_K < 1 or cls.RETRIEVAL_K > 10:
            errors.append(f"RETRIEVAL_K={cls.RETRIEVAL_K} is unreasonable (use 1-10)")

        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            errors.append(f"TEMPERATURE={cls.TEMPERATURE} must be between 0 and 1")

        # If there are errors, raise exception with all of them
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create necessary directories if they don't exist.

        Why? Prevents "FileNotFoundError: No such directory" when saving files.
        """
        cls.SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        """
        Print current configuration (for debugging).
        Hides API keys for security.
        """
        print("=== Configuration ===")
        print(f"OpenAI API Key: {cls.OPENAI_API_KEY[:20]}..." if cls.OPENAI_API_KEY else "Missing")
        print(f"Anthropic API Key: {cls.ANTHROPIC_API_KEY[:25]}..." if cls.ANTHROPIC_API_KEY else "Missing")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Sample Docs Dir: {cls.SAMPLE_DOCS_DIR}")
        print(f"Vector Store Dir: {cls.VECTOR_STORE_DIR}")
        print("=" * 40)


# Auto-validate on import (fail fast if config is wrong)
# You can disable this by commenting out the line below
try:
    Config.validate()
    Config.ensure_directories()
    if Config.DEBUG:
        Config.print_config()
except ValueError as e:
    print(f"\nConfiguration Error:\n{e}\n")
    print("💡 Fix your .env file and try again.\n")
    raise


# Convenience: Export config instance for easy importing
# Usage: from src.config import config
config = Config()
