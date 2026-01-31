"""Configuration management with environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration from environment variables."""
    
    AZURE_DOCINTEL_ENDPOINT = os.getenv("AZURE_DOCINTEL_ENDPOINT")
    AZURE_DOCINTEL_KEY = os.getenv("AZURE_DOCINTEL_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set."""
        missing = []
        
        if not cls.AZURE_DOCINTEL_ENDPOINT:
            missing.append("AZURE_DOCINTEL_ENDPOINT")
        if not cls.AZURE_DOCINTEL_KEY:
            missing.append("AZURE_DOCINTEL_KEY")
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Please check your .env file."
            )
        
        return True
