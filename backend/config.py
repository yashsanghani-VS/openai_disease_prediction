import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    LOG_DIR = BASE_DIR / "logs"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", "8000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    # Model Settings
    GPT4_MODEL = "gpt-4.1"
    GEMINI_MODEL = "gemini-2.0-flash"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Fallback Models
    FALLBACK_MODELS = {
        "gpt4": "gpt-4.1",
        "gemini": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile"
    }
    
    # Model Parameters
    MODEL_PARAMS = {
        "gpt4": {
            "temperature": 0,
            "max_tokens": 4096,
            "timeout": 60,
            "retry_attempts": 3
        },
        "gemini": {
            "temperature": 0,
            "max_output_tokens": 2048,
            "timeout": 60,
            "retry_attempts": 3
        },
        "groq": {
            "temperature": 0,
            "max_tokens": 4096,
            "timeout": 60,
            "retry_attempts": 3
        }
    }
    
    # Image Settings
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    IMAGE_QUALITY = 85  # JPEG quality (0-100)
    
    # API Settings
    CORS_ORIGINS = ["http://localhost:8501"]  # Add your frontend URL
    API_RATE_LIMIT = 100  # requests per minute
    REQUEST_TIMEOUT = 30  # seconds
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOG_DIR / "app.log"
    
    # Cache Settings
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # 1 hour in seconds
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration settings."""
        return {
            "gpt4": {
                "model": cls.GPT4_MODEL,
                **cls.MODEL_PARAMS["gpt4"],
                "fallback_model": cls.FALLBACK_MODELS["gpt4"]
            },
            "gemini": {
                "model": cls.GEMINI_MODEL,
                **cls.MODEL_PARAMS["gemini"],
                "fallback_model": cls.FALLBACK_MODELS["gemini"]
            },
            "groq": {
                "model": cls.GROQ_MODEL,
                **cls.MODEL_PARAMS["groq"],
                "fallback_model": cls.FALLBACK_MODELS["groq"]
            }
        }
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings."""
        # Check required environment variables
        required_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        # Validate image settings
        if not all(cls.SUPPORTED_FORMATS):
            raise ValueError("Invalid image format configuration")
        
        if cls.MAX_IMAGE_SIZE <= 0:
            raise ValueError("Invalid maximum image size configuration")
            
        if not 0 <= cls.IMAGE_QUALITY <= 100:
            raise ValueError("Invalid image quality setting")
        
        # Create log directory if it doesn't exist
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Validate API settings
        if cls.API_RATE_LIMIT <= 0:
            raise ValueError("Invalid API rate limit configuration")
            
        if cls.REQUEST_TIMEOUT <= 0:
            raise ValueError("Invalid request timeout configuration")
    
    @classmethod
    def setup_logging(cls) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        ) 