"""
Configuration file for Azure LLM Twitter Bot
Store your credentials and configuration settings here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Azure LLM Twitter Bot"""
    
    # Azure ML Configuration
    AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
    AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "")
    
    # Azure LLM Endpoint Configuration
    AZURE_LLM_ENDPOINT = os.getenv("AZURE_LLM_ENDPOINT", "")
    AZURE_LLM_API_KEY = os.getenv("AZURE_LLM_API_KEY", "")
    
    # Twitter API Configuration
    TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY", "")
    TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET", "")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
    
    # Bot Configuration
    BOT_CHECK_INTERVAL = int(os.getenv("BOT_CHECK_INTERVAL", "60"))  # seconds
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "280"))  # Twitter character limit
    
    # LLM Configuration
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "150"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "bot.log")
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_fields = [
            "AZURE_LLM_ENDPOINT",
            "AZURE_LLM_API_KEY",
            "TWITTER_CONSUMER_KEY",
            "TWITTER_CONSUMER_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_TOKEN_SECRET"
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return True

# Example .env file content (create this file and fill in your actual values)
ENV_TEMPLATE = """
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-azure-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group-name
AZURE_WORKSPACE_NAME=your-workspace-name

# Azure LLM Endpoint
AZURE_LLM_ENDPOINT=https://your-endpoint.azureml.net/score
AZURE_LLM_API_KEY=your-azure-api-key

# Twitter API Credentials
TWITTER_CONSUMER_KEY=your-twitter-consumer-key
TWITTER_CONSUMER_SECRET=your-twitter-consumer-secret
TWITTER_ACCESS_TOKEN=your-twitter-access-token
TWITTER_ACCESS_TOKEN_SECRET=your-twitter-access-token-secret

# Bot Configuration (Optional)
BOT_CHECK_INTERVAL=60
MAX_RESPONSE_LENGTH=280
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=150
LOG_LEVEL=INFO
LOG_FILE=bot.log
"""

if __name__ == "__main__":
    # Create a template .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(ENV_TEMPLATE)
        print("Created .env template file. Please fill in your actual credentials.")
    
    # Test configuration validation
    try:
        Config.validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and ensure all required fields are filled.")

