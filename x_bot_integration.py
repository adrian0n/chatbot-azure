"""
X (Twitter) Bot Integration Script
This script demonstrates how to create a Twitter bot that uses an Azure-deployed LLM
"""

import tweepy
import os
import time
import json
import logging
from typing import Optional, Dict, Any
from azure_llm_deployment import AzureLLMDeployment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XBot:
    """
    A Twitter bot that uses Azure LLM for generating responses
    """
    
    def __init__(self, 
                 consumer_key: str,
                 consumer_secret: str, 
                 access_token: str,
                 access_token_secret: str,
                 azure_endpoint: str,
                 azure_api_key: str):
        """
        Initialize the X Bot with Twitter API credentials and Azure LLM client
        
        Args:
            consumer_key (str): Twitter API consumer key
            consumer_secret (str): Twitter API consumer secret
            access_token (str): Twitter API access token
            access_token_secret (str): Twitter API access token secret
            azure_endpoint (str): Azure ML endpoint URL
            azure_api_key (str): Azure ML API key
        """
        
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Initialize Azure LLM client
        self.llm_client = AzureLLMDeployment(azure_endpoint, azure_api_key)
        
        # Bot configuration
        self.bot_username = self.twitter_api.verify_credentials().screen_name
        self.last_mention_id = None
        
        logger.info(f"Bot initialized as @{self.bot_username}")
    
    def generate_tweet_response(self, mention_text: str, context: str = "") -> str:
        """
        Generate a response using Azure LLM for a Twitter mention
        
        Args:
            mention_text (str): The text of the mention
            context (str): Additional context for the response
            
        Returns:
            str: Generated response text
        """
        # Create a prompt that encourages concise, Twitter-appropriate responses
        prompt = f"""You are a helpful AI assistant on Twitter. Someone mentioned you with this message: "{mention_text}"
        
        Context: {context}
        
        Please provide a helpful, concise response that is appropriate for Twitter (under 280 characters). 
        Be friendly, informative, and engaging. Do not include hashtags unless specifically relevant.
        
        Response:"""
        
        try:
            response = self.llm_client.generate_response(
                prompt, 
                max_tokens=100, 
                temperature=0.7
            )
            
            if "error" not in response and "choices" in response:
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                # Ensure the response fits Twitter's character limit
                if len(generated_text) > 280:
                    generated_text = generated_text[:277] + "..."
                
                return generated_text
            else:
                logger.error(f"Error in LLM response: {response}")
                return "Thanks for reaching out! I'm having trouble generating a response right now. Please try again later."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Thanks for your message! I'm experiencing technical difficulties. Please try again later."
    
    def reply_to_mention(self, mention):
        """
        Reply to a Twitter mention using Azure LLM
        
        Args:
            mention: Twitter mention object from tweepy
        """
        try:
            # Extract the mention text (remove @bot_username)
            mention_text = mention.text.replace(f"@{self.bot_username}", "").strip()
            
            # Generate response using Azure LLM
            response_text = self.generate_tweet_response(mention_text)
            
            # Reply to the mention
            reply_text = f"@{mention.user.screen_name} {response_text}"
            
            self.twitter_api.update_status(
                status=reply_text,
                in_reply_to_status_id=mention.id,
                auto_populate_reply_metadata=True
            )
            
            logger.info(f"Replied to @{mention.user.screen_name}: {reply_text}")
            
        except Exception as e:
            logger.error(f"Error replying to mention: {e}")
    
    def check_mentions(self):
        """
        Check for new mentions and respond to them
        """
        try:
            # Get mentions since the last processed mention
            mentions = tweepy.Cursor(
                self.twitter_api.mentions_timeline,
                since_id=self.last_mention_id,
                tweet_mode='extended'
            ).items()
            
            new_mentions = []
            for mention in mentions:
                new_mentions.append(mention)
            
            # Process mentions in chronological order (oldest first)
            new_mentions.reverse()
            
            for mention in new_mentions:
                # Skip if it's a retweet or if the bot is replying to itself
                if hasattr(mention, 'retweeted_status') or mention.user.screen_name == self.bot_username:
                    continue
                
                logger.info(f"Processing mention from @{mention.user.screen_name}: {mention.full_text}")
                self.reply_to_mention(mention)
                
                # Update the last processed mention ID
                self.last_mention_id = mention.id
                
                # Add delay to respect rate limits
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error checking mentions: {e}")
    
    def post_scheduled_tweet(self, content: str):
        """
        Post a scheduled tweet using Azure LLM to generate content
        
        Args:
            content (str): Topic or prompt for the tweet
        """
        try:
            prompt = f"""Generate an engaging tweet about: {content}
            
            The tweet should be:
            - Under 280 characters
            - Informative and interesting
            - Appropriate for a general audience
            - Include relevant information or insights
            
            Tweet:"""
            
            response = self.llm_client.generate_response(prompt, max_tokens=100, temperature=0.8)
            
            if "error" not in response and "choices" in response:
                tweet_text = response["choices"][0]["message"]["content"].strip()
                
                # Ensure it fits Twitter's character limit
                if len(tweet_text) > 280:
                    tweet_text = tweet_text[:277] + "..."
                
                self.twitter_api.update_status(tweet_text)
                logger.info(f"Posted scheduled tweet: {tweet_text}")
            else:
                logger.error(f"Error generating scheduled tweet: {response}")
                
        except Exception as e:
            logger.error(f"Error posting scheduled tweet: {e}")
    
    def run_bot(self, check_interval: int = 60):
        """
        Run the bot continuously, checking for mentions at regular intervals
        
        Args:
            check_interval (int): Time in seconds between mention checks
        """
        logger.info(f"Starting bot with {check_interval}s check interval")
        
        while True:
            try:
                self.check_mentions()
                logger.info(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in bot loop: {e}")
                time.sleep(check_interval)

# Example usage
if __name__ == "__main__":
    # Twitter API credentials (replace with your actual credentials)
    CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY", "your-consumer-key")
    CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET", "your-consumer-secret")
    ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "your-access-token")
    ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "your-access-token-secret")
    
    # Azure credentials (replace with your actual credentials)
    AZURE_ENDPOINT = os.getenv("AZURE_LLM_ENDPOINT", "https://your-endpoint.azureml.net/score")
    AZURE_API_KEY = os.getenv("AZURE_LLM_API_KEY", "your-azure-api-key")
    
    # Initialize and run the bot
    try:
        bot = XBot(
            consumer_key=CONSUMER_KEY,
            consumer_secret=CONSUMER_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET,
            azure_endpoint=AZURE_ENDPOINT,
            azure_api_key=AZURE_API_KEY
        )
        
        # Test the bot's LLM connection
        if bot.llm_client.health_check():
            logger.info("✅ Azure LLM connection successful")
            
            # Example: Post a test tweet
            # bot.post_scheduled_tweet("The future of AI and social media")
            
            # Start the bot
            bot.run_bot(check_interval=60)  # Check for mentions every 60 seconds
        else:
            logger.error("❌ Failed to connect to Azure LLM endpoint")
            
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")

