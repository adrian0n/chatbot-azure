"""
# 🚀 Azure LLM Deployment Script
# 👨‍💻 @adrian0n
"""

import os
import requests
import json
from typing import Dict, Any, List
import time

class AzureLLMDeployment:
    """
    A class to handle Azure LLM deployment and inference operations
    """
    
    def __init__(self, endpoint_url: str, api_key: str):
        """
        Initialize the Azure LLM deployment client
        
        Args:
            endpoint_url (str): The Azure ML endpoint URL for your deployed model
            api_key (str): The API key for authentication
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response from the deployed LLM
        
        Args:
            prompt (str): The input prompt for the model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            Dict containing the model's response
        """
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Azure LLM endpoint: {e}")
            return {"error": str(e)}
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts (List[str]): List of input prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            List of responses from the model
        """
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
            # Add a small delay to respect rate limits
            time.sleep(0.1)
        
        return responses
    
    def health_check(self) -> bool:
        """
        Check if the deployed model endpoint is healthy
        
        Returns:
            bool: True if endpoint is healthy, False otherwise
        """
        try:
            test_response = self.generate_response("Hey hey ce faci draga?", max_tokens=10)
            return "error" not in test_response
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Replace with your actual Azure ML endpoint URL and API key
    ENDPOINT_URL = "https://your-endpoint.azureml.net/score"
    API_KEY = "api-key-here"
    
    # Initialize the deployment client
    llm_client = AzureLLMDeployment(ENDPOINT_URL, API_KEY)
    
    # Test the health of the endpoint
    if llm_client.health_check():
        print("✅ Azure LLM endpoint is healthy")
        
        # Generate a sample response
        test_prompt = "Make a flirty sentance about LoveSIRI that ends with 'draga' "
        response = llm_client.generate_response(test_prompt)
        
        if "error" not in response:
            print(f"🤖 Model Response: {response}")
        else:
            print(f"❌ Error generating response: {response['error']}")
    else:
        print("❌ Azure LLM endpoint is fucked")

