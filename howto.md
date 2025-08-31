# Azure LLM Twitter Bot

A comprehensive solution for deploying a Large Language Model (LLM) on Azure and integrating it with X (Twitter) to create an intelligent bot similar to Grok.

## Project Overview

This project provides:
- Scripts for deploying LLMs on Azure AI Foundry
- Fine-tuning capabilities using Azure Machine Learning
- Twitter bot integration with Azure-deployed LLMs
- Complete configuration and deployment templates

## Prerequisites

1. **Azure Account**: Active Azure subscription with sufficient credits
2. **Twitter Developer Account**: Approved X Developer account with API access
3. **Python 3.8+**: Python environment with pip
4. **Azure CLI**: For authentication and resource management

## Quick Start

### 1. Clone and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create configuration file
python config.py
```

### 2. Configure Credentials

Edit the `.env` file created by the config script:

```env
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
```

### 3. Deploy Your LLM on Azure

```python
from azure_llm_deployment import AzureLLMDeployment

# Initialize deployment client
llm_client = AzureLLMDeployment(
    endpoint_url="your-azure-endpoint",
    api_key="your-api-key"
)

# Test the deployment
response = llm_client.generate_response("Hello, how are you?")
print(response)
```

### 4. Run the Twitter Bot

```python
from x_bot_integration import XBot

# Initialize the bot
bot = XBot(
    consumer_key="your-consumer-key",
    consumer_secret="your-consumer-secret",
    access_token="your-access-token",
    access_token_secret="your-access-token-secret",
    azure_endpoint="your-azure-endpoint",
    azure_api_key="your-azure-api-key"
)

# Start the bot
bot.run_bot(check_interval=60)
```

## File Structure

```
├── azure_llm_deployment.py    # Azure LLM deployment and inference
├── x_bot_integration.py       # Twitter bot with Azure LLM integration
├── azure_ml_fine_tuning.py    # Fine-tuning scripts for Azure ML
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── howto.md                   # Comprehensive deployment guide
└── README.md                  # README file
```

## Key Features

### Azure LLM Deployment (`azure_llm_deployment.py`)
- Connect to Azure-deployed LLM endpoints
- Generate responses with configurable parameters
- Batch processing capabilities
- Health check functionality

### Twitter Bot Integration (`x_bot_integration.py`)
- Automated mention monitoring and responses
- Context-aware conversation handling
- Rate limit management
- Scheduled tweet posting
- Error handling and logging

### Fine-tuning Support (`azure_ml_fine_tuning.py`)
- Azure ML workspace integration
- Custom model training pipelines
- Hyperparameter configuration
- Model registration and versioning

## Deployment Steps

### Step 1: Set Up Azure Resources

1. Create an Azure AI Foundry workspace
2. Deploy a foundation model (e.g., GPT-3.5, Llama 2)
3. Note the endpoint URL and API key

### Step 2: Configure Twitter API

1. Apply for Twitter Developer access
2. Create a new app in the Developer Portal
3. Generate API keys and access tokens
4. Configure bot account permissions

### Step 3: Fine-tune Your Model (Optional)

```python
from azure_ml_fine_tuning import AzureMLFineTuner

fine_tuner = AzureMLFineTuner(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    workspace_name="your-workspace"
)

# Submit fine-tuning job
job = fine_tuner.create_fine_tuning_job(
    base_model_name="microsoft/DialoGPT-medium",
    training_data_name="your-training-data",
    compute_target="gpu-cluster",
    output_model_name="fine-tuned-bot"
)
```

### Step 4: Deploy and Monitor

1. Test your LLM endpoint
2. Start the Twitter bot
3. Monitor logs and performance
4. Scale as needed

## Configuration Options

### Bot Behavior
- `BOT_CHECK_INTERVAL`: How often to check for mentions (seconds)
- `MAX_RESPONSE_LENGTH`: Maximum tweet length
- `DEFAULT_TEMPERATURE`: LLM creativity level (0.0-1.0)
- `DEFAULT_MAX_TOKENS`: Maximum response tokens

### Logging
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `LOG_FILE`: Log file location

## Troubleshooting

### Common Issues

1. **Azure Authentication Errors**
   - Ensure Azure CLI is logged in: `az login`
   - Verify subscription and resource group names
   - Check API key validity

2. **Twitter API Errors**
   - Verify API keys and tokens
   - Check rate limits and usage
   - Ensure proper app permissions

3. **LLM Response Issues**
   - Test endpoint health
   - Adjust temperature and max_tokens
   - Check input prompt formatting

### Monitoring and Logs

The bot logs all activities to help with debugging:
- Mention processing
- Response generation
- API errors
- Rate limit warnings

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for sensitive data
3. **Regularly rotate API keys**
4. **Monitor usage and costs**
5. **Implement content moderation**

## Cost Optimization

- Use pay-as-you-go billing for variable workloads
- Consider provisioned throughput for consistent usage
- Monitor Azure costs regularly
- Optimize response length and frequency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Azure and Twitter API documentation
3. Create an issue in the repository
4. Consult the comprehensive deployment guide

## Acknowledgments

- Microsoft Azure AI team for the foundational services
- Twitter/X for the API platform
- Open-source community for the underlying libraries


#  ────────────────────────────────
#      a d r i a n 0 n
#  ────────────────────────────────

