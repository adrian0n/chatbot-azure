"""
# 🚀 Azure ML Fine-tuning Script
# 👨‍💻 @adrian0n
"""

import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Job,
    Environment,
    BuildContext,
    CommandJob,
    Data,
    Model
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from typing import Dict, Any, List

class AzureMLFineTuner:
    """
    A class to handle LLM fine-tuning operations on Azure Machine Learning
    """
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize the Azure ML client
        
        Args:
            subscription_id (str): Azure subscription ID
            resource_group (str): Azure resource group name
            workspace_name (str): Azure ML workspace name
        """
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
    def prepare_training_data(self, data_path: str, data_name: str) -> Data:
        """
        Register training data in Azure ML
        
        Args:
            data_path (str): Path to the training data
            data_name (str): Name for the registered data asset
            
        Returns:
            Data: Registered data asset
        """
        training_data = Data(
            path=data_path,
            type=AssetTypes.URI_FILE,
            description="Fine-tuning dataset for LLM",
            name=data_name,
            version="1"
        )
        
        return self.ml_client.data.create_or_update(training_data)
    
    def create_fine_tuning_environment(self) -> Environment:
        """
        Create a custom environment for fine-tuning
        
        Returns:
            Environment: Azure ML environment for fine-tuning
        """
        environment = Environment(
            name="llm-fine-tuning-env",
            description="Environment for LLM fine-tuning",
            build=BuildContext(
                path="./environment",
                dockerfile_path="Dockerfile"
            ),
            version="1"
        )
        
        return self.ml_client.environments.create_or_update(environment)
    
    def create_fine_tuning_job(self, 
                              base_model_name: str,
                              training_data_name: str,
                              compute_target: str,
                              output_model_name: str,
                              hyperparameters: Dict[str, Any] = None) -> CommandJob:
        """
        Create a fine-tuning job
        
        Args:
            base_model_name (str): Name of the base model to fine-tune
            training_data_name (str): Name of the registered training data
            compute_target (str): Name of the compute target for training
            output_model_name (str): Name for the output fine-tuned model
            hyperparameters (Dict): Training hyperparameters
            
        Returns:
            CommandJob: The fine-tuning job
        """
        if hyperparameters is None:
            hyperparameters = {
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "max_seq_length": 512,
                "fp16": True
            }
        
        # Create the fine-tuning command
        command = f"""
        python fine_tune_script.py \\
            --base_model {base_model_name} \\
            --training_data ${{inputs.training_data}} \\
            --output_dir ${{outputs.model}} \\
            --learning_rate {hyperparameters['learning_rate']} \\
            --num_train_epochs {hyperparameters['num_train_epochs']} \\
            --per_device_train_batch_size {hyperparameters['per_device_train_batch_size']} \\
            --gradient_accumulation_steps {hyperparameters['gradient_accumulation_steps']} \\
            --warmup_steps {hyperparameters['warmup_steps']} \\
            --max_seq_length {hyperparameters['max_seq_length']} \\
            --fp16 {hyperparameters['fp16']}
        """
        
        job = CommandJob(
            command=command,
            environment="llm-fine-tuning-env:1",
            compute=compute_target,
            inputs={
                "training_data": f"azureml:{training_data_name}:1"
            },
            outputs={
                "model": f"azureml://datastores/workspaceblobstore/paths/models/{output_model_name}"
            },
            display_name=f"fine-tune-{base_model_name}",
            description=f"Fine-tuning job for {base_model_name}",
            tags={"model_type": "llm", "task": "fine_tuning"}
        )
        
        return job
    
    def submit_fine_tuning_job(self, job: CommandJob) -> Job:
        """
        Submit the fine-tuning job to Azure ML
        
        Args:
            job (CommandJob): The fine-tuning job to submit
            
        Returns:
            Job: The submitted job
        """
        submitted_job = self.ml_client.jobs.create_or_update(job)
        print(f"Fine-tuning job submitted: {submitted_job.name}")
        print(f"Job URL: {submitted_job.studio_url}")
        
        return submitted_job
    
    def monitor_job(self, job_name: str) -> str:
        """
        Monitor the status of a fine-tuning job
        
        Args:
            job_name (str): Name of the job to monitor
            
        Returns:
            str: Final job status
        """
        job = self.ml_client.jobs.get(job_name)
        print(f"Job Status: {job.status}")
        
        # You can add more sophisticated monitoring here
        # For example, streaming logs or checking progress
        
        return job.status
    
    def register_fine_tuned_model(self, 
                                 job_name: str, 
                                 model_name: str, 
                                 model_description: str = None) -> Model:
        """
        Register the fine-tuned model from a completed job
        
        Args:
            job_name (str): Name of the completed fine-tuning job
            model_name (str): Name for the registered model
            model_description (str): Description for the model
            
        Returns:
            Model: The registered model
        """
        job = self.ml_client.jobs.get(job_name)
        
        if job.status != "Completed":
            raise ValueError(f"Job {job_name} is not completed. Current status: {job.status}")
        
        model = Model(
            path=f"azureml://jobs/{job_name}/outputs/model",
            name=model_name,
            description=model_description or f"Fine-tuned model from job {job_name}",
            type=AssetTypes.CUSTOM_MODEL,
            tags={"fine_tuned": "true", "base_job": job_name}
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        print(f"Model registered: {registered_model.name}:{registered_model.version}")
        
        return registered_model

# Fine-tuning script template (save as fine_tune_script.py)
FINE_TUNING_SCRIPT = '''
"""
Fine-tuning script for LLMs on Azure ML
This script should be saved as fine_tune_script.py in your training environment
"""

import argparse
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

def load_and_prepare_data(data_path, tokenizer, max_length=512):
    """Load and tokenize the training data"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--fp16", type=bool, default=True)
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    train_dataset = load_and_prepare_data(args.training_data, tokenizer, args.max_seq_length)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
'''

# Example usage
if __name__ == "__main__":
    # Azure ML workspace configuration
    SUBSCRIPTION_ID = "your-subscription-id"
    RESOURCE_GROUP = "your-resource-group"
    WORKSPACE_NAME = "your-workspace-name"
    
    # Initialize the fine-tuner
    fine_tuner = AzureMLFineTuner(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    
    # Example workflow
    try:
        # 1. Prepare training data
        training_data = fine_tuner.prepare_training_data(
            data_path="./training_data.json",
            data_name="llm-training-data"
        )
        print(f"Training data registered: {training_data.name}")
        
        # 2. Create fine-tuning job
        job = fine_tuner.create_fine_tuning_job(
            base_model_name="microsoft/DialoGPT-medium",
            training_data_name="llm-training-data",
            compute_target="gpu-cluster",
            output_model_name="fine-tuned-chatbot"
        )
        
        # 3. Submit the job
        submitted_job = fine_tuner.submit_fine_tuning_job(job)
        
        # 4. Monitor job (you would typically do this in a separate script or notebook)
        # status = fine_tuner.monitor_job(submitted_job.name)
        
        # 5. Register the fine-tuned model (after job completion)
        # model = fine_tuner.register_fine_tuned_model(
        #     job_name=submitted_job.name,
        #     model_name="my-fine-tuned-llm",
        #     model_description="Fine-tuned LLM for Twitter bot"
        # )
        
    except Exception as e:
        print(f"Error in fine-tuning workflow: {e}")
    
    # Save the fine-tuning script template
    with open("fine_tune_script.py", "w") as f:
        f.write(FINE_TUNING_SCRIPT)
    print("Fine-tuning script template saved as fine_tune_script.py")

