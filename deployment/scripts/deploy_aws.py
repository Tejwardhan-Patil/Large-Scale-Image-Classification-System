import boto3
import json
import os
import logging
import time
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load AWS configuration from environment variables or a config file
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Set up paths
model_path = "/models/model.tar.gz"
bucket_name = "model-deployment-bucket"
s3_model_key = "deployed_models/model.tar.gz"
ec2_key_name = "aws-key-pair"
security_group_id = "sg-0123456789abcdef0"
instance_type = "t2.large"
ami_id = "ami-12345678" 
lambda_role_arn = "arn:aws:iam::123456789012:role/lambda_execution_role"

# Retry policy configuration
RETRY_COUNT = 3
RETRY_DELAY = 5  # seconds

# Initialize AWS clients
def init_clients():
    for attempt in range(RETRY_COUNT):
        try:
            s3_client = boto3.client(
                "s3",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            ec2_client = boto3.client(
                "ec2",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            lambda_client = boto3.client(
                "lambda",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            logger.info("AWS clients initialized successfully.")
            return s3_client, ec2_client, lambda_client
        except ClientError as e:
            logger.error(f"Error initializing AWS clients, attempt {attempt + 1}: {e}")
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e

# Upload model to S3
def upload_model_to_s3(s3_client):
    for attempt in range(RETRY_COUNT):
        try:
            s3_client.upload_file(model_path, bucket_name, s3_model_key)
            logger.info(f"Model uploaded successfully to S3 bucket: {bucket_name}")
            break
        except ClientError as e:
            logger.error(f"Error uploading model to S3, attempt {attempt + 1}: {e}")
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e

# Create and tag EC2 instance
def deploy_ec2_instance(ec2_client):
    try:
        instance = ec2_client.run_instances(
            ImageId=ami_id,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            KeyName=ec2_key_name,
            SecurityGroupIds=[security_group_id],
            UserData=open('startup_script.sh', 'r').read(),
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [{'Key': 'Name', 'Value': 'ModelDeploymentEC2'}]
            }]
        )
        instance_id = instance["Instances"][0]["InstanceId"]
        logger.info(f"EC2 Instance {instance_id} deployed successfully.")
        return instance_id
    except ClientError as e:
        logger.error(f"Error deploying EC2 instance: {e}")
        raise e

# Check EC2 instance status
def wait_for_instance(ec2_client, instance_id):
    logger.info(f"Waiting for EC2 instance {instance_id} to be in 'running' state...")
    waiter = ec2_client.get_waiter('instance_running')
    try:
        waiter.wait(InstanceIds=[instance_id])
        logger.info(f"EC2 instance {instance_id} is now running.")
    except ClientError as e:
        logger.error(f"Error waiting for EC2 instance {instance_id} to run: {e}")
        raise e

# Monitor instance health and metrics
def monitor_instance_health(ec2_client, instance_id):
    logger.info(f"Starting health monitoring for EC2 instance {instance_id}...")
    try:
        health_status = ec2_client.describe_instance_status(InstanceIds=[instance_id])
        logger.info(f"Instance {instance_id} health status: {json.dumps(health_status, indent=2)}")
    except ClientError as e:
        logger.error(f"Error monitoring health of EC2 instance {instance_id}: {e}")

# Deploy Lambda function for inference
def create_lambda_function(lambda_client):
    try:
        with open('lambda_deployment_package.zip', 'rb') as f:
            lambda_client.create_function(
                FunctionName="ModelInferenceLambda",
                Runtime="python3.8",
                Role=lambda_role_arn,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": f.read()},
                Timeout=300,
                MemorySize=128,
                Tags={
                    'Project': 'ImageClassification',
                    'Environment': 'Production'
                }
            )
        logger.info("Lambda function created successfully.")
    except ClientError as e:
        logger.error(f"Error creating Lambda function: {e}")
        raise e

# Create CloudWatch alarms for monitoring
def create_cloudwatch_alarm(instance_id):
    cloudwatch = boto3.client("cloudwatch", region_name=aws_region)
    try:
        cloudwatch.put_metric_alarm(
            AlarmName=f"HighCPUUtilization-{instance_id}",
            AlarmDescription="Triggered when CPU utilization exceeds 80% for 5 minutes.",
            ActionsEnabled=True,
            MetricName="CPUUtilization",
            Namespace="AWS/EC2",
            Statistic="Average",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            Period=300,
            EvaluationPeriods=1,
            Threshold=80.0,
            ComparisonOperator="GreaterThanThreshold",
            AlarmActions=[
                f"arn:aws:swf:{aws_region}:123456789012:action/actions/AWS_EC2.InstanceId.Reboot/1.0"
            ],
        )
        logger.info(f"CloudWatch alarm for CPU utilization created for instance {instance_id}.")
    except ClientError as e:
        logger.error(f"Error creating CloudWatch alarm for instance {instance_id}: {e}")

# Main function to coordinate deployment
def main():
    # Step 1: Initialize AWS clients
    s3_client, ec2_client, lambda_client = init_clients()

    # Step 2: Upload model to S3
    upload_model_to_s3(s3_client)

    # Step 3: Deploy EC2 instance
    instance_id = deploy_ec2_instance(ec2_client)

    # Step 4: Wait for instance to be in running state
    wait_for_instance(ec2_client, instance_id)

    # Step 5: Monitor EC2 instance health
    monitor_instance_health(ec2_client, instance_id)

    # Step 6: Create CloudWatch alarms
    create_cloudwatch_alarm(instance_id)

    # Step 7: Deploy Lambda function
    create_lambda_function(lambda_client)

    logger.info("AWS deployment process completed successfully.")

if __name__ == "__main__":
    main()