import os
import subprocess
import logging
from google.cloud import storage
from googleapiclient import discovery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError
import yaml

# GCP project and deployment configuration
PROJECT_ID = 'project-id'
CLUSTER_NAME = 'cluster-name'
ZONE = 'cluster-zone'
DOCKER_IMAGE_NAME = 'gcr.io/project-id/image-name'
GCS_BUCKET = 'bucket-name'
MODEL_FILE_PATH = '/model/file'
MODEL_ARTIFACT_NAME = 'model_artifacts.tar.gz'
DEPLOYMENT_NAME = 'deployment-name'
SERVICE_ACCOUNT_FILE = '/service_account.json'
SERVICE_NAME = f"{DEPLOYMENT_NAME}-service"
NAMESPACE = 'default'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authenticate and initialize GCP clients
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
storage_client = storage.Client(credentials=credentials)
gke_client = discovery.build('container', 'v1', credentials=credentials)

def upload_model_to_gcs():
    """Uploads the model artifact to a Google Cloud Storage bucket."""
    try:
        logging.info(f"Uploading model artifacts: {MODEL_FILE_PATH} to GCS bucket: {GCS_BUCKET}")
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(MODEL_ARTIFACT_NAME)
        blob.upload_from_filename(MODEL_FILE_PATH)
        logging.info(f"Model successfully uploaded to {GCS_BUCKET}/{MODEL_ARTIFACT_NAME}")
    except GoogleAPIError as e:
        logging.error(f"Error uploading model to GCS: {e}")
        raise

def create_kubernetes_deployment_yaml():
    """Creates the Kubernetes deployment YAML file."""
    deployment_yaml = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": DEPLOYMENT_NAME
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": DEPLOYMENT_NAME
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": DEPLOYMENT_NAME
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": DEPLOYMENT_NAME,
                            "image": DOCKER_IMAGE_NAME,
                            "ports": [
                                {
                                    "containerPort": 80
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }
    with open('deployment.yaml', 'w') as yaml_file:
        yaml.dump(deployment_yaml, yaml_file, default_flow_style=False)
    logging.info("Deployment YAML file created successfully.")

def create_kubernetes_service_yaml():
    """Creates the Kubernetes service YAML file."""
    service_yaml = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": SERVICE_NAME
        },
        "spec": {
            "selector": {
                "app": DEPLOYMENT_NAME
            },
            "ports": [
                {
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 80
                }
            ],
            "type": "LoadBalancer"
        }
    }
    with open('service.yaml', 'w') as yaml_file:
        yaml.dump(service_yaml, yaml_file, default_flow_style=False)
    logging.info("Service YAML file created successfully.")

def get_gke_cluster_credentials():
    """Fetches GKE cluster credentials."""
    try:
        logging.info(f"Fetching credentials for GKE cluster: {CLUSTER_NAME}")
        subprocess.run(
            f"gcloud container clusters get-credentials {CLUSTER_NAME} --zone {ZONE} --project {PROJECT_ID}",
            shell=True, check=True)
        logging.info("GKE credentials fetched successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error fetching GKE credentials: {e}")
        raise

def apply_kubernetes_yaml(file_path):
    """Applies the Kubernetes YAML file using kubectl."""
    try:
        logging.info(f"Applying Kubernetes YAML: {file_path}")
        subprocess.run(f"kubectl apply -f {file_path}", shell=True, check=True)
        logging.info(f"{file_path} applied successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying Kubernetes YAML: {file_path}. Error: {e}")
        raise

def check_deployment_status():
    """Checks the deployment status in GKE."""
    try:
        logging.info(f"Checking status of deployment: {DEPLOYMENT_NAME}")
        result = subprocess.run(
            f"kubectl get deployments {DEPLOYMENT_NAME} --namespace={NAMESPACE} -o=jsonpath='{{.status.availableReplicas}}'",
            shell=True, check=True, capture_output=True, text=True)
        if result.stdout.strip() == '1':
            logging.info(f"Deployment {DEPLOYMENT_NAME} is successfully running.")
        else:
            logging.warning(f"Deployment {DEPLOYMENT_NAME} is not fully available.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking deployment status: {e}")
        raise

def expose_service():
    """Exposes the deployment as a LoadBalancer service."""
    try:
        apply_kubernetes_yaml('service.yaml')
        logging.info(f"Service {SERVICE_NAME} exposed successfully.")
    except Exception as e:
        logging.error(f"Failed to expose service {SERVICE_NAME}: {e}")
        raise

def deploy_to_gke():
    """Deploys the application to GKE."""
    try:
        get_gke_cluster_credentials()
        create_kubernetes_deployment_yaml()
        apply_kubernetes_yaml('deployment.yaml')
        check_deployment_status()
        create_kubernetes_service_yaml()
        expose_service()
    except Exception as e:
        logging.error(f"Deployment to GKE failed: {e}")
        raise

def main():
    """Main function to trigger the deployment process."""
    try:
        logging.info("Starting GCP model deployment process.")
        upload_model_to_gcs()
        deploy_to_gke()
        logging.info("Deployment completed successfully.")
    except Exception as e:
        logging.error(f"Deployment process encountered an error: {e}")

if __name__ == "__main__":
    main()