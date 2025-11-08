#!/bin/bash

# Exit on error
set -e

# Get the project ID
PROJECT_ID=$(gcloud config get-value project)

# Set the service name, repository name, and region
SERVICE_NAME="hack"
AR_REPOSITORY="hack"
DEPLOY_REGION="us-west1"

# Build the Docker image
docker build -t us-west1-docker.pkg.dev/$PROJECT_ID/$AR_REPOSITORY/$SERVICE_NAME:latest .

# Configure Docker to use the gcloud CLI
gcloud auth configure-docker us-west1-docker.pkg.dev

# Push the Docker image
docker push us-west1-docker.pkg.dev/$PROJECT_ID/$AR_REPOSITORY/$SERVICE_NAME:latest
