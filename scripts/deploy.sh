#!/bin/bash

# Exit on error
set -e

# Get the project ID
PROJECT_ID=$(gcloud config get-value project)

# Set the service name, repository name, and region
SERVICE_NAME="hack"
AR_REPOSITORY="hack"
DEPLOY_REGION="us-west1"

# Create the Artifact Registry repository if it doesn't exist
gcloud artifacts repositories create $AR_REPOSITORY \
  --repository-format=docker \
  --location=$DEPLOY_REGION \
  --description="Docker repository for $SERVICE_NAME" \
  --quiet || true

# Submit the build to Google Cloud Build
gcloud builds submit --config cloudbuild.yaml --substitutions=_SERVICE_NAME=$SERVICE_NAME,_AR_REPOSITORY=$AR_REPOSITORY,_DEPLOY_REGION=$DEPLOY_REGION
