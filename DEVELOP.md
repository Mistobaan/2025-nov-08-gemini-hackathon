# Project Setup

This section describes how to create a new Google Cloud project from the command line and configure it for this application.

1.  **Create a new project:**

    ```bash
    gcloud projects create proj-2025-nov-08-gemini-hackathon --name="2025-nov-08-gemini-hackathon"
    ```

    Replace `<your-project-id>` with a unique ID for your project and `<your-project-name>` with a display name for your project.

2.  **Set the project:**

    ```bash
    gcloud config set project <your-project-id>
    ```

3.  **Enable the Cloud Build and Cloud Run APIs:**

    ```bash
    gcloud services enable cloudbuild.googleapis.com run.googleapis.com
    ```

4.  **Create a new Artifact Registry repository:**

    ```bash
    gcloud artifacts repositories create tpu-kernel-bench \
      --repository-format=docker \
      --location=us-west1 \
      --description="Docker repository for tpu-kernel-bench"
    ```

    Replace `<your-repo-name>` with a name for your repository and `<your-app-name>` with the name of your application.

5.  **Set the region to California:**

    The region for California is `us-west1`. You will use this value for the `_DEPLOY_REGION` substitution when running the Cloud Build pipeline.

# Deploying to Google Cloud Run

This document provides the steps to deploy the application to Google Cloud Run.

## Prerequisites

*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured.
*   A Google Cloud project with the Cloud Run and Cloud Build APIs enabled.
*   A container registry (e.g., Google Artifact Registry) to store the Docker image.

## Using Cloud Build

This project includes a `cloudbuild.yaml` file that allows you to build and deploy the application with a single command.

1.  **Run the build:**

    ```bash
    gcloud builds submit --config cloudbuild.yaml --substitutions=_SERVICE_NAME=<your-app-name>,_AR_REPOSITORY=<your-repo-name>,_DEPLOY_REGION=us-west1
    ```

    Replace `<your-app-name>` with the name of your application and `<your-repo-name>` with the name of your repository.

## Manual Steps

If you prefer to build and deploy the application manually, you can follow these steps.

1.  **Build the Docker image:**

    ```bash
    gcloud builds submit --tag us-west1-docker.pkg.dev/$(gcloud config get-value project)/<your-repo-name>/<your-app-name>:$(git rev-parse --short HEAD)
    ```

    Replace `<your-repo-name>` with the name of your repository and `<your-app-name>` with the name of your application.

2.  **Deploy to Cloud Run:**

    ```bash
    gcloud run deploy <your-app-name> \
      --image us-west1-docker.pkg.dev/$(gcloud config get-value project)/<your-repo-name>/<your-app-name>:$(git rev-parse --short HEAD) \
      --platform managed \
      --region us-west1 \
      --allow-unauthenticated
    ```

    Replace `<your-app-name>` with the name of your application.

## Additional Information

*   The `Dockerfile` in the root of the project is a multi-stage Dockerfile that builds the Next.js application and creates a production-ready image.
*   The `gcloud builds submit` command builds the Docker image using Google Cloud Build and pushes it to the Artifact Registry.
*   The `gcloud run deploy` command deploys the container image to Google Cloud Run.
