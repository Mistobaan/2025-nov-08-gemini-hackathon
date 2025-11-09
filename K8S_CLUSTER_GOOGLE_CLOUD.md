# Setting up a GKE cluster with a TPU node pool for queue-based jobs

This document explains how to set up a Google Kubernetes Engine (GKE) cluster with a TPU node pool that autoscales based on a job queue. The node pool will have a default size of 0 and will scale up to 1 when there are jobs in the queue. The node will remain active for 10 minutes after the queue is empty before scaling down.

## 1. Prerequisites

*   A Google Cloud project with billing enabled.
*   The `gcloud` command-line tool installed and configured.
*   The `kubectl` command-line tool installed.

## 2. Enable Google Kubernetes Engine API and Cloud TPU API

Before creating a GKE cluster with a TPU node pool, ensure both the Google Kubernetes Engine API and the Cloud TPU API are enabled for your project.

```bash
gcloud services enable container.googleapis.com
gcloud services enable tpu.googleapis.com
```

## 3. Check TPU Quotas

TPU resources are often limited and require explicit quota requests. Before proceeding, ensure you have sufficient TPU quota in your project for the desired TPU type and topology.

You can check your current quotas and request increases via the Google Cloud Console:
[https://console.cloud.google.com/iam-admin/quotas](https://console.cloud.google.com/iam-admin/quotas)

Look for "Cloud TPU API" quotas, specifically for "TPU v2 cores" or "TPU v3 cores" depending on your chosen `TPU_TYPE`.

## 4. Create the GKE cluster

First, create a GKE cluster. Choose a machine type that is suitable for your needs. For this example, we will use `e2-medium`.

```bash
export CLUSTER_NAME="tpu-cluster"
export ZONE="us-west1-c"

gcloud container clusters create $CLUSTER_NAME \
    --zone $ZONE \
    --machine-type "e2-medium" \
    --num-nodes "1"
```

## 5. Configure kubectl to connect to the cluster

After creating the GKE cluster, you need to configure `kubectl` to connect to it. This command downloads the cluster credentials and configures your local `kubectl` to use them:

```bash
export CLUSTER_NAME="tpu-cluster"
export ZONE="us-west1-c"
gcloud components install gke-gcloud-auth-plugin -y
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
```

This command will:

*   Fetch the cluster endpoint and authentication data
*   Update your `~/.kube/config` file with the cluster information
*   Set the current context to the newly created cluster

You can verify the connection by running:

```bash
kubectl cluster-info
kubectl get nodes
```

The first command will show the cluster endpoint, and the second will display the nodes in your cluster.

## 6. Create the TPU node pool

Next, create a TPU node pool with autoscaling enabled. We will set the minimum number of nodes to 0 and the maximum to 1. We will also set the `scale-down-unneeded-time` to 10 minutes.

```bash
export NODE_POOL_NAME="tpu-node-pool"
export TPU_TYPE="ct5lp-hightpu-1t" # these are the ones in us-west1-e 
export ZONE="us-west1-c"
export CLUSTER_NAME="tpu-cluster"

gcloud container node-pools create $NODE_POOL_NAME \
    --cluster $CLUSTER_NAME \
    --zone $ZONE \
    --machine-type $TPU_TYPE \
    --enable-autoscaling \
    --min-nodes "0" \
    --max-nodes "1"

# gcloud container node-pools update $NODE_POOL_NAME \
#     --cluster $CLUSTER_NAME \
#     --zone $ZONE \
#     --scale-down-unneeded-time=10m
```

**Note:** The `optimize-utilization` profile will remove nodes aggressively after 10 minutes of being unneeded.

## 7. Set up the job queue

We will use a Google Cloud Storage (GCS) bucket as a job queue. When a new file is uploaded to the bucket, a Pub/Sub notification will be sent. A Kubernetes Deployment will be listening to these notifications and will create a Kubernetes Job to process the file.

### 7.1. Create a GCS bucket

```bash
export BUCKET_NAME="tpu-job-queue-bucket"
gsutil mb gs://$BUCKET_NAME
****```

### 7.2. Create a Pub/Sub topic

```bash
export PUBSUB_TOPIC="tpu-job-queue-topic"
gcloud pubsub topics create $PUBSUB_TOPIC
```

### 7.3. Configure GCS notifications

```bash
gsutil notification create -t $PUBSUB_TOPIC -f json gs://$BUCKET_NAME
```

### 7.4. Create a Service Account for the Job Creator

Create a service account that has permissions to create Kubernetes Jobs.

```bash
export SA_NAME="job-creator-sa"
gcloud iam service-accounts create $SA_NAME

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member "serviceAccount:$SA_NAME@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role "roles/container.developer"

gcloud iam service-accounts keys create key.json --iam-account "$SA_NAME@$(gcloud config get-value project).iam.gserviceaccount.com"
kubectl create secret generic job-creator-sa-key --from-file=key.json
```

### 7.5. Create the Job Creator Deployment

This deployment will run a script that listens to the Pub/Sub topic and creates a Kubernetes Job for each message.

Create a file named `job-creator-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: job-creator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: job-creator
  template:
    metadata:
      labels:
        app: job-creator
    spec:
      containers:
      - name: job-creator
        image: google/cloud-sdk:slim
        command: ["/bin/sh", "-c"]
        args:
        - >
          gcloud pubsub subscriptions create job-creator-sub --topic tpu-job-queue-topic --ack-deadline=600 &&
          while true; do
            MESSAGES=$(gcloud pubsub subscriptions pull job-creator-sub --auto-ack --limit=1 --format="value(message.data)");
            for MESSAGE in $MESSAGES; do
              FILE_NAME=$(echo $MESSAGE | base64 --decode | jq -r .name);
              JOB_NAME="tpu-job-$(date +%s)";
              echo "Creating job $JOB_NAME for file $FILE_NAME";
              envsubst < job-template.yaml | kubectl apply -f -;
            done;
            sleep 10;
          done
        env:
        - name: BUCKET_NAME
          value: "tpu-job-queue-bucket"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/etc/gcp/key.json"
        volumeMounts:
        - name: gcp-key
          mountPath: /etc/gcp
          readOnly: true
      volumes:
      - name: gcp-key
        secret:
          secretName: job-creator-sa-key
```

Create a file named `job-template.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  template:
    spec:
      containers:
      - name: tpu-job
        image: your-tpu-processing-image:latest # Replace with your image
        command: ["python", "process.py", "--file", "gs://${BUCKET_NAME}/${FILE_NAME}"]
        resources:
          limits:
            google.com/tpu: "8" # Number of TPU cores
      restartPolicy: Never
  backoffLimit: 4
```

Now apply the deployment:

```bash
kubectl apply -f job-creator-deployment.yaml
```

## 8. Submit a job

To submit a job, upload a file to the GCS bucket:

```bash
gsutil cp my-file.txt gs://$BUCKET_NAME/
```

This will trigger the `job-creator` to create a new Kubernetes Job, which will in turn cause the TPU node pool to scale up.

## 9. Monitor the system

You can monitor the system using `kubectl` and the Google Cloud Console.

*   **Check the nodes:** `kubectl get nodes`
*   **Check the pods:** `kubectl get pods`
*   **Check the jobs:** `kubectl get jobs`
*   **Check the GKE cluster and node pool in the Google Cloud Console.**
*   **Check the Pub/Sub topic and subscriptions in the Google Cloud Console.**
*   **Check the GCS bucket in the Google Cloud Console.**

After 10 minutes of inactivity (no jobs in the queue), the TPU node will be automatically removed.

## 10. Deleting the GKE cluster

To clean up your GKE cluster, you can first list all clusters in the `us` regions to identify the one you want to delete.

```bash
gcloud container clusters list --regions=us-*
```

Once you have identified the cluster, you can delete it using the following command. Replace `YOUR_CLUSTER_NAME` and `YOUR_ZONE` with the appropriate values.

```bash
gcloud container clusters delete YOUR_CLUSTER_NAME --zone YOUR_ZONE
```
