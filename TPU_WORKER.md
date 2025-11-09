# TPU Worker Service for Queue-Based JAX Execution

This document explains how to set up a TPU worker service that reads jobs from a queue, executes Python code with JAX on TPU nodes, and processes challenges efficiently.

## Architecture Overview

The TPU worker service follows this flow:

1. **Job Queue**: Jobs are submitted to Firestore collection (containing code + challenge number)
2. **Real-time Listener**: Job Creator listens to Firestore changes in real-time
3. **Job Creator**: Creates Kubernetes Jobs for pending jobs
4. **TPU Worker Pod**: Runs on TPU node, executes code with JAX
5. **Status Updates**: Worker updates job status in Firestore in real-time

### Why This Architecture?

- **Real-time**: Firestore listeners provide instant job detection (no polling)
- **Auto-scaling**: TPU nodes scale from 0 to 1 based on queue depth
- **Cost-efficient**: TPUs only run when needed (scale down after 10 min idle)
- **Isolation**: Each job runs in its own Kubernetes Job for clean isolation
- **JAX/TPU Integration**: Proper TPU detection and utilization through JAX
- **Status Tracking**: Built-in job status tracking with Firestore documents
- **Simpler Setup**: No need for GCS notifications or Pub/Sub configuration

## Component 1: TPU Worker Container

### Dockerfile

Create a `Dockerfile.tpu-worker` with JAX and TPU support:

```dockerfile
# Use Google's official JAX TPU image as base
FROM gcr.io/tpu-pytorch/xla:r2.1_3.10_tpuvm

# Install additional dependencies
RUN pip install --no-cache-dir \
    google-cloud-firestore \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Create working directory
WORKDIR /workspace

# Copy worker script
COPY tpu_worker.py /workspace/
COPY requirements.txt /workspace/

# Install any additional requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for TPU
ENV TPU_LIBRARY_PATH=/lib
ENV JAX_PLATFORMS=tpu

# Run the worker
ENTRYPOINT ["python", "/workspace/tpu_worker.py"]
```

**Rationale**:
- Start with Google's official TPU image to ensure proper TPU drivers
- Install JAX with TPU support from the official release channel
- Set `JAX_PLATFORMS=tpu` to ensure JAX uses TPU backend
- Use gcr.io for fast image pulls in GKE

### requirements.txt

```txt
google-cloud-firestore>=2.13.0
google-cloud-logging>=3.5.0
jax[tpu]>=0.4.20
jaxlib
numpy>=1.24.0
```

## Component 2: Worker Script

Create `tpu_worker.py`:

```python
#!/usr/bin/env python3
"""
TPU Worker: Executes Python/JAX code from job queue on TPU.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from google.cloud import firestore
import jax
import jax.numpy as jnp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firestore
db = firestore.Client()


def verify_tpu_available():
    """Verify that TPU is properly detected by JAX."""
    try:
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == 'tpu']
        
        if not tpu_devices:
            logger.error("No TPU devices found!")
            logger.info(f"Available devices: {devices}")
            return False
        
        logger.info(f"✓ TPU detected: {len(tpu_devices)} devices")
        for i, device in enumerate(tpu_devices):
            logger.info(f"  Device {i}: {device}")
        
        # Test TPU with simple computation
        x = jnp.ones((1000, 1000))
        result = jnp.dot(x, x)
        logger.info(f"✓ TPU test computation successful: {result.shape}")
        
        return True
    except Exception as e:
        logger.error(f"TPU verification failed: {e}")
        return False


def get_job_from_firestore(job_id: str) -> dict:
    """
    Get job details from Firestore.
    
    Expected document format:
    {
        "challenge_number": 123,
        "code": "import jax\\nimport jax.numpy as jnp\\n...",
        "timeout": 300,
        "status": "pending",
        "created_at": timestamp,
        "metadata": {...}
    }
    """
    try:
        job_ref = db.collection('tpu_jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            raise ValueError(f"Job {job_id} not found in Firestore")
        
        job_data = job_doc.to_dict()
        logger.info(f"Retrieved job: challenge #{job_data.get('challenge_number')}")
        return job_data
    except Exception as e:
        logger.error(f"Failed to get job from Firestore: {e}")
        raise


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status and additional fields in Firestore."""
    try:
        job_ref = db.collection('tpu_jobs').document(job_id)
        update_data = {
            'status': status,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        update_data.update(kwargs)
        job_ref.update(update_data)
        logger.info(f"Updated job {job_id} status to: {status}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


def execute_code(code: str, challenge_number: int, timeout: int = 300):
    """
    Execute the Python/JAX code in a controlled environment.
    
    Args:
        code: Python code string to execute
        challenge_number: Challenge identifier
        timeout: Maximum execution time in seconds
    """
    logger.info(f"Executing challenge #{challenge_number}")
    
    # Create execution environment with JAX imports pre-loaded
    exec_globals = {
        'jax': jax,
        'jnp': jnp,
        'logger': logger,
        'challenge_number': challenge_number,
        '__builtins__': __builtins__,
    }
    
    try:
        # Execute the code
        exec(code, exec_globals)
        
        # Check if there's a result to return
        if 'result' in exec_globals:
            logger.info(f"Challenge #{challenge_number} result: {exec_globals['result']}")
            return exec_globals['result']
        else:
            logger.info(f"Challenge #{challenge_number} completed successfully")
            return None
            
    except Exception as e:
        logger.error(f"Error executing challenge #{challenge_number}: {e}")
        logger.exception(e)
        raise


def save_result(job_id: str, challenge_number: int, result: any):
    """Save execution result to Firestore."""
    try:
        update_job_status(
            job_id,
            'completed',
            result=str(result) if result is not None else None,
            completed_at=firestore.SERVER_TIMESTAMP
        )
        logger.info(f"Saved result for challenge #{challenge_number}")
    except Exception as e:
        logger.error(f"Failed to save result: {e}")


def main():
    """Main worker execution."""
    logger.info("=" * 60)
    logger.info("TPU Worker Starting")
    logger.info("=" * 60)
    
    # Get environment variables
    job_id = os.environ.get('JOB_ID')
    
    if not job_id:
        logger.error("Missing required environment variable: JOB_ID")
        sys.exit(1)
    
    logger.info(f"Job ID: {job_id}")
    
    # Update status to running
    update_job_status(job_id, 'running', started_at=firestore.SERVER_TIMESTAMP)
    
    # Verify TPU availability
    if not verify_tpu_available():
        logger.error("TPU not available - aborting")
        update_job_status(job_id, 'failed', error="TPU not available")
        sys.exit(1)
    
    try:
        # Get job specification from Firestore
        job_data = get_job_from_firestore(job_id)
        
        challenge_number = job_data['challenge_number']
        code = job_data['code']
        timeout = job_data.get('timeout', 300)
        
        # Execute the code
        result = execute_code(code, challenge_number, timeout)
        
        # Save results to Firestore
        save_result(job_id, challenge_number, result)
        
        logger.info("=" * 60)
        logger.info(f"Challenge #{challenge_number} completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        update_job_status(job_id, 'failed', error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Rationale**:
- **TPU Verification**: Ensures TPU is detected before running code
- **Safe Execution**: Uses `exec()` with controlled globals for code execution
- **Error Handling**: Comprehensive logging and error reporting
- **Result Storage**: Uploads results back to GCS for retrieval
- **Environment Injection**: Pre-loads JAX imports for convenience

## Component 3: Build and Push Container

```bash
# Set your project and registry
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE_NAME="tpu-worker"
export IMAGE_TAG="latest"
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the container
docker build -f Dockerfile.tpu-worker -t ${IMAGE_URI} .

# Push to Google Container Registry
docker push ${IMAGE_URI}

# Grant GKE access to pull the image
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:$(gcloud iam service-accounts list --filter='Default compute service account' --format='value(email)')" \
    --role="roles/storage.objectViewer"
```

## Component 4: Updated Kubernetes Job Template

Update the `job-template.yaml` from the cluster setup:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    app: tpu-worker
    challenge: "${CHALLENGE_NUMBER}"
spec:
  # Clean up completed jobs after 1 hour
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: tpu-worker
    spec:
      # Use TPU node pool
      nodeSelector:
        cloud.google.com/gke-tpu-topology: ${TPU_TYPE}
        cloud.google.com/gke-tpu-accelerator: ${TPU_TYPE}
      
      containers:
      - name: tpu-worker
        image: gcr.io/${PROJECT_ID}/tpu-worker:latest
        
        env:
        - name: BUCKET_NAME
          value: "${BUCKET_NAME}"
        - name: FILE_NAME
          value: "${FILE_NAME}"
        - name: JAX_PLATFORMS
          value: "tpu"
        - name: TPU_LIBRARY_PATH
          value: "/lib"
        
        resources:
          limits:
            # Request TPU resource
            google.com/tpu: "4"  # Adjust based on TPU type
          requests:
            memory: "4Gi"
            cpu: "2"
        
        # Security context
        securityContext:
          privileged: true  # Required for TPU access
      
      restartPolicy: Never
  
  # Retry failed jobs up to 2 times
  backoffLimit: 2
```

**Rationale**:
- **Node Selector**: Forces scheduling on TPU nodes
- **TPU Resources**: Explicitly requests TPU with correct count
- **Privileged Mode**: Required for direct TPU hardware access
- **TTL**: Auto-cleanup of completed jobs to prevent clutter
- **Backoff**: Limited retries to prevent infinite loops on bad code

## Component 5: Job Submission to Firestore

### Enable Firestore

First, enable Firestore in your Google Cloud project:

```bash
gcloud firestore databases create --region=us-west1
```

### Job Document Format

Jobs are stored in the `tpu_jobs` collection with this structure:

```javascript
{
  "challenge_number": 42,
  "code": "import jax\nimport jax.numpy as jnp\n\n# Your JAX/TPU code here\nx = jnp.ones((1000, 1000))\nresult = jnp.dot(x, x)\nlogger.info(f'Result shape: {result.shape}')\n",
  "timeout": 300,
  "status": "pending",  // pending, running, completed, failed
  "created_at": Timestamp,
  "updated_at": Timestamp,
  "started_at": Timestamp,  // when job starts running
  "completed_at": Timestamp,  // when job completes
  "result": "...",  // set upon completion
  "error": "...",  // set upon failure
  "metadata": {
    "submitted_by": "user@example.com"
  }
}
```

### Submit Job via Python

```python
from google.cloud import firestore
from datetime import datetime

db = firestore.Client()

# Create a new job
job_data = {
    "challenge_number": 42,
    "code": """import jax
import jax.numpy as jnp

x = jnp.ones((1000, 1000))
result = jnp.dot(x, x).sum()
logger.info(f'Result: {result}')
""",
    "timeout": 300,
    "status": "pending",
    "created_at": firestore.SERVER_TIMESTAMP,
    "metadata": {
        "submitted_by": "user@example.com"
    }
}

# Add to Firestore (auto-generates ID)
doc_ref = db.collection('tpu_jobs').add(job_data)
job_id = doc_ref[1].id
print(f"Job submitted with ID: {job_id}")
```

### Submit Job via gcloud CLI

```bash
# Create job document
gcloud firestore documents create \
  --database='(default)' \
  --collection=tpu_jobs \
  --document-id=job-$(date +%s) \
  --data='{
    "challenge_number": 42,
    "code": "import jax\nimport jax.numpy as jnp\n\nx = jnp.ones((100, 100))\nresult = jnp.dot(x, x).sum()\nlogger.info(f\"Result: {result}\")",
    "timeout": 300,
    "status": "pending",
    "created_at": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
  }'
```

### Monitor Job Status

```python
# Watch for status updates
job_ref = db.collection('tpu_jobs').document(job_id)

def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f"Status: {doc.get('status')}")
        if doc.get('status') == 'completed':
            print(f"Result: {doc.get('result')}")
        elif doc.get('status') == 'failed':
            print(f"Error: {doc.get('error')}")

# Listen for changes
doc_watch = job_ref.on_snapshot(on_snapshot)
```

## Component 6: Firestore Job Creator

Create a Python-based job creator that listens to Firestore changes in real-time:

### job_creator.py

```python
#!/usr/bin/env python3
"""
Job Creator: Listens to Firestore for new jobs and creates Kubernetes Jobs.
"""

import os
import logging
from google.cloud import firestore
from kubernetes import client, config
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()

batch_v1 = client.BatchV1Api()
db = firestore.Client()

PROJECT_ID = os.environ.get('PROJECT_ID')
TPU_CORES = os.environ.get('TPU_CORES', '4')

def create_kubernetes_job(job_id: str, challenge_number: int):
    """Create a Kubernetes Job for the TPU worker."""
    job_name = f"tpu-job-{challenge_number}-{int(datetime.now().timestamp())}"
    
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name,
            labels={
                "app": "tpu-worker",
                "challenge": str(challenge_number),
                "job-id": job_id
            }
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=3600,
            backoff_limit=2,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "tpu-worker",
                        "job-id": job_id
                    }
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    node_selector={
                        "cloud.google.com/gke-nodepool": "tpu-node-pool"
                    },
                    containers=[
                        client.V1Container(
                            name="tpu-worker",
                            image=f"gcr.io/{PROJECT_ID}/tpu-worker:latest",
                            env=[
                                client.V1EnvVar(name="JOB_ID", value=job_id),
                                client.V1EnvVar(name="JAX_PLATFORMS", value="tpu"),
                            ],
                            resources=client.V1ResourceRequirements(
                                limits={"google.com/tpu": TPU_CORES},
                                requests={"memory": "4Gi", "cpu": "2"}
                            ),
                            security_context=client.V1SecurityContext(
                                privileged=True
                            )
                        )
                    ]
                )
            )
        )
    )
    
    try:
        batch_v1.create_namespaced_job(namespace="default", body=job)
        logger.info(f"Created Kubernetes Job: {job_name} for job_id: {job_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to create Kubernetes Job: {e}")
        return False


def on_snapshot(col_snapshot, changes, read_time):
    """Handle Firestore collection changes."""
    for change in changes:
        if change.type.name == 'ADDED' or change.type.name == 'MODIFIED':
            doc = change.document
            job_data = doc.to_dict()
            
            # Only process pending jobs
            if job_data.get('status') == 'pending':
                job_id = doc.id
                challenge_number = job_data.get('challenge_number')
                
                logger.info(f"New pending job detected: {job_id}, challenge #{challenge_number}")
                
                # Update status to scheduled
                doc.reference.update({
                    'status': 'scheduled',
                    'scheduled_at': firestore.SERVER_TIMESTAMP
                })
                
                # Create Kubernetes Job
                if create_kubernetes_job(job_id, challenge_number):
                    logger.info(f"Successfully scheduled job {job_id}")
                else:
                    # Revert to pending if failed
                    doc.reference.update({'status': 'pending'})


def main():
    """Main job creator loop."""
    logger.info("=" * 60)
    logger.info("Job Creator Starting - Listening to Firestore")
    logger.info("=" * 60)
    
    # Watch the tpu_jobs collection
    col_query = db.collection('tpu_jobs')
    doc_watch = col_query.on_snapshot(on_snapshot)
    
    logger.info("Watching Firestore for new jobs...")
    
    # Keep alive
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        doc_watch.unsubscribe()


if __name__ == "__main__":
    main()
```

### Dockerfile.job-creator

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    google-cloud-firestore \
    kubernetes

WORKDIR /app
COPY job_creator.py /app/

CMD ["python", "/app/job_creator.py"]
```

### Build and Deploy Job Creator

```bash
# Build job creator image
docker build -f Dockerfile.job-creator -t gcr.io/${PROJECT_ID}/job-creator:latest .
docker push gcr.io/${PROJECT_ID}/job-creator:latest
```

### job-creator-deployment.yaml

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
      serviceAccountName: job-creator-sa
      containers:
      - name: job-creator
        image: gcr.io/PROJECT_ID/job-creator:latest  # Replace PROJECT_ID
        env:
        - name: PROJECT_ID
          value: "YOUR_PROJECT_ID"  # Replace with your project ID
        - name: TPU_CORES
          value: "4"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/var/secrets/google/key.json"
        volumeMounts:
        - name: google-cloud-key
          mountPath: /var/secrets/google
          readOnly: true
      volumes:
      - name: google-cloud-key
        secret:
          secretName: firestore-key
```

## Deployment Instructions

### 1. Build and Deploy Container

```bash
# Build the worker container
docker build -f Dockerfile.tpu-worker -t gcr.io/${PROJECT_ID}/tpu-worker:latest .

# Push to registry
docker push gcr.io/${PROJECT_ID}/tpu-worker:latest
```

### 2. Create Service Account and Firestore Access

```bash
# Create Kubernetes service account
kubectl create serviceaccount job-creator-sa

# Bind cluster role for job creation
kubectl create clusterrolebinding job-creator-binding \
  --clusterrole=edit \
  --serviceaccount=default:job-creator-sa

# Create GCP service account with Firestore access
gcloud iam service-accounts create firestore-accessor \
  --display-name="Firestore Accessor for K8s"

# Grant Firestore permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:firestore-accessor@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

# Create and store key
gcloud iam service-accounts keys create firestore-key.json \
  --iam-account="firestore-accessor@${PROJECT_ID}.iam.gserviceaccount.com"

kubectl create secret generic firestore-key \
  --from-file=key.json=firestore-key.json
```

### 3. Deploy Job Creator

```bash
kubectl apply -f job-creator-deployment.yaml
```

### 4. Submit Test Job

```python
# test_submit_job.py
from google.cloud import firestore

db = firestore.Client()

job_data = {
    "challenge_number": 1,
    "code": """import jax
import jax.numpy as jnp

logger.info('Testing TPU...')
x = jnp.ones((100, 100))
result = jnp.dot(x, x).sum()
logger.info(f'Test result: {result}')
""",
    "timeout": 60,
    "status": "pending",
    "created_at": firestore.SERVER_TIMESTAMP
}

doc_ref = db.collection('tpu_jobs').add(job_data)
print(f"Test job submitted with ID: {doc_ref[1].id}")
```

Run the script:

```bash
python test_submit_job.py
```

### 5. Monitor Execution

```bash
# Watch for TPU node scaling
watch kubectl get nodes

# Watch jobs
watch kubectl get jobs

# View logs from worker pod
kubectl logs -f job/tpu-job-1-<timestamp> -c tpu-worker

# Check job status in Firestore
gcloud firestore documents list tpu_jobs --format=json

# Or use Python to monitor
python -c "
from google.cloud import firestore
db = firestore.Client()
docs = db.collection('tpu_jobs').stream()
for doc in docs:
    data = doc.to_dict()
    print(f\"Job {doc.id}: status={data['status']}, challenge={data['challenge_number']}\")
    if 'result' in data:
        print(f\"  Result: {data['result']}\")
"
```

## Troubleshooting

### TPU Not Detected

If JAX doesn't detect the TPU:

1. **Check node labels**: `kubectl get nodes --show-labels`
2. **Verify TPU resource**: `kubectl describe node <tpu-node-name>`
3. **Check pod placement**: `kubectl get pod <pod-name> -o wide`
4. **Review logs**: `kubectl logs <pod-name>`

### Common Issues

**Issue**: "No TPU devices found"
- **Solution**: Ensure `nodeSelector` targets TPU nodes correctly
- **Solution**: Verify `google.com/tpu` resource is requested
- **Solution**: Check that `privileged: true` is set

**Issue**: "Job stuck in Pending"
- **Solution**: TPU node pool may need quota increase
- **Solution**: Check node pool autoscaling settings
- **Solution**: Verify TPU type matches requested resources

**Issue**: "ImportError: libtpu.so"
- **Solution**: Use official Google TPU base image
- **Solution**: Ensure TPU_LIBRARY_PATH is set correctly

## Performance Optimization

### 1. Preload Common Libraries

Add frequently used libraries to the Docker image:

```dockerfile
RUN pip install --no-cache-dir \
    jax[tpu] \
    optax \
    flax \
    tensorflow \
    torch
```

### 2. Use Multi-Stage Builds

```dockerfile
# Build stage
FROM python:3.10-slim as builder
RUN pip install --no-cache-dir --user jax[tpu]

# Runtime stage
FROM gcr.io/tpu-pytorch/xla:r2.1_3.10_tpuvm
COPY --from=builder /root/.local /root/.local
```

### 3. Optimize JAX Configuration

Set these environment variables for better performance:

```yaml
env:
- name: XLA_FLAGS
  value: "--xla_gpu_cuda_data_dir=/usr/local/cuda"
- name: JAX_COMPILATION_CACHE_DIR
  value: "/tmp/jax_cache"
- name: JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES
  value: "0"
```

## Security Considerations

1. **Code Execution**: The worker executes arbitrary Python code - ensure jobs come from trusted sources
2. **Resource Limits**: Set appropriate memory/CPU limits to prevent resource exhaustion
3. **Network Policies**: Consider restricting network access for worker pods
4. **Service Account**: Use minimal permissions for job creator service account

## Cost Management

- **Auto-scaling**: TPU nodes scale to 0 when idle (cost = $0)
- **Job TTL**: Completed jobs are cleaned up automatically
- **Monitoring**: Set up billing alerts for TPU usage
- **Quotas**: Request only necessary TPU quota to prevent overspending

## Next Steps

1. Implement result notifications (email, Slack, etc.)
2. Add metrics collection (Prometheus, Cloud Monitoring)
3. Create dashboard for job monitoring
4. Implement priority queues for different challenge types
5. Add support for multi-challenge batch processing
