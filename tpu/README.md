# TPU Worker Service Implementation

This directory contains the complete implementation of a TPU worker service for executing JAX code on Google Cloud TPU nodes, based on the plan in `TPU_WORKER.md`.

## 📁 Directory Structure

```
tpu/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── Dockerfile.tpu-worker              # TPU worker container
├── Dockerfile.job-creator             # Job creator container
├── tpu_worker.py                      # TPU worker Python script
├── job_creator.py                     # Job creator Python script
├── k8s/
│   ├── service-account.yaml           # Kubernetes service account & RBAC
│   └── job-creator-deployment.yaml    # Job creator deployment
└── scripts/
    ├── deploy.sh                      # Deployment script
    ├── submit_test_job.py             # Submit test job to Firestore
    └── monitor_job.py                 # Monitor job status
```

## 🏗️ Architecture

The system uses Firestore for real-time job queue management:

1. **Job Submission** → Job document created in Firestore with `status: "pending"`
2. **Job Creator** → Listens to Firestore, detects pending jobs, creates K8s Jobs
3. **TPU Worker** → Executes on TPU node, updates status in real-time
4. **Results** → Saved back to Firestore with completion status

### Why Firestore?

- ✅ Real-time listeners (no polling)
- ✅ Built-in status tracking
- ✅ Simpler than GCS + Pub/Sub
- ✅ Easy monitoring and debugging

## 🚀 Quick Start

### Prerequisites

1. GKE cluster with TPU node pool (see `K8S_CLUSTER_GOOGLE_CLOUD.md`)
2. Firestore enabled in your project
3. Docker installed locally
4. kubectl configured for your cluster
5. gcloud CLI authenticated

### 1. Enable Firestore

```bash
gcloud firestore databases create --region=us-west1
```

### 2. Create Firestore Service Account

```bash
PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create firestore-accessor \
  --display-name="Firestore Accessor for K8s"

# Grant permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:firestore-accessor@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

# Create key
gcloud iam service-accounts keys create firestore-key.json \
  --iam-account="firestore-accessor@${PROJECT_ID}.iam.gserviceaccount.com"

# Create Kubernetes secret
kubectl create secret generic firestore-key \
  --from-file=key.json=firestore-key.json

# Clean up local key file
rm firestore-key.json
```

### 3. Deploy the Service

```bash
cd tpu
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This script will:
- Build both Docker images
- Push to Google Container Registry
- Create Kubernetes resources
- Deploy the job creator

### 4. Verify Deployment

```bash
# Check job creator pod
kubectl get pods -l app=job-creator

# View logs
kubectl logs -f -l app=job-creator

# Check service account
kubectl get serviceaccount job-creator-sa
```

## 📝 Usage

### Submit a Job

**Using Python script:**

```bash
# Submit test job for challenge #1
python scripts/submit_test_job.py 1

# Submit for different challenge
python scripts/submit_test_job.py 42
```

**Using Python code:**

```python
from google.cloud import firestore

db = firestore.Client()

job_data = {
    "challenge_number": 42,
    "code": """
import jax
import jax.numpy as jnp

# Your JAX/TPU code here
x = jnp.ones((1000, 1000))
result = jnp.dot(x, x).sum()
logger.info(f'Result: {result}')
""",
    "timeout": 300,
    "status": "pending",
    "created_at": firestore.SERVER_TIMESTAMP
}

_, doc_ref = db.collection('tpu_jobs').add(job_data)
print(f"Job ID: {doc_ref.id}")
```

### Monitor Jobs

**List recent jobs:**

```bash
python scripts/monitor_job.py --list
```

**Monitor specific job:**

```bash
python scripts/monitor_job.py <job_id>
```

**Follow job in real-time:**

```bash
python scripts/monitor_job.py <job_id> --follow
```

**Using Kubernetes:**

```bash
# Watch for TPU node scaling
watch kubectl get nodes

# Watch jobs
watch kubectl get jobs

# View worker logs (replace with actual job name)
kubectl logs -f job/tpu-job-1-<timestamp> -c tpu-worker
```

## 🔍 Job Lifecycle

Jobs progress through these states:

```
pending → scheduled → running → completed/failed
```

1. **pending**: Job created, waiting for job creator
2. **scheduled**: Kubernetes Job created, waiting for TPU node
3. **running**: Executing on TPU
4. **completed**: Successfully finished
5. **failed**: Error occurred

## 📊 Job Document Structure

```javascript
{
  "challenge_number": 42,
  "code": "import jax...",
  "timeout": 300,
  "status": "pending",
  "created_at": Timestamp,
  "scheduled_at": Timestamp,  // when K8s job created
  "started_at": Timestamp,    // when worker starts
  "completed_at": Timestamp,  // when finished
  "updated_at": Timestamp,    // last update
  "result": "...",            // output (if completed)
  "error": "...",             // error message (if failed)
  "metadata": {
    "submitted_by": "user@example.com"
  }
}
```

## 🧪 Testing

### Test TPU Detection

Create a simple test job to verify TPU is detected:

```python
from google.cloud import firestore

db = firestore.Client()
job_data = {
    "challenge_number": 0,
    "code": """
import jax
devices = jax.devices()
logger.info(f'Available devices: {devices}')
logger.info(f'TPU cores: {len([d for d in devices if d.platform == "tpu"])}')
result = f'Found {len(devices)} devices'
""",
    "timeout": 60,
    "status": "pending",
    "created_at": firestore.SERVER_TIMESTAMP
}
_, doc_ref = db.collection('tpu_jobs').add(job_data)
print(f"Test job ID: {doc_ref.id}")
```

### Test JAX Computation

```python
test_code = """
import jax.numpy as jnp

# Matrix multiplication on TPU
x = jnp.ones((1000, 1000))
y = jnp.ones((1000, 1000))
result = jnp.dot(x, y).sum()
logger.info(f'Computation result: {result}')
"""
```

## 🔧 Troubleshooting

### Job Creator Not Starting

```bash
# Check pod status
kubectl describe pod -l app=job-creator

# Check logs
kubectl logs -l app=job-creator

# Verify secret exists
kubectl get secret firestore-key
```

### Jobs Stuck in Pending

```bash
# Check if Firestore is accessible
gcloud firestore databases list

# Verify service account permissions
gcloud projects get-iam-policy $(gcloud config get-value project) \
  --flatten="bindings[].members" \
  --filter="bindings.members:firestore-accessor" 

# Check job creator is running
kubectl get pods -l app=job-creator
```

### TPU Not Detected

```bash
# Verify TPU node pool exists
kubectl get nodes -l cloud.google.com/gke-nodepool=tpu-node-pool

# Check TPU quota
gcloud compute tpus quota list

# Verify job requests TPU resources
kubectl describe job <job-name>
```

### Worker Fails to Start

```bash
# Check worker logs
kubectl logs job/<job-name> -c tpu-worker

# Verify image exists
gcloud container images list --repository=gcr.io/$(gcloud config get-value project)

# Check node has TPU
kubectl describe node <tpu-node-name>
```

## 🔄 Updating the Service

### Update TPU Worker Code

```bash
cd tpu

# Make changes to tpu_worker.py
# Then rebuild and deploy

PROJECT_ID=$(gcloud config get-value project)
docker build -f Dockerfile.tpu-worker -t gcr.io/${PROJECT_ID}/tpu-worker:latest .
docker push gcr.io/${PROJECT_ID}/tpu-worker:latest

# No need to restart - new jobs will use new image
```

### Update Job Creator

```bash
# Make changes to job_creator.py
# Then rebuild and redeploy

PROJECT_ID=$(gcloud config get-value project)
docker build -f Dockerfile.job-creator -t gcr.io/${PROJECT_ID}/job-creator:latest .
docker push gcr.io/${PROJECT_ID}/job-creator:latest

# Restart job creator
kubectl rollout restart deployment job-creator
```

## 🧹 Cleanup

### Delete All Jobs

```bash
kubectl delete jobs -l app=tpu-worker
```

### Remove Completed Jobs

```bash
# Jobs auto-cleanup after 1 hour due to ttlSecondsAfterFinished
# Or manually:
kubectl delete jobs -l app=tpu-worker --field-selector status.successful=1
```

### Uninstall Service

```bash
kubectl delete deployment job-creator
kubectl delete clusterrolebinding job-creator-binding
kubectl delete clusterrole job-creator-role
kubectl delete serviceaccount job-creator-sa
kubectl delete secret firestore-key
```

### Clear Firestore Data

```bash
# Using gcloud (careful!)
gcloud firestore databases delete "(default)"

# Or selectively delete jobs via console or Python script
```

## 📈 Monitoring & Observability

### View Job Creator Logs

```bash
kubectl logs -f -l app=job-creator
```

### Monitor TPU Node Scaling

```bash
# Watch nodes scale up/down
watch kubectl get nodes -l cloud.google.com/gke-nodepool=tpu-node-pool

# Check node pool autoscaling status
gcloud container node-pools describe tpu-node-pool \
  --cluster=tpu-cluster \
  --zone=us-west1-c
```

### Firestore Console

View jobs in real-time:
https://console.cloud.google.com/firestore/data

### Cloud Monitoring

Set up alerts for:
- Failed jobs
- TPU usage costs
- Long-running jobs
- Job creator health

## 🎯 Production Considerations

### Security

- [ ] Use Workload Identity instead of service account keys
- [ ] Restrict network policies for worker pods
- [ ] Validate code before execution (sandbox)
- [ ] Set resource limits appropriately

### Performance

- [ ] Add JAX compilation cache
- [ ] Preload common libraries in container
- [ ] Use multi-stage builds for smaller images
- [ ] Configure TPU topology appropriately

### Reliability

- [ ] Add health checks for job creator
- [ ] Implement retry logic with exponential backoff
- [ ] Set up alerting for failures
- [ ] Monitor Firestore quotas

### Cost Optimization

- [ ] Monitor TPU usage costs
- [ ] Adjust scale-down time
- [ ] Use preemptible nodes where possible
- [ ] Set appropriate timeout values

## 📚 Additional Resources

- [TPU_WORKER.md](../TPU_WORKER.md) - Original design document
- [K8S_CLUSTER_GOOGLE_CLOUD.md](../K8S_CLUSTER_GOOGLE_CLOUD.md) - Cluster setup
- [JAX Documentation](https://jax.readthedocs.io/)
- [Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [Firestore Documentation](https://cloud.google.com/firestore/docs)

## 💡 Tips

1. **Start small**: Test with simple jobs before complex ones
2. **Monitor costs**: TPUs are expensive, watch your spending
3. **Use timeouts**: Prevent runaway jobs
4. **Check logs**: Both job creator and worker logs are valuable
5. **Real-time monitoring**: Use `--follow` flag to watch jobs execute

## 🐛 Known Issues

- Workers require privileged mode for TPU access
- First TPU node startup can take 5-10 minutes
- Firestore has rate limits (500 writes/second per database)

## 🤝 Contributing

To extend this system:

1. **Add new features** to `tpu_worker.py` or `job_creator.py`
2. **Create helper scripts** in `scripts/`
3. **Update documentation** in this README
4. **Test thoroughly** before deploying

## 📧 Support

For issues:
1. Check troubleshooting section
2. Review logs from job creator and workers
3. Verify Firestore and GKE configurations
4. Check Google Cloud quotas and limits
