#!/bin/bash
set -e

# TPU Worker Deployment Script
# This script deploys the TPU worker service to GKE

echo "=========================================="
echo "TPU Worker Service Deployment"
echo "=========================================="

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No GCP project configured"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✓ Project ID: $PROJECT_ID"

# Navigate to tpu directory
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Building TPU Worker Container"
echo "--------------------------------------"
docker build -f Dockerfile.tpu-worker -t gcr.io/${PROJECT_ID}/tpu-worker:latest .
echo "✓ TPU worker image built"

echo ""
echo "Step 2: Building Job Creator Container"
echo "--------------------------------------"
docker build -f Dockerfile.job-creator -t gcr.io/${PROJECT_ID}/job-creator:latest .
echo "✓ Job creator image built"

echo ""
echo "Step 3: Pushing Images to GCR"
echo "--------------------------------------"
docker push gcr.io/${PROJECT_ID}/tpu-worker:latest
echo "✓ TPU worker image pushed"
docker push gcr.io/${PROJECT_ID}/job-creator:latest
echo "✓ Job creator image pushed"

echo ""
echo "Step 4: Creating Kubernetes Resources"
echo "--------------------------------------"

# Apply service account and RBAC
kubectl apply -f k8s/service-account.yaml
echo "✓ Service account created"

# Update deployment with project ID
sed "s/PROJECT_ID/${PROJECT_ID}/g" k8s/job-creator-deployment.yaml | \
sed "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" | \
kubectl apply -f -
echo "✓ Job creator deployed"

echo ""
echo "Step 5: Verifying Deployment"
echo "--------------------------------------"
kubectl get pods -l app=job-creator
echo ""
kubectl get serviceaccount job-creator-sa

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure Firestore is enabled:"
echo "   gcloud firestore databases create --region=us-west1"
echo ""
echo "2. Submit a test job:"
echo "   python scripts/submit_test_job.py"
echo ""
echo "3. Monitor jobs:"
echo "   python scripts/monitor_job.py --list"
echo ""
echo "4. Watch job creator logs:"
echo "   kubectl logs -f -l app=job-creator"
