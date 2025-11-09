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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()

batch_v1 = client.BatchV1Api()
db = firestore.Client()

PROJECT_ID = os.environ.get("PROJECT_ID")
TPU_CORES = os.environ.get("TPU_CORES", "4")


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
                "job-id": job_id,
            },
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=3600,
            backoff_limit=2,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": "tpu-worker", "job-id": job_id}
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    node_selector={"cloud.google.com/gke-nodepool": "tpu-node-pool"},
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
                                requests={"memory": "4Gi", "cpu": "2"},
                            ),
                            security_context=client.V1SecurityContext(privileged=True),
                        )
                    ],
                ),
            ),
        ),
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
        if change.type.name == "ADDED" or change.type.name == "MODIFIED":
            doc = change.document
            job_data = doc.to_dict()

            # Only process pending jobs
            if job_data.get("status") == "pending":
                job_id = doc.id
                challenge_number = job_data.get("challenge_number")

                logger.info(
                    f"New pending job detected: {job_id}, challenge #{challenge_number}"
                )

                # Update status to scheduled
                doc.reference.update(
                    {"status": "scheduled", "scheduled_at": firestore.SERVER_TIMESTAMP}
                )

                # Create Kubernetes Job
                if create_kubernetes_job(job_id, challenge_number):
                    logger.info(f"Successfully scheduled job {job_id}")
                else:
                    # Revert to pending if failed
                    doc.reference.update({"status": "pending"})


def main():
    """Main job creator loop."""
    logger.info("=" * 60)
    logger.info("Job Creator Starting - Listening to Firestore")
    logger.info("=" * 60)

    # Watch the tpu_jobs collection
    col_query = db.collection("tpu_jobs")
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
