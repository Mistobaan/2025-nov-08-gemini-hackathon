#!/usr/bin/env python3
"""
Submit a test job to Firestore to trigger TPU execution.
"""

from google.cloud import firestore
import sys


def submit_job(challenge_number: int = 1):
    """Submit a test job to Firestore."""
    db = firestore.Client()

    # Test JAX/TPU code
    test_code = """import jax
import jax.numpy as jnp

logger.info('Testing TPU...')
x = jnp.ones((100, 100))
result = jnp.dot(x, x).sum()
logger.info(f'Test result: {result}')
"""

    job_data = {
        "challenge_number": challenge_number,
        "code": test_code,
        "timeout": 60,
        "status": "pending",
        "created_at": firestore.SERVER_TIMESTAMP,
        "metadata": {"submitted_by": "test_script", "description": "Test TPU job"},
    }

    # Add to Firestore
    _, doc_ref = db.collection("tpu_jobs").add(job_data)
    job_id = doc_ref.id

    print(f"✓ Test job submitted successfully!")
    print(f"  Job ID: {job_id}")
    print(f"  Challenge: #{challenge_number}")
    print(f"\nMonitor status with:")
    print(f"  python monitor_job.py {job_id}")

    return job_id


if __name__ == "__main__":
    challenge = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    submit_job(challenge)
