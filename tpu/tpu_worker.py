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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Firestore
db = firestore.Client()


def verify_tpu_available():
    """Verify that TPU is properly detected by JAX."""
    try:
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]

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
        job_ref = db.collection("tpu_jobs").document(job_id)
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
        job_ref = db.collection("tpu_jobs").document(job_id)
        update_data = {"status": status, "updated_at": firestore.SERVER_TIMESTAMP}
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
        "jax": jax,
        "jnp": jnp,
        "logger": logger,
        "challenge_number": challenge_number,
        "__builtins__": __builtins__,
    }

    try:
        # Execute the code
        exec(code, exec_globals)

        # Check if there's a result to return
        if "result" in exec_globals:
            logger.info(
                f"Challenge #{challenge_number} result: {exec_globals['result']}"
            )
            return exec_globals["result"]
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
            "completed",
            result=str(result) if result is not None else None,
            completed_at=firestore.SERVER_TIMESTAMP,
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
    job_id = os.environ.get("JOB_ID")

    if not job_id:
        logger.error("Missing required environment variable: JOB_ID")
        sys.exit(1)

    logger.info(f"Job ID: {job_id}")

    # Update status to running
    update_job_status(job_id, "running", started_at=firestore.SERVER_TIMESTAMP)

    # Verify TPU availability
    if not verify_tpu_available():
        logger.error("TPU not available - aborting")
        update_job_status(job_id, "failed", error="TPU not available")
        sys.exit(1)

    try:
        # Get job specification from Firestore
        job_data = get_job_from_firestore(job_id)

        challenge_number = job_data["challenge_number"]
        code = job_data["code"]
        timeout = job_data.get("timeout", 300)

        # Execute the code
        result = execute_code(code, challenge_number, timeout)

        # Save results to Firestore
        save_result(job_id, challenge_number, result)

        logger.info("=" * 60)
        logger.info(f"Challenge #{challenge_number} completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Job failed: {e}")
        update_job_status(job_id, "failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
