#!/usr/bin/env python3
"""
Monitor a Firestore job's status and display results.
"""

from google.cloud import firestore
import sys
import time
from datetime import datetime


def format_timestamp(ts):
    """Format Firestore timestamp."""
    if ts:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return "N/A"


def monitor_job(job_id: str, follow: bool = False):
    """Monitor a specific job."""
    db = firestore.Client()
    job_ref = db.collection("tpu_jobs").document(job_id)

    print(f"Monitoring job: {job_id}")
    print("=" * 60)

    last_status = None

    while True:
        job_doc = job_ref.get()

        if not job_doc.exists:
            print(f"❌ Job {job_id} not found")
            return

        job_data = job_doc.to_dict()
        status = job_data.get("status", "unknown")

        # Only print if status changed or first time
        if status != last_status:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")

            if status == "pending":
                print("  ⏳ Waiting for job creator to schedule...")
            elif status == "scheduled":
                print("  🚀 Job scheduled, waiting for TPU node...")
                print(
                    f"  Scheduled at: {format_timestamp(job_data.get('scheduled_at'))}"
                )
            elif status == "running":
                print("  ▶️  Job running on TPU...")
                print(f"  Started at: {format_timestamp(job_data.get('started_at'))}")
            elif status == "completed":
                print("  ✅ Job completed successfully!")
                print(
                    f"  Completed at: {format_timestamp(job_data.get('completed_at'))}"
                )
                if "result" in job_data:
                    print(f"\n  Result:\n  {job_data['result']}")
                break
            elif status == "failed":
                print("  ❌ Job failed!")
                if "error" in job_data:
                    print(f"\n  Error:\n  {job_data['error']}")
                break

            last_status = status

        if not follow or status in ["completed", "failed"]:
            break

        time.sleep(2)

    print("\n" + "=" * 60)


def list_recent_jobs(limit: int = 10):
    """List recent jobs."""
    db = firestore.Client()

    print("Recent Jobs:")
    print("=" * 80)
    print(f"{'Job ID':<25} {'Status':<12} {'Challenge':<10} {'Created':<20}")
    print("-" * 80)

    jobs = (
        db.collection("tpu_jobs")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    for job in jobs:
        data = job.to_dict()
        job_id = job.id
        status = data.get("status", "unknown")
        challenge = data.get("challenge_number", "?")
        created = format_timestamp(data.get("created_at"))

        # Color coding
        status_emoji = {
            "pending": "⏳",
            "scheduled": "📅",
            "running": "▶️",
            "completed": "✅",
            "failed": "❌",
        }.get(status, "❓")

        print(f"{job_id:<25} {status_emoji} {status:<10} #{challenge:<9} {created}")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python monitor_job.py <job_id>          # Monitor specific job")
        print("  python monitor_job.py <job_id> --follow # Follow job status")
        print("  python monitor_job.py --list            # List recent jobs")
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_recent_jobs()
    else:
        job_id = sys.argv[1]
        follow = "--follow" in sys.argv or "-f" in sys.argv
        monitor_job(job_id, follow)
