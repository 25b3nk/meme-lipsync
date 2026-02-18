#!/usr/bin/env python3
"""E2E test for the meme lip-sync generator API."""

import sys
import time
import requests

BASE_URL = "http://localhost:8000"
TEST_FILE = "test_input.mp4"
TEST_TEXT = "Hello world this is a test"
POLL_INTERVAL = 5    # seconds between status polls
POLL_TIMEOUT = 1800  # 30 min — CPU inference is slow


def check_health():
    print("[1/4] Checking /health ...")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    assert data.get("status") == "ok", f"Unexpected health response: {data}"
    print("      OK")


def upload_file():
    print(f"[2/4] Uploading '{TEST_FILE}' ...")
    with open(TEST_FILE, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"file": (TEST_FILE, f, "video/mp4")},
            timeout=60,
        )
    if r.status_code != 200:
        print(f"      UPLOAD FAILED: {r.status_code} {r.text}")
        sys.exit(1)
    data = r.json()
    job_id = data["job_id"]
    print(f"      job_id = {job_id}")
    return job_id


def trigger_generation(job_id):
    print(f"[3/4] Triggering generation with text: '{TEST_TEXT}' ...")
    r = requests.post(
        f"{BASE_URL}/generate",
        json={"job_id": job_id, "text": TEST_TEXT},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"      GENERATE FAILED: {r.status_code} {r.text}")
        sys.exit(1)
    data = r.json()
    task_id = data["task_id"]
    print(f"      task_id = {task_id}")
    return task_id


def poll_status(task_id):
    print("[4/4] Polling /status/<task_id> ...")
    deadline = time.time() + POLL_TIMEOUT
    last_status = None

    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/status/{task_id}", timeout=15)
        if r.status_code != 200:
            print(f"      STATUS ERROR: {r.status_code} {r.text}")
            sys.exit(1)

        data = r.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        error = data.get("error")

        if status != last_status:
            print(f"      status={status}  progress={progress}%")
            last_status = status

        if status == "done":
            return data
        if status == "error":
            print(f"      PIPELINE ERROR: {error}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    print(f"      TIMEOUT: job did not complete within {POLL_TIMEOUT}s")
    sys.exit(1)


def verify_output(data):
    output_url = data.get("output_url")
    assert output_url, "Missing output_url in done response"
    print(f"\nOutput URL: {output_url}")

    full_url = f"{BASE_URL}{output_url}"
    print(f"Downloading output from {full_url} ...")
    r = requests.get(full_url, timeout=60)
    if r.status_code != 200:
        print(f"OUTPUT DOWNLOAD FAILED: {r.status_code} {r.text}")
        sys.exit(1)

    size = len(r.content)
    assert size > 0, "Output file is empty"
    print(f"Output file downloaded successfully ({size:,} bytes)")

    # Basic GIF header check
    content_type = r.headers.get("content-type", "")
    print(f"Content-Type: {content_type}")


def main():
    print("=" * 60)
    print("Meme Lip-Sync E2E Test")
    print("=" * 60)
    try:
        check_health()
        job_id = upload_file()
        task_id = trigger_generation(job_id)
        result = poll_status(task_id)
        verify_output(result)
        print("\n[PASS] Full pipeline completed successfully!")
    except SystemExit:
        print("\n[FAIL] Test failed — see errors above.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[FAIL] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
