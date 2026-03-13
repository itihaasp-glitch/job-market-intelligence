import os
import json
import requests

print("=" * 55)
print("APIFY ACTOR FINDER — Testing multiple Indeed scrapers")
print("=" * 55)

apify_token = os.getenv("APIFY_TOKEN")
if not apify_token:
    print("✗ APIFY_TOKEN not set. Set it and rerun.")
    exit()

print(f"✓ Token: {apify_token[:10]}...\n")

# List of actors to try, in order of preference
ACTORS_TO_TRY = [
    {
        "id": "misceres~indeed-scraper",
        "input": {"keyword": "Data Engineer", "location": "Remote", "maxItems": 3},
        "title_field": ["job_title", "title", "positionName"],
    },
    {
        "id": "hynekhruza~indeed-scraper",
        "input": {"keyword": "Data Engineer", "location": "Remote", "maxItems": 3},
        "title_field": ["job_title", "title", "positionName"],
    },
    {
        "id": "borderline~indeed-scraper",
        "input": {"position": "Data Engineer", "country": "us", "location": "Remote", "maxItems": 3},
        "title_field": ["positionName", "title"],
    },
    {
        "id": "dhrumil~indeed-jobs",
        "input": {"searchQuery": "Data Engineer", "location": "Remote", "maxResults": 3},
        "title_field": ["title", "job_title", "positionName"],
    },
    {
        "id": "compass~crawler-google-places",  # fallback test
        "input": {"searchStringsArray": ["Data Engineer jobs Remote"]},
        "title_field": ["title"],
    },
]

working_actor = None

for actor in ACTORS_TO_TRY:
    actor_id = actor["id"]
    print(f"Testing: {actor_id}")
    print(f"  Input: {json.dumps(actor['input'])}")

    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"

    try:
        resp = requests.post(
            url,
            json=actor["input"],
            params={"token": apify_token},
            timeout=90,
        )

        print(f"  Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                # Try to find the job title in the response
                sample = data[0]
                title = None
                for field in actor["title_field"]:
                    title = sample.get(field)
                    if title:
                        break

                print(f"  ✓ SUCCESS — got {len(data)} results")
                print(f"  ✓ Sample title: {title or 'unknown field'}")
                print(f"  ✓ Response keys: {list(sample.keys())[:8]}")
                working_actor = actor_id
                print(f"\n{'='*55}")
                print(f"WORKING ACTOR FOUND: {actor_id}")
                print(f"Full sample result:")
                print(json.dumps(sample, indent=2, default=str)[:800])
                break
            else:
                print(f"  ✗ Returned 0 results or unexpected format: {str(data)[:150]}")

        elif resp.status_code == 400:
            err = resp.json().get("error", {})
            print(f"  ✗ Bad request: {err.get('message', resp.text[:150])}")
        elif resp.status_code == 402:
            print(f"  ✗ Payment required (free credits exhausted)")
        elif resp.status_code == 404:
            print(f"  ✗ Actor not found")
        elif resp.status_code == 408:
            print(f"  ✗ Timeout")
        else:
            print(f"  ✗ Error {resp.status_code}: {resp.text[:150]}")

    except requests.exceptions.Timeout:
        print(f"  ✗ Timed out after 90s")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

    print()

if not working_actor:
    print("=" * 55)
    print("NO WORKING ACTOR FOUND")
    print("Options:")
    print("  1. Check apify.com/store for updated Indeed scrapers")
    print("  2. Check your free credit balance at apify.com/billing")
    print("  3. Use the sample data runner (run_with_sample_data.py)")
    print("=" * 55)