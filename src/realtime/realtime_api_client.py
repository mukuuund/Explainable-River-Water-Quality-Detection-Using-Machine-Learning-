import requests
import urllib3
import json
import os
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_URL = "https://rtwqmsdb1.cpcb.gov.in/data/internet/layers/10/index.json"
RAW_DATA_PATH = "data/raw/realtime/cpcb_realtime_raw_response.json"
METADATA_PATH = "reports/realtime/realtime_api_fetch_metadata.json"

def fetch_realtime_data():
    """
    Fetches real-time water quality data from CPCB endpoint.
    Saves the raw JSON response and a metadata file.
    Falls back to a cached file if the live fetch fails.
    """
    print("Fetching live data from CPCB...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    metadata = {
        "endpoint_url": DATA_URL,
        "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
        "http_status_code": None,
        "number_of_records_fetched": 0,
        "success": False,
        "error_message": None,
        "fetch_mode": "failed"
    }

    try:
        response = requests.get(DATA_URL, timeout=30, verify=False)
        metadata["http_status_code"] = response.status_code

        if response.status_code == 200:
            data = response.json()
            metadata["number_of_records_fetched"] = len(data)
            metadata["success"] = True
            metadata["fetch_mode"] = "live_api"

            with open(RAW_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Success! Fetched {len(data)} points.")
        else:
            metadata["error_message"] = f"HTTP Error: {response.status_code}"
            print(f"Status Code: {response.status_code}")

    except Exception as e:
        metadata["error_message"] = str(e)
        print(f"Error fetching live data: {e}")

    # Fallback logic
    if not metadata["success"]:
        print("Live fetch failed. Attempting fallback to cached data...")
        if os.path.exists(RAW_DATA_PATH):
            print(f"Fallback successful. Using cached data at {RAW_DATA_PATH}.")
            metadata["fetch_mode"] = "cached_fallback"
            metadata["success"] = True # Marking success because we have data to proceed with
            with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata["number_of_records_fetched"] = len(data)
        else:
            error_msg = f"No cached data found at {RAW_DATA_PATH}. Pipeline cannot proceed."
            print(error_msg)
            metadata["fetch_mode"] = "failed"
            metadata["success"] = False
            metadata["error_message"] = error_msg

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return metadata

if __name__ == "__main__":
    fetch_realtime_data()
