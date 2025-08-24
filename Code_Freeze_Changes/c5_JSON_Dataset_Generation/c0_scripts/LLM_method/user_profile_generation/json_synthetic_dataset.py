import os, json, time, random, uuid, argparse
from typing import List, Optional, Dict, Any
from datetime import datetime
from dateutil.parser import isoparse

from pydantic import BaseModel, Field, ValidationError, field_validator
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
DEFAULT_INPUT  = "final_checkin_file.json"      # your dataset
DEFAULT_OUTPUT = "app_metadata_all_users_openai.jsonl"
MODEL = "gpt-4o-mini"   # JSON-mode capable, fast + cost-effective
MAX_RETRIES = 4
TEMPERATURE = 0.2

client = OpenAI()

# -----------------------------
# Pydantic schema for validation
# -----------------------------

class Coordinates(BaseModel):
    latitude: float
    longitude: float

class NearestStation(BaseModel):
    stationName: str
    stationCode: str
    coordinates: Coordinates

class POILocation(BaseModel):
    latitude: float
    longitude: float
    address: str

class Deal(BaseModel):
    dealId: str
    discount: str
    validUntil: str

class POI(BaseModel):
    poiId: str
    name: str
    categories: List[str]
    subcategories: List[str]
    location: POILocation
    nearestStation: NearestStation
    rating: float
    deal: Deal

class Device(BaseModel):
    platform: str
    appVersion: str

class UserLocation(BaseModel):
    latitude: float
    longitude: float

class View(BaseModel):
    timestamp: str
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str]
    duration: int
    referrer: str
    userLocation: UserLocation

    @field_validator("timestamp")
    def check_iso(cls, v):
        # must be ISO 8601 and Z or offset
        try:
            isoparse(v)
        except Exception:
            raise ValueError("timestamp must be ISO-8601")
        return v

class Transaction(BaseModel):
    timestamp: str
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str]
    transactionId: str
    amount: float
    currency: str
    paymentMethod: str
    userLocation: UserLocation

    @field_validator("timestamp")
    def check_iso(cls, v):
        try:
            isoparse(v)
        except Exception:
            raise ValueError("timestamp must be ISO-8601")
        return v

class Review(BaseModel):
    timestamp: str
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str]
    rating: float
    reviewText: str
    userLocation: UserLocation

    @field_validator("timestamp")
    def check_iso(cls, v):
        try:
            isoparse(v)
        except Exception:
            raise ValueError("timestamp must be ISO-8601")
        return v

class Interaction(BaseModel):
    views: List[View]
    transactions: List[Transaction]
    reviews: List[Review]

class UserBlock(BaseModel):
    userId: str
    age: int
    gender: str
    location: Dict[str, str]
    device: Device

class AppRecord(BaseModel):
    user: UserBlock
    interaction: Interaction
    pois: List[POI]

# -----------------------------
# Prompt assembly
# -----------------------------

SYSTEM_MSG = """You are a data product engineer that must return STRICT JSON ONLY.
Task: Convert a user's check-in history into an application metadata JSON with keys:
- user
- interaction (views, transactions, reviews)
- pois

Rules:
- Return ONLY a single JSON object that conforms to the schema below.
- Generate realistic but consistent fields when missing.
- Ensure ISO-8601 timestamps (UTC 'Z' or with offset).
- Use Singapore (SG) context for locations.
- No extra commentary, no markdown—JSON only.

Schema (shape, not types):
{
  "user": {
    "userId": "string",
    "age": number,
    "gender": "male|female|other",
    "location": {"city": "Singapore", "country": "SG"},
    "device": {"platform": "iOS|Android", "appVersion": "x.y.z"}
  },
  "interaction": {
    "views": [ { "timestamp": "...", "poiId": "...", "poiCategories": [...],
                 "poiSubcategories": [...], "duration": number, "referrer": "...",
                 "userLocation": {"latitude": number, "longitude": number} } ],
    "transactions": [ { "timestamp": "...", "poiId": "...", "poiCategories": [...],
                        "poiSubcategories": [...], "transactionId": "...",
                        "amount": number, "currency": "SGD",
                        "paymentMethod": "credit_card|mobile_wallet|cash",
                        "userLocation": {"latitude": number, "longitude": number} } ],
    "reviews": [ { "timestamp": "...", "poiId": "...", "poiCategories": [...],
                   "poiSubcategories": [...], "rating": number (3.0-5.0),
                   "reviewText": "string",
                   "userLocation": {"latitude": number, "longitude": number} } ]
  },
  "pois": [
    { "poiId": "...", "name": "...", "categories": [...], "subcategories": [...],
      "location": {"latitude": number, "longitude": number, "address": "string"},
      "nearestStation": {"stationName": "...", "stationCode": "...",
                         "coordinates": {"latitude": number, "longitude": number}},
      "rating": number (3.0-5.0),
      "deal": {"dealId": "...", "discount": "string", "validUntil": "YYYY-MM-DD"}
    }
  ]
}
"""

USER_INSTRUCTIONS = """You will be given:
1) A single user's check-ins (user_metadata: poi_category, planning_area, day_of_week, time_of_day, month_of_year).
2) The user's cluster hints (cluster_metadata: top categories, most active day/time, mean planning area).
3) A profile tag (Probable_user_profile_tag).

Produce an app metadata JSON as per the schema.
Guidelines:
- Derive 6–20 `views` from the check-ins; you may collapse similar visits.
- Create 30–70% as `transactions` from a subset of `views`.
- For each transaction, produce one `review`.
- Build `pois` for all distinct poiIds used in views/transactions/reviews (no duplicates).
- `poiId` can be derived deterministically (e.g., hash of category+planning_area) or simple unique IDs.
- Use Singapore lat/lon; jitter slightly per record.
- When uncertain, make reasonable, consistent choices.
- Keep values realistic (e.g., prices by category; ratings 3.5–5.0; common MRT stations).
- Output STRICT JSON ONLY."""

def make_user_message(user_obj: Dict[str, Any]) -> str:
    # Minimal, lossless payload; the system message has the schema/rules.
    return json.dumps({
        "user_id": user_obj.get("user_id"),
        "user_metadata": user_obj.get("user_metadata", []),
        "cluster_metadata": user_obj.get("cluster_metadata", {}),
        "Probable_user_profile_tag": user_obj.get("Probable_user_profile_tag", "General")
    }, ensure_ascii=False)

# -----------------------------
# OpenAI call with JSON mode
# -----------------------------

def call_openai_json(system_prompt: str, user_payload: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},  # 🔒 JSON-only
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_INSTRUCTIONS},
            {"role": "user", "content": user_payload},
        ],
    )
    return json.loads(resp.choices[0].message.content)

# -----------------------------
# Validation + repair loop
# -----------------------------

def validate_or_repair(obj: Dict[str, Any], raw_payload: str) -> Dict[str, Any]:
    # Try direct validation first
    try:
        AppRecord.model_validate(obj)
        return obj
    except ValidationError as e:
        last_error = str(e)

    # If validation fails, ask the model to repair it once or twice
    repair_prompt = f"""
You previously returned JSON that did not pass validation.
Validation errors:
{last_error}

Your task: return a corrected JSON that STRICTLY conforms to the schema.
Return only the corrected JSON. Here is the original user payload again:

{raw_payload}
""".strip()

    for _ in range(2):
        repaired = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": repair_prompt},
            ],
        )
        fixed = json.loads(repaired.choices[0].message.content)
        try:
            AppRecord.model_validate(fixed)
            return fixed
        except ValidationError as e:
            last_error = str(e)

    # If still failing, raise with context
    raise ValueError(f"JSON did not validate after repair attempts:\n{last_error}")

# -----------------------------
# Main
# -----------------------------

def process_file(input_path: str, output_path: str, max_users: Optional[int] = None):
    with open(input_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    n_total = len(users)
    if max_users is not None:
        users = users[:max_users]

    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, u in enumerate(users, start=1):
            # Retry on API/network hiccups
            delay = 2.0
            for attempt in range(MAX_RETRIES):
                try:
                    user_payload = make_user_message(u)
                    raw = call_openai_json(SYSTEM_MSG, user_payload)
                    valid = validate_or_repair(raw, user_payload)
                    out.write(json.dumps(valid, ensure_ascii=False) + "\n")
                    written += 1
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        # Write a minimal fallback record (so the pipeline is resilient)
                        fallback = {
                            "user": {
                                "userId": str(u.get("user_id")),
                                "age": 30,
                                "gender": "other",
                                "location": {"city": "Singapore", "country": "SG"},
                                "device": {"platform": "iOS", "appVersion": "3.1.0"}
                            },
                            "interaction": {"views": [], "transactions": [], "reviews": []},
                            "pois": []
                        }
                        out.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                        print(f"[WARN] User {u.get('user_id')} failed after retries: {e}")
                    else:
                        time.sleep(delay)
                        delay *= 1.7

            if idx % 20 == 0:
                print(f"Processed {idx}/{n_total}")

    print(f"Done. Wrote {written} records to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",  default=DEFAULT_INPUT,  help="Path to final_checkin_file.json")
    parser.add_argument("--outfile", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--max_users", type=int, default=None, help="Limit number of users")
    args = parser.parse_args()

    process_file(args.infile, args.outfile, args.max_users)
