import os
import json
import csv
from openai import OpenAI
from pydantic import BaseModel, Field
from c0_Configuration.s00_config_paths import POI_UNIQUE_ID_MAPPING, OUTPUT_SAMPLED_FSQ_PATH

PROMPT_TEMPLATE_PATH = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/prompt_for_user_summary.json"
CHECKIN_FILE = OUTPUT_SAMPLED_FSQ_PATH
POI_MAPPING_FILE = POI_UNIQUE_ID_MAPPING
OUTPUT_FILE = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_llm.json"
# for testing, limiting to first user
MAX_USERS = 1

client = OpenAI()

class FrequentArea(BaseModel):
    area: str
    activity: str
    time_pattern: str

    class Config:
        extra = "forbid"

class PersonaCharacteristics(BaseModel):
    age_group: str
    likely_profession: str
    lifestyle: str
    personality_traits: list[str]

    class Config:
        extra = "forbid"

class UserSummary(BaseModel):
    user_id: str
    user_summary: str
    poi_preferences: list[str]
    frequent_areas: list[FrequentArea] = Field(default_factory=list)
    persona_characteristics: PersonaCharacteristics = Field(default_factory=PersonaCharacteristics)

    class Config:
        extra = "forbid"

def load_prompt_template():
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(template, user_id, checkins, poi_mapping, example_summary=None):
    context = template["context"]
    instructions = "\n".join([f"{i+1}. {inst}" for i, inst in enumerate(template["instructions"])])
    constraints = f"- Output format: {template['constraints']['output_format']}\n- Do not include {template['constraints']['forbidden']}"
    user_data = f"\nUser ID: {user_id}\nUser check-ins:\n{json.dumps(checkins, ensure_ascii=False)}\nPOI mapping:\n{json.dumps(poi_mapping, ensure_ascii=False)}"
    example = f"\nExample summary:\n{json.dumps(example_summary, ensure_ascii=False)}" if example_summary else ""
    prompt = (
        f"You are {template['role']}. Your goal is to {template['task']}.\n\n"
        f"Context:\n{context}\n\n"
        f"Instructions:\n{instructions}\n\n"
        f"Constraints:\n{constraints}\n"
        f"{user_data}"
        f"{example}\n\n"
        "Now, based on the above, provide the output:"
    )
    return prompt

def main():
    # Load prompt template
    template = load_prompt_template()

    # Load example summary for reference (optional)
    example_summary = None
    if os.path.exists("example_user_summary.json"):
        with open("example_user_summary.json", "r", encoding="utf-8") as f:
            example_summary = json.load(f)

    # Load POI mapping
    poi_mapping = {}
    with open(POI_MAPPING_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                poi_mapping[row[0]] = row[1]

    # Load user check-ins
    user_checkins = {}
    with open(CHECKIN_FILE, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            user_id, place_id, datetime, _ = parts[:4]
            if user_id not in user_checkins:
                user_checkins[user_id] = []
            user_checkins[user_id].append({
                "place_id": place_id,
                "datetime": datetime
            })

    # Generate summaries
    summaries = []
    for user_id, checkins in user_checkins.items():
        if not checkins:
            continue
        checkins = checkins[-MAX_USERS:]  # Limit to MAX_USERS for testing
        user_poi_ids = {c['place_id'] for c in checkins}
        user_poi_mapping = {pid: poi_mapping[pid] for pid in user_poi_ids if pid in poi_mapping}
        prompt = build_prompt(template, user_id, checkins, user_poi_mapping, example_summary)
        print(f"Processing user {user_id} with {len(checkins)} check-ins...")
        try:
            response = client.responses.parse(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": "Extract the user summary as per the prompt."},
                    {"role": "user", "content": prompt},
                ],
                text_format=UserSummary,
            )
            summary = response.output_parsed
            summaries.append(summary.dict())
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            continue
        break

    # Save all summaries
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(summaries)} user summaries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()