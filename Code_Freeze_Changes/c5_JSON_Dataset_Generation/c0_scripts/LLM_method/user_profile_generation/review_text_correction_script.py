import json
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Paths
JSON_PATH = r'C:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset\Code_Freeze_Changes\c5_JSON_Dataset_Generation\c0_scripts\LLM_method\user_profile_generation\app_profiles_all_users_LLM_reviews_unique_per_poi_time_updated.json'
CSV_PATH = r'C:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset\Code_Freeze_Changes\c5_JSON_Dataset_Generation\c0_scripts\LLM_method\user_summary_extraction\user_summaries_hf.csv'

# 1. Load user summaries
user_summaries = pd.read_csv(CSV_PATH)
user_summary_dict = dict(zip(user_summaries['user_id'], user_summaries['summary']))

# 2. Load app profiles JSON
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    app_profiles = json.load(f)

# 3. Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # set device=-1 for CPU

def generate_unique_review(user_id, poi_id, time, original_review, user_summary):
    """
    Generate a unique review text using user summary and original review.
    """
    prompt = (
        f"User summary: {user_summary}\n"
        f"POI: {poi_id}\n"
        f"Time: {time}\n"
        f"Original review: {original_review}\n"
        "Write a unique, personalized review for this POI and time, reflecting the user's style."
    )
    # Summarize (you can adjust max_length/min_length as needed)
    summary = summarizer(prompt, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# 4. Replace review texts
for user_profile in tqdm(app_profiles, desc="Processing user profiles"):
    user_id = user_profile.get('user_id')
    user_summary = user_summary_dict.get(user_id, "")
    if not user_summary:
        continue  # Skip if no summary available

    for review in user_profile.get('reviews', []):
        poi_id = review.get('poi_id')
        time = review.get('time')
        original_review = review.get('reviewText', "")
        # Generate unique review
        unique_review = generate_unique_review(user_id, poi_id, time, original_review, user_summary)
        review['reviewText'] = unique_review

# 5. Save updated JSON
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(app_profiles, f, ensure_ascii=False, indent=2)

print("All reviews updated with unique, summary-based texts.")
