import json
from transformers import pipeline
from tqdm import tqdm

# Paths
JSON_PATH = r'C:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset\Code_Freeze_Changes\c5_JSON_Dataset_Generation\c0_scripts\LLM_method\user_profile_generation\app_profiles_all_users_LLM_reviews_unique_per_poi_time_updated.json'
USER_SUMMARY_PATH = r'C:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset\Code_Freeze_Changes\c5_JSON_Dataset_Generation\c0_scripts\LLM_method\user_summary_extraction\user_summaries_hf.csv'

# 1. Load user summaries (as JSON)
with open(USER_SUMMARY_PATH, 'r', encoding='utf-8') as f:
    user_summary_dict = json.load(f)

# 2. Load app profiles JSON
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    app_profiles = json.load(f)

# 3. Initialize summarization pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # set device=-1 for CPU
summarizer = pipeline("summarization", model="google/flan-t5-large", device=0)  # set device=-1 for CPU


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
    #prompt = prompt[:900]  # Truncate to fit model input size
    prompt = prompt[:4096]  # Truncate to fit model input size
    summary = summarizer(prompt, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# 4. Replace review texts
for user_profile in tqdm(app_profiles[:2], desc="Processing first 2 user profiles"):
    # Try both 'user_id' and nested 'user'->'userId'
    user_id = user_profile.get('user_id') or user_profile.get('user', {}).get('userId')
    if not user_id:
        continue
    user_id = str(user_id)
    user_summary = user_summary_dict.get(user_id, "")
    if not user_summary:
        continue  # Skip if no summary available

    reviews = user_profile.get('reviews') or user_profile.get('interaction', {}).get('reviews', [])
    for review in reviews:
        poi_id = review.get('poi_id') or review.get('poiId')
        time = review.get('time') or review.get('timestamp')
        original_review = review.get('reviewText', "")
        # Generate unique review
        unique_review = generate_unique_review(user_id, poi_id, time, original_review, user_summary)
        review['reviewText'] = unique_review

# 5. Save updated JSON
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(app_profiles, f, ensure_ascii=False, indent=2)

print("All reviews updated with unique, summary-based texts.")