#!/usr/bin/env python3
"""
LLM-based Synthetic Data Generation

Generates realistic user journeys using GPT-4 with structured outputs.
Each user gets a persona and realistic POI visit sequences.
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("Please create a .env file with: OPENAI_API_KEY=your_key")
    sys.exit(1)

try:
    import instructor
except ImportError:
    print("❌ Error: instructor library not installed")
    print("Please run: pip install instructor")
    sys.exit(1)

# Setup OpenAI with instructor
client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Define data structures
class ViewEvent(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp")
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str] = []
    duration: int = Field(description="View duration in seconds")
    referrer: str = Field(description="One of: map, search, ad, friend")
    userLocation: dict = Field(description="Lat/lon where user viewed from (home/work)")

class TransactionEvent(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp, 20-90 min after view")
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str] = []
    transactionId: str
    amount: float
    currency: str = "SGD"
    paymentMethod: str = Field(description="One of: credit_card, mobile_wallet, cash")
    userLocation: dict = Field(description="Lat/lon of POI location")

class ReviewEvent(BaseModel):
    timestamp: str = Field(description="ISO 8601 timestamp, 1-7 days after transaction")
    poiId: str
    poiCategories: List[str]
    poiSubcategories: List[str] = []
    rating: float = Field(description="Rating between 3.5 and 5.0")
    reviewText: str = Field(description="Brief realistic review text")
    userLocation: dict = Field(description="Lat/lon where review written (home)")

class UserJourney(BaseModel):
    userId: str
    age: int = Field(ge=18, le=75)
    gender: str = Field(description="One of: male, female, other")
    persona: str = Field(description="User type: Foodie Explorer, Convenience Seeker, Social Butterfly, Tourist, or Power Reviewer")
    homeLocation: dict = Field(description="User's home lat/lon in Singapore")
    views: List[ViewEvent]
    transactions: List[TransactionEvent]
    reviews: List[ReviewEvent]
    reasoning: str = Field(description="Brief explanation of user behavior pattern")


def load_real_fsq_data() -> dict:
    """Load real Foursquare Singapore data to learn patterns"""
    try:
        import pandas as pd
        
        # Load checkins
        checkins = pd.read_csv(
            'data/raw/FSQ_SG_2013_Checkins.csv',
            sep=',',
            names=['userId', 'poiId', 'timestamp', 'timezone'],
            encoding='latin1'
        )
        
        # Load POIs
        pois = pd.read_csv(
            'data/raw/FSQ_SG_2013_POI.csv',
            sep=',',
            names=['poiId', 'poiName', 'latitude', 'longitude', 'poiCategory', 'country'],
            encoding='latin1'
        )
        
        # Analyze patterns
        user_checkin_counts = checkins.groupby('userId').size()
        poi_popularity = checkins.groupby('poiId').size().sort_values(ascending=False)
        
        # Get top POIs with names
        top_pois = poi_popularity.head(100).index.tolist()
        poi_data = pois[pois['poiId'].isin(top_pois)].to_dict('records')
        
        # Calculate statistics
        stats = {
            'total_users': len(user_checkin_counts),
            'total_checkins': len(checkins),
            'total_pois': len(pois),
            'avg_checkins_per_user': user_checkin_counts.mean(),
            'median_checkins_per_user': user_checkin_counts.median(),
            'top_categories': checkins.merge(pois, on='poiId')['poiCategory'].value_counts().head(10).to_dict()
        }
        
        return {
            'pois': poi_data[:50],  # Top 50 POIs
            'stats': stats,
            'success': True
        }
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load FSQ data: {e}")
        return {'pois': [], 'stats': {}, 'success': False}


def load_poi_sample(poi_file: Path, sample_size: int = 50) -> List[dict]:
    """Load sample POIs for LLM to use (fallback if FSQ data unavailable)"""
    try:
        with open(poi_file, 'r') as f:
            all_pois = json.load(f)
        
        # Take diverse sample
        import random
        sample = random.sample(all_pois, min(sample_size, len(all_pois)))
        
        # Simplify for LLM context
        simplified = []
        for poi in sample:
            simplified.append({
                'poiId': poi['poiId'],
                'poiName': poi['poiName'],
                'poiCategories': poi['poiCategories'],
                'location': poi['userLocation']
            })
        
        return simplified
    except Exception as e:
        print(f"⚠️  Warning: Could not load POIs: {e}")
        return []


def generate_user(user_id: int, sample_pois: List[dict], fsq_stats: dict, num_visits: int = 10) -> dict:
    """Generate one realistic user journey using LLM"""
    
    stats_context = ""
    if fsq_stats:
        stats_context = f"""
Real Foursquare Singapore Dataset Statistics (2013):
- Total users: {fsq_stats.get('total_users', 'N/A')}
- Average checkins per user: {fsq_stats.get('avg_checkins_per_user', 'N/A'):.1f}
- Top categories: {', '.join(list(fsq_stats.get('top_categories', {}).keys())[:5])}

Use these patterns to generate realistic behavior.
"""
    
    prompt = f"""Generate a realistic Singapore user's POI interactions for user ID {user_id}.

{stats_context}

Available POIs (use actual IDs from this list - these are REAL Foursquare POIs):
{json.dumps(sample_pois[:30], indent=2)}

CRITICAL REQUIREMENTS:

1. MANDATORY FIELDS - DO NOT OMIT:
   - homeLocation: MUST be a dict with "latitude" and "longitude" (Singapore coords: 1.2-1.5, 103.6-104.0)
   - userLocation in ALL views/transactions/reviews: MUST be dict with "latitude" and "longitude"
   
2. CORRECT SCHEMA EXAMPLES:
   ✓ homeLocation: {{"latitude": 1.3521, "longitude": 103.8198}}
   ✓ userLocation: {{"latitude": 1.3048, "longitude": 103.8318}}
   ✗ WRONG: userLocation: "home" (must be coordinates dict)

3. FUNNEL LOGIC:
   - Generate exactly {num_visits} POI visits
   - Each transaction MUST have a matching view 20-90 minutes before
   - Same poiId for view and transaction pair
   - View timestamp < transaction timestamp (20-90 min gap)
   - Views from home/work location, transactions at POI actual location
   - Reviews for ~40% of transactions (1-7 days after)

4. REALISTIC BEHAVIOR:
   - Morning (7-10am): coffee/breakfast near home
   - Lunch (12-2pm): restaurants near work  
   - Evening (6-10pm): dinner, entertainment, shopping
   - Weekends: leisure, malls, outdoors
   - Use ONLY POI IDs from the provided list
   - Vary timestamps - don't use same time

5. PERSONA:
   - Assign ONE persona: Foodie Explorer, Convenience Seeker, Social Butterfly, Tourist, or Power Reviewer
   - Stay consistent with persona throughout all interactions

Return complete structured data with reasoning.
"""

    try:
        journey = client.chat.completions.create(
            model="gpt-4o",
            response_model=UserJourney,
            messages=[{
                "role": "system",
                "content": "You are an expert synthetic data generator for location-based services. You MUST follow the exact schema specifications. All location fields MUST be dictionaries with latitude/longitude keys, never strings. You ensure perfect temporal ordering: views before transactions, reviews after transactions."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_retries=2
        )
        
        # Format to match existing schema
        return {
            "user": {
                "userId": str(user_id),
                "age": journey.age,
                "gender": journey.gender,
                "location": {"city": "Singapore", "country": "SG"},
                "device": {
                    "platform": "iOS" if user_id % 2 == 0 else "Android",
                    "appVersion": "3.0.0"
                }
            },
            "interaction": {
                "views": [v.model_dump() for v in journey.views],
                "transactions": [t.model_dump() for t in journey.transactions],
                "reviews": [r.model_dump() for r in journey.reviews]
            },
            "metadata": {
                "persona": journey.persona,
                "homeLocation": journey.homeLocation,
                "reasoning": journey.reasoning,
                "generatedBy": "LLM-GPT4",
                "generatedAt": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"❌ Error generating user {user_id}: {e}")
        return None


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("LLM-BASED SYNTHETIC DATA GENERATION")
    print("="*60 + "\n")
    
    # Configuration
    POI_FILE = Path('src/utils/all_pois_final_dataset.json')
    OUTPUT_DIR = Path('data/llm_generated')
    OUTPUT_FILE = OUTPUT_DIR / 'synthetic_data_llm.json'
    NUM_USERS = 100  # Generate 100 users (~$5.25)
    
    # Load real FSQ data first (preferred)
    print(f"📖 Loading REAL Foursquare Singapore dataset...")
    fsq_data = load_real_fsq_data()
    
    if fsq_data['success']:
        print(f"✅ Loaded real FSQ data:")
        print(f"   • {len(fsq_data['pois'])} top POIs")
        print(f"   • {fsq_data['stats']['total_users']:,} users in dataset")
        print(f"   • {fsq_data['stats']['total_checkins']:,} total checkins")
        print(f"   • Avg {fsq_data['stats']['avg_checkins_per_user']:.1f} checkins/user\n")
        sample_pois = fsq_data['pois']
        fsq_stats = fsq_data['stats']
    else:
        print(f"⚠️  FSQ data not available, using fallback POI list...")
        sample_pois = load_poi_sample(POI_FILE, sample_size=50)
        fsq_stats = {}
    
    if not sample_pois:
        print("❌ No POIs loaded. Please ensure data files exist.")
        return
    
    # Generate users
    print(f"🤖 Generating {NUM_USERS} users with GPT-4...")
    print("⏱️  This may take 2-3 minutes...\n")
    
    users = []
    for i in range(1, NUM_USERS + 1):
        print(f"  Generating user {i}/{NUM_USERS}...", end=" ", flush=True)
        
        user = generate_user(i, sample_pois, fsq_stats, num_visits=8)
        
        if user:
            users.append(user)
            views = len(user['interaction']['views'])
            txns = len(user['interaction']['transactions'])
            reviews = len(user['interaction']['reviews'])
            persona = user['metadata']['persona']
            print(f"✅ {persona} | {views}V {txns}T {reviews}R")
        else:
            print("❌ Failed")
    
    # Save output
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"✅ Generated {len(users)} users")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    
    total_views = sum(len(u['interaction']['views']) for u in users)
    total_txns = sum(len(u['interaction']['transactions']) for u in users)
    total_reviews = sum(len(u['interaction']['reviews']) for u in users)
    
    print(f"\n📊 Statistics:")
    print(f"   Total views:        {total_views}")
    print(f"   Total transactions: {total_txns}")
    print(f"   Total reviews:      {total_reviews}")
    if total_txns > 0:
        print(f"   Review rate:        {total_reviews/total_txns*100:.1f}%")
    else:
        print(f"   Review rate:        N/A (no transactions)")
    
    print("\n💡 Next step: Run validation")
    print("   python scripts/validate_llm_output.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
