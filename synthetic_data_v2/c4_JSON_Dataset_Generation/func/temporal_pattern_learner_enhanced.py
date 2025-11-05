"""
Enhanced Temporal Pattern Learner - Best of Both Approaches

Combines:
- Direct TSV parsing with real timestamps (from new code)
- Rich pattern learning: categories, transitions, seasonality (from old code)
- Week schedule generation for simulation (from new code)
- All improvements: BIC, log-space intervals, joint distributions
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import os
import sys
from sklearn.mixture import GaussianMixture

# Config paths
try:
    from c0_Configuration.config_paths import JSON_INPUT, JSON_FUNC, SAMPLED_FSQ_PLANNING_AREA
except:
    JSON_INPUT = None
    JSON_FUNC = "."

DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']


class EnhancedTemporalPatternLearner:
    """
    Learns comprehensive temporal patterns from FSQ data (TSV or JSON).
    """

    def __init__(self):
        self.patterns = {
            'visit_frequency': None,          # GMM (BIC-selected)
            'day_of_week': None,              # {days, probabilities}
            'time_of_day': None,              # {hours, probabilities}
            'month_of_year': None,            # {months, probabilities} ✅ KEPT
            'day_hour_joint': None,           # {matrix: 7x24, day_names} ✅ NEW
            'inter_visit_days': None,         # GMM on log(days) ✅ IMPROVED
            'category_temporal': {},          # per-category patterns ✅ KEPT
            'category_transitions': {},       # Markov chains ✅ KEPT
            'planning_area_patterns': {},     # area patterns ✅ ENHANCED
            'burst_probability': 0.0,         # data-driven ✅ IMPROVED
            'weekly_pattern': None,           # {probability, consistency}
            'poi_loyalty': {},                # {revisit_rate, avg_revisits} ✅ NEW
        }
        self.fitted = False

    def learn_from_tsv(self, tsv_path: str, has_categories=True):
        """
        Learn from TSV with columns:
        user_id, place_id, datetime, timezone, lat, lon, cluster_id, sampled_count, planning_area
        Optional: poi_category (if available)
        """
        print("\n=== Enhanced Learning from FSQ TSV ===")
        
        # Read TSV
        df = pd.read_csv(tsv_path, sep='\t', dtype={'timezone': 'Int64'})
        
        # Parse timestamps with timezone
        df['dt_utc'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        if 'timezone' in df.columns:
            df['tz_delta'] = pd.to_timedelta(df['timezone'].fillna(0), unit='m')
        else:
            df['tz_delta'] = pd.to_timedelta(0, unit='m')
        
        df['dt_local'] = df['dt_utc'] + df['tz_delta']
        df = df.dropna(subset=['dt_local'])
        
        # Derive temporal fields
        df['day_name'] = df['dt_local'].dt.day_name()
        df['hour'] = df['dt_local'].dt.hour
        df['month_name'] = df['dt_local'].dt.month_name()
        df['date'] = df['dt_local'].dt.date
        df['day_idx'] = df['dt_local'].dt.weekday
        
        n_checkins = len(df)
        n_users = df['user_id'].nunique()
        print(f"✅ Loaded {n_checkins:,} check-ins from {n_users} users")
        
        # ========== 1. Visit Frequency (BIC-selected GMM) ==========
        user_visit_counts = df.groupby('user_id').size().values.reshape(-1, 1)
        best_gmm, best_bic = None, np.inf
        for k in range(2, 6):
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
            gmm.fit(user_visit_counts)
            bic = gmm.bic(user_visit_counts)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        
        self.patterns['visit_frequency'] = best_gmm
        print(f"✅ Visit frequency: mean={user_visit_counts.mean():.1f}, "
              f"std={user_visit_counts.std():.1f}, components={best_gmm.n_components}")
        
        # ========== 2. Day-of-Week Distribution ==========
        day_counts = df['day_name'].value_counts().reindex(DAYS_ORDER, fill_value=0)
        day_probs = (day_counts / day_counts.sum()).tolist()
        self.patterns['day_of_week'] = {'days': DAYS_ORDER, 'probabilities': day_probs}
        
        # ========== 3. Time-of-Day Distribution ==========
        hour_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
        hour_probs = (hour_counts / hour_counts.sum()).tolist()
        self.patterns['time_of_day'] = {'hours': list(range(24)), 'probabilities': hour_probs}
        
        # ========== 4. Month Seasonality ==========
        month_counts = df['month_name'].value_counts().reindex(MONTHS_ORDER, fill_value=0)
        month_probs = (month_counts / month_counts.sum()).tolist()
        self.patterns['month_of_year'] = {'months': MONTHS_ORDER, 'probabilities': month_probs}
        print("✅ Learned month seasonality")
        
        # ========== 5. Joint Day×Hour Distribution ==========
        pivot = df.pivot_table(index='day_name', columns='hour', 
                               values='user_id', aggfunc='count', fill_value=0)
        pivot = pivot.reindex(index=DAYS_ORDER, columns=range(24), fill_value=0)
        joint_matrix = pivot.values.astype(float)
        joint_matrix /= joint_matrix.sum()
        self.patterns['day_hour_joint'] = {
            'matrix': joint_matrix.tolist(),
            'day_names': DAYS_ORDER
        }
        print("✅ Learned joint day×hour distribution (7×24)")
        
        # ========== 6. Inter-Visit Intervals (log-space GMM) ==========
        diffs_days = []
        for _, g in df.sort_values('dt_local').groupby('user_id'):
            dt_vals = g['dt_local'].values
            if len(dt_vals) >= 2:
                diffs = np.diff(dt_vals).astype('timedelta64[s]').astype(float) / 86400.0
                diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
                diffs_days.extend(diffs.tolist())
        
        if len(diffs_days) < 10:
            # Fallback power law
            diffs_days = list(np.clip(np.random.pareto(1.7, 1000) * 2 + 0.5, 0.5, 90.0))
        
        # Model in log-space (more stable)
        log_diffs = np.log(np.clip(diffs_days, 0.01, None)).reshape(-1, 1)
        best_iv_gmm, best_iv_bic = None, np.inf
        for k in range(2, 5):
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
            gmm.fit(log_diffs)
            bic = gmm.bic(log_diffs)
            if bic < best_iv_bic:
                best_iv_bic, best_iv_gmm = bic, gmm
        
        self.patterns['inter_visit_days'] = best_iv_gmm
        print(f"✅ Inter-visit intervals: log-GMM with {best_iv_gmm.n_components} components "
              f"(N={len(log_diffs)} samples)")
        
        # ========== 7. POI Loyalty Metrics ==========
        user_poi_counts = df.groupby(['user_id', 'place_id']).size()
        revisited = user_poi_counts[user_poi_counts >= 2]
        revisit_rate = len(revisited) / len(user_poi_counts) if len(user_poi_counts) > 0 else 0.0
        avg_revisits = revisited.mean() if len(revisited) > 0 else 0.0
        
        self.patterns['poi_loyalty'] = {
            'revisit_rate': float(revisit_rate),
            'avg_revisits': float(avg_revisits)
        }
        print(f"✅ POI loyalty: {revisit_rate*100:.1f}% revisited, avg {avg_revisits:.2f} times")
        
        # ========== 8. Burst Probability (data-driven) ==========
        user_daily_counts = df.groupby(['user_id', 'date']).size()
        burst_users = user_daily_counts[user_daily_counts >= 2]['user_id'].unique()
        burst_prob_emp = len(burst_users) / n_users if n_users > 0 else 0.20
        burst_prob = float(np.clip(burst_prob_emp, 0.15, 0.30))
        self.patterns['burst_probability'] = burst_prob
        
        # ========== 9. Weekly Habit Probability ==========
        weekly_users = 0
        for uid, g in df.groupby('user_id'):
            day_counts = g['day_name'].value_counts()
            if day_counts.sum() >= 3 and (day_counts.max() / day_counts.sum()) > 0.30:
                weekly_users += 1
        
        weekly_prob = weekly_users / n_users if n_users > 0 else 0.30
        self.patterns['weekly_pattern'] = {
            'probability': float(weekly_prob),
            'consistency': 0.7
        }
        
        # ========== 10. Category-Specific Patterns ==========
        if has_categories and 'poi_category' in df.columns:
            category_hours = df.groupby('poi_category')['hour'].apply(list)
            category_days = df.groupby('poi_category')['day_name'].apply(list)
            category_months = df.groupby('poi_category')['month_name'].apply(list)
            
            for cat in df['poi_category'].unique():
                if pd.isna(cat):
                    continue
                
                hours = category_hours.get(cat, [])
                days = category_days.get(cat, [])
                months = category_months.get(cat, [])
                
                if len(hours) >= 5:
                    # Hour distribution
                    hour_counts = pd.Series(hours).value_counts().reindex(range(24), fill_value=0)
                    hour_dist = (hour_counts / hour_counts.sum()).tolist()
                    
                    # Day distribution
                    day_counts = pd.Series(days).value_counts().reindex(DAYS_ORDER, fill_value=0)
                    day_dist = (day_counts / day_counts.sum()).to_dict()
                    
                    # Month distribution
                    month_counts = pd.Series(months).value_counts().reindex(MONTHS_ORDER, fill_value=0)
                    month_dist = (month_counts / month_counts.sum()).to_dict()
                    
                    self.patterns['category_temporal'][cat] = {
                        'hour_distribution': hour_dist,
                        'day_distribution': day_dist,
                        'month_distribution': month_dist,
                        'sample_count': len(hours)
                    }
            
            print(f"✅ Category patterns: {len(self.patterns['category_temporal'])} categories")
            
            # ========== 11. Category Transitions (Markov Chain) ==========
            category_sequences = []
            for uid, g in df.sort_values('dt_local').groupby('user_id'):
                cats = g['poi_category'].dropna().tolist()
                if len(cats) > 1:
                    category_sequences.append(cats)
            
            transitions = defaultdict(lambda: defaultdict(int))
            for seq in category_sequences:
                for i in range(len(seq) - 1):
                    transitions[seq[i]][seq[i+1]] += 1
            
            # Normalize to probabilities
            transition_probs = {}
            for cat1, next_cats in transitions.items():
                total = sum(next_cats.values())
                if total > 0:
                    transition_probs[cat1] = {
                        cat2: count/total for cat2, count in next_cats.items()
                    }
            
            self.patterns['category_transitions'] = transition_probs
            print(f"✅ Category transitions: {len(transition_probs)} learned")
        
        # ========== 12. Planning Area Patterns (Enhanced) ==========
        area_counts = df['planning_area'].value_counts()
        area_day_hour = df.groupby(['planning_area', 'day_name', 'hour']).size()
        
        area_patterns = {}
        for area in area_counts.index:
            if area_counts[area] >= 5:
                # Day preferences
                day_data = df[df['planning_area'] == area]['day_name'].value_counts()
                day_probs = (day_data.reindex(DAYS_ORDER, fill_value=0) / day_data.sum()).to_dict()
                
                # Hour preferences
                hour_data = df[df['planning_area'] == area]['hour'].value_counts()
                hour_probs = (hour_data.reindex(range(24), fill_value=0) / hour_data.sum()).tolist()
                
                area_patterns[area] = {
                    'visit_count': int(area_counts[area]),
                    'day_preferences': day_probs,
                    'hour_preferences': hour_probs  # ✅ NEW
                }
        
        self.patterns['planning_area_patterns'] = area_patterns
        print(f"✅ Planning area patterns: {len(area_patterns)} areas (with hour prefs)")
        
        self.fitted = True
        print("\n✅ Enhanced temporal pattern learning complete!")
        print(f"📊 Burst: {burst_prob*100:.1f}%  |  Weekly habits: {weekly_prob*100:.1f}%  "
              f"|  POI loyalty: {revisit_rate*100:.1f}%")
        
        return self

    def learn_from_json(self, json_path: str):
        """
        Learn from JSON input file (user_metadata format).
        This converts JSON to DataFrame format and uses same learning logic.
        """
        print("\n=== Enhanced Learning from JSON ===")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for user in data:
            user_id = user.get('user_id')
            metadata = user.get('user_metadata', [])
            
            for visit in metadata:
                # Parse day_of_week and time_of_day to reconstruct datetime
                day_name = visit.get('day_of_week', 'Monday')
                time_str = visit.get('time_of_day', '12:00 PM')
                
                # Create synthetic datetime (we don't have real timestamps in JSON)
                # Use current year, map day names to week days
                try:
                    day_idx = DAYS_ORDER.index(day_name)
                except ValueError:
                    day_idx = 0
                
                try:
                    time_obj = datetime.strptime(time_str, '%I:%M %p')
                    hour = time_obj.hour
                    minute = time_obj.minute
                except:
                    hour = 12
                    minute = 0
                
                row = {
                    'user_id': user_id,
                    'place_id': visit.get('poi_id', ''),
                    'poi_category': visit.get('poi_category', ''),
                    'planning_area': visit.get('planning_area', ''),
                    'lat': visit.get('lat', 0.0),
                    'lon': visit.get('lon', 0.0),
                    'day_name': day_name,
                    'day_idx': day_idx,
                    'hour': hour,
                    'time_of_day': time_str
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        n_checkins = len(df)
        n_users = df['user_id'].nunique()
        print(f"✅ Loaded {n_checkins:,} visits from {n_users} users (JSON format)")
        
        # ========== Learn patterns (same logic as TSV) ==========
        
        # 1. Visit Frequency
        user_visit_counts = df.groupby('user_id').size().values.reshape(-1, 1)
        best_gmm, best_bic = None, np.inf
        for k in range(2, 6):
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
            gmm.fit(user_visit_counts)
            bic = gmm.bic(user_visit_counts)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        
        self.patterns['visit_frequency'] = best_gmm
        print(f"✅ Visit frequency: mean={user_visit_counts.mean():.1f}, "
              f"std={user_visit_counts.std():.1f}, components={best_gmm.n_components}")
        
        # 2. Day-of-Week
        day_counts = df['day_name'].value_counts().reindex(DAYS_ORDER, fill_value=0)
        day_probs = (day_counts / day_counts.sum()).tolist()
        self.patterns['day_of_week'] = {'days': DAYS_ORDER, 'probabilities': day_probs}
        
        # 3. Time-of-Day
        hour_counts = df['hour'].value_counts().reindex(range(24), fill_value=0)
        hour_probs = (hour_counts / hour_counts.sum()).tolist()
        self.patterns['time_of_day'] = {'hours': list(range(24)), 'probabilities': hour_probs}
        
        # 4. Month Seasonality (use uniform since JSON lacks dates)
        month_probs = [1.0/12] * 12
        self.patterns['month_of_year'] = {'months': MONTHS_ORDER, 'probabilities': month_probs}
        print("✅ Month seasonality: uniform (no real dates in JSON)")
        
        # 5. Joint Day×Hour
        pivot = df.pivot_table(index='day_name', columns='hour', 
                               values='user_id', aggfunc='count', fill_value=0)
        pivot = pivot.reindex(index=DAYS_ORDER, columns=range(24), fill_value=0)
        joint_matrix = pivot.values.astype(float)
        joint_matrix /= joint_matrix.sum()
        self.patterns['day_hour_joint'] = {
            'matrix': joint_matrix.tolist(),
            'day_names': DAYS_ORDER
        }
        print("✅ Learned joint day×hour distribution (7×24)")
        
        # 6. Inter-Visit Intervals (estimate from visit counts)
        diffs_days = []
        for _, g in df.groupby('user_id'):
            n_visits = len(g)
            if n_visits >= 2:
                # Assume 180 days total, distribute evenly with noise
                avg_gap = 180.0 / n_visits
                for _ in range(n_visits - 1):
                    gap = max(0.5, np.random.gamma(2, avg_gap/2))
                    diffs_days.append(gap)
        
        if len(diffs_days) < 10:
            diffs_days = list(np.clip(np.random.pareto(1.7, 1000) * 2 + 0.5, 0.5, 90.0))
        
        log_diffs = np.log(np.clip(diffs_days, 0.01, None)).reshape(-1, 1)
        best_iv_gmm, best_iv_bic = None, np.inf
        for k in range(2, 5):
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
            gmm.fit(log_diffs)
            bic = gmm.bic(log_diffs)
            if bic < best_iv_bic:
                best_iv_bic, best_iv_gmm = bic, gmm
        
        self.patterns['inter_visit_days'] = best_iv_gmm
        print(f"✅ Inter-visit intervals: log-GMM with {best_iv_gmm.n_components} components (estimated)")
        
        # 7. POI Loyalty
        user_poi_counts = df.groupby(['user_id', 'place_id']).size()
        revisited = user_poi_counts[user_poi_counts >= 2]
        revisit_rate = len(revisited) / len(user_poi_counts) if len(user_poi_counts) > 0 else 0.0
        avg_revisits = revisited.mean() if len(revisited) > 0 else 0.0
        
        self.patterns['poi_loyalty'] = {
            'revisit_rate': float(revisit_rate),
            'avg_revisits': float(avg_revisits)
        }
        print(f"✅ POI loyalty: {revisit_rate*100:.1f}% revisited, avg {avg_revisits:.2f} times")
        
        # 8. Burst Probability (estimate from day/hour clustering)
        burst_prob = 0.20  # Default
        self.patterns['burst_probability'] = burst_prob
        
        # 9. Weekly Habit
        weekly_prob = 0.30  # Default
        self.patterns['weekly_pattern'] = {
            'probability': float(weekly_prob),
            'consistency': 0.7
        }
        
        # 10. Category-Specific Patterns
        category_hours = df.groupby('poi_category')['hour'].apply(list)
        category_days = df.groupby('poi_category')['day_name'].apply(list)
        
        for cat in df['poi_category'].unique():
            if pd.isna(cat) or cat == '':
                continue
            
            hours = category_hours.get(cat, [])
            days = category_days.get(cat, [])
            
            if len(hours) >= 5:
                hour_counts = pd.Series(hours).value_counts().reindex(range(24), fill_value=0)
                hour_dist = (hour_counts / hour_counts.sum()).tolist()
                
                day_counts = pd.Series(days).value_counts().reindex(DAYS_ORDER, fill_value=0)
                day_dist = (day_counts / day_counts.sum()).to_dict()
                
                month_dist = {m: 1.0/12 for m in MONTHS_ORDER}  # Uniform
                
                self.patterns['category_temporal'][cat] = {
                    'hour_distribution': hour_dist,
                    'day_distribution': day_dist,
                    'month_distribution': month_dist,
                    'sample_count': len(hours)
                }
        
        print(f"✅ Category patterns: {len(self.patterns['category_temporal'])} categories")
        
        # 11. Category Transitions
        category_sequences = []
        for uid, g in df.groupby('user_id'):
            cats = g['poi_category'].dropna().tolist()
            if len(cats) > 1:
                category_sequences.append(cats)
        
        transitions = defaultdict(lambda: defaultdict(int))
        for seq in category_sequences:
            for i in range(len(seq) - 1):
                transitions[seq[i]][seq[i+1]] += 1
        
        transition_probs = {}
        for cat1, next_cats in transitions.items():
            total = sum(next_cats.values())
            if total > 0:
                transition_probs[cat1] = {
                    cat2: count/total for cat2, count in next_cats.items()
                }
        
        self.patterns['category_transitions'] = transition_probs
        print(f"✅ Category transitions: {len(transition_probs)} learned")
        
        # 12. Planning Area Patterns
        area_counts = df['planning_area'].value_counts()
        area_patterns = {}
        
        for area in area_counts.index:
            if area and area_counts[area] >= 5:
                day_data = df[df['planning_area'] == area]['day_name'].value_counts()
                day_probs = (day_data.reindex(DAYS_ORDER, fill_value=0) / day_data.sum()).to_dict()
                
                hour_data = df[df['planning_area'] == area]['hour'].value_counts()
                hour_probs = (hour_data.reindex(range(24), fill_value=0) / hour_data.sum()).tolist()
                
                area_patterns[area] = {
                    'visit_count': int(area_counts[area]),
                    'day_preferences': day_probs,
                    'hour_preferences': hour_probs
                }
        
        self.patterns['planning_area_patterns'] = area_patterns
        print(f"✅ Planning area patterns: {len(area_patterns)} areas")
        
        self.fitted = True
        print("\n✅ Enhanced temporal pattern learning complete (from JSON)!")
        print(f"📊 Burst: {burst_prob*100:.1f}%  |  Weekly habits: {weekly_prob*100:.1f}%  "
              f"|  POI loyalty: {revisit_rate*100:.1f}%")
        
        return self

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.patterns, f)
        print(f"✅ Saved enhanced patterns to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.patterns = pickle.load(f)
        self.fitted = True
        print(f"✅ Loaded enhanced patterns from {path}")
        return self


class EnhancedTimeSeriesGenerator:
    """
    Generates temporal sequences using enhanced learned patterns.
    """

    def __init__(self, pattern_learner):
        if not pattern_learner.fitted:
            raise ValueError("Pattern learner must be fitted first!")
        self.learner = pattern_learner
        self.patterns = pattern_learner.patterns

    def generate_visit_count(self, min_visits=1, max_visits=100):
        """Sample visit count from learned GMM."""
        x = self.patterns['visit_frequency'].sample(1)[0][0]
        return int(np.clip(np.round(x), min_visits, max_visits))

    def _sample_day_hour_joint(self):
        """Sample (day_idx, hour) from learned joint distribution."""
        mat = np.array(self.patterns['day_hour_joint']['matrix'])
        flat = mat.ravel()
        if flat.sum() <= 0:
            return np.random.randint(0, 7), np.random.randint(0, 24)
        
        idx = np.random.choice(len(flat), p=flat/flat.sum())
        day_idx = idx // 24
        hour = idx % 24
        return int(day_idx), int(hour)

    def _sample_intervisit_days(self):
        """Sample inter-visit gap from log-GMM."""
        log_days = self.patterns['inter_visit_days'].sample(1)[0][0]
        days = float(np.exp(log_days))
        return float(np.clip(days, 0.25, 90.0))

    def generate_week_schedule(self, num_visits: int, week_start: datetime):
        """
        Generate visits within a 7-day window using joint day×hour distribution.
        Perfect for simulation weeks.
        
        Returns: list of (visit_idx, datetime, day_of_week)
        """
        if num_visits <= 0:
            return []

        visits = []
        used_slots = set()

        for i in range(num_visits):
            # Sample from joint distribution
            day_idx, hour = self._sample_day_hour_joint()
            
            # Map to actual week
            visit_day = week_start + timedelta(days=day_idx)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            # Avoid exact collisions
            slot_key = (day_idx, hour, minute // 5)
            if slot_key in used_slots:
                minute = (minute + np.random.randint(1, 10)) % 60
            used_slots.add(slot_key)
            
            visit_dt = visit_day.replace(hour=hour, minute=minute, second=second)
            visits.append((i, visit_dt, DAYS_ORDER[day_idx]))
            
            # Burst behavior: add same-day visit
            if np.random.random() < self.patterns['burst_probability']:
                burst_hour = (hour + np.random.choice([2, 3, 4])) % 24
                burst_dt = visit_day.replace(
                    hour=burst_hour,
                    minute=np.random.randint(0, 60),
                    second=np.random.randint(0, 60)
                )
                visits.append((i + 100000, burst_dt, DAYS_ORDER[day_idx]))
        
        visits.sort(key=lambda x: x[1])
        # Clean reindex
        return [(k, dt, day_name) for k, (_, dt, day_name) in enumerate(visits)]

    def generate_temporal_sequence(self, num_visits, start_date, num_days=180):
        """
        Long-span temporal sequence (legacy, still useful for 6-month histories).
        """
        if num_visits <= 0:
            return []

        sequence = []
        current = start_date

        # Weekly habit?
        has_weekly = np.random.random() < self.patterns['weekly_pattern']['probability']
        preferred_day = None
        if has_weekly:
            days = self.patterns['day_of_week']['days']
            probs = self.patterns['day_of_week']['probabilities']
            preferred_day = np.random.choice(days, p=probs)

        for i in range(num_visits):
            if i == 0:
                offset = np.random.randint(0, min(30, num_days))
            else:
                gap = self._sample_intervisit_days()
                offset = int(round(gap))
                if np.random.random() < self.patterns['burst_probability']:
                    offset = 0

            current = current + timedelta(days=offset)
            
            if (current - start_date).days >= num_days:
                current = start_date + timedelta(days=np.random.randint(0, num_days))

            # Weekly habit adjustment
            if has_weekly and preferred_day:
                if np.random.random() < self.patterns['weekly_pattern']['consistency']:
                    target_idx = DAYS_ORDER.index(preferred_day)
                    diff = (target_idx - current.weekday()) % 7
                    current = current + timedelta(days=diff)

            # Sample hour from joint
            day_idx, hour = self._sample_day_hour_joint()
            if day_idx != current.weekday() and np.random.random() < 0.5:
                hours = self.patterns['time_of_day']['hours']
                hprobs = self.patterns['time_of_day']['probabilities']
                hour = int(np.random.choice(hours, p=hprobs))

            visit_dt = current.replace(
                hour=hour,
                minute=np.random.randint(0, 60),
                second=np.random.randint(0, 60)
            )
            sequence.append((i, visit_dt, visit_dt.strftime('%A')))

        sequence.sort(key=lambda x: x[1])
        return sequence

    def generate_category_specific_time(self, category, base_datetime):
        """Adjust time for specific category using learned patterns."""
        if category in self.patterns['category_temporal']:
            cat_patterns = self.patterns['category_temporal'][category]
            hour_dist = cat_patterns['hour_distribution']
            hour = np.random.choice(range(24), p=hour_dist)
            
            return base_datetime.replace(
                hour=hour,
                minute=np.random.randint(0, 60)
            )
        return base_datetime

    def generate_category_sequence(self, start_category, length=5):
        """Generate category sequence using Markov transitions."""
        if 'category_transitions' not in self.patterns:
            return [start_category] * length

        sequence = [start_category]
        current = start_category

        for _ in range(length - 1):
            transitions = self.patterns['category_transitions'].get(current, {})
            if transitions:
                next_cats = list(transitions.keys())
                probs = [transitions[cat] for cat in next_cats]
                probs = np.array(probs) / sum(probs)
                next_cat = np.random.choice(next_cats, p=probs)
                sequence.append(next_cat)
                current = next_cat
            else:
                all_cats = list(self.patterns['category_temporal'].keys())
                if all_cats:
                    current = np.random.choice(all_cats)
                    sequence.append(current)
                else:
                    sequence.append(start_category)

        return sequence

    def should_revisit_poi(self):
        """Decide if user should revisit POI based on learned loyalty."""
        loyalty = self.patterns.get('poi_loyalty', {})
        revisit_rate = loyalty.get('revisit_rate', 0.3)
        return np.random.random() < revisit_rate


def train_enhanced_model(input_path: str, output_path: str, input_format='auto'):
    """
    Train and save enhanced model from JSON or TSV input.
    
    Args:
        input_path: Path to input file (JSON or TSV)
        output_path: Path to save model
        input_format: 'auto', 'json', or 'tsv'
    """
    learner = EnhancedTemporalPatternLearner()
    
    # Auto-detect format
    if input_format == 'auto':
        if input_path.endswith('.json'):
            input_format = 'json'
        elif input_path.endswith('.txt') or input_path.endswith('.tsv'):
            input_format = 'tsv'
        else:
            raise ValueError(f"Cannot auto-detect format for {input_path}")
    
    # Learn from appropriate format
    if input_format == 'json':
        print(f"📚 Training from JSON: {input_path}")
        learner.learn_from_json(input_path)
    elif input_format == 'tsv':
        print(f"📚 Training from TSV: {input_path}")
        learner.learn_from_tsv(input_path, has_categories=True)
    else:
        raise ValueError(f"Unknown input format: {input_format}")
    
    learner.save(output_path)
    
    # Print summary
    print("\n=== Enhanced Pattern Summary ===")
    print("Day-of-week:")
    for day, prob in zip(learner.patterns['day_of_week']['days'],
                        learner.patterns['day_of_week']['probabilities']):
        print(f"  {day:10s}: {prob*100:5.1f}%")
    
    print("\nPeak hours:")
    hours = learner.patterns['time_of_day']['hours']
    hprobs = learner.patterns['time_of_day']['probabilities']
    for h, p in sorted(zip(hours, hprobs), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {h:02d}:00 → {p*100:5.1f}%")
    
    if learner.patterns['category_transitions']:
        print(f"\nCategory transitions: {len(learner.patterns['category_transitions'])}")
    
    return learner


if __name__ == "__main__":
    # Use JSON_INPUT from config (the processed input file)
    input_path = JSON_INPUT
    out_dir = os.path.join(JSON_FUNC, 'temporal_models')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'temporal_patterns_enhanced.pkl')
    
    print(f"Training enhanced temporal model from: {input_path}")
    learner = train_enhanced_model(input_path, model_path, input_format='auto')
    
    print("\n=== Testing Week Schedule ===")
    gen = EnhancedTimeSeriesGenerator(learner)
    num_visits = gen.generate_visit_count()
    
    week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = week_start - timedelta(days=week_start.weekday())
    
    week_seq = gen.generate_week_schedule(num_visits, week_start)
    print(f"\nGenerated {len(week_seq)} visits for week {week_start.date()}:")
    for i, (idx, dt, day) in enumerate(week_seq[:10]):
        print(f"  {dt.strftime('%Y-%m-%d %H:%M')} ({day})")
