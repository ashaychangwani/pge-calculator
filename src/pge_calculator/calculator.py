"""
PG&E Rate Calculator Core Logic

This module contains the main calculator class and rate calculation logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Union, Dict, Any
import os
from pathlib import Path

# Territory baseline allowances (kWh per day)
BASELINE_ALLOWANCES = {
    'X': {'basic_electric': {'summer': 9.8, 'winter': 9.7}, 'all_electric': {'summer': 8.5, 'winter': 14.6}},
    'P': {'basic_electric': {'summer': 13.5, 'winter': 11.0}, 'all_electric': {'summer': 15.2, 'winter': 26.0}},
    'Q': {'basic_electric': {'summer': 9.8, 'winter': 11.0}, 'all_electric': {'summer': 8.5, 'winter': 26.0}},
    'R': {'basic_electric': {'summer': 17.7, 'winter': 10.4}, 'all_electric': {'summer': 19.9, 'winter': 26.7}},
    'S': {'basic_electric': {'summer': 15.0, 'winter': 10.2}, 'all_electric': {'summer': 17.8, 'winter': 23.7}},
    'T': {'basic_electric': {'summer': 6.5, 'winter': 7.5}, 'all_electric': {'summer': 7.1, 'winter': 12.9}},
    'V': {'basic_electric': {'summer': 7.1, 'winter': 8.1}, 'all_electric': {'summer': 10.4, 'winter': 19.1}},
    'W': {'basic_electric': {'summer': 19.2, 'winter': 9.8}, 'all_electric': {'summer': 22.4, 'winter': 19.0}},
    'Y': {'basic_electric': {'summer': 10.5, 'winter': 11.1}, 'all_electric': {'summer': 12.0, 'winter': 24.0}},
    'Z': {'basic_electric': {'summer': 5.9, 'winter': 7.8}, 'all_electric': {'summer': 6.7, 'winter': 15.7}}
}


class PGERateCalculator:
    """Main calculator class for PG&E rate calculations."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the calculator with rate data.
        
        Args:
            data_path: Optional path to the rate data CSV file
        """
        self.data_path = data_path or self._get_default_data_path()
        self.rates_df = None
        self.load_rate_data()
        
    def _get_default_data_path(self) -> str:
        """Get the default path to the rate data file."""
        # Look for the file in the project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_file = project_root / "data" / "pge_rates.csv"
        
        if data_file.exists():
            return str(data_file)
        
        # Fallback to old location for backward compatibility
        old_data_file = project_root / "pge_rates_new.csv"
        if old_data_file.exists():
            return str(old_data_file)
            
        raise FileNotFoundError("Rate data file not found. Please ensure pge_rates.csv exists in the data directory.")
        
    def load_rate_data(self) -> None:
        """Load PG&E rate data from CSV file."""
        try:
            self.rates_df = pd.read_csv(self.data_path)

            # Pre-processing to simplify later look-ups
            def _time_to_hour(t_str):
                """Convert time string to fractional hour."""
                if pd.isna(t_str) or t_str == "":
                    return None  # signifies "all day"
                t_obj = datetime.strptime(t_str.strip(), "%I:%M %p").time()
                return t_obj.hour + t_obj.minute / 60.0

            # Clean column names for consistency
            self.rates_df.rename(columns={
                'Rate Plan Name': 'plan_name',
                'Rate Plan Code': 'plan_code',
                'Before/After Baseline': 'baseline_cat',
                'Season Name': 'season',
                'Start Time': 'start_time',
                'End Time': 'end_time',
                'Cost in cents': 'cost_cents'
            }, inplace=True)

            # Derive numeric hours and $ cost
            self.rates_df['start_hour'] = self.rates_df['start_time'].apply(_time_to_hour)
            self.rates_df['end_hour'] = self.rates_df['end_time'].apply(_time_to_hour)
            self.rates_df['cost_$'] = self.rates_df['cost_cents'] / 100.0

            # Normalize baseline category values
            self.rates_df['baseline_cat'] = self.rates_df['baseline_cat'].fillna('').str.strip()
            self.rates_df['has_baseline'] = self.rates_df['baseline_cat'] != ''
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Rate data file not found at {self.data_path}")
        except Exception as e:
            raise ValueError(f"Error loading rate data: {str(e)}")
            
    def parse_usage_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Parse uploaded usage CSV file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame with parsed usage data or None if parsing fails
        """
        try:
            # Read the CSV file
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Find the start of usage data
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('TYPE,DATE,START TIME'):
                    data_start = i
                    break
            
            # Create DataFrame from usage data
            usage_data = []
            for line in lines[data_start + 1:]:
                if line.strip() and not line.startswith('Electric usage'):
                    continue
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 6:
                        usage_data.append({
                            'date': parts[1],
                            'start_time': parts[2],
                            'end_time': parts[3],
                            'usage_kwh': float(parts[4]) if parts[4] else 0,
                            'cost': float(parts[5].replace('$', '')) if parts[5].replace('$', '') else 0
                        })
            
            df = pd.DataFrame(usage_data)
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['start_time'])
            df['hour'] = df['datetime'].dt.hour
            return df
            
        except Exception as e:
            raise ValueError(f"Error parsing usage data: {str(e)}")
    
    def get_season(self, date: datetime) -> str:
        """
        Determine if date is in summer or winter season.
        
        Args:
            date: Date to check
            
        Returns:
            'summer' or 'winter'
        """
        month = date.month
        # Summer: June 1 - September 30
        return 'summer' if 6 <= month <= 9 else 'winter'
    
    def get_time_period(self, hour: int, rate_plan: str, season: str) -> str:
        """
        Get time period for a given hour and rate plan.
        
        Args:
            hour: Hour of the day (0-23)
            rate_plan: Rate plan code
            season: Season string
            
        Returns:
            Time period string
        """
        if rate_plan == 'E-TOU-C':
            if season == 'Summer Season (June 1–Sept 30)':
                if 0 <= hour < 16:
                    return '12 a.m.–4 p.m.'
                elif 16 <= hour < 21:
                    return '4 p.m.–9 p.m.'
                else:
                    return '9 p.m.–12 a.m.'
            else:  # Winter
                if 0 <= hour < 16:
                    return '12 a.m.–4 p.m.'
                elif 16 <= hour < 21:
                    return '4 p.m.–9 p.m.'
                else:
                    return '9 p.m.–12 a.m.'
        elif rate_plan == 'E-TOU-D':
            if season == 'Summer Season (June 1–Sept 30)':
                if 0 <= hour < 17:
                    return '12 a.m.–5 p.m.'
                elif 17 <= hour < 20:
                    return '5 p.m.–8 p.m.'
                else:
                    return '8 p.m.–12 a.m.'
            else:  # Winter
                if 0 <= hour < 17:
                    return '12 a.m.–5 p.m.'
                elif 17 <= hour < 20:
                    return '5 p.m.–8 p.m.'
                else:
                    return '8 p.m.–12 a.m.'
        elif rate_plan == 'E-ELEC':
            if season == 'Summer Season (June 1–Sept 30)':
                if 0 <= hour < 15:
                    return '12 a.m.–3 p.m.'
                elif 15 <= hour < 16:
                    return '3 p.m.–4 p.m.'
                else:
                    return '4 p.m.–9 p.m.'
            else:  # Winter
                if 0 <= hour < 15:
                    return '12 a.m.–3 p.m.'
                elif 15 <= hour < 16:
                    return '3 p.m.–4 p.m.'
                else:
                    return '4 p.m.–9 p.m.'
        elif rate_plan == 'EV2-A':
            if season == 'Summer Season (June 1–Sept 30)':
                if 0 <= hour < 15:
                    return '12 a.m.–3 p.m.'
                elif 15 <= hour < 16:
                    return '3 p.m.–4 p.m.'
                else:
                    return '4 p.m.–9 p.m.'
            else:  # Winter
                if 0 <= hour < 15:
                    return '12 a.m.–3 p.m.'
                elif 15 <= hour < 16:
                    return '3 p.m.–4 p.m.'
                else:
                    return '4 p.m.–9 p.m.'
        elif rate_plan == 'EV-B':
            if season == 'Summer Season (June 1–Sept 30)':
                if 0 <= hour < 7:
                    return '12 a.m.–7 a.m.'
                elif 7 <= hour < 14:
                    return '7 a.m.–2 p.m.'
                elif 14 <= hour < 21:
                    return '2 p.m.–9 p.m.'
                else:
                    return '9 p.m.–12 a.m.'
            else:  # Winter
                if 0 <= hour < 7:
                    return '12 a.m.–7 a.m.'
                elif 7 <= hour < 14:
                    return '7 a.m.–2 p.m.'
                elif 14 <= hour < 21:
                    return '2 p.m.–9 p.m.'
                else:
                    return '9 p.m.–12 a.m.'
        
        return 'All Day'  # Default for tiered plans
    
    def calculate_tiered_cost(self, usage_df: pd.DataFrame, territory: str, heating_type: str) -> Dict[str, Any]:
        """
        Calculate cost for tiered rate plan (E-1).
        
        Args:
            usage_df: DataFrame with usage data
            territory: Territory code
            heating_type: 'basic_electric' or 'all_electric'
            
        Returns:
            Dictionary with cost calculation results
        """
        # Calculate daily usage by date
        usage_df['date_only'] = usage_df['datetime'].dt.date
        daily_usage = usage_df.groupby('date_only')['usage_kwh'].sum()
        
        total_cost = 0
        tier1_kwh = 0
        tier2_kwh = 0
        
        for date, usage in daily_usage.items():
            season = self.get_season(datetime.combine(date, datetime.min.time()))
            baseline_allowance = BASELINE_ALLOWANCES[territory][heating_type][season]
            
            # Calculate tier 1 and tier 2 usage
            tier1_usage = min(usage, baseline_allowance)
            tier2_usage = max(0, usage - baseline_allowance)
            
            tier1_kwh += tier1_usage
            tier2_kwh += tier2_usage
            
            # Get rates
            tier1_rate = self.get_rate('E-1', None, None, 'Below Baseline')
            tier2_rate = self.get_rate('E-1', None, None, 'Above Baseline')
            
            total_cost += tier1_usage * tier1_rate + tier2_usage * tier2_rate
        
        return {
            'plan': 'E-1',
            'total_cost': total_cost,
            'tier1_kwh': tier1_kwh,
            'tier2_kwh': tier2_kwh,
            'total_kwh': tier1_kwh + tier2_kwh
        }
    
    def calculate_tou_cost(self, usage_df: pd.DataFrame, plan_code: str, 
                          territory: Optional[str] = None, heating_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate cost for time-of-use rate plans.
        
        Args:
            usage_df: DataFrame with usage data
            plan_code: Rate plan code
            territory: Territory code (for baseline plans)
            heating_type: Heating type (for baseline plans)
            
        Returns:
            Dictionary with cost calculation results
        """
        total_cost = 0
        hourly_costs = []
        
        for _, row in usage_df.iterrows():
            season = 'Summer Season' if self.get_season(row['datetime']) == 'summer' else 'Winter Season'
            hour = row['hour']
            usage = row['usage_kwh']
            
            # Handle baseline plans (E-TOU-C)
            if plan_code == 'E-TOU-C':
                # Calculate baseline for the day
                date_only = row['datetime'].date()
                daily_usage = usage_df[usage_df['datetime'].dt.date == date_only]['usage_kwh'].sum()
                season_key = self.get_season(row['datetime'])
                baseline_allowance = BASELINE_ALLOWANCES[territory][heating_type][season_key]
                
                # Determine if this hour contributes to above or below baseline
                baseline_cat = 'Below Baseline' if daily_usage <= baseline_allowance else 'Above Baseline'
                rate = self.get_rate(plan_code, season, hour, baseline_cat)
            else:
                rate = self.get_rate(plan_code, season, hour, None)
            
            cost = usage * rate
            total_cost += cost
            hourly_costs.append(cost)
        
        return {
            'plan': plan_code,
            'total_cost': total_cost,
            'total_kwh': usage_df['usage_kwh'].sum(),
            'hourly_costs': hourly_costs
        }
    
    def calculate_all_plans(self, usage_df: pd.DataFrame, territory: str, heating_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate costs for all available rate plans.
        
        Args:
            usage_df: DataFrame with usage data
            territory: Territory code
            heating_type: Heating type
            
        Returns:
            Dictionary with results for all plans
        """
        results = {}
        
        # Tiered plan
        results['E-1'] = self.calculate_tiered_cost(usage_df, territory, heating_type)
        
        # Time-of-use plans
        tou_plans = ['E-TOU-C', 'E-TOU-D', 'E-ELEC', 'EV2-A', 'EV-B']
        for plan in tou_plans:
            if plan == 'E-TOU-C':
                results[plan] = self.calculate_tou_cost(usage_df, plan, territory, heating_type)
            else:
                results[plan] = self.calculate_tou_cost(usage_df, plan)
        
        return results
    
    def get_rate(self, plan_code: str, season: Optional[str], hour: Optional[Union[int, float]], 
                 baseline_cat: Optional[str]) -> float:
        """
        Get the rate for specific conditions.
        
        Args:
            plan_code: Rate plan code
            season: Season string
            hour: Hour of day
            baseline_cat: Baseline category
            
        Returns:
            Rate in $/kWh
        """
        # Filter by plan
        plan_rates = self.rates_df[self.rates_df['plan_code'] == plan_code]
        
        if plan_rates.empty:
            return 0.0
        
        # Filter by baseline category if specified
        if baseline_cat:
            baseline_rates = plan_rates[plan_rates['baseline_cat'] == baseline_cat]
            if not baseline_rates.empty:
                plan_rates = baseline_rates
        
        # Filter by season if specified
        if season:
            season_rates = plan_rates[plan_rates['season'] == season]
            if not season_rates.empty:
                plan_rates = season_rates
        
        # Filter by time if specified
        if hour is not None:
            def _matches_time(row):
                start_h = row['start_hour']
                end_h = row['end_hour']
                
                if start_h is None or end_h is None:
                    return True  # All day rate
                
                # Handle midnight crossing
                if start_h > end_h:
                    return hour >= start_h or hour < end_h
                else:
                    return start_h <= hour < end_h
            
            time_matched = plan_rates[plan_rates.apply(_matches_time, axis=1)]
            if not time_matched.empty:
                plan_rates = time_matched
        
        # Return the first matching rate
        if not plan_rates.empty:
            return plan_rates.iloc[0]['cost_$']
        
        return 0.0 