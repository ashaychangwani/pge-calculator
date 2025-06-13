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
    'P': {'basic_electric': {'summer': 13.5, 'winter': 11.0}, 'all_electric': {'summer': 15.2, 'winter': 26.0}},
    'Q': {'basic_electric': {'summer': 9.8, 'winter': 11.0}, 'all_electric': {'summer': 8.5, 'winter': 26.0}},
    'R': {'basic_electric': {'summer': 17.7, 'winter': 10.4}, 'all_electric': {'summer': 19.9, 'winter': 26.7}},
    'S': {'basic_electric': {'summer': 15.0, 'winter': 10.2}, 'all_electric': {'summer': 17.8, 'winter': 23.7}},
    'T': {'basic_electric': {'summer': 6.5, 'winter': 7.5}, 'all_electric': {'summer': 7.1, 'winter': 12.9}},
    'V': {'basic_electric': {'summer': 7.1, 'winter': 8.1}, 'all_electric': {'summer': 10.4, 'winter': 19.1}},
    'W': {'basic_electric': {'summer': 19.2, 'winter': 9.8}, 'all_electric': {'summer': 22.4, 'winter': 19.0}},
    'X': {'basic_electric': {'summer': 9.8, 'winter': 9.7}, 'all_electric': {'summer': 8.5, 'winter': 14.6}},
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
        # Load generation component rates (PG&E, GreenSource, TotalGreen)
        self._load_generation_rates()

        # Default provider when none specified
        self.default_provider = "pge"
        
        # Performance optimization: pre-build rate lookup tables (after all data is loaded)
        self._build_rate_lookup_tables()
        
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
    
    def _build_rate_lookup_tables(self) -> None:
        """Pre-build lookup tables for faster rate retrieval."""
        self.rate_lookup = {}
        self.gen_rate_lookup = {}
        
        # Build total rate lookup table
        for _, row in self.rates_df.iterrows():
            key = (
                row['plan_code'],
                row.get('season', ''),
                row.get('baseline_cat', ''),
                row.get('start_hour'),
                row.get('end_hour')
            )
            self.rate_lookup[key] = row['cost_$']
        
        # Build generation rate lookup table
        if hasattr(self, 'gen_rates_df'):
            for _, row in self.gen_rates_df.iterrows():
                key = (
                    row['plan_code'],
                    row.get('season', ''),
                    row.get('baseline_cat', ''),
                    row.get('start_hour'),
                    row.get('end_hour'),
                    row['provider'].lower()
                )
                self.gen_rate_lookup[key] = row['cost_$']
            
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
    
    def calculate_tiered_cost(
        self,
        usage_df: pd.DataFrame,
        territory: str,
        heating_type: str,
        provider: str = "pge",
    ) -> Dict[str, Any]:
        """
        Calculate cost for tiered rate plan (E-1).
        
        Args:
            usage_df: DataFrame with usage data
            territory: Territory code
            heating_type: 'basic_electric' or 'all_electric'
            provider: Provider code
            
        Returns:
            Dictionary with cost calculation results
        """
        # Calculate daily usage by date - VECTORIZED
        usage_df_copy = usage_df.copy()
        usage_df_copy['date_only'] = usage_df_copy['datetime'].dt.date
        daily_usage = usage_df_copy.groupby('date_only')['usage_kwh'].sum()
        
        # Vectorized season calculation
        daily_usage_df = daily_usage.reset_index()
        daily_usage_df['season'] = pd.to_datetime(daily_usage_df['date_only']).dt.month.apply(
            lambda x: 'summer' if 6 <= x <= 9 else 'winter'
        )
        
        # Vectorized baseline allowance calculation
        daily_usage_df['baseline_allowance'] = daily_usage_df['season'].apply(
            lambda season: BASELINE_ALLOWANCES[territory][heating_type][season]
        )
        
        # Vectorized tier calculations
        daily_usage_df['tier1_usage'] = np.minimum(daily_usage_df['usage_kwh'], daily_usage_df['baseline_allowance'])
        daily_usage_df['tier2_usage'] = np.maximum(0, daily_usage_df['usage_kwh'] - daily_usage_df['baseline_allowance'])
        
        tier1_kwh = daily_usage_df['tier1_usage'].sum()
        tier2_kwh = daily_usage_df['tier2_usage'].sum()
        
        provider = provider.lower()
        
        # Get rates (cached lookup)
        tier1_rate_total = self._get_rate_fast('E-1', None, None, 'Below Baseline')
        tier2_rate_total = self._get_rate_fast('E-1', None, None, 'Above Baseline')
        
        if provider == 'pge':
            total_cost = tier1_kwh * tier1_rate_total + tier2_kwh * tier2_rate_total
        else:
            gen_pge_tier1 = self._get_generation_rate_fast('E-1', None, None, 'Below Baseline', 'pge')
            gen_pge_tier2 = self._get_generation_rate_fast('E-1', None, None, 'Above Baseline', 'pge')
            gen_provider_tier1 = self._get_generation_rate_fast('E-1', None, None, 'Below Baseline', provider)
            gen_provider_tier2 = self._get_generation_rate_fast('E-1', None, None, 'Above Baseline', provider)
            
            total_cost = (tier1_kwh * (tier1_rate_total - gen_pge_tier1 + gen_provider_tier1) +
                         tier2_kwh * (tier2_rate_total - gen_pge_tier2 + gen_provider_tier2))
        
        return {
            'plan': 'E-1',
            'total_cost': total_cost,
            'tier1_kwh': tier1_kwh,
            'tier2_kwh': tier2_kwh,
            'total_kwh': tier1_kwh + tier2_kwh
        }
    
    def calculate_tou_cost(
        self,
        usage_df: pd.DataFrame,
        plan_code: str,
        territory: Optional[str] = None,
        heating_type: Optional[str] = None,
        provider: str = "pge",
    ) -> Dict[str, Any]:
        """
        Calculate cost for time-of-use rate plans - OPTIMIZED VERSION.
        
        Args:
            usage_df: DataFrame with usage data
            plan_code: Rate plan code
            territory: Territory code (for baseline plans)
            heating_type: Heating type (for baseline plans)
            provider: Provider code
            
        Returns:
            Dictionary with cost calculation results
        """
        provider = provider.lower()
        
        # Make a copy to avoid modifying original
        df = usage_df.copy()
        
        # Vectorized season calculation
        df['season_key'] = df['datetime'].dt.month.apply(lambda x: 'summer' if 6 <= x <= 9 else 'winter')
        df['season'] = df['season_key'].map({'summer': 'Summer Season', 'winter': 'Winter Season'})
        
        # Pre-calculate daily usage for E-TOU-C baseline determination
        if plan_code == 'E-TOU-C':
            df['date_only'] = df['datetime'].dt.date
            daily_usage_map = df.groupby('date_only')['usage_kwh'].sum().to_dict()
            df['daily_usage'] = df['date_only'].map(daily_usage_map)
            
            # Vectorized baseline allowance calculation
            df['baseline_allowance'] = df.apply(
                lambda row: BASELINE_ALLOWANCES[territory][heating_type][row['season_key']], axis=1
            )
            df['baseline_cat'] = np.where(df['daily_usage'] <= df['baseline_allowance'], 'Below Baseline', 'Above Baseline')
        
        # Vectorized rate lookup using time periods
        df['rate_total'] = df.apply(
            lambda row: self._get_rate_fast(
                plan_code, 
                row['season'], 
                row['hour'], 
                row.get('baseline_cat') if plan_code == 'E-TOU-C' else None
            ), axis=1
        )
        
        if provider == 'pge':
            df['cost'] = df['usage_kwh'] * df['rate_total']
        else:
            df['gen_pge_rate'] = df.apply(
                lambda row: self._get_generation_rate_fast(
                    plan_code, 
                    row['season'], 
                    row['hour'], 
                    row.get('baseline_cat') if plan_code == 'E-TOU-C' else None, 
                    'pge'
                ), axis=1
            )
            df['gen_provider_rate'] = df.apply(
                lambda row: self._get_generation_rate_fast(
                    plan_code, 
                    row['season'], 
                    row['hour'], 
                    row.get('baseline_cat') if plan_code == 'E-TOU-C' else None, 
                    provider
                ), axis=1
            )
            df['cost'] = df['usage_kwh'] * (df['rate_total'] - df['gen_pge_rate'] + df['gen_provider_rate'])
        
        total_cost = df['cost'].sum()
        
        return {
            'plan': plan_code,
            'total_cost': total_cost,
            'total_kwh': df['usage_kwh'].sum(),
            'hourly_costs': df['cost'].tolist()
        }
    
    def _get_rate_fast(self, plan_code: str, season: Optional[str], hour: Optional[Union[int, float]], 
                      baseline_cat: Optional[str]) -> float:
        """Fast rate lookup using pre-built lookup table."""
        # Try exact matches first
        for key, rate in self.rate_lookup.items():
            if (key[0] == plan_code and 
                (not season or key[1] == season) and
                (not baseline_cat or key[2] == baseline_cat)):
                
                start_hour, end_hour = key[3], key[4]
                
                if hour is None or start_hour is None or end_hour is None:
                    return rate
                
                # Check if hour falls within time range
                if start_hour > end_hour:  # Crosses midnight
                    if hour >= start_hour or hour < end_hour:
                        return rate
                else:
                    if start_hour <= hour < end_hour:
                        return rate
        
        return 0.0
    
    def _get_generation_rate_fast(self, plan_code: str, season: Optional[str], hour: Optional[Union[int, float]], 
                                 baseline_cat: Optional[str], provider: str) -> float:
        """Fast generation rate lookup using pre-built lookup table."""
        provider = provider.lower()
        
        # Try exact matches first
        for key, rate in self.gen_rate_lookup.items():
            if (key[0] == plan_code and 
                (not season or key[1] == season) and
                (not baseline_cat or key[2] == baseline_cat) and
                key[5] == provider):
                
                start_hour, end_hour = key[3], key[4]
                
                if hour is None or start_hour is None or end_hour is None:
                    return rate
                
                # Check if hour falls within time range
                if start_hour > end_hour:  # Crosses midnight
                    if hour >= start_hour or hour < end_hour:
                        return rate
                else:
                    if start_hour <= hour < end_hour:
                        return rate
        
        return 0.0
    
    def calculate_all_plans(
        self,
        usage_df: pd.DataFrame,
        territory: str,
        heating_type: str,
        provider: str = "pge",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate costs for all available rate plans.
        
        Args:
            usage_df: DataFrame with usage data
            territory: Territory code
            heating_type: Heating type
            provider: Provider code
            
        Returns:
            Dictionary with results for all plans
        """
        results = {}
        
        # Tiered plan
        results['E-1'] = self.calculate_tiered_cost(usage_df, territory, heating_type, provider)
        
        # Time-of-use plans
        tou_plans = ['E-TOU-C', 'E-TOU-D', 'E-ELEC', 'EV2-A', 'EV-B']
        for plan in tou_plans:
            if plan == 'E-TOU-C':
                results[plan] = self.calculate_tou_cost(usage_df, plan, territory, heating_type, provider)
            else:
                results[plan] = self.calculate_tou_cost(usage_df, plan, provider=provider)
        
        return results
    
    def get_rate(self, plan_code: str, season: Optional[str], hour: Optional[Union[int, float]], 
                 baseline_cat: Optional[str]) -> float:
        """
        Get the rate for specific conditions - LEGACY METHOD (slower).
        
        Args:
            plan_code: Rate plan code
            season: Season string
            hour: Hour of day
            baseline_cat: Baseline category
            
        Returns:
            Rate in $/kWh
        """
        return self._get_rate_fast(plan_code, season, hour, baseline_cat)

    # ------------------------------------------------------------------
    # Generation rates loading & lookup
    # ------------------------------------------------------------------

    def _get_generation_data_path(self) -> str:
        """Return path to generation rate CSV."""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        gen_file = project_root / "data" / "generation_rates.csv"
        if gen_file.exists():
            return str(gen_file)
        raise FileNotFoundError("Generation rate data file not found. Please ensure generation_rates.csv exists in the data directory.")

    def _load_generation_rates(self) -> None:
        """Load generation component rates for PG&E, GreenSource, TotalGreen."""
        gen_path = self._get_generation_data_path()
        try:
            gen_df = pd.read_csv(gen_path)

            # Standardize column names to match total-rate dataframe for easier look-up
            gen_df.rename(columns={
                'Rate Plan Name': 'plan_name',
                'Rate Plan Code': 'plan_code',
                'Before/After Baseline': 'baseline_cat',
                'Season Name': 'season',
                'Start Time': 'start_time',
                'End Time': 'end_time',
                'Cost in cents': 'cost_cents',
                'Source': 'provider'
            }, inplace=True)

            def _time_to_hour(t_str):
                if pd.isna(t_str) or t_str == "":
                    return None
                t_obj = datetime.strptime(t_str.strip(), "%I:%M %p").time()
                return t_obj.hour + t_obj.minute / 60.0

            gen_df['start_hour'] = gen_df['start_time'].apply(_time_to_hour)
            gen_df['end_hour'] = gen_df['end_time'].apply(_time_to_hour)
            gen_df['cost_$'] = gen_df['cost_cents'] / 100.0

            gen_df['baseline_cat'] = gen_df['baseline_cat'].fillna('').str.strip()

            self.gen_rates_df = gen_df
        except Exception as e:
            raise ValueError(f"Error loading generation rate data: {str(e)}")

    # ------------------------------------------------------------------
    # Public helper – generation rate lookup
    # ------------------------------------------------------------------

    def get_generation_rate(
        self,
        plan_code: str,
        season: Optional[str],
        hour: Optional[Union[int, float]],
        baseline_cat: Optional[str],
        provider: str,
    ) -> float:
        """Return generation component rate ($/kWh) for given conditions and provider - LEGACY METHOD."""
        return self._get_generation_rate_fast(plan_code, season, hour, baseline_cat, provider) 