"""
Streamlit Web Application for PG&E Rate Calculator

This module contains the Streamlit interface and visualization logic.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from typing import Optional
import calendar
import logging
import sys

from .calculator import PGERateCalculator, BASELINE_ALLOWANCES

# ------------------------------------------------------------------
# Logging configuration (stdout) for debugging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Force to stdout
        logging.StreamHandler(sys.stderr)   # Also to stderr for errors
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)

# For Streamlit debugging, also log to console
logger.addHandler(logging.StreamHandler(sys.stdout))

# ------------------------------------------------------------------
# Provider-aware labelling helpers and best-result aggregation
# ------------------------------------------------------------------

def plan_label_provider(plan_code: str, provider_key: str) -> str:
    """Return a user-friendly plan label that also includes the cheapest provider."""
    provider_key = provider_key.lower()
    provider_label = PROVIDERS.get(provider_key, {}).get('label', provider_key)
    return f"{plan_label(plan_code)} [{provider_label}]"


def compute_best_results(calculator: PGERateCalculator, usage_df: pd.DataFrame, territory: str, heating_type: str):
    """Compute the cheapest cost for every plan across all providers.

    Returns (best_results, provider_results) where:
    • best_results: dict mapping plan_code -> result dict extended with 'provider'.
    • provider_results: dict mapping provider_key -> full results from calculator.
    """
    # Check if we have cached results for this combination
    cache_key = (id(usage_df), territory, heating_type, len(usage_df))
    if not hasattr(compute_best_results, '_cache'):
        compute_best_results._cache = {}
    
    if cache_key in compute_best_results._cache:
        return compute_best_results._cache[cache_key]
    
    provider_results = {}
    
    # Calculate results for each provider
    for key in PROVIDERS.keys():
        provider_results[key] = calculator.calculate_all_plans(usage_df, territory, heating_type, key)

    best_results = {}
    for prov_key, plans in provider_results.items():
        for plan_code, data in plans.items():
            if (plan_code not in best_results) or (data['total_cost'] < best_results[plan_code]['total_cost']):
                entry = data.copy()
                entry['provider'] = prov_key
                best_results[plan_code] = entry

    # Cache the results
    result = (best_results, provider_results)
    compute_best_results._cache[cache_key] = result
    
    return result

# Mapping of PG&E rate plan codes to human-friendly names. Feel free to extend whenever more plans are added.
PLAN_NAMES = {
    "E-1": "Tiered Rate Plan",
    "E-TOU-C": "Time-of-Use Plan C",
    "E-TOU-D": "Time-of-Use Plan D",
    "E-ELEC": "Electric Home Rate Plan",
    "EV2-A": "Electric Vehicle Plan 2-A",
    "EV-B": "Electric Vehicle Plan B",
}

# Generation providers and descriptions
PROVIDERS = {
    "pge": {
        "label": "PG&E (Bundled Service)",
        "description": "Electricity generation provided by PG&E. Mix of traditional and renewable sources."
    },
    "greensource": {
        "label": "GreenSource (60% renewable)",
        "description": "Generation service from San José Clean Energy with at least 60% renewable content."
    },
    "totalgreen": {
        "label": "TotalGreen (100% renewable) (SJCE)",
        "description": "SJCE's 100% renewable generation service, sourced entirely from renewable resources."
    },
}

def plan_label(plan_code: str) -> str:
    """Return a user-friendly label of the form 'Plan Name (CODE)'."""
    return f"{PLAN_NAMES.get(plan_code, plan_code)} ({plan_code})"


def format_hour_12(hour: int) -> str:
    """Convert 24-hour format to 12-hour format with AM/PM."""
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


def calculate_monthly_cost(total_cost: float, usage_df: pd.DataFrame) -> float:
    """Convert total cost to average monthly cost based on data duration."""
    days_of_data = len(usage_df['datetime'].dt.date.unique())
    daily_cost = total_cost / days_of_data
    monthly_cost = daily_cost * 30  # Assume 30 days per month
    return monthly_cost

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="PG&E Rate Calculator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling (theme-aware)
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .best-plan-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .sidebar-metric {
        background: var(--background-color, #353436);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: var(--text-color, white);
    }
    
    .chart-container {
        background: transparent;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Theme-aware styling */
    [data-theme="dark"] .chart-container {
        box-shadow: 0 2px 10px rgba(255,255,255,0.1);
    }
    
    [data-theme="light"] .chart-container {
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Provider info tooltip style
    st.markdown("""
    <style>
    .provider-card {
        background: var(--background-color, #353436);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: var(--text-color, white);
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the main header and description."""
    st.markdown('<h1 class="main-header">⚡ PG&E Rate Plan Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    <strong>Find the optimal PG&E electricity rate plan based on your actual usage patterns.</strong><br>
    Upload your usage data and compare costs across all major PG&E rate plans to maximize your savings.
    </div>
    """, unsafe_allow_html=True)


def display_sidebar_stats(usage_df: Optional[pd.DataFrame] = None, results: Optional[dict] = None, territory: str = "X", heating_type: str = "basic_electric"):
    """Display comprehensive statistics in sidebar."""
    st.sidebar.markdown("## 📊 Usage Statistics")
    
    if usage_df is not None:
        # Calculate key metrics
        total_usage = usage_df['usage_kwh'].sum()
        days_of_data = len(usage_df['datetime'].dt.date.unique())
        avg_daily = total_usage / days_of_data
        peak_hour = usage_df['usage_kwh'].max()
        
        # Peak usage times
        peak_hours = usage_df.groupby('hour')['usage_kwh'].sum()
        peak_hour_of_day = peak_hours.idxmax()
        peak_hour_formatted = format_hour_12(peak_hour_of_day)
        
        # Weekend vs weekday usage
        usage_df_copy = usage_df.copy()
        usage_df_copy['is_weekend'] = usage_df_copy['datetime'].dt.dayofweek >= 5
        weekday_avg = usage_df_copy[~usage_df_copy['is_weekend']]['usage_kwh'].sum()
        weekend_avg = usage_df_copy[usage_df_copy['is_weekend']]['usage_kwh'].sum()
        
        # Baseline comparison
        baseline_allowances = BASELINE_ALLOWANCES[territory][heating_type]
        summer_baseline = baseline_allowances['summer']
        winter_baseline = baseline_allowances['winter']
        
        # Display metrics in sidebar
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>📈 Days of Data</strong><br>
            {days_of_data} days
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>📈 Total Usage</strong><br>
            {total_usage:.1f} kWh
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>📅 Daily Average</strong><br>
            {avg_daily:.2f} kWh/day
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>⏰ Peak Hour</strong><br>
            {peak_hour_formatted} ({peak_hours[peak_hour_of_day]:.2f} kWh avg)
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>🏠 Weekday vs Weekend</strong><br>
            Weekday: {weekday_avg:.3f} kWh/hr<br>
            Weekend: {weekend_avg:.3f} kWh/hr
        </div>
        """, unsafe_allow_html=True)
        
        # Baseline analysis
        baseline_exceeded_days = 0
        daily_usage = usage_df.groupby(usage_df['datetime'].dt.date)['usage_kwh'].sum()
        for date, daily_kwh in daily_usage.items():
            # Determine season
            month = date.month
            if 6 <= month <= 9:  # Summer
                baseline = summer_baseline
            else:  # Winter
                baseline = winter_baseline
            
            if daily_kwh > baseline:
                baseline_exceeded_days += 1
        
        baseline_exceed_percent = (baseline_exceeded_days / len(daily_usage)) * 100
        
        st.sidebar.markdown(f"""
        <div class="sidebar-metric">
            <strong>📊 Baseline Analysis</strong><br>
            Days exceeding baseline: {baseline_exceeded_days}<br>
            ({baseline_exceed_percent:.1f}% of days)
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.sidebar.info("Upload usage data to see detailed statistics")
    
    # Rate comparison results in sidebar
    if results is not None and usage_df is not None:
        st.sidebar.markdown("## 💰 Rate Comparison (Monthly)")
        
        # Sort results by cost
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_cost'])
        
        for i, (plan, data) in enumerate(sorted_results[:3]):  # Top 3 plans
            if i == 0:
                color = "#11998e"  # Best plan - green
                icon = "🏆"
            elif i == 1:
                color = "#f39c12"  # Second best - orange
                icon = "🥈"
            else:
                color = "#e74c3c"  # Third - red
                icon = "🥉"
            
            label = plan_label_provider(plan, data.get('provider', 'pge'))
            monthly_cost = calculate_monthly_cost(data['total_cost'], usage_df)
            st.sidebar.markdown(f"""
            <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>{icon} {label}</strong><br>
                ${monthly_cost:.2f}/month<br>
                <small>${data['total_cost'] / data['total_kwh']:.4f}/kWh</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Configuration info
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.markdown(f"""
    <div class="sidebar-metric">
        <strong>📍 Territory:</strong> {territory}<br>
        <strong>🏠 Heating:</strong> {heating_type.replace('_', ' ').title()}
    </div>
    """, unsafe_allow_html=True)

    # Baseline allowances
    allowances = BASELINE_ALLOWANCES[territory][heating_type]
    st.sidebar.markdown("## 📏 Baseline Allowances")
    st.sidebar.markdown(f"""
    <div class="sidebar-metric">
        <strong>☀️ Summer:</strong> {allowances['summer']:.1f} kWh/day<br>
        <strong>❄️ Winter:</strong> {allowances['winter']:.1f} kWh/day
    </div>
    """, unsafe_allow_html=True)


def display_baseline_territory_selection():
    """Let users select their baseline territory and heating type."""
    st.markdown("## 🔧 Configuration Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Baseline Territory")
        
        # Instructions for finding baseline territory
        with st.expander("💡 How to find your Baseline Territory"):
            st.markdown("""
            **To find your Baseline Territory:**
            
            1. Navigate to [https://myaccount.pge.com/myaccount/s/bill-and-payment-history](https://myaccount.pge.com/myaccount/s/bill-and-payment-history)
            2. Click "View Bill PDF" for your most recent bill
            3. Go to **page 3** of your bill PDF
            4. On the **right side** of the page, look for the **"Baseline Territory"** code
            
            Your territory will be a single letter (P, Q, R, S, T, V, W, X, Y, or Z).
            """)
        
        territory_choice = st.selectbox(
            "Select your baseline territory:",
            options=list(BASELINE_ALLOWANCES.keys()),
            index=7,  # Default to 'X'
            help="Your baseline territory determines your daily allowance for tier 1 pricing"
        )
    
    with col2:
        st.markdown("### 🏠 Heating System Type")
        
        # Explanation of heating types
        with st.expander("💡 Heating System Explanation"):
            st.markdown("""
            **Basic Electric**: No permanent electric space heating
            - Standard electric service for homes with gas heating, wood heating, or no central heating
            
            **All Electric**: Includes permanent electric space heating
            - Homes with electric furnaces, heat pumps, or other permanent electric heating systems
            """)
        
        heating_choice = st.radio(
            "Select your heating system:",
            options=["basic_electric", "all_electric"],
            format_func=lambda x: "Basic Electric" if x == "basic_electric" else "All Electric",
            help="This affects your baseline allowances"
        )
    
    # Baseline allowances are now displayed in the sidebar to save space on the main page
    
    return territory_choice, heating_choice


def display_file_uploader():
    """Display file uploader and instructions."""
    st.markdown("## 📁 Upload Your Usage Data")
    
    # Instructions
    with st.expander("📖 How to get your usage data", expanded=False):
        st.markdown("""
        1. Visit [PG&E My Account](https://myaccount.pge.com/)
        2. Log in to your account
        3. Navigate to "Usage & Consumption"
        4. Select "Download Data" or "Export"
        5. Choose CSV format and download
        6. Upload the file below
        """)
    
    uploaded_file = st.file_uploader(
        "Choose your PG&E usage CSV file",
        type=['csv'],
        help="Upload your downloaded PG&E usage data in CSV format"
    )
    
    return uploaded_file


def create_enhanced_usage_charts(usage_df: pd.DataFrame):
    """Create comprehensive and beautiful usage visualizations."""
    
    # 1. Hourly heatmap by day of week
    usage_pivot = usage_df.pivot_table(
        values='usage_kwh', 
        index='hour', 
        columns=usage_df['datetime'].dt.day_name(),
        aggfunc='mean'
    )
    
    # Reorder columns to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    usage_pivot = usage_pivot.reindex(columns=[day for day in day_order if day in usage_pivot.columns])
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=usage_pivot.values,
        x=usage_pivot.columns,
        y=usage_pivot.index,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title="kWh", title_side="right")
    ))
    
    fig_heatmap.update_layout(
        title={
            'text': '🔥 Usage Heatmap: Hour of Day vs Day of Week',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        height=500,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. Enhanced hourly pattern with TOU periods
    hourly_avg = usage_df.groupby('hour')['usage_kwh'].sum().reset_index()
    hourly_std = usage_df.groupby('hour')['usage_kwh'].std().reset_index()
    
    # Format hours for 12-hour display
    hourly_avg['hour_formatted'] = hourly_avg['hour'].apply(format_hour_12)
    hourly_std['hour_formatted'] = hourly_std['hour'].apply(format_hour_12)
    
    fig_hourly = go.Figure()
    
    # Add confidence interval
    fig_hourly.add_trace(go.Scatter(
        x=hourly_avg['hour_formatted'],
        y=hourly_avg['usage_kwh'] + hourly_std['usage_kwh'],
        mode='lines',
        line=dict(color='rgba(0,100,80,0)'),
        showlegend=False,
        name='Upper bound'
    ))
    
    fig_hourly.add_trace(go.Scatter(
        x=hourly_avg['hour_formatted'],
        y=hourly_avg['usage_kwh'] - hourly_std['usage_kwh'],
        mode='lines',
        line=dict(color='rgba(0,100,80,0)'),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        showlegend=False,
        name='Confidence interval'
    ))
    
    # Main line
    fig_hourly.add_trace(go.Scatter(
        x=hourly_avg['hour_formatted'],
        y=hourly_avg['usage_kwh'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6, color='#ff7f0e'),
        name='Average Usage'
    ))
    
    # Note: TOU period annotations removed - these are now shown in the monthly analysis with plan-specific periods
    
    fig_hourly.update_layout(
        title={
            'text': '⏰ Average Hourly Usage Pattern with Time-of-Use Periods',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Hour of Day",
        yaxis_title="Usage (kWh)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    # 3. Daily usage with moving average
    daily_usage = usage_df.groupby(usage_df['datetime'].dt.date)['usage_kwh'].sum().reset_index()
    daily_usage.columns = ['date', 'usage_kwh']
    daily_usage['moving_avg'] = daily_usage['usage_kwh'].rolling(window=7, center=True).mean()
    
    fig_daily = go.Figure()
    
    # Add daily usage bars
    fig_daily.add_trace(go.Bar(
        x=daily_usage['date'],
        y=daily_usage['usage_kwh'],
        name='Daily Usage',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add moving average line
    fig_daily.add_trace(go.Scatter(
        x=daily_usage['date'],
        y=daily_usage['moving_avg'],
        mode='lines',
        line=dict(color='red', width=3),
        name='7-Day Moving Average'
    ))
    
    fig_daily.update_layout(
        title={
            'text': '📅 Daily Usage Trend with 7-Day Moving Average',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Date",
        yaxis_title="Daily Usage (kWh)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    # 4. Usage distribution histogram
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=usage_df['usage_kwh'],
        nbinsx=50,
        name='Usage Distribution',
        marker_color='skyblue',
        opacity=0.7
    ))
    
    fig_dist.add_vline(
        x=usage_df['usage_kwh'].sum(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {usage_df['usage_kwh'].sum():.3f} kWh"
    )
    
    fig_dist.update_layout(
        title={
            'text': '📊 Usage Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Usage (kWh)",
        yaxis_title="Frequency",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_heatmap, fig_hourly, fig_daily, fig_dist


def display_usage_analysis(usage_df: pd.DataFrame):
    """Display a simplified usage analysis (daily trend only)."""
    st.markdown("## 📊 Usage Analysis – Daily Trend")

    daily_usage = usage_df.groupby(usage_df['datetime'].dt.date)['usage_kwh'].sum().reset_index()
    daily_usage.columns = ['date', 'usage_kwh']
    
    daily_usage['moving_avg'] = daily_usage['usage_kwh'].rolling(window=7, center=True).mean()

    fig_daily = go.Figure()

    fig_daily.add_trace(go.Bar(
        x=daily_usage['date'],
        y=daily_usage['usage_kwh'],
        name='Daily Usage',
        marker_color='lightblue',
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Usage: %{y:.2f} kWh<extra></extra>'
    ))

    fig_daily.add_trace(go.Scatter(
        x=daily_usage['date'],
        y=daily_usage['moving_avg'],
        mode='lines',
        line=dict(color='red', width=3),
        name='7-Day Moving Average',
        hovertemplate='<b>%{x}</b><br>7-Day Avg: %{y:.2f} kWh<extra></extra>'
    ))

    fig_daily.update_layout(
        title={'text': '📅 Daily Usage Trend', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Daily Usage (kWh)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return usage_df


def create_enhanced_rate_comparison_charts(results: dict, usage_df: pd.DataFrame):
    """Create enhanced rate comparison visualizations."""
    
    # Create comparison dataframe
    comparison_data = []
    for plan, data in results.items():
        monthly_cost = calculate_monthly_cost(data['total_cost'], usage_df)
        comparison_data.append({
            'Plan': plan_label_provider(plan, data.get('provider', 'pge')),
            'Monthly Cost': monthly_cost,
            'Cost per kWh': data['total_cost'] / data['total_kwh'] if data['total_kwh'] > 0 else 0,
            'Total kWh': data['total_kwh']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    # Sort and reset index so row indices align with ranking
    comparison_df = comparison_df.sort_values('Monthly Cost').reset_index(drop=True)
    
    # 1. Enhanced cost comparison bar chart
    colors = ['#2ecc71' if i == 0 else '#f39c12' if i == 1 else '#e74c3c' if i == 2 else '#95a5a6' 
              for i in range(len(comparison_df))]
    
    fig_cost = go.Figure()
    
    fig_cost.add_trace(go.Bar(
        x=comparison_df['Plan'],
        y=comparison_df['Monthly Cost'],
        marker_color=colors,
        text=[f'${cost:.2f}/mo' for cost in comparison_df['Monthly Cost']],
        textposition='auto',
        name='Monthly Cost'
    ))
    
    fig_cost.update_layout(
        title={
            'text': '💰 Rate Plan Monthly Cost Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title="Rate Plan",
        yaxis_title="Monthly Cost ($)",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=False
    )
    
    # 2. Cost per kWh comparison
    fig_rate = go.Figure()
    
    fig_rate.add_trace(go.Scatter(
        x=comparison_df['Plan'],
        y=comparison_df['Cost per kWh'],
        mode='markers+lines',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=2, color='white')
        ),
        line=dict(width=3, color='#34495e'),
        text=[f'${rate:.4f}/kWh' for rate in comparison_df['Cost per kWh']],
        textposition='top center',
        name='Cost per kWh'
    ))
    
    fig_rate.update_layout(
        title={
            'text': '⚡ Cost per kWh Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Rate Plan",
        yaxis_title="Cost per kWh ($)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    # 3. Savings comparison (waterfall chart style)
    best_cost = comparison_df.iloc[0]['Monthly Cost']
    savings_data = []
    for idx, row in comparison_df.iterrows():
        if idx == 0:
            savings_data.append({
                'Plan': row['Plan'],
                'Additional Cost': 0,
                'Savings': 0
            })
        else:
            savings = row['Monthly Cost'] - best_cost
            savings_data.append({
                'Plan': row['Plan'],
                'Additional Cost': savings,
                'Savings': -savings if savings > 0 else 0
            })
    
    savings_df = pd.DataFrame(savings_data)
    
    fig_savings = go.Figure()
    
    fig_savings.add_trace(go.Bar(
        x=savings_df['Plan'],
        y=savings_df['Additional Cost'],
        marker_color=['#2ecc71' if x == 0 else '#e74c3c' for x in savings_df['Additional Cost']],
        text=[f'${cost:.2f}/mo' if cost > 0 else 'Best Plan!' for cost in savings_df['Additional Cost']],
        textposition='auto',
        name='Additional Monthly Cost vs Best Plan'
    ))
    
    fig_savings.update_layout(
        title={
            'text': '💡 Monthly Savings Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Rate Plan",
        yaxis_title="Additional Monthly Cost vs Best Plan ($)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    return fig_cost, fig_rate, fig_savings, comparison_df


def display_rate_comparison(results: dict, usage_df: pd.DataFrame):
    """Display enhanced rate plan comparison results."""
    st.markdown("## 💰 Comprehensive Rate Plan Analysis")
    
    # Create enhanced comparison charts
    fig_cost, fig_rate, fig_savings, comparison_df = create_enhanced_rate_comparison_charts(results, usage_df)
    
    # Best plan recommendation with enhanced styling
    best_plan = comparison_df.iloc[0]
    worst_plan = comparison_df.iloc[-1]
    savings = worst_plan['Monthly Cost'] - best_plan['Monthly Cost']
    
    st.markdown(f"""
    <div class="best-plan-card">
        <h2>🏆 RECOMMENDED PLAN</h2>
        <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{best_plan['Plan']}</h1>
        <h3 style="margin: 0;">Monthly Cost: ${best_plan['Monthly Cost']:.2f}</h3>
        <p style="font-size: 1.2rem; margin: 1rem 0;">
            Cost per kWh: ${best_plan['Cost per kWh']:.4f}<br>
            <strong>💰 You could save ${savings:.2f}/month</strong> vs worst plan ({worst_plan['Plan']})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display charts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_cost, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_rate, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_savings, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced detailed comparison table
    st.markdown("### 📋 Detailed Rate Plan Comparison")
    
    # Format the dataframe for display
    display_df = comparison_df.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df['Monthly Cost'] = display_df['Monthly Cost'].apply(lambda x: f"${x:.2f}")
    display_df['Cost per kWh'] = display_df['Cost per kWh'].apply(lambda x: f"${x:.4f}")
    display_df['Total kWh'] = display_df['Total kWh'].apply(lambda x: f"{x:.1f}")
    
    # Calculate monthly savings vs best
    best_monthly = comparison_df.iloc[0]['Monthly Cost']
    monthly_savings = []
    for idx, row in comparison_df.iterrows():
        if idx == 0:
            monthly_savings.append('Best Plan')
        else:
            savings = row['Monthly Cost'] - best_monthly
            monthly_savings.append(f"${savings:.2f}/mo more")
    display_df['Monthly Savings vs Best'] = monthly_savings
    
    # Reorder columns
    display_df = display_df[['Rank', 'Plan', 'Monthly Cost', 'Cost per kWh', 'Total kWh', 'Monthly Savings vs Best']]
    
    # Style the dataframe
    def color_rank(val):
        if val == 1:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif val == 2:
            return 'background-color: #fff3cd; color: #856404'
        elif val == 3:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_df = display_df.style.map(color_rank, subset=['Rank'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_plan_details():
    """Display information about different rate plans."""
    st.markdown("## 📖 Rate Plan Information")
    
    with st.expander("📋 Comprehensive Rate Plan Guide", expanded=False):
        st.markdown("""
        ### 🏠 Tiered Rate Plan (E-1)
        - **Structure**: Fixed rates regardless of time of use
        - **Tier 1**: Below baseline allowance (lower rate)
        - **Tier 2**: Above baseline allowance (higher rate)
        - **Best for**: Customers with low, consistent usage patterns
        - **Pros**: Predictable pricing, simple to understand
        - **Cons**: No incentive for off-peak usage
        
        ### ⏰ Time-of-Use Plans
        Variable rates based on time of day and season:
        
        **E-TOU-C**: Time-of-Use with baseline allowances
        - **Peak hours**: 4-9 PM (summer), 4-9 PM (winter)
        - **Structure**: Different rates for usage above/below baseline
        - **Best for**: Customers who can shift usage to off-peak hours
        
        **E-TOU-D**: Simple time-of-use
        - **Peak hours**: 5-8 PM (summer), 5-8 PM (winter)
        - **Structure**: No baseline differentiation
        - **Best for**: Moderate usage customers with flexible schedules
        
        **E-ELEC**: Electric home rate plan
        - **Peak hours**: 4-9 PM with super-peak 3-4 PM (summer)
        - **Structure**: Designed for all-electric homes
        - **Best for**: Homes with electric heating/cooling systems
        
        ### 🚗 Electric Vehicle Plans
        
        **EV2-A**: Home charging plan
        - **Super off-peak**: 12-3 PM (very low rates)
        - **Peak**: 4-9 PM (highest rates)
        - **Best for**: EV owners who can charge during midday
        
        **EV-B**: Time-of-use for EV owners
        - **Super off-peak**: 12-7 AM (overnight charging)
        - **Peak**: 2-9 PM (summer), 2-9 PM (winter)
        - **Best for**: EV owners with overnight charging capability
        
        ### 💡 Tips for Choosing the Right Plan
        1. **Analyze your usage patterns** - Look at when you use the most electricity
        2. **Consider your flexibility** - Can you shift usage to off-peak hours?
        3. **Account for future changes** - EV purchase, home additions, etc.
        4. **Monitor baseline usage** - Staying below baseline saves money on tiered plans
        5. **Review seasonally** - Usage patterns may change with weather
        """)


def calculate_monthly_costs(usage_df: pd.DataFrame, calculator: PGERateCalculator, territory: str, heating_type: str):
    """
    Calculate monthly costs for each rate plan with forecasting for incomplete months.
    
    Args:
        usage_df: DataFrame with usage data
        calculator: PGERateCalculator instance
        territory: Territory code
        heating_type: Heating type
        
    Returns:
        Dictionary with monthly cost data for each plan
    """
    # Add month-year column
    usage_df['month_year'] = usage_df['datetime'].dt.to_period('M')
    
    # Get unique months
    months = usage_df['month_year'].unique()
    
    monthly_results = {}
    current_date = datetime.now()
    
    for month in months:
        # Filter data for this month
        month_data = usage_df[usage_df['month_year'] == month].copy()
        
        # Determine if month is complete
        month_start = pd.to_datetime(str(month))
        month_end = month_start + pd.offsets.MonthEnd(0)
        
        # Check if we have data for the entire month
        actual_start = month_data['datetime'].min()
        actual_end = month_data['datetime'].max()
        
        # Calculate days in month and days we have data for
        days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
        unique_days = len(month_data['datetime'].dt.date.unique())
        
        is_complete = (unique_days >= days_in_month * 0.95)  # 95% threshold for "complete"

        # Calculate costs for this month's data
        if len(month_data) > 0:
            # Use selected provider stored in session_state if available
            provider_choice = st.session_state.get('selected_provider', 'pge') if hasattr(st, 'session_state') else 'pge'
            month_results = calculator.calculate_all_plans(month_data, territory, heating_type, provider_choice)
            
            # If month is incomplete, forecast remaining days
            if not is_complete and month_start.month == current_date.month and month_start.year == current_date.year:
                # Calculate daily averages for forecasting
                days_remaining = days_in_month - unique_days
                
                if days_remaining > 0:
                    # Create forecast data for remaining days
                    forecast_data = []
                    last_date = actual_end
                    
                    for i in range(int(days_remaining)):
                        forecast_date = last_date + timedelta(days=i+1)
                        # Use hourly averages to create realistic hourly forecast
                        hourly_avg = month_data.groupby('hour')['usage_kwh'].sum()
                        
                        for hour in range(24):
                            avg_usage = hourly_avg.get(hour, 0)
                            forecast_data.append({
                                'datetime': forecast_date.replace(hour=hour),
                                'usage_kwh': avg_usage,
                                'hour': hour
                            })
                    
                    if forecast_data:
                        forecast_df = pd.DataFrame(forecast_data)
                        forecast_results = calculator.calculate_all_plans(forecast_df, territory, heating_type, provider_choice)
                        
                        # Add forecasted costs to actual costs
                        for plan in month_results:
                            if plan in forecast_results:
                                month_results[plan]['total_cost'] += forecast_results[plan]['total_cost']
                                month_results[plan]['total_kwh'] += forecast_results[plan]['total_kwh']
            
            monthly_results[str(month)] = {
                'month': month,
                'is_complete': is_complete,
                'days_with_data': unique_days,
                'days_in_month': days_in_month,
                'results': month_results
            }
    
    return monthly_results


def create_monthly_cost_charts(monthly_data: dict):
    """
    Create comprehensive monthly cost visualizations.
    
    Args:
        monthly_data: Dictionary with monthly cost data
        
    Returns:
        Plotly figures for monthly cost analysis
    """
    # Prepare data for plotting
    months = sorted(monthly_data.keys())
    plans = list(monthly_data[months[0]]['results'].keys()) if months else []
    
    # Create subplots for each plan
    fig_monthly = make_subplots(
        rows=len(plans), 
        cols=1,
        subplot_titles=[f"{plan} Monthly Costs" for plan in plans],
        shared_xaxes=True,
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}]] * len(plans)
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, plan in enumerate(plans):
        # Extract data for this plan
        x_values = []
        y_actual = []
        y_forecasted = []
        
        for month in months:
            month_info = monthly_data[month]
            cost = month_info['results'][plan]['total_cost']
            
            x_values.append(str(month_info['month']))
            
            if month_info['is_complete']:
                y_actual.append(cost)
                y_forecasted.append(None)
            else:
                y_actual.append(None)
                y_forecasted.append(cost)
        
        # Add actual costs line
        fig_monthly.add_trace(
            go.Scatter(
                x=x_values,
                y=y_actual,
                mode='lines+markers',
                name=f'{plan} (Actual)',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
        
        # Add forecasted costs line (dotted)
        fig_monthly.add_trace(
            go.Scatter(
                x=x_values,
                y=y_forecasted,
                mode='lines+markers',
                name=f'{plan} (Forecasted)',
                line=dict(color=colors[i % len(colors)], width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond'),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
    
    fig_monthly.update_layout(
        title={
            'text': '📅 Monthly Cost Analysis by Rate Plan',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=200 * len(plans) + 100,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis_title="Month",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis titles
    for i in range(len(plans)):
        fig_monthly.update_yaxes(title_text="Cost ($)", row=i+1, col=1)
    
    # Create comparative monthly chart
    fig_comparison = go.Figure()
    
    plan_colors = {plan: colors[i % len(colors)] for i, plan in enumerate(plans)}
    
    for plan in plans:
        x_values = []
        y_actual = []
        y_forecasted = []
        
        for month in months:
            month_info = monthly_data[month]
            cost = month_info['results'][plan]['total_cost']
            
            x_values.append(str(month_info['month']))
            
            if month_info['is_complete']:
                y_actual.append(cost)
                y_forecasted.append(None)
            else:
                y_actual.append(None)
                y_forecasted.append(cost)
        
        # Add actual costs
        fig_comparison.add_trace(
            go.Scatter(
                x=x_values,
                y=y_actual,
                mode='lines+markers',
                name=f'{plan}',
                line=dict(color=plan_colors[plan], width=3),
                marker=dict(size=6)
            )
        )
        
        # Add forecasted costs (dotted)
        fig_comparison.add_trace(
            go.Scatter(
                x=x_values,
                y=y_forecasted,
                mode='lines+markers',
                name=f'{plan} (Forecast)',
                line=dict(color=plan_colors[plan], width=3, dash='dot'),
                marker=dict(size=6, symbol='diamond'),
                showlegend=False
            )
        )
    
    fig_comparison.update_layout(
        title={
            'text': '📊 Monthly Cost Comparison: All Rate Plans',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Month",
        yaxis_title="Monthly Cost ($)",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        hovermode='x unified'
    )
    
    # Create monthly usage vs cost scatter plot
    fig_scatter = go.Figure()
    
    for plan in plans:
        x_usage = []
        y_costs = []
        month_labels = []
        
        for month in months:
            month_info = monthly_data[month]
            cost = month_info['results'][plan]['total_cost']
            usage = month_info['results'][plan]['total_kwh']
            
            x_usage.append(usage)
            y_costs.append(cost)
            month_labels.append(f"{month} ({'Complete' if month_info['is_complete'] else 'Forecasted'})")
        
        fig_scatter.add_trace(
            go.Scatter(
                x=x_usage,
                y=y_costs,
                mode='markers',
                name=plan,
                marker=dict(
                    size=12,
                    color=plan_colors[plan],
                    line=dict(width=2, color='white')
                ),
                text=month_labels,
                hovertemplate='<b>%{text}</b><br>Usage: %{x:.1f} kWh<br>Cost: $%{y:.2f}<extra></extra>'
            )
        )
    
    fig_scatter.update_layout(
        title={
            'text': '⚡ Monthly Usage vs Cost Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Monthly Usage (kWh)",
        yaxis_title="Monthly Cost ($)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig_monthly, fig_comparison, fig_scatter


def time_string_to_hour(time_str: str, is_end_time: bool = False) -> int:
    """Convert time string like '3:00 PM' to hour index (0-23)."""
    try:
        if 'AM' in time_str:
            hour = int(time_str.split(':')[0])
            if hour == 12:
                # 12:00 AM can be start of day (0) or end of day (24)
                return 24 if is_end_time else 0
            return hour
        else:  # PM
            hour = int(time_str.split(':')[0])
            if hour == 12:
                return 12  # 12:00 PM = 12
            return hour + 12
    except:
        return 0  # Fallback


def get_rate_plan_time_periods(plan_code: str, season: str = "summer"):
    """
    Get the time periods for a specific rate plan based on the CSV data.
    Returns a list of dictionaries with time periods and their labels.
    """
    # Define time periods for each rate plan based on pge_rates.csv
    # Note: Times are structured to avoid crossing midnight for vrect compatibility
    time_periods = {
        "E-1": [],  # No time-based periods for tiered plan
        "E-TOU-C": [
            {"start": "4:00 PM", "end": "9:00 PM", "label": "Peak Hours", "color": "red"}
        ],
        "E-TOU-D": [
            {"start": "5:00 PM", "end": "8:00 PM", "label": "Peak Hours", "color": "red"}
        ],
        "E-ELEC": [
            {"start": "4:00 PM", "end": "9:00 PM", "label": "Peak Hours", "color": "red"},
            {"start": "3:00 PM", "end": "4:00 PM", "label": "Super-Peak" if season == "summer" else "Partial-Peak", "color": "darkred" if season == "summer" else "orange"},
            {"start": "9:00 PM", "end": "12:00 AM", "label": "Partial-Peak", "color": "orange"}
        ],
        "EV2-A": [
            {"start": "4:00 PM", "end": "9:00 PM", "label": "Peak Hours", "color": "red"},
            {"start": "3:00 PM", "end": "4:00 PM", "label": "Partial-Peak", "color": "orange"},
            {"start": "9:00 PM", "end": "12:00 AM", "label": "Partial-Peak", "color": "orange"}
        ],
        "EV-B": [
            {"start": "2:00 PM", "end": "9:00 PM", "label": "Peak Hours", "color": "red"},
            {"start": "7:00 AM", "end": "2:00 PM", "label": "Partial-Peak", "color": "orange"},
            {"start": "9:00 PM", "end": "11:00 PM", "label": "Partial-Peak", "color": "orange"}
        ]
    }
    
    return time_periods.get(plan_code, [])


def display_monthly_analysis(usage_df: pd.DataFrame, calculator: PGERateCalculator, territory: str, heating_type: str):
    """Interactive monthly cost analysis with forecasting for missing end-of-month data."""
    st.markdown("## 📅 Monthly Cost Analysis")

    # Determine which months have any data
    usage_df['month_year'] = usage_df['datetime'].dt.to_period('M')
    available_months = sorted({str(p) for p in usage_df['month_year'].unique()})
    if not available_months:
        st.warning("No monthly data available.")
        return

    # Month picker (latest month pre-selected)
    month_choice = st.selectbox("Select Year-Month for analysis:", options=available_months, index=len(available_months) - 1)
    month_period = pd.Period(month_choice)

    month_data = usage_df[usage_df['month_year'] == month_period].copy()
    # Guarantee an 'hour' column exists for grouping
    if 'hour' not in month_data.columns:
        month_data['hour'] = month_data['datetime'].dt.hour
    if month_data.empty:
        st.warning("No data for the selected month.")
        return

    month_start = pd.to_datetime(str(month_period))
    # Forecast up to the very first moment of the next month
    month_end = month_start + pd.offsets.MonthBegin(1)
    earliest = month_data['datetime'].min()
    latest = month_data['datetime'].max()

    # Decide whether to forecast remaining days (missing END) – only if we have data starting very close to the first of the month
    forecast_needed = (earliest.date() <= month_start.date() + timedelta(days=1)) and (latest.date() < month_end.date())

    if forecast_needed:
        st.info("🚀 Forecasting usage for the remainder of the month based on historical hourly averages…")
        # Build 15-minute intervals until (but not including) the next-month boundary
        remaining_intervals = pd.date_range(
            start=latest + timedelta(minutes=15),
            end=month_end - timedelta(minutes=15),
            freq='15min'
        )
        if not remaining_intervals.empty:
            # Use hourly averages from ALL data, not just current month
            if 'hour' not in usage_df.columns:
                usage_df['hour'] = usage_df['datetime'].dt.hour
            hourly_avg = usage_df.groupby('hour')['usage_kwh'].mean()
            forecast_rows = [{
                'datetime': ts,
                'usage_kwh': hourly_avg.get(ts.hour, 0),
                'hour': ts.hour,
                'month_year': month_period
            } for ts in remaining_intervals]
            month_data = pd.concat([month_data, pd.DataFrame(forecast_rows)], ignore_index=True)

    # Use cheapest provider for each plan in monthly analysis
    results, provider_results_all = compute_best_results(calculator, month_data, territory, heating_type)

    # Calculate total kWh for the month
    total_kwh = month_data['usage_kwh'].sum()
    total_wh = total_kwh * 1000  # Convert to watt hours

    # Display energy consumption summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Energy Consumption", f"{total_kwh:.1f} kWh", f"{total_wh:,.0f} Wh")
    with col2:
        st.metric("Average Daily Usage", f"{total_kwh / calendar.monthrange(month_start.year, month_start.month)[1]:.2f} kWh")
    with col3:
        peak_hour_usage = month_data.groupby('hour')['usage_kwh'].sum().max()
        peak_hour = month_data.groupby('hour')['usage_kwh'].sum().idxmax()
        peak_hour_formatted = format_hour_12(peak_hour)
        st.metric("Peak Hour Usage", f"{peak_hour_usage:.3f} kWh", f"at {peak_hour_formatted}")

    # PROMINENT COST ANALYSIS - Moved up and made more prominent
    st.markdown("---")
    st.markdown("### 💰 Monthly Cost Analysis")

    # Build dataframe for cost analysis
    costs_df = pd.DataFrame([
        {
            'Plan': plan_label_provider(code, data.get('provider', 'pge')),
            'Cost': data['total_cost'],
            'kWh': data['total_kwh']
        }
        for code, data in results.items()
    ]).sort_values('Cost')

    # Best plan recommendation with enhanced styling - PROMINENT
    best_plan_row = costs_df.iloc[0]
    worst_plan_row = costs_df.iloc[-1]
    savings = worst_plan_row['Cost'] - best_plan_row['Cost']
    
    st.markdown(f"""
    <div class="best-plan-card">
        <h2>🏆 BEST PLAN FOR {month_choice.upper()}</h2>
        <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{best_plan_row['Plan']}</h1>
        <h3 style="margin: 0;">Monthly Cost: ${best_plan_row['Cost']:.2f}</h3>
        <p style="font-size: 1.2rem; margin: 1rem 0;">
            Cost per kWh: ${best_plan_row['Cost'] / best_plan_row['kWh']:.4f}<br>
            <strong>💰 You could save ${savings:.2f}/month</strong> vs worst plan ({worst_plan_row['Plan']})
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick cost comparison table - PROMINENT
    st.markdown("#### 📊 All Plans Cost Comparison")
    display_costs_df = costs_df.copy()
    display_costs_df['Rank'] = range(1, len(display_costs_df) + 1)
    display_costs_df['Cost'] = display_costs_df['Cost'].apply(lambda x: f"${x:.2f}")
    display_costs_df['Cost per kWh'] = (costs_df['Cost'] / costs_df['kWh']).apply(lambda x: f"${x:.4f}")
    display_costs_df['Savings vs Best'] = ['Best Plan'] + [f"${costs_df.iloc[i]['Cost'] - costs_df.iloc[0]['Cost']:.2f} more" for i in range(1, len(costs_df))]
    
    # Style the dataframe
    def color_rank(val):
        if val == 1:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif val == 2:
            return 'background-color: #fff3cd; color: #856404'
        elif val == 3:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_costs_df = display_costs_df[['Rank', 'Plan', 'Cost', 'Cost per kWh', 'Savings vs Best']].style.map(color_rank, subset=['Rank'])
    st.dataframe(styled_costs_df, use_container_width=True, hide_index=True)

    # Bar chart of costs - PROMINENT
    bar_colors = ['#2ecc71' if i == 0 else '#f39c12' if i == 1 else '#e74c3c' if i == 2 else '#95a5a6' for i in range(len(costs_df))]
    fig_bar = go.Figure(go.Bar(
        x=costs_df['Plan'],
        y=costs_df['Cost'],
        marker_color=bar_colors,
        text=[f"${c:.2f}" for c in costs_df['Cost']],
        textposition='auto'
    ))
    fig_bar.update_layout(
        title={'text': f"💰 Cost Comparison – {month_choice}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis_title="Rate Plan",
        yaxis_title="Monthly Cost ($)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Detailed Energy Usage Analysis")

    # Create hourly energy consumption chart with plan-specific time periods
    hourly_avg = month_data.groupby('hour')['usage_kwh'].mean().reset_index()
    hourly_avg['hour_formatted'] = hourly_avg['hour'].apply(format_hour_12)
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_avg['hour_formatted'],
        y=hourly_avg['usage_kwh'],
        marker_color='skyblue',
        name='Average Hourly Usage'
    ))
    
    # Add time period shading based on the best rate plan
    # Extract plan code from format: "Plan Name (CODE) [Provider]"
    plan_text = best_plan_row['Plan']
    
    # Find the last occurrence of (CODE) before any [Provider] bracket
    if ' [' in plan_text:
        # Remove provider info first: "Plan Name (CODE) [Provider]" -> "Plan Name (CODE)"
        plan_text = plan_text.split(' [')[0]
    
    # Now extract the code from the parentheses: "Plan Name (CODE)" -> "CODE"
    if '(' in plan_text and ')' in plan_text:
        # Find the last parentheses pair
        last_open = plan_text.rfind('(')
        last_close = plan_text.rfind(')')
        if last_open < last_close:
            best_plan_code = plan_text[last_open + 1:last_close]
        else:
            best_plan_code = plan_text  # Fallback if parsing fails
    else:
        best_plan_code = plan_text  # Fallback if no parentheses found
    
    # Determine season based on month
    month_num = month_start.month
    season = "summer" if 6 <= month_num <= 9 else "winter"
    time_periods = get_rate_plan_time_periods(best_plan_code, season)
    
    # Add time period annotations
    if time_periods:
        for period in time_periods:
            # Convert time strings to hour indices for proper alignment
            start_hour = time_string_to_hour(period["start"], is_end_time=False)
            end_hour = time_string_to_hour(period["end"], is_end_time=True)
            
            fig_hourly.add_vrect(
                x0=start_hour - 0.5,  # Offset by 0.5 to align with bar edges
                x1=end_hour - 0.5, 
                fillcolor=period["color"], 
                opacity=0.15, 
                annotation_text=period["label"], 
                annotation_position="top left"
            )
        
        # Add a note about the rate plan
        fig_hourly.add_annotation(
            text=f"Time periods shown for: {best_plan_row['Plan']} ({season.title()} Season)",
            xref="paper", yref="paper",
            x=0.5, y=1.08,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center"
        )
    
    fig_hourly.update_layout(
        title={'text': f"Energy Consumption by Hour – {month_choice}", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Hour of Day",
        yaxis_title="Average Usage (kWh)",
        height=450,  # Slightly taller to accommodate annotation
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Create cumulative kWh consumption chart
    st.markdown("### 📈 Cumulative Energy Consumption")
    
    # Group by date and calculate daily totals
    daily_usage = month_data.groupby(month_data['datetime'].dt.date)['usage_kwh'].sum().reset_index()
    daily_usage.columns = ['date', 'daily_kwh']
    daily_usage = daily_usage.sort_values('date')
    daily_usage['cumulative_kwh'] = daily_usage['daily_kwh'].cumsum()
    
    # Determine which days are forecasted
    # Use the latest actual date (before forecasting) to determine the cutoff
    actual_latest_date = latest.date()
    daily_usage['is_forecasted'] = daily_usage['date'] > actual_latest_date
    
    # Split into actual and forecasted data
    actual_data = daily_usage[~daily_usage['is_forecasted']]
    forecasted_data = daily_usage[daily_usage['is_forecasted']]
    
    fig_cumulative = go.Figure()
    
    # Add actual consumption line
    if not actual_data.empty:
        fig_cumulative.add_trace(go.Scatter(
            x=actual_data['date'],
            y=actual_data['cumulative_kwh'],
            mode='lines+markers',
            name='Actual Consumption',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.1f} kWh<extra></extra>'
        ))
    
    # Add forecasted consumption line (dotted)
    if not forecasted_data.empty and forecast_needed:
        # Connect the last actual point with the first forecasted point
        if not actual_data.empty:
            connection_data = pd.concat([
                actual_data.iloc[-1:],
                forecasted_data.iloc[:1]
            ])
            fig_cumulative.add_trace(go.Scatter(
                x=connection_data['date'],
                y=connection_data['cumulative_kwh'],
                mode='lines',
                name='Forecast Connection',
                line=dict(color='#ff7f0e', width=3, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig_cumulative.add_trace(go.Scatter(
            x=forecasted_data['date'],
            y=forecasted_data['cumulative_kwh'],
            mode='lines+markers',
            name='Forecasted Consumption',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Forecasted Cumulative: %{y:.1f} kWh<extra></extra>'
        ))
    
    fig_cumulative.update_layout(
        title={'text': f"Cumulative Energy Consumption – {month_choice}", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Cumulative Usage (kWh)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)


def display_provider_selection():
    """Sidebar section for selecting generation provider and comparison options."""
    st.sidebar.markdown("## ⚡️ Generation Provider")

    provider_keys = list(PROVIDERS.keys())
    provider_labels = [PROVIDERS[k]["label"] for k in provider_keys]

    selected_index = 0  # Default PG&E
    provider_choice_label = st.sidebar.radio(
        "Select your electricity generation provider:",
        options=provider_labels,
        index=selected_index,
    )

    # Map label back to key
    provider_choice = provider_keys[provider_labels.index(provider_choice_label)]

    # Show description card
    with st.sidebar.expander("About this provider", expanded=False):
        st.markdown(f"<div class='provider-card'>{PROVIDERS[provider_choice]['description']}</div>", unsafe_allow_html=True)

    return provider_choice


def main():
    """Main application function."""
    configure_page()
    display_header()
    
    # Initialize calculator
    try:
        calculator = PGERateCalculator()
    except Exception as e:
        st.error(f"Error initializing calculator: {str(e)}")
        st.stop()
    
    # Configuration
    territory, heating_type = display_baseline_territory_selection()
    provider_choice = display_provider_selection()
    
    # Persist provider selection for other components (like monthly analysis)
    if hasattr(st, 'session_state'):
        st.session_state['selected_provider'] = provider_choice
    
    st.markdown("---")
    
    # File upload
    uploaded_file = display_file_uploader()
    
    if uploaded_file is not None:
        try:
            # Parse usage data
            with st.spinner("🔄 Processing usage data..."):
                usage_df = calculator.parse_usage_data(uploaded_file)
            
            if usage_df is not None and not usage_df.empty:
                # Calculate rates for all plans
                with st.spinner("💰 Calculating rates for all plans..."):
                    results, provider_results_all = compute_best_results(calculator, usage_df, territory, heating_type)
                
                # Update sidebar with stats
                display_sidebar_stats(usage_df, results, territory, heating_type)
                
                # Display usage analysis
                filtered_df = display_usage_analysis(usage_df)
                
                # Display comparison for selected provider
                display_rate_comparison(results, usage_df)
                

                
                # Display monthly analysis
                display_monthly_analysis(usage_df, calculator, territory, heating_type)
                
            else:
                st.error("❌ No usage data found in the uploaded file. Please check the file format.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure you've uploaded a valid PG&E usage CSV file.")
    
    else:
        # Initialize sidebar without data
        display_sidebar_stats()
        
        # Show sample data or instructions when no file is uploaded
        st.info("👆 Please upload your PG&E usage CSV file to begin the comprehensive analysis.")
        
        # Display plan details
        display_plan_details()


if __name__ == "__main__":
    main() 