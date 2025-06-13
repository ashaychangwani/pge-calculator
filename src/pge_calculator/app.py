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

from .calculator import PGERateCalculator, BASELINE_ALLOWANCES


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="PG&E Rate Calculator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def display_header():
    """Display the main header and description."""
    st.title("⚡ PG&E Rate Plan Calculator")
    st.markdown("""
    **Find the best PG&E electricity rate plan based on your actual usage patterns.**
    
    Upload your usage data and compare costs across all major PG&E rate plans to maximize your savings.
    """)


def display_sidebar(calculator: PGERateCalculator):
    """Display sidebar with configuration options."""
    st.sidebar.header("Configuration")
    
    # Territory selection
    territory = st.sidebar.selectbox(
        "📍 Select your baseline territory:",
        options=list(BASELINE_ALLOWANCES.keys()),
        index=1,  # Default to 'P'
        help="Find your territory on your PG&E bill"
    )
    
    # Heating type selection
    heating_type = st.sidebar.radio(
        "🏠 Select your heating system:",
        options=["basic_electric", "all_electric"],
        format_func=lambda x: "Basic Electric" if x == "basic_electric" else "All Electric",
        help="Basic Electric: No permanent electric space heating. All Electric: Includes permanent electric space heating."
    )
    
    # Display baseline allowances
    st.sidebar.subheader("📊 Your Baseline Allowances")
    allowances = BASELINE_ALLOWANCES[territory][heating_type]
    st.sidebar.metric("Summer (kWh/day)", f"{allowances['summer']:.1f}")
    st.sidebar.metric("Winter (kWh/day)", f"{allowances['winter']:.1f}")
    
    return territory, heating_type


def display_file_uploader():
    """Display file uploader and instructions."""
    st.header("📁 Upload Your Usage Data")
    
    # Instructions
    with st.expander("How to get your usage data", expanded=False):
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


def display_usage_analysis(usage_df: pd.DataFrame):
    """Display usage data analysis and visualizations."""
    st.header("📊 Usage Analysis")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Usage", f"{usage_df['usage_kwh'].sum():.2f} kWh")
    with col2:
        st.metric("Average Daily", f"{usage_df['usage_kwh'].sum() / len(usage_df['datetime'].dt.date.unique()):.2f} kWh")
    with col3:
        st.metric("Peak Hour Usage", f"{usage_df['usage_kwh'].max():.2f} kWh")
    with col4:
        st.metric("Data Points", f"{len(usage_df):,}")
    
    # Date range slider
    min_date = usage_df['datetime'].min().date()
    max_date = usage_df['datetime'].max().date()
    
    date_range = st.slider(
        "📅 Select date range for analysis:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter data based on date range
    filtered_df = usage_df[
        (usage_df['datetime'].dt.date >= date_range[0]) &
        (usage_df['datetime'].dt.date <= date_range[1])
    ]
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly usage pattern
        hourly_avg = filtered_df.groupby('hour')['usage_kwh'].mean().reset_index()
        fig_hourly = px.line(
            hourly_avg, 
            x='hour', 
            y='usage_kwh',
            title='Average Hourly Usage Pattern',
            labels={'hour': 'Hour of Day', 'usage_kwh': 'Usage (kWh)'}
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Daily usage trend
        daily_usage = filtered_df.groupby(filtered_df['datetime'].dt.date)['usage_kwh'].sum().reset_index()
        daily_usage.columns = ['date', 'usage_kwh']
        
        fig_daily = px.line(
            daily_usage,
            x='date',
            y='usage_kwh',
            title='Daily Usage Trend',
            labels={'date': 'Date', 'usage_kwh': 'Daily Usage (kWh)'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    return filtered_df


def display_rate_comparison(results: dict):
    """Display rate plan comparison results."""
    st.header("💰 Rate Plan Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for plan, data in results.items():
        comparison_data.append({
            'Plan': plan,
            'Total Cost': data['total_cost'],
            'Cost per kWh': data['total_cost'] / data['total_kwh'] if data['total_kwh'] > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Total Cost')
    
    # Best plan recommendation
    best_plan = comparison_df.iloc[0]
    worst_plan = comparison_df.iloc[-1]
    savings = worst_plan['Total Cost'] - best_plan['Total Cost']
    
    st.success(f"""
    🏆 **Recommended Plan: {best_plan['Plan']}**
    
    Total Cost: ${best_plan['Total Cost']:.2f}  
    Cost per kWh: ${best_plan['Cost per kWh']:.4f}  
    Potential Savings: ${savings:.2f} vs worst plan ({worst_plan['Plan']})
    """)
    
    # Comparison chart
    fig = px.bar(
        comparison_df,
        x='Plan',
        y='Total Cost',
        title='Rate Plan Cost Comparison',
        color='Total Cost',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    # Format the dataframe for display
    display_df = comparison_df.copy()
    display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:.2f}")
    display_df['Cost per kWh'] = display_df['Cost per kWh'].apply(lambda x: f"${x:.4f}")
    display_df['Rank'] = range(1, len(display_df) + 1)
    
    # Reorder columns
    display_df = display_df[['Rank', 'Plan', 'Total Cost', 'Cost per kWh']]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def display_plan_details():
    """Display information about different rate plans."""
    st.header("📖 Rate Plan Details")
    
    with st.expander("Rate Plan Information", expanded=False):
        st.markdown("""
        ### Tiered Rate Plan (E-1)
        - **Structure**: Fixed rates regardless of time of use
        - **Tier 1**: Below baseline allowance
        - **Tier 2**: Above baseline allowance
        - **Best for**: Customers with low, consistent usage
        
        ### Time-of-Use Plans
        Variable rates based on time of day and season:
        
        **E-TOU-C**: Time-of-Use with baseline allowances
        - Peak hours: 4-9 PM (summer), 4-9 PM (winter)
        - Different rates for usage above/below baseline
        
        **E-TOU-D**: Simple time-of-use
        - Peak hours: 5-8 PM (summer), 5-8 PM (winter)
        - No baseline differentiation
        
        **E-ELEC**: Electric home rate plan
        - Peak hours: 4-9 PM with super-peak 3-4 PM (summer)
        - Designed for all-electric homes
        
        ### Electric Vehicle Plans
        
        **EV2-A**: Home charging plan
        - Super off-peak: 12-3 PM
        - Peak: 4-9 PM
        
        **EV-B**: Time-of-use for EV owners
        - Super off-peak: 12-7 AM
        - Peak: 2-9 PM (summer), 2-9 PM (winter)
        """)


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
    
    # Sidebar configuration
    territory, heating_type = display_sidebar(calculator)
    
    # File upload
    uploaded_file = display_file_uploader()
    
    if uploaded_file is not None:
        try:
            # Parse usage data
            with st.spinner("Processing usage data..."):
                usage_df = calculator.parse_usage_data(uploaded_file)
            
            if usage_df is not None and not usage_df.empty:
                # Display usage analysis
                filtered_df = display_usage_analysis(usage_df)
                
                # Calculate rates for all plans
                with st.spinner("Calculating rates for all plans..."):
                    results = calculator.calculate_all_plans(filtered_df, territory, heating_type)
                
                # Display comparison
                display_rate_comparison(results)
                
            else:
                st.error("No usage data found in the uploaded file. Please check the file format.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure you've uploaded a valid PG&E usage CSV file.")
    
    else:
        # Show sample data or instructions when no file is uploaded
        st.info("👆 Please upload your PG&E usage CSV file to begin the analysis.")
        
        # Display plan details
        display_plan_details()


if __name__ == "__main__":
    main() 