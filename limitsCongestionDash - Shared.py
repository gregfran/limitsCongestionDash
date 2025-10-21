import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime as dt, timedelta
from sqlalchemy import create_engine, engine
import urllib

# Constants
AIS_LIMITS = {
    'Flow Into Ottawa [FIO]': 2900.0,
    'Flow North [FN]': 1500.0,
    'Flow South [FS]': 2100.0,
    'P502X': 1585.0,
    'Flow East To Toronto [FETT]': 5500.0,
    'Queenston Flow West [QFW]': 2500.0,
    'Ontario-New York Export Summer': 1700.0,
    'Ontario-New York Import Summer': 1300.0,
    'Ontario-New York Export Winter': 1900.0,
    'Ontario-New York Import Winter': 1550.0,
    'Ontario-Michigan Export Summer': 1450.0,
    'Ontario-Michigan Import Summer': 1350.0,
    'Ontario-Michigan Export Winter': 1450.0,
    'Ontario-Michigan Import Winter': 1500.0
}

ZONES = ['NORTHWEST', 'NORTHEAST', 'WEST', 'SOUTHWEST', 'NIAGARA', 'ESSA', 'TORONTO', 'EAST', 'OTTAWA']

# Database connection
@st.cache_data
def load_data():
    """Load data from SQL Service database"""

    params = urllib.parse.quote_plus(
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        'Encrypt=yes;'
        'TrustServerCertificate=no;'
        'Connection Timeout=30;'
    )

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    ##### LOAD DA DATA #####
    da_query = """
    SELECT EstTimeStamp,
        Zone,
        ZonePrice,
        LossPrice,
        CongestionPrice
    FROM dbo.IESO_DaVirtualZonalPrice
    ORDER BY EstTimeStamp DESC, Zone
    """
    da_price_df = pd.read_sql_query(da_query, engine)
    
    ##### LOAD RT DATA #####
    rt_query = """
    SELECT
        EstTimeStamp,
        Zone,
        AVG(ZonePrice) AS ZonePrice,
        AVG(LossPrice) AS LossPrice,
        AVG(CongestionPrice) AS CongestionPrice
    FROM dbo.IESO_RealtimeVirtualZonalPrices_5min
    GROUP BY EstTimeStamp, Zone
    ORDER BY EstTimeStamp, Zone
    """
    rt_price_df = pd.read_sql_query(rt_query, engine)

    ##### LOAD LIMITS DATA #####
    limits_query = "SELECT * FROM dbo.IESO_InterfaceLimits"
    limits_df = pd.read_sql_query(limits_query, engine)

    ##### KILL ENGINE #####
    engine.dispose()

    return limits_df, da_price_df, rt_price_df

def process_limits_data(limits_df, interface_name):
    """Process limits data for a specific interface"""
    interface_limits_df = limits_df[limits_df['Interface'] == interface_name].copy()
    
    if interface_limits_df.empty:
        return pd.DataFrame(columns=['datetime', 'OperatingLimit'])
    
    interface_limits_df['IssueDate'] = pd.to_datetime(interface_limits_df['IssueDate'])
    interface_limits_df['StartDate'] = pd.to_datetime(interface_limits_df['StartDate'])
    interface_limits_df['EndDate'] = pd.to_datetime(interface_limits_df['EndDate'])
    interface_limits_df['OperatingLimit'] = pd.to_numeric(interface_limits_df['OperatingLimit'], errors='coerce')
    
    # Expand limits to hourly data
    expanded_records = []
    for _, row in interface_limits_df.iterrows():
        hour_range = pd.date_range(start=row['StartDate'], end=row['EndDate'], freq='h')
        for hour in hour_range:
            expanded_records.append({
                'datetime': hour,
                'OperatingLimit': row['OperatingLimit']
            })
    
    if not expanded_records:
        return pd.DataFrame(columns=['datetime', 'OperatingLimit'])
    
    expanded_df = pd.DataFrame(expanded_records)
    expanded_df['datetime'] = expanded_df['datetime'].dt.floor('h')
    hourly_min_limits = expanded_df.groupby('datetime')['OperatingLimit'].min().reset_index()
    
    return hourly_min_limits

def process_price_data(price_df, zone, price_type='DA'):
    """Process pricing data for a specific zone"""
    price_df = price_df.copy()
    price_df['datetime'] = pd.to_datetime(price_df['EstTimeStamp'])
    price_df['Date'] = price_df['EstTimeStamp'].dt.date
    price_df['Hour'] = price_df['EstTimeStamp'].dt.hour + 1
    price_df = price_df[['datetime', 'Date', 'Hour', 'Zone', 'CongestionPrice']]
    price_df = price_df.rename(columns={'CongestionPrice': 'Value'})
    price_df['Value'] = pd.to_numeric(price_df['Value'], errors='coerce')
    price_df['Zone'] = price_df['Zone'].str.replace(':HUB', '', regex=False)

    zone_price_df = price_df[(price_df['Zone'] == zone)]
    zone_price_df = zone_price_df[['datetime', 'Value']].rename(columns={'Value': f'{price_type}_cg'})
    
    return zone_price_df

def create_datetime_shell(start_date, end_date):
    """Create datetime range DataFrame"""
    datetime_range = pd.date_range(start=start_date, end=end_date, freq='h')
    return pd.DataFrame({'datetime': datetime_range})

def calculate_interface_correlations(limits_df, da_price_df, rt_price_df, selected_zone, start_date, end_date):
    """Calculate correlations between all interfaces and zonal prices"""
    # Create datetime shell for the analysis period
    datetime_df = create_datetime_shell(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date) + timedelta(hours=23, minutes=59)
    )
    
    # Process pricing data for selected zone
    da_price_data = process_price_data(da_price_df, selected_zone, 'DA')
    rt_price_data = process_price_data(rt_price_df, selected_zone, 'RT')
    
    # Merge pricing data
    price_df = datetime_df.merge(da_price_data, on='datetime', how='left')
    price_df = price_df.merge(rt_price_data, on='datetime', how='left')
    
    correlations = []
    
    for interface_name in AIS_LIMITS.keys():
        # Process limits data for each interface
        limits_data = process_limits_data(limits_df, interface_name)
        
        if limits_data.empty or 'OperatingLimit' not in limits_data.columns:
            # If no limits data, use AIS as constant
            temp_df = datetime_df.copy()
            temp_df['OperatingLimit'] = AIS_LIMITS[interface_name]
            limits_data = temp_df[['datetime', 'OperatingLimit']]
        
        # Merge with pricing data
        analysis_df = price_df.merge(limits_data, on='datetime', how='left')
        
        # COnfirm OperatingLimit col and fill NaNs
        if 'OperatingLimit' not in analysis_df.columns:
            analysis_df['OperatingLimit'] = AIS_LIMITS[interface_name]
        else:
            analysis_df['OperatingLimit'] = analysis_df['OperatingLimit'].fillna(AIS_LIMITS[interface_name])
        
        # Calculate correlations if pricing data exists
        valid_data = analysis_df.dropna(subset=['DA_cg', 'RT_cg'])
        
        if len(valid_data) > 10:  # Need more than 10 pts
            # Calculate inverse correlation (lower limits often correlate with higher prices)
            da_corr = valid_data['OperatingLimit'].corr(valid_data['DA_cg'])
            rt_corr = valid_data['OperatingLimit'].corr(valid_data['RT_cg'])
            
            # Calculate constraint frequency (how often is limit below AIS limit)
            constraint_freq = (valid_data['OperatingLimit'] < AIS_LIMITS[interface_name]).mean()
            
            correlations.append({
                'Interface': interface_name,
                'DA_Correlation': da_corr,
                'RT_Correlation': rt_corr,
                'Avg_Correlation': (abs(da_corr) + abs(rt_corr)) / 2,
                'Constraint_Frequency': constraint_freq,
                'Data_Points': len(valid_data)
            })
        else:
            correlations.append({
                'Interface': interface_name,
                'DA_Correlation': np.nan,
                'RT_Correlation': np.nan,
                'Avg_Correlation': np.nan,
                'Constraint_Frequency': 0,
                'Data_Points': len(valid_data)
            })
    
    correlation_df = pd.DataFrame(correlations)
    
    # Sort by best to worst corr, rank
    correlation_df = correlation_df.sort_values('Avg_Correlation', ascending=False, na_position='last')
    correlation_df.reset_index(drop=True, inplace=True)
    correlation_df['Rank'] = correlation_df.index + 1
    
    return correlation_df

def corr_matrixer(df):
    """Generate correlation matrix for specified columns"""
    # Determine which operating limit column to use
    operating_limit_col = None
    if 'OperatingLimit_1' in df.columns:
        operating_limit_col = 'OperatingLimit_1'
    elif 'OperatingLimit' in df.columns:
        operating_limit_col = 'OperatingLimit'
    
    if operating_limit_col and 'DA_cg' in df.columns and 'RT_cg' in df.columns:
        corr_matrix = df[[operating_limit_col, 'DA_cg', 'RT_cg']].corr()
        return corr_matrix
    else:
        # Return empty correlation matrix if required columns don't exist
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="IESO Limits & Zonal Congestion Dashboard", layout="wide")

    st.title("IESO Limits & Zonal Congestion Dashboard")

    # Load data
    with st.spinner("Loading data..."):
        limits_df, da_price_df, rt_price_df = load_data()
    
    # Sidebar controls
    st.sidebar.header("Chart Controls")
    
    # Date range inputs
    min_date = pd.to_datetime('2025-05-03').date()
    max_date = dt.now().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=max_date - timedelta(days=30),
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        return
    
    # Zone selection for pricing (needed for correlation analysis)
    selected_zone = st.sidebar.selectbox(
        "Zone for Pricing",
        options=ZONES,
        index=ZONES.index('OTTAWA') if 'OTTAWA' in ZONES else 0
    )
    
    # Correlation Analysis Section
    st.sidebar.subheader("Correlation Analysis")
    show_correlation = st.sidebar.checkbox("Show Interface-Price Correlations", value=False)  # Changed default to False for debugging
    
    correlation_df = None
    if show_correlation:
        try:
            with st.spinner("Calculating correlations..."):
                correlation_df = calculate_interface_correlations(
                    limits_df, da_price_df, rt_price_df, selected_zone, start_date, end_date
                )
        except Exception as e:
            st.sidebar.error(f"Error calculating correlations: {str(e)}")
            show_correlation = False  # Disable correlation display if it fails
    
    # Interface/Limits selection (two dropdowns for multiple selection)
    st.sidebar.subheader("Interface Selection")
    interface_options = list(AIS_LIMITS.keys())
    
    # Get correlation-based recommendation if analysis is enabled
    recommended_interface = None
    if show_correlation and correlation_df is not None:
        top_corr = correlation_df[correlation_df['Avg_Correlation'].notna()].head(1)
        if not top_corr.empty:
            recommended_interface = top_corr.iloc[0]['Interface']
    
    # Default selection logic
    if recommended_interface and recommended_interface in interface_options:
        default_index_1 = interface_options.index(recommended_interface)
        # st.sidebar.success(f"Recommended: {recommended_interface}")
    elif 'Flow Into Ottawa [FIO]' in interface_options:
        default_index_1 = interface_options.index('Flow Into Ottawa [FIO]')
    else:
        default_index_1 = 0
    
    interface_1 = st.sidebar.selectbox(
        "Primary Interface",
        options=interface_options,
        index=default_index_1,
        help="Select primary interface to display"
    )
    
    interface_2 = st.sidebar.selectbox(
        "Secondary Interface (Optional)",
        options=['None'] + interface_options,
        index=0,
        help="Optionally select a second interface for comparison"
    )
    
    # Display correlation analysis if enabled
    if show_correlation and correlation_df is not None:
        # Display correlation analysis
        with st.expander("Interface-Price Correlation Analysis", expanded=True):
            st.markdown(f"""
            **Correlation Period:** {start_date} to {end_date}  
            **Zone:** {selected_zone}  
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Interface Rank and Sort")
                
                # Format the dataframe for display
                display_corr_df = correlation_df.copy()
                display_corr_df['DA_Correlation'] = display_corr_df['DA_Correlation'].round(3)
                display_corr_df['RT_Correlation'] = display_corr_df['RT_Correlation'].round(3)
                display_corr_df['Avg_Correlation'] = display_corr_df['Avg_Correlation'].round(3)
                display_corr_df['Constraint_Frequency'] = (display_corr_df['Constraint_Frequency'] * 100).round(1)
                                
                # Rename columns for display
                display_corr_df = display_corr_df.rename(columns={
                    'DA_Correlation': 'DA Corr',
                    'RT_Correlation': 'RT Corr', 
                    'Avg_Correlation': 'Avg |Corr|',
                    'Constraint_Frequency': 'Constrained %',
                    'Data_Points': 'Data Points'
                })
                
                st.dataframe(
                    display_corr_df[['Rank', 'Interface', 'DA Corr', 'RT Corr', 'Avg |Corr|', 'Constrained %']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add explanation
                st.caption("""
                **Legend:** 
                - **DA/RT Corr**: Correlation between interface limits and prices (-1 to 1)
                - **Avg |Corr|**: Average absolute correlation (higher = stronger relationship)
                - **Constrained %**: Percentage of time interface was below AIS limit
                """)
            
            with col2:
                st.subheader("Recommended Interfaces/Interties")
                
                # Get top 3 interfaces with valid correlations
                top_interfaces = correlation_df[correlation_df['Avg_Correlation'].notna()].head(3)
                
                for i, (_, row) in enumerate(top_interfaces.iterrows(), 1):
                    with st.container():
                        st.markdown(f"**#{i}. {row['Interface']}**")
                        st.markdown(f"Avg Correlation: {row['Avg_Correlation']:.3f}")
                        st.markdown(f"Constrained: {row['Constraint_Frequency']*100:.1f}% of analysis period")
                        if i < 3:  # Don't add divider after last item
                            st.markdown("---")
                
                if len(top_interfaces) > 0:
                    st.info("Higher absolute correlation values indicate stronger relationships between interface limits and pricing.\n" \
                    "Verify that the correlated interface makes physical sense before drawing conclusions.")
    
    st.markdown("---")  # Separator before main chart
    
    # Process data based on selections
    with st.spinner("Processing data..."):
        # Create datetime shell
        datetime_df = create_datetime_shell(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date) + timedelta(hours=23, minutes=59)
        )
        
        # Process limits data for selected interfaces
        limits_1 = process_limits_data(limits_df, interface_1)
        display_df = datetime_df.merge(limits_1, on='datetime', how='left')
        display_df['AISLimit_1'] = AIS_LIMITS[interface_1]
        
        # Handle the case where OperatingLimit column may not exist
        if 'OperatingLimit' in display_df.columns:
            display_df['OperatingLimit_1'] = display_df['OperatingLimit'].fillna(AIS_LIMITS[interface_1])
            display_df = display_df.drop('OperatingLimit', axis=1)
        else:
            display_df['OperatingLimit_1'] = AIS_LIMITS[interface_1]
        
        # Add second interface if selected
        if interface_2 != 'None':
            limits_2 = process_limits_data(limits_df, interface_2)
            limits_2 = limits_2.rename(columns={'OperatingLimit': 'OperatingLimit_2'})
            display_df = display_df.merge(limits_2, on='datetime', how='left')
            display_df['AISLimit_2'] = AIS_LIMITS[interface_2]
            display_df['OperatingLimit_2'] = display_df['OperatingLimit_2'].fillna(AIS_LIMITS[interface_2])
        
        # Process pricing data
        da_price_data = process_price_data(da_price_df, selected_zone, 'DA')
        rt_price_data = process_price_data(rt_price_df, selected_zone, 'RT')
        
        display_df = display_df.merge(da_price_data, on='datetime', how='left')
        display_df = display_df.merge(rt_price_data, on='datetime', how='left')

        # # Correlation matrix
        # corr_matrix = corr_matrixer(display_df)
    
    # Create the plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add limits traces
    fig.add_trace(
        go.Scatter(
            x=display_df['datetime'],
            y=display_df['OperatingLimit_1'],
            name=f'{interface_1} - Operating Limit',
            line=dict(color='red', dash='dash')
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=display_df['datetime'],
            y=display_df['AISLimit_1'],
            name=f'{interface_1} - AIS Limit',
            line=dict(color='orange', dash='dot')
        ),
        secondary_y=False,
    )
    
    # Add second interface if selected
    if interface_2 != 'None':
        fig.add_trace(
            go.Scatter(
                x=display_df['datetime'],
                y=display_df['OperatingLimit_2'],
                name=f'{interface_2} - Operating Limit',
                line=dict(color='darkred', dash='dash')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=display_df['datetime'],
                y=display_df['AISLimit_2'],
                name=f'{interface_2} - AIS Limit',
                line=dict(color='darkorange', dash='dot')
            ),
            secondary_y=False,
        )
    
    # Add pricing traces
    fig.add_trace(
        go.Scatter(
            x=display_df['datetime'],
            y=display_df['DA_cg'],
            name=f'{selected_zone} - DA Cg Price',
            line=dict(color='blue')
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=display_df['datetime'],
            y=display_df['RT_cg'],
            name=f'{selected_zone} - RT Cg Price',
            line=dict(color='green')
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_zone} Zone",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date"
        ),
        height=600
    )
    
    # Set y-axes
    max_ais_limit = max(AIS_LIMITS[interface_1], AIS_LIMITS.get(interface_2, 0) if interface_2 != 'None' else 0)
    fig.update_yaxes(
        title_text="Limits (MW)",
        range=[0, max_ais_limit + 200],
        secondary_y=False
    )
    
    # Dynamic pricing axis scaling
    price_columns = ['DA_cg', 'RT_cg']
    price_data = display_df[price_columns].dropna()
    
    if not price_data.empty:
        max_val = price_data.max().max()
        min_val = price_data.min().min()

        # Price range dynamic scaling logic to ensure visibility
        if max_val > 200 and min_val < -200:
            price_range=[-200, 200]
        elif max_val > 200 and min_val >= 0:
            price_range=[0, 200]
        elif min_val < -200 and max_val <= 0:
            price_range=[-200, 0]
        elif max_val > 200 and min_val > -200:
            price_range=[None, 200]
        elif min_val < -200 and max_val <= 200:
            price_range=[-200, None]
        else:
            price_range = None
                
        if price_range:
            fig.update_yaxes(
                title_text="Price ($/MWh)",
                range=price_range,
                side="right",
                secondary_y=True
            )
        else:
            fig.update_yaxes(
                title_text="Price ($/MWh)",
                side="right",
                autorange=True,
                secondary_y=True
            )
    else:
        fig.update_yaxes(
            title_text="Price ($/MWh)",
            side="right",
            secondary_y=True
        )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Limits Summary")
        if not display_df.empty:
            avg_limit_1 = display_df['OperatingLimit_1'].mean()
            min_limit_1 = display_df['OperatingLimit_1'].min()
            st.metric(f"{interface_1} Avg Limit", f"{avg_limit_1:.0f} MW")
            st.metric(f"{interface_1} Min Limit", f"{min_limit_1:.0f} MW")
            
            if interface_2 != 'None':
                avg_limit_2 = display_df['OperatingLimit_2'].mean()
                min_limit_2 = display_df['OperatingLimit_2'].min()
                st.metric(f"{interface_2} Avg Limit", f"{avg_limit_2:.0f} MW")
                st.metric(f"{interface_2} Min Limit", f"{min_limit_2:.0f} MW")
    
    with col2:
        st.subheader("DA Price Summary")
        if not display_df['DA_cg'].isna().all():
            da_avg = display_df['DA_cg'].mean()
            da_max = display_df['DA_cg'].max()
            da_min = display_df['DA_cg'].min()
            st.metric("Average DA Price", f"${da_avg:.2f}/MWh")
            st.metric("Max DA Price", f"${da_max:.2f}/MWh")
            st.metric("Min DA Price", f"${da_min:.2f}/MWh")
    
    with col3:
        st.subheader("RT Price Summary")
        if not display_df['RT_cg'].isna().all():
            rt_avg = display_df['RT_cg'].mean()
            rt_max = display_df['RT_cg'].max()
            rt_min = display_df['RT_cg'].min()
            st.metric("Average RT Price", f"${rt_avg:.2f}/MWh")
            st.metric("Max RT Price", f"${rt_max:.2f}/MWh")
            st.metric("Min RT Price", f"${rt_min:.2f}/MWh")

if __name__ == "__main__":
    main()