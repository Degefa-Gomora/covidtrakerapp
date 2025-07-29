# # dashboard_app.py

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import plotly.express as px # For the optional Choropleth Map

# # --- Streamlit App Configuration ---
# st.set_page_config(
#     layout="wide", # Use wide layout for better visualization of plots
#     page_title="COVID-19 Global Data Tracker",
#     page_icon="🦠"
# )

# # --- Title and Introduction ---
# st.title("🦠 COVID-19 Global Data Tracker")
# st.markdown("""
# This interactive dashboard allows you to explore global COVID-19 trends, including cases, deaths, and vaccination progress for selected countries and date ranges.
# Data is sourced from Our World in Data.
# """)

# # --- Data Loading Function with Caching ---
# # @st.cache_data decorator caches the function's output.
# # This means the data will only be loaded and processed once,
# # speeding up subsequent runs after initial load.
# @st.cache_data
# def load_and_prepare_data(file_path='cov.csv'):
#     """
#     Loads the COVID-19 data, converts 'date' column, and sorts.
#     """
#     try:
#         df_raw = pd.read_csv(file_path)
#         df_raw['date'] = pd.to_datetime(df_raw['date'])
#         df_raw = df_raw.sort_values(by=['location', 'date']).reset_index(drop=True)
#         return df_raw
#     except FileNotFoundError:
#         st.error(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory as this script.")
#         st.stop()
#     except Exception as e:
#         st.error(f"An error occurred while loading or preparing data: {e}")
#         st.stop()

# df = load_and_prepare_data()

# # --- Sidebar for User Inputs ---
# st.sidebar.header("📊 Filter Data")

# # Get available countries from the loaded data
# available_countries_raw = sorted(df['location'].unique().tolist())

# # Remove continents and other non-country entities for selection, but keep 'World'
# # You might need to adjust this list based on what appears in your 'location' column
# countries_to_exclude_from_selection = [
#     'World', 'High income', 'Upper middle income', 'Lower middle income', 'Low income',
#     'European Union', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania'
# ]
# selectable_countries = [
#     country for country in available_countries_raw
#     if country not in countries_to_exclude_from_selection
# ]

# # Add 'World' as a default option if it exists
# default_countries = ['United States', 'India', 'World']
# default_countries_filtered = [c for c in default_countries if c in available_countries_raw]

# selected_countries = st.sidebar.multiselect(
#     "Select Countries:",
#     options=selectable_countries,
#     default=default_countries_filtered
# )

# # Ensure 'World' is always included if the user doesn't explicitly select it,
# # and if it's available in the raw data
# if 'World' in available_countries_raw and 'World' not in selected_countries:
#     selected_countries.append('World')


# # Date Range Selection
# min_date_data = df['date'].min()
# max_date_data = df['date'].max()

# date_range = st.sidebar.date_input(
#     "Select Date Range:",
#     value=(min_date_data, max_date_data),
#     min_value=min_date_data,
#     max_value=max_date_data,
#     key='date_selector' # Add a key to prevent "DuplicateWidgetID" errors if used elsewhere
# )

# # Ensure date_range has two dates before proceeding
# if len(date_range) == 2:
#     start_date = pd.to_datetime(date_range[0])
#     end_date = pd.to_datetime(date_range[1])
#     if start_date > end_date: # Swap if dates are inverted
#         start_date, end_date = end_date, start_date
# else: # Fallback if only one date is selected (e.g., initial state)
#     start_date = min_date_data
#     end_date = max_date_data

# # --- Dynamic Data Filtering and Cleaning ---
# # Filter the DataFrame based on user selections
# df_filtered_dashboard = df[
#     (df['location'].isin(selected_countries)) &
#     (df['date'] >= start_date) &
#     (df['date'] <= end_date)
# ].copy()

# # Define numerical columns to fill with 0 (including new ones for hospitalization/ICU)
# numerical_cols_to_fill = [
#     'total_cases', 'new_cases', 'new_cases_smoothed',
#     'total_deaths', 'new_deaths', 'new_deaths_smoothed',
#     'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
#     'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million',
#     'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
#     'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
#     'population', # Important for per million calculations
#     'icu_patients', 'icu_patients_per_million', # ICU data
#     'hosp_patients', 'hosp_patients_per_million', # Hospitalization data
#     'reproduction_rate', 'positive_rate', 'stringency_index' # Other potentially useful numerical data
# ]

# # Apply fillna(0) to relevant numerical columns
# for col in numerical_cols_to_fill:
#     if col in df_filtered_dashboard.columns: # Check if column exists in the dataframe
#         df_filtered_dashboard[col] = df_filtered_dashboard[col].fillna(0)


# # --- Dashboard Layout and Visualizations ---

# if df_filtered_dashboard.empty:
#     st.warning("No data available for the selected criteria. Please adjust your filters.")
# else:
#     st.subheader(f"Analyzing Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

#     # Use columns for a better layout
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("### Total Cases Over Time")
#         fig_cases, ax_cases = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=df_filtered_dashboard, x='date', y='total_cases', hue='location', ax=ax_cases)
#         ax_cases.set_xlabel('Date')
#         ax_cases.set_ylabel('Total Cases')
#         ax_cases.grid(True, linestyle='--', alpha=0.7)
#         ax_cases.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
#         st.pyplot(fig_cases)

#     with col2:
#         st.markdown("### Total Deaths Over Time")
#         fig_deaths, ax_deaths = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=df_filtered_dashboard, x='date', y='total_deaths', hue='location', ax=ax_deaths)
#         ax_deaths.set_xlabel('Date')
#         ax_deaths.set_ylabel('Total Deaths')
#         ax_deaths.grid(True, linestyle='--', alpha=0.7)
#         ax_deaths.ticklabel_format(style='plain', axis='y')
#         st.pyplot(fig_deaths)

#     st.markdown("---") # Separator

#     col3, col4 = st.columns(2)

#     with col3:
#         st.markdown("### Daily New Cases (7-day smoothed)")
#         fig_new_cases, ax_new_cases = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=df_filtered_dashboard, x='date', y='new_cases_smoothed', hue='location', ax=ax_new_cases)
#         ax_new_cases.set_xlabel('Date')
#         ax_new_cases.set_ylabel('Daily New Cases (Smoothed)')
#         ax_new_cases.grid(True, linestyle='--', alpha=0.7)
#         ax_new_cases.ticklabel_format(style='plain', axis='y')
#         st.pyplot(fig_new_cases)

#     with col4:
#         st.markdown("### Case Fatality Ratio (CFR)")
#         # Calculate CFR for the latest available data for each country in the filtered view
#         cfr_data = df_filtered_dashboard.groupby('location').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
#         cfr_data['case_fatality_ratio'] = np.where(
#             cfr_data['total_cases'] > 0,
#             (cfr_data['total_deaths'] / cfr_data['total_cases']) * 100,
#             0
#         )
#         cfr_result_display = cfr_data[['location', 'total_cases', 'total_deaths', 'case_fatality_ratio']].sort_values(by='case_fatality_ratio', ascending=False).round(2)
#         st.dataframe(cfr_result_display)
#         st.info("CFR calculated based on the latest available data point for each selected country within the chosen date range.")


#     st.markdown("---") # Separator

#     st.header("💉 Vaccination Progress")

#     col5, col6 = st.columns(2)

#     with col5:
#         st.markdown("### Total Vaccine Doses Administered Per 100 People")
#         fig_total_vacc, ax_total_vacc = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=df_filtered_dashboard, x='date', y='total_vaccinations_per_hundred', hue='location', ax=ax_total_vacc)
#         ax_total_vacc.set_xlabel('Date')
#         ax_total_vacc.set_ylabel('Total Doses Administered Per 100 People')
#         ax_total_vacc.grid(True, linestyle='--', alpha=0.7)
#         ax_total_vacc.set_ylim(0, df_filtered_dashboard['total_vaccinations_per_hundred'].max() * 1.1) # Set reasonable y-limit
#         st.pyplot(fig_total_vacc)

#     with col6:
#         st.markdown("### Share of Population Fully Vaccinated Per 100 People")
#         fig_fully_vacc, ax_fully_vacc = plt.subplots(figsize=(10, 5))
#         sns.lineplot(data=df_filtered_dashboard, x='date', y='people_fully_vaccinated_per_hundred', hue='location', ax=ax_fully_vacc)
#         ax_fully_vacc.set_xlabel('Date')
#         ax_fully_vacc.set_ylabel('People Fully Vaccinated Per 100 People')
#         ax_fully_vacc.grid(True, linestyle='--', alpha=0.7)
#         ax_fully_vacc.set_ylim(0, 100) # Percentage from 0-100
#         st.pyplot(fig_fully_vacc)

#     st.markdown("### Latest Full Vaccination Coverage (%)")
#     latest_vaccination_data = df_filtered_dashboard.groupby('location').last().reset_index()
#     latest_vaccination_data_countries = latest_vaccination_data[latest_vaccination_data['location'] != 'World']
#     latest_vaccination_data_countries = latest_vaccination_data_countries.sort_values(by='people_fully_vaccinated_per_hundred', ascending=False)

#     fig_bar_vacc, ax_bar_vacc = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=latest_vaccination_data_countries, x='people_fully_vaccinated_per_hundred', y='location', palette='viridis', ax=ax_bar_vacc)
#     ax_bar_vacc.set_xlabel('Percentage of Population Fully Vaccinated')
#     ax_bar_vacc.set_ylabel('Country')
#     ax_bar_vacc.set_xlim(0, 100)
#     st.pyplot(fig_bar_vacc)
#     st.dataframe(latest_vaccination_data_countries[['location', 'people_fully_vaccinated_per_hundred']].round(2))

#     st.markdown("---") # Separator

#     # --- Optional: Hospitalization & ICU Data ---
#     st.header("🏥 Hospitalization & ICU Data (if available)")

#     # Check if hospitalization/ICU columns are actually present and have non-zero data
#     has_hosp_data = 'hosp_patients' in df_filtered_dashboard.columns and df_filtered_dashboard['hosp_patients'].sum() > 0
#     has_icu_data = 'icu_patients' in df_filtered_dashboard.columns and df_filtered_dashboard['icu_patients'].sum() > 0

#     if has_hosp_data or has_icu_data:
#         col7, col8 = st.columns(2)

#         if has_hosp_data:
#             with col7:
#                 st.markdown("### Hospital Patients Over Time")
#                 fig_hosp, ax_hosp = plt.subplots(figsize=(10, 5))
#                 sns.lineplot(data=df_filtered_dashboard, x='date', y='hosp_patients', hue='location', ax=ax_hosp)
#                 ax_hosp.set_xlabel('Date')
#                 ax_hosp.set_ylabel('Hospital Patients')
#                 ax_hosp.grid(True, linestyle='--', alpha=0.7)
#                 ax_hosp.ticklabel_format(style='plain', axis='y')
#                 st.pyplot(fig_hosp)
#         else:
#             with col7:
#                 st.info("Hospitalization data not available for the selected criteria.")

#         if has_icu_data:
#             with col8:
#                 st.markdown("### ICU Patients Over Time")
#                 fig_icu, ax_icu = plt.subplots(figsize=(10, 5))
#                 sns.lineplot(data=df_filtered_dashboard, x='date', y='icu_patients', hue='location', ax=ax_icu)
#                 ax_icu.set_xlabel('Date')
#                 ax_icu.set_ylabel('ICU Patients')
#                 ax_icu.grid(True, linestyle='--', alpha=0.7)
#                 ax_icu.ticklabel_format(style='plain', axis='y')
#                 st.pyplot(fig_icu)
#         else:
#             with col8:
#                 st.info("ICU data not available for the selected criteria.")
#     else:
#         st.info("No hospitalization or ICU data available for the selected countries and date range.")

#     st.markdown("---") # Separator

#     # --- Optional: Choropleth Map (requires plotly) ---
#     st.header("🌍 Global Overview Maps (Latest Data)")
#     st.info("These maps display the latest available data across all countries (not just your selected ones) for a global context.")

#     # Get the latest data point for ALL countries in the original df for mapping
#     df_latest_map = df.dropna(subset=['iso_code', 'total_cases_per_million', 'people_fully_vaccinated_per_hundred']).copy()
#     df_latest_map = df_latest_map.sort_values(by='date').groupby('iso_code').last().reset_index()

#     # Filter out non-country entities like continents and income levels for mapping
#     df_latest_map_countries = df_latest_map[~df_latest_map['location'].isin(countries_to_exclude_from_selection)]

#     if not df_latest_map_countries.empty:
#         # Map 1: Total Cases per Million (Latest)
#         fig_cases_map = px.choropleth(df_latest_map_countries,
#                                       locations="iso_code",
#                                       color="total_cases_per_million",
#                                       hover_name="location",
#                                       color_continuous_scale=px.colors.sequential.Plasma,
#                                       title='Total COVID-19 Cases Per Million People (Latest Data)',
#                                       projection="natural earth")
#         st.plotly_chart(fig_cases_map, use_container_width=True)

#         # Map 2: Fully Vaccinated Per Hundred (Latest)
#         fig_vacc_map = px.choropleth(df_latest_map_countries,
#                                      locations="iso_code",
#                                      color="people_fully_vaccinated_per_hundred",
#                                      hover_name="location",
#                                      color_continuous_scale=px.colors.sequential.Viridis,
#                                      title='Share of Population Fully Vaccinated Per 100 People (Latest Data)',
#                                      projection="natural earth")
#         st.plotly_chart(fig_vacc_map, use_container_width=True)
#     else:
#         st.warning("Could not generate global maps. Check 'iso_code' and data availability in the raw dataset.")

#     st.markdown("---") # Separator

#     # --- Final Deliverable Notes ---
#     st.header("✅ Final Deliverable:")
#     st.markdown("""
#     This Streamlit dashboard serves as an interactive data report that:
#     * **Loads, cleans, analyzes, and visualizes COVID-19 data.**
#     * **Communicates insights** with clear visuals based on user selections.
#     * Is **interactive, easy to use**, and provides a dynamic view of the data.

#     **Key Objectives Achieved:**
#     * ✅ Collected global COVID-19 data from Our World in Data.
#     * ✅ Loaded and explored the dataset using pandas.
#     * ✅ Cleaned and prepared the data by handling missing values and filtering relevant countries/dates.
#     * ✅ Performed exploratory data analysis (EDA) to identify trends in cases, deaths, and vaccinations.
#     * ✅ Created interactive visualizations (line charts, bar charts, choropleth maps) to illustrate key metrics.
#     * ✅ Calculated critical indicators such as death rates and vaccination coverage.
#     * ✅ Presented insights through a user-friendly, interactive dashboard.
#     """)

# dashboard_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px # For interactive maps

# --- 1. Streamlit App Configuration (Best Practice) ---
# Set page configuration early for wide layout and custom title/icon
st.set_page_config(
    layout="wide", # Use wide layout for better visualization of plots
    page_title="COVID-19 Global Data Tracker",
    page_icon="🦠"
)

# --- 2. Title and Introduction ---
st.title("🦠 COVID-19 Global Data Tracker")
st.markdown("""
This interactive dashboard allows you to explore global COVID-19 trends, including cases, deaths, and vaccination progress for selected countries and date ranges.
Data is sourced from [Our World in Data](https://ourworldindata.org/covid-vaccinations).
""")

# --- 3. Data Loading and Initial Preparation (Best Practice: Caching) ---
# @st.cache_data decorator caches the function's output.
# This means the data will only be loaded and processed once across all user sessions,
# speeding up subsequent runs and interactions after the initial load.
# It's specifically for dataframes and other serializable data.
@st.cache_data
def load_and_prepare_data(file_path='cov.csv'):
    """
    Loads the COVID-19 data, performs initial data type conversions,
    and sorts the DataFrame.
    """
    try:
        df_raw = pd.read_csv(file_path)

        # Convert 'date' column to datetime objects
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Sort values for consistent time series analysis
        df_raw = df_raw.sort_values(by=['location', 'date']).reset_index(drop=True)

        return df_raw
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the CSV file is in the same directory as this script.")
        st.stop() # Stop the app execution if data cannot be loaded
    except Exception as e:
        st.error(f"An error occurred during data loading or initial preparation: {e}")
        st.stop() # Stop the app execution on other data errors

df = load_and_prepare_data()

# --- 4. Sidebar for User Inputs (Best Practice: Clear UI for filters) ---
st.sidebar.header("📊 Filter Data")

# Get available countries from the loaded data for the multiselect options
available_countries_for_selection = sorted(df['location'].unique().tolist())

# Define entities to exclude from the main country selection dropdown.
# 'World' is specifically kept out here because we handle its inclusion separately
# to ensure it's always available as a global comparison.
entities_to_exclude_from_dropdown = [
    'High income', 'Upper middle income', 'Lower middle income', 'Low income',
    'European Union', 'Africa', 'Asia', 'Europe', 'North America', 'South America',
    'Oceania', 'World' # Explicitly exclude 'World' from the dropdown options
]

# Create the list of countries actually displayed in the multiselect dropdown
selectable_countries_in_dropdown = [
    country for country in available_countries_for_selection
    if country not in entities_to_exclude_from_dropdown
]

# Set default selection for the multiselect.
# Ensure 'World' is NOT in this default list if it's excluded from selectable_countries_in_dropdown
initial_default_countries = ['United States', 'India', 'Germany', 'Kenya']
default_countries_filtered = [c for c in initial_default_countries if c in selectable_countries_in_dropdown]


# Multiselect widget for country selection
selected_countries = st.sidebar.multiselect(
    "Select Countries for Detailed Analysis:",
    options=selectable_countries_in_dropdown,
    default=default_countries_filtered,
    help="Choose countries to view their specific trends. 'World' is always included for global comparison."
)

# Best Practice: Always include 'World' for global context if it exists in the raw data
# This adds 'World' to the list of `selected_countries` *after* the user's multiselect input,
# regardless of whether they chose it, as long as 'World' data is available.
if 'World' in df['location'].unique() and 'World' not in selected_countries:
    selected_countries.append('World')


# Date Range Selection
min_date_data = df['date'].min()
max_date_data = df['date'].max()

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date_data, max_date_data), # Default to full range
    min_value=min_date_data,
    max_value=max_date_data,
    help=f"Data available from {min_date_data.strftime('%Y-%m-%d')} to {max_date_data.strftime('%Y-%m-%d')}.",
    key='date_selector' # Unique key for the widget
)

# Validate and process date range input
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    if start_date > end_date: # Swap if dates are inverted by user
        start_date, end_date = end_date, start_date
else: # Fallback if only one date is selected (e.g., initial state or incomplete selection)
    start_date = min_date_data
    end_date = max_date_data

# --- 5. Data Filtering and Cleaning (Best Practice: Use .loc for explicit copies/views) ---
# Filter the DataFrame based on user selections
# Using .loc for clear filtering and .copy() to prevent SettingWithCopyWarning
df_filtered_dashboard = df.loc[
    (df['location'].isin(selected_countries)) &
    (df['date'] >= start_date) &
    (df['date'] <= end_date)
].copy() # .copy() ensures we are working on a separate DataFrame

# Define numerical columns to fill with 0 (includes cases, deaths, vaccinations, and hospitalization/ICU)
# This list can be expanded if more numerical columns are relevant
numerical_cols_to_fill = [
    'total_cases', 'new_cases', 'new_cases_smoothed',
    'total_deaths', 'new_deaths', 'new_deaths_smoothed',
    'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
    'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million',
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
    'population', # Essential for per-capita metrics
    'icu_patients', 'icu_patients_per_million', # ICU data
    'hosp_patients', 'hosp_patients_per_million', # Hospitalization data
    'reproduction_rate', 'positive_rate', 'stringency_index' # Other potentially useful metrics
]

# Apply fillna(0) to relevant numerical columns that exist in the filtered DataFrame
for col in numerical_cols_to_fill:
    if col in df_filtered_dashboard.columns: # Check if column exists to prevent KeyError
        # Using .loc with boolean indexing to avoid SettingWithCopyWarning if it's already a view
        df_filtered_dashboard.loc[:, col] = df_filtered_dashboard[col].fillna(0)


# --- 6. Dashboard Layout and Visualizations ---
if df_filtered_dashboard.empty:
    st.warning("No data available for the selected criteria. Please adjust your filters in the sidebar.")
else:
    st.subheader(f"📅 Displaying Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # --- Cases and Deaths Overview ---
    st.header("📈 Case and Death Trends")
    col1, col2 = st.columns(2) # Two columns for side-by-side plots

    with col1:
        st.markdown("#### Total COVID-19 Cases Over Time")
        # Best Practice: Create figure and axes explicitly for Matplotlib plots
        fig_cases, ax_cases = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtered_dashboard, x='date', y='total_cases', hue='location', ax=ax_cases)
        ax_cases.set_xlabel('Date')
        ax_cases.set_ylabel('Total Cases')
        ax_cases.set_title('Total Cases (Cumulative)')
        ax_cases.grid(True, linestyle='--', alpha=0.7)
        ax_cases.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
        plt.xticks(rotation=45) # Rotate x-axis labels for readability
        st.pyplot(fig_cases) # Display Matplotlib figure in Streamlit
        plt.close(fig_cases) # Best practice: Close figures to free up memory

    with col2:
        st.markdown("#### Total COVID-19 Deaths Over Time")
        fig_deaths, ax_deaths = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtered_dashboard, x='date', y='total_deaths', hue='location', ax=ax_deaths)
        ax_deaths.set_xlabel('Date')
        ax_deaths.set_ylabel('Total Deaths')
        ax_deaths.set_title('Total Deaths (Cumulative)')
        ax_deaths.grid(True, linestyle='--', alpha=0.7)
        ax_deaths.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig_deaths)
        plt.close(fig_deaths)

    st.markdown("---") # Separator

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Daily New Cases (7-day smoothed)")
        fig_new_cases, ax_new_cases = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtered_dashboard, x='date', y='new_cases_smoothed', hue='location', ax=ax_new_cases)
        ax_new_cases.set_xlabel('Date')
        ax_new_cases.set_ylabel('Daily New Cases (Smoothed)')
        ax_new_cases.set_title('Daily New Cases (7-Day Smoothed Average)')
        ax_new_cases.grid(True, linestyle='--', alpha=0.7)
        ax_new_cases.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig_new_cases)
        plt.close(fig_new_cases)

    with col4:
        st.markdown("#### Case Fatality Ratio (CFR)")
        # Calculate CFR for the latest available data for each country in the filtered view
        # Using .drop_duplicates(subset='location', keep='last') for robustness
        cfr_data = df_filtered_dashboard.drop_duplicates(subset='location', keep='last').copy()
        
        # Calculate CFR, handling division by zero for total_cases
        cfr_data.loc[:, 'case_fatality_ratio'] = np.where(
            cfr_data['total_cases'] > 0,
            (cfr_data['total_deaths'] / cfr_data['total_cases']) * 100,
            0
        )
        
        # Filter out 'World' if not explicitly needed in CFR table, or keep it
        cfr_result_display = cfr_data[cfr_data['location'] != 'World'][['location', 'total_cases', 'total_deaths', 'case_fatality_ratio']].sort_values(by='case_fatality_ratio', ascending=False).round(2)

        if not cfr_result_display.empty:
            st.dataframe(cfr_result_display, hide_index=True, use_container_width=True)
            st.info("CFR calculated based on the latest available data point for each selected country within the chosen date range.")
        else:
            st.info("CFR data not available for the selected countries/date range.")

    st.markdown("---") # Separator

    # --- Vaccination Progress ---
    st.header("💉 Vaccination Progress")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Total Vaccine Doses Administered Per 100 People")
        fig_total_vacc, ax_total_vacc = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtered_dashboard, x='date', y='total_vaccinations_per_hundred', hue='location', ax=ax_total_vacc)
        ax_total_vacc.set_xlabel('Date')
        ax_total_vacc.set_ylabel('Total Doses Administered Per 100 People')
        ax_total_vacc.set_title('Total Vaccine Doses Administered (Cumulative)')
        ax_total_vacc.grid(True, linestyle='--', alpha=0.7)
        ax_total_vacc.set_ylim(bottom=0) # Ensure y-axis starts from 0
        plt.xticks(rotation=45)
        st.pyplot(fig_total_vacc)
        plt.close(fig_total_vacc)

    with col6:
        st.markdown("#### Share of Population Fully Vaccinated Per 100 People")
        fig_fully_vacc, ax_fully_vacc = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_filtered_dashboard, x='date', y='people_fully_vaccinated_per_hundred', hue='location', ax=ax_fully_vacc)
        ax_fully_vacc.set_xlabel('Date')
        ax_fully_vacc.set_ylabel('People Fully Vaccinated Per 100 People')
        ax_fully_vacc.set_title('Share of Population Fully Vaccinated')
        ax_fully_vacc.grid(True, linestyle='--', alpha=0.7)
        ax_fully_vacc.set_ylim(0, 100) # Percentage from 0-100
        plt.xticks(rotation=45)
        st.pyplot(fig_fully_vacc)
        plt.close(fig_fully_vacc)

    st.markdown("#### Latest Full Vaccination Coverage by Country")
    latest_vaccination_data = df_filtered_dashboard.drop_duplicates(subset='location', keep='last').copy()
    latest_vaccination_data_countries = latest_vaccination_data[latest_vaccination_data['location'] != 'World'].sort_values(by='people_fully_vaccinated_per_hundred', ascending=False)

    if not latest_vaccination_data_countries.empty:
        fig_bar_vacc, ax_bar_vacc = plt.subplots(figsize=(12, max(6, len(latest_vaccination_data_countries) * 0.5))) # Dynamic height
        sns.barplot(data=latest_vaccination_data_countries, x='people_fully_vaccinated_per_hundred', y='location', palette='viridis', ax=ax_bar_vacc)
        ax_bar_vacc.set_xlabel('Percentage of Population Fully Vaccinated')
        ax_bar_vacc.set_ylabel('Country')
        ax_bar_vacc.set_xlim(0, 100)
        ax_bar_vacc.set_title('Latest Full Vaccination Coverage (%)')
        st.pyplot(fig_bar_vacc)
        plt.close(fig_bar_vacc)

        st.dataframe(latest_vaccination_data_countries[['location', 'people_fully_vaccinated_per_hundred']].round(2), hide_index=True, use_container_width=True)
    else:
        st.info("No latest vaccination data available for the selected countries.")

    st.markdown("---") # Separator

    # --- Optional: Hospitalization & ICU Data ---
    st.header("🏥 Hospitalization & ICU Data")
    st.markdown("*(Note: Availability of this data varies by country and reporting period)*")

    # Check if hospitalization/ICU columns are actually present and have significant non-zero data
    has_hosp_data = 'hosp_patients' in df_filtered_dashboard.columns and df_filtered_dashboard['hosp_patients'].sum() > 0
    has_icu_data = 'icu_patients' in df_filtered_dashboard.columns and df_filtered_dashboard['icu_patients'].sum() > 0

    if has_hosp_data or has_icu_data:
        col7, col8 = st.columns(2)

        if has_hosp_data:
            with col7:
                st.markdown("#### Hospital Patients Over Time")
                fig_hosp, ax_hosp = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_filtered_dashboard, x='date', y='hosp_patients', hue='location', ax=ax_hosp)
                ax_hosp.set_xlabel('Date')
                ax_hosp.set_ylabel('Hospital Patients')
                ax_hosp.set_title('COVID-19 Patients in Hospital')
                ax_hosp.grid(True, linestyle='--', alpha=0.7)
                ax_hosp.ticklabel_format(style='plain', axis='y')
                plt.xticks(rotation=45)
                st.pyplot(fig_hosp)
                plt.close(fig_hosp)
        else:
            with col7:
                st.info("Hospitalization data not available for the selected countries or date range.")

        if has_icu_data:
            with col8:
                st.markdown("#### ICU Patients Over Time")
                fig_icu, ax_icu = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=df_filtered_dashboard, x='date', y='icu_patients', hue='location', ax=ax_icu)
                ax_icu.set_xlabel('Date')
                ax_icu.set_ylabel('ICU Patients')
                ax_icu.set_title('COVID-19 Patients in ICU')
                ax_icu.grid(True, linestyle='--', alpha=0.7)
                ax_icu.ticklabel_format(style='plain', axis='y')
                plt.xticks(rotation=45)
                st.pyplot(fig_icu)
                plt.close(fig_icu)
        else:
            with col8:
                st.info("ICU data not available for the selected countries or date range.")
    else:
        st.info("No hospitalization or ICU data available for the selected countries and date range in the dataset.")

    st.markdown("---") # Separator

    # --- Optional: Global Overview Maps (Plotly) ---
    st.header("🌍 Global Overview Maps (Latest Available Data)")
    st.markdown("These interactive maps visualize the **latest available data for all countries** in the dataset, providing a broader global context.")

    # Get the latest data point for ALL countries in the original df for mapping
    # Ensure 'iso_code' is present for Plotly choropleth maps
    df_latest_map = df.dropna(subset=['iso_code']).copy()
    # Use the last available non-zero value for key metrics for a more representative "latest"
    df_latest_map = df_latest_map.sort_values(by='date').groupby('iso_code').last().reset_index()

    # Further filter out non-country entities for mapping, using 'location'
    df_latest_map_countries = df_latest_map[~df_latest_map['location'].isin(entities_to_exclude_from_dropdown)]
    
    # Ensure the columns for mapping exist and have some data
    has_cases_per_million = 'total_cases_per_million' in df_latest_map_countries.columns and df_latest_map_countries['total_cases_per_million'].sum() > 0
    has_fully_vaccinated_per_hundred = 'people_fully_vaccinated_per_hundred' in df_latest_map_countries.columns and df_latest_map_countries['people_fully_vaccinated_per_hundred'].sum() > 0

    if not df_latest_map_countries.empty and (has_cases_per_million or has_fully_vaccinated_per_hundred):
        if has_cases_per_million:
            # Map 1: Total Cases per Million (Latest)
            fig_cases_map = px.choropleth(df_latest_map_countries,
                                          locations="iso_code",
                                          color="total_cases_per_million",
                                          hover_name="location",
                                          color_continuous_scale=px.colors.sequential.Plasma,
                                          title='Total COVID-19 Cases Per Million People (Latest Data)',
                                          projection="natural earth",
                                          hover_data={'total_cases_per_million': ':.2f'}) # Format hover data
            st.plotly_chart(fig_cases_map, use_container_width=True)

        if has_fully_vaccinated_per_hundred:
            # Map 2: Fully Vaccinated Per Hundred (Latest)
            fig_vacc_map = px.choropleth(df_latest_map_countries,
                                         locations="iso_code",
                                         color="people_fully_vaccinated_per_hundred",
                                         hover_name="location",
                                         color_continuous_scale=px.colors.sequential.Viridis,
                                         title='Share of Population Fully Vaccinated Per 100 People (Latest Data)',
                                         projection="natural earth",
                                         hover_data={'people_fully_vaccinated_per_hundred': ':.2f'})
            st.plotly_chart(fig_vacc_map, use_container_width=True)
    else:
        st.warning("Could not generate global maps. Ensure 'iso_code' is present and relevant data exists in the raw dataset.")

    st.markdown("---") # Separator

    # --- Final Deliverable Notes ---
    st.header("✅ Project Deliverables & Objectives Achieved:")
    st.markdown("""
    This Streamlit dashboard serves as an interactive data report that:

    * **Loads, cleans, analyzes, and visualizes COVID-19 data.**
    * **Communicates insights** with clear, interactive visuals based on user selections.
    * Is **user-friendly, well-structured, and reproducible.**

    **Key Objectives Achieved:**
    * ✅ Collected global COVID-19 data from Our World in Data.
    * ✅ Loaded and explored the dataset using pandas.
    * ✅ Cleaned and prepared the data by handling missing values and filtering relevant countries/dates.
    * ✅ Performed exploratory data analysis (EDA) to identify trends in cases, deaths, and vaccinations.
    * ✅ Created interactive visualizations (line charts, bar charts, choropleth maps) to illustrate key metrics.
    * ✅ Calculated critical indicators such as death rates and vaccination coverage.
    * ✅ Presented insights through a user-friendly, interactive dashboard.
    * ⭐ **Implemented user input** for country and date range selection.
    * ⭐ **Built an interactive dashboard** using Streamlit.
    * ⭐ **Included hospitalization and ICU data** if available in the dataset.
    """)