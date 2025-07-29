# covidtrakerapp

An interactive Streamlit dashboard for global COVID-19 data analysis.
# COVID-19 Global Data Tracker: Interactive Dashboard

## Project Overview

This project delivers a dynamic and interactive **COVID-19 Global Data Tracker** dashboard, built with **Streamlit** and powered by Python's robust data science ecosystem. The application provides a user-friendly interface for exploring critical epidemiological and vaccination trends worldwide, leveraging up-to-date data from Our World in Data.

The primary goal was to transform complex pandemic figures into clear, actionable, and customizable visualizations, enabling users to analyze trends across selected countries and specific date ranges.

## Objectives of the Project

* **Data Acquisition & Cleaning:** To source, load, clean, and preprocess a large-scale COVID-19 dataset, handling missing values and ensuring data quality for analysis.
* **Exploratory Data Analysis (EDA):** To identify key trends in total cases, deaths, new cases, vaccination rates, and hospitalization data across various geographical locations.
* **Interactive Dashboard Development:** To build an intuitive and dynamic web application using Streamlit that allows users to filter data by country and date range.
* **Comprehensive Visualization:** To create informative and engaging visualizations (line charts, bar charts, choropleth maps) using Matplotlib, Seaborn, and Plotly Express.
* **Insights Communication:** To present clear and concise insights derived from the data analysis within both a Jupyter Notebook report and an interactive dashboard.
* **Reproducibility & Deployment:** To ensure the project is well-documented, reproducible, and easily deployable for wider access.

## Tools and Libraries Used

* **Programming Language:** Python
* **Interactive Dashboard Framework:** Streamlit
* **Data Manipulation & Analysis:** Pandas
* **Static Data Visualization:** Matplotlib, Seaborn
* **Interactive Data Visualization (Maps):** Plotly Express
* **Version Control:** Git & GitHub

## How to Run/View the Project

### Live Dashboard

You can access the live interactive dashboard deployed on Streamlit Community Cloud (or your chosen platform) here:
* [**Live Demo: COVID-19 Global Data Tracker**](https://covidtrakerapp-hxewrbvwxxmakvv6jfrezq.streamlit.app/)
  

### Running Locally

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Degefa-Gomora/covidtrakerapp
    cd your-repository-name
    ```
    

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run dashboard_app.py
    ```
    This will open the dashboard in your web browser, usually at `http://localhost:8501`.

5.  **Explore the Jupyter Notebook Report:**
    You can also open and run the `covid.ipynb` Jupyter Notebook to see the step-by-step data cleaning, analysis, and visualization process.
    ```bash
    jupyter notebook covid.ipynb
    ```
    *(Ensure you have Jupyter installed: `pip install jupyter`)*

## Insights and Reflections

This project provided deep insights into the dynamics of the COVID-19 pandemic, including:

* **Varied Pandemic Trajectories:** Different countries experienced distinct waves of cases and deaths, influenced by factors like public health policies, population density, and vaccination rates.
* **Vaccination Impact:** The dashboard clearly illustrates the correlation between increasing vaccination coverage and a potential flattening of case/death curves in many regions.
* **Data Availability Challenges:** The project highlighted the real-world challenge of inconsistent data reporting, especially for metrics like hospitalization and ICU admissions across different countries and time periods.
* **Power of Interactivity:** Building an interactive dashboard significantly enhances data exploration, allowing users to pose their own questions and visualize answers without needing to write code.
* **Streamlit's Efficiency:** Streamlit proved to be an incredibly efficient tool for rapidly developing and deploying data applications, bridging the gap between data analysis and user-friendly interfaces.

This project significantly strengthened my skills in end-to-end data science project execution, from raw data to deployed interactive application.

---
