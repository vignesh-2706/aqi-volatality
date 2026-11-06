# AQI Volatility and Spike Prediction

A time-series analytics and machine learning project for identifying AQI volatility patterns and detecting potential air-quality spike events across Indian cities.

The project combines historical city-level AQI data, time-series feature engineering, volatility analysis, and a Random Forest classifier. The notebook in this repository is a small-scale Pandas implementation of the workflow. For larger datasets, the same ETL/ELT and feature-engineering stages can be implemented on Databricks using PySpark and SQL.

## Project Overview

Air-quality datasets contain observations from multiple cities over time. Instead of only predicting the AQI value, this project focuses on two related questions:

1. **Which cities exhibit higher or lower AQI volatility?**
2. **Can an upcoming AQI spike be detected from recent historical AQI behavior?**

The pipeline therefore transforms raw city-level AQI records into a structured time-series dataset, calculates rolling volatility and lag-based features, defines spike events, and trains a classification model to identify those events.

### Key Results

- Historical dataset processed at approximately **239K city-level AQI records** in the notebook implementation.
- **7-day rolling mean and standard deviation** used to characterize local AQI behavior and volatility.
- **Lag-1, lag-2, and lag-3 AQI features** used to capture short-term temporal dependencies.
- AQI spike events defined using a threshold above the city's recent 7-day rolling mean.
- Random Forest classifier used for spike-event classification.
- Final evaluation on the selected high- and low-volatility city subset achieved approximately **77% accuracy** and **75% recall for spike events**.

> **Scale note:** The GitHub repository contains the lightweight/local Pandas implementation so that the workflow and analysis can be reproduced without a distributed environment. For larger historical datasets, the same transformations are designed to be executed using Databricks, PySpark, and SQL.

---

## Architecture

```text
                Raw AQI CSV Files
                       |
                       v
              Data Ingestion / ETL
                       |
             +---------+---------+
             |                   |
             v                   v
       Data Cleaning        Data Validation
             |                   |
             +---------+---------+
                       |
                       v
             Time-Series Preparation
                       |
             +---------+---------+
             |                   |
             v                   v
       Lag Features       Rolling Statistics
       lag1 / lag2 /      7-day mean / std
          lag3                  |
             |                   |
             +---------+---------+
                       |
                       v
                Spike Definition
                       |
                       v
             Volatility Analysis
                       |
              High / Low Volatility
                   Cities
                       |
                       v
              Feature Preparation
                       |
                       v
             Random Forest Classifier
                       |
                       v
          AQI Spike Event Prediction
                       |
                       v
          Accuracy / Recall Analysis
```

---

## Dataset

The repository contains state-wise CSV files containing city-level AQI observations.

The notebook combines the available CSV files into a single dataset and standardizes the important columns into:

| Column | Description |
|---|---|
| `date` | Date of AQI observation |
| `city` | City associated with the observation |
| `status` | AQI category/status |
| `aqi` | Numeric AQI value |

The notebook implementation produces approximately **239,079 rows** after combining and cleaning the available data.

The data is then sorted by city and date so that time-series features are calculated independently for each city.

---

## Data Pipeline

### 1. Data Ingestion

The local implementation reads multiple CSV files using Pandas and combines them into a single DataFrame.

```python
files = glob.glob("*.csv")

df_list = []

for file in files:
    temp = pd.read_csv(file)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
```

For a larger implementation, this ingestion stage can be moved to Databricks using PySpark and distributed storage.

### 2. Data Cleaning

The raw dataset contains additional columns that are not required for the analysis. The pipeline:

- Removes unused columns.
- Standardizes column names.
- Converts dates into datetime format.
- Removes invalid/missing dates.
- Sorts records by city and date.
- Converts AQI values to numeric values.
- Removes rows that cannot support the required time-series features.

Example:

```python
df.columns = ['date', 'city', 'status', 'aqi']

df['date'] = pd.to_datetime(
    df['date'],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['date'])

df = df.sort_values(
    by=['city', 'date']
)

df['aqi'] = df['aqi'].astype(int)
```

---

## Feature Engineering

The project uses historical AQI behavior rather than treating each observation independently.

### Lag Features

Three short-term lag features are created independently for every city:

- `lag1` — previous AQI observation
- `lag2` — AQI two observations earlier
- `lag3` — AQI three observations earlier

```python
df['lag1'] = df.groupby('city')['aqi'].shift(1)
df['lag2'] = df.groupby('city')['aqi'].shift(2)
df['lag3'] = df.groupby('city')['aqi'].shift(3)
```

These features allow the classifier to learn short-term changes in AQI.

### Rolling Statistics

A 7-observation rolling window is calculated for every city.

```python
df['rolling_mean'] = (
    df.groupby('city')['aqi']
      .transform(lambda x: x.rolling(7).mean())
)

df['rolling_std'] = (
    df.groupby('city')['aqi']
      .transform(lambda x: x.rolling(7).std())
)
```

The rolling mean represents the recent AQI baseline, while the rolling standard deviation provides a measure of short-term AQI volatility.

---

## AQI Spike Definition

A spike event is defined when the current AQI exceeds the recent rolling baseline by a configurable threshold.

The current implementation uses:

```python
threshold = 20

df['spike'] = (
    df['aqi'] > (df['rolling_mean'] + threshold)
).astype(int)
```

Therefore:

```text
AQI > 7-day rolling mean + 20
                    |
                    v
              Spike = 1

Otherwise
                    |
                    v
              Spike = 0
```

The threshold can be tuned depending on the objective of the analysis.

---

## Volatility Analysis

The project's volatility analysis uses the mean 7-day rolling standard deviation for each city.

```python
volatility = (
    df.groupby('city')['rolling_std']
      .mean()
)
```

Cities are then ranked based on their average rolling volatility.

The analysis selects:

- Top 5 high-volatility cities
- Bottom 5 low-volatility cities

This provides a focused comparison between locations with substantially different AQI behavior.

The selected cities in the notebook include examples such as:

**High volatility**
- Mandi Gobindgarh
- Byrnihat
- Sri Ganganagar
- Dharuhera
- Begusarai

**Low volatility**
- Bagalkot
- Chamarajanagar
- Tirunelveli
- Kannur
- Kozhikode

The exact selected cities depend on the processed dataset and volatility calculation.

---

## Spike Rate Analysis

For each selected city, the pipeline calculates the proportion of observations classified as spike events.

This helps compare the relationship between volatility and the frequency of AQI spikes.

For example, the notebook evaluates:

```python
spike_rate = city_data['spike'].mean()
```

The project also visualizes the relationship between:

```text
Average Rolling Volatility
            vs.
       Spike Rate
```

This analysis provides an additional view of whether cities with greater AQI variability also experience more frequent spike events.

---

## Machine Learning

### Model

The project uses a **Random Forest Classifier**.

The feature set is:

```python
features = [
    'lag1',
    'lag2',
    'lag3',
    'rolling_mean',
    'rolling_std'
]
```

The classifier is configured with 100 trees:

```python
model = RandomForestClassifier(
    n_estimators=100
)
```

Random Forest was selected because it can capture non-linear relationships between recent AQI behavior, rolling statistics, and the likelihood of a spike event.

---

## Time-Series Train/Test Strategy

Because this is a time-series problem, the project avoids randomly shuffling observations when evaluating the main workflow.

For the selected cities, observations are sorted chronologically and each city's data is divided into:

- **80% training data**
- **20% testing data**

```python
split_idx = int(len(city_df) * 0.8)

train_list.append(
    city_df.iloc[:split_idx]
)

test_list.append(
    city_df.iloc[split_idx:]
)
```

The resulting training and testing datasets are then combined across the selected cities.

This preserves the chronological nature of the problem and makes the evaluation more representative of predicting future spike events from historical observations.

---

## Model Evaluation

The final selected-city experiment produced:

| Metric | Result |
|---|---:|
| Accuracy | **77%** |
| Spike-event Recall | **75%** |
| Spike-event Precision | 39% |
| Spike-event F1-score | 51% |

The spike-event class is the positive class (`1`).

The recall of approximately **75%** indicates that the model identified a substantial proportion of the actual spike events in the evaluation set.

Because spike events are less frequent than normal AQI observations in the selected dataset, accuracy alone is not sufficient. Recall is therefore an important metric for evaluating the usefulness of the model as a spike-detection system.

---

## Why Recall Matters

For an AQI spike detection system, a false negative can be more important than a false positive.

```text
Actual Spike
     |
     +----> Model detects spike
     |          |
     |          +----> Useful early warning
     |
     +----> Model misses spike
                |
                +----> Potentially important event missed
```

Therefore, the project emphasizes **spike-event recall** alongside overall accuracy.

---

## High- vs Low-Volatility Analysis

The project compares cities with different volatility characteristics.

The analysis shows that high-volatility cities can have substantially higher spike rates than low-volatility cities, while low-volatility cities may contain very few spike events.

This difference is important for modeling because the same classification threshold can behave differently across cities with very different AQI distributions.

The project therefore treats volatility as an analytical dimension rather than simply predicting a single AQI value for every city.

---

## Scaling the Pipeline with Databricks and PySpark

The repository contains a Pandas-based implementation because it is easier to reproduce locally. The same workflow can be scaled to significantly larger datasets using **Databricks, PySpark, and SQL**.

### Local / Prototype

```text
CSV
 |
 v
Pandas
 |
 v
Feature Engineering
 |
 v
Scikit-learn
```

### Scalable Implementation

```text
Raw AQI Data
     |
     v
Databricks
     |
     v
PySpark DataFrames
     |
     +----------------------+
     |                      |
     v                      v
Data Cleaning          Data Validation
     |                      |
     +----------+-----------+
                |
                v
       Distributed Feature
          Engineering
                |
       +--------+--------+
       |                 |
       v                 v
   Lag Windows      Rolling Windows
       |                 |
       +--------+--------+
                |
                v
       Feature Dataset
                |
                v
       ML Training / Scoring
                |
                v
       Spike Predictions
```

PySpark window functions can be used to reproduce the lag and rolling operations while distributing computation across a Databricks cluster.

For example, the conceptual PySpark transformation is:

```python
from pyspark.sql.window import Window
from pyspark.sql import functions as F

window = (
    Window
    .partitionBy("city")
    .orderBy("date")
)

df = (
    df
    .withColumn("lag1", F.lag("aqi", 1).over(window))
    .withColumn("lag2", F.lag("aqi", 2).over(window))
    .withColumn("lag3", F.lag("aqi", 3).over(window))
)
```

A rolling window can similarly be used to calculate the recent AQI mean and standard deviation.

This allows the pipeline to move from a local single-machine workflow to a distributed ETL/ELT architecture suitable for larger historical datasets.

---

## ETL / ELT Workflow

The overall data engineering workflow can be summarized as:

### Extract
- Ingest state-wise/city-wise AQI datasets.
- Load raw records into the processing environment.

### Transform
- Standardize schema.
- Parse dates.
- Remove invalid records.
- Sort/partition data by city and date.
- Generate lag features.
- Generate rolling statistics.
- Calculate volatility.
- Define AQI spike labels.

### Validate
- Check missing dates and AQI values.
- Validate data types.
- Verify city-level chronological ordering.
- Validate rolling-feature availability.
- Remove records that cannot support the required features.

### Load
- Store the structured feature dataset for downstream analytics and machine learning.

In a Databricks implementation, these stages can be organized as reusable ETL/ELT jobs with PySpark and SQL.

---

## SQL Analytics

SQL can also be used for downstream aggregation and validation.

Example:

```sql
SELECT
    city,
    AVG(aqi) AS avg_aqi,
    STDDEV(aqi) AS aqi_stddev
FROM aqi_data
GROUP BY city
ORDER BY aqi_stddev DESC;
```

This makes it possible to identify cities with higher overall AQI variability before applying the machine learning workflow.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core analysis and ML workflow |
| Pandas | Local data processing and prototyping |
| NumPy | Numerical operations |
| SQL | Data transformation and analytical queries |
| PySpark | Distributed data processing at scale |
| Databricks | Scalable ETL/ELT and analytics environment |
| Scikit-learn | Random Forest classification |
| Matplotlib | Data visualization |

---

## Project Structure

```text
aqi-volatality/
│
├── aqi-volatality.ipynb
│
├── *_state_cities_aqi_data.csv
│
├── state_names.csv
│
└── README.md
```

The repository intentionally keeps the implementation lightweight enough to run locally. The larger-scale architecture is represented through the Databricks/PySpark scaling approach described above.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vignesh-2706/aqi-volatality.git
cd aqi-volatality
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
aqi-volatality.ipynb
```

### 4. Run the notebook

Make sure the CSV files are available in the same working directory as the notebook.

The notebook will:

1. Load the CSV files.
2. Combine the datasets.
3. Clean and standardize the data.
4. Generate lag features.
5. Generate rolling statistics.
6. Define AQI spike events.
7. Calculate city-level volatility.
8. Select high- and low-volatility cities.
9. Train the Random Forest classifier.
10. Evaluate spike detection performance.
11. Analyze spike rates and city-level performance.

---

## Key Takeaways

This project demonstrates an end-to-end workflow combining data engineering, time-series feature engineering, analytics, and machine learning.

The main insight is that **AQI volatility and spike frequency vary significantly across cities**, making city-level historical behavior useful for detecting abnormal AQI events.

The workflow can be implemented locally with Pandas for experimentation and scaled using **Databricks + PySpark + SQL** when working with substantially larger datasets.

---

## Future Improvements

Potential extensions include:

- Add weather variables such as temperature, humidity, wind speed, and rainfall.
- Add pollutant-level features such as PM2.5, PM10, NO2, SO2, CO, and O3.
- Tune the spike threshold dynamically by city.
- Address class imbalance using class weighting or resampling.
- Compare Random Forest with XGBoost, LightGBM, and other classifiers.
- Use MLflow for experiment tracking and model management.
- Build a production Databricks pipeline with scheduled jobs.
- Add automated data-quality checks.
- Implement model monitoring and drift detection.
- Create an AQI spike early-warning dashboard.

---

## Author

**Vignesh**

Computer Science / Cloud & Data Engineering

GitHub: https://github.com/vignesh-2706
