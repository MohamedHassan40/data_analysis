from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes

app = Flask(__name__)

# Datasets Dictionary (Built-in or Fetchable)
DATASETS = {
    "Iris": load_iris(as_frame=True).frame,
    "Wine": load_wine(as_frame=True).frame,
    "Titanic": sns.load_dataset("titanic"),  # Load Titanic dataset from seaborn
    "Diabetes": load_diabetes(as_frame=True).frame,
    "Penguins": sns.load_dataset("penguins"),
    "Boston Housing": fetch_openml(name="Boston", as_frame=True).frame,



}

# Preprocess the data (handle missing and non-numeric values)
def preprocess_data(data):
    numeric_data = data.select_dtypes(include=["number"])
    numeric_data = numeric_data.fillna(0)  # Fill missing values with 0
    return numeric_data

# Calculate the 5 Vs of data
def calculate_5vs(data):
    volume = f"{data.shape[0]} rows x {data.shape[1]} columns"
    velocity = "Processing speed: 500 rows/sec (simulated)"
    numeric_cols = len(data.select_dtypes(include=["number"]).columns)
    categorical_cols = len(data.select_dtypes(include=["object"]).columns)
    variety = f"Numeric: {numeric_cols}, Categorical: {categorical_cols}"
    missing_values = data.isnull().sum().sum()
    veracity = f"Missing Values: {missing_values} ({missing_values / data.size:.2%})"
    value = "Insights generated during analysis."
    return {"Volume": volume, "Velocity": velocity, "Variety": variety, "Veracity": veracity, "Value": value}

# Generate basic graphs
def generate_basic_graphs(data):
    graphs = []
    for column in data.columns[:3]:  # Limit to the first 3 numeric columns
        fig = px.histogram(data, x=column, title=f"Distribution of {column}")
        graphs.append(fig.to_html(full_html=False))
    return graphs

# Generate advanced analytics and visualizations
def generate_advanced_analytics(data):
    analytics = []

    # 1. Correlation Matrix
    correlation = data.corr()
    fig = px.imshow(correlation, text_auto=True, title="Correlation Matrix")
    analytics.append(fig.to_html(full_html=False))

    # 2. K-Means Clustering
    if data.shape[1] > 1:  # Ensure sufficient columns for clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        clusters = kmeans.fit_predict(data)
        data['Cluster'] = clusters
        fig = px.scatter_matrix(data, dimensions=data.columns[:-1], color='Cluster', title="K-Means Clustering")
        analytics.append(fig.to_html(full_html=False))

    # 3. Linear Regression (if 2+ columns available)
    if data.shape[1] > 1:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        fig = px.scatter(x=y, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title="Regression Analysis")
        analytics.append(fig.to_html(full_html=False))

    # 4. Time-Series Forecasting (if a date-like column exists)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        fig = px.line(data, x='date', y=data.columns[1], title="Time-Series Forecasting")
        analytics.append(fig.to_html(full_html=False))

    return analytics

@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS)

@app.route("/analyze", methods=["POST"])
def analyze():
    dataset_name = request.form["dataset"]

    # Load the dataset from memory
    data = DATASETS[dataset_name]
    data_cleaned = preprocess_data(data)
    five_vs = calculate_5vs(data)

    # Generate basic and advanced analytics
    basic_graphs = generate_basic_graphs(data_cleaned)
    advanced_analytics = generate_advanced_analytics(data_cleaned)

    return render_template(
        "results.html",
        dataset_name=dataset_name,
        five_vs=five_vs,
        basic_graphs=basic_graphs,
        advanced_analytics=advanced_analytics,
    )

if __name__ == "__main__":
    from gunicorn.app.wsgiapp import run
    run()

