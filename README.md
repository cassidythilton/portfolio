# Portfolio — Machine Learning & Software Engineering

A curated collection of end-to-end machine learning projects demonstrating applied AI in financial services and real estate. Each project spans the full development lifecycle—from problem framing and model architecture through deployment infrastructure and production monitoring.

---

## Table of Contents

| # | Project | Domain | Key Technologies |
|---|---------|--------|-----------------|
| 1 | [po001_mrm — Automated Model Risk Management Reports](#po001_mrm--automated-model-risk-management-reports) | Financial Services | GPT-4, Python, MLOps, HTML |
| 2 | [po002_hou — Home Price Prediction Engine](#po002_hou--home-price-prediction-engine) | Real Estate | CNN, MLP, FastAPI, Streamlit, MLflow |

---

## po001_mrm — Automated Model Risk Management Reports

### Overview

An automated report generation pipeline for Model Risk Management (MRM) in financial services. The system ingests an MRM template document, parses each section's analytical requirements using GPT-4 function calling, executes the appropriate data retrieval and computation tasks against an MLOps platform, and assembles a fully rendered HTML report—reducing what was previously a multi-FTE manual process to a single notebook execution.

### Problem

MRM reports are a regulatory obligation under [SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) (Board of Governors of the Federal Reserve System / OCC). They require cross-referencing model performance metrics, feature drift statistics, segment-level analyses, and training metadata across quarters. Producing them manually is labor-intensive, error-prone, and difficult to standardize across teams.

### Solution Architecture

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1P-eHM2_z1wHfrq-WQEb3Z9gTyW2h52c6">
</div>

1. **Template Parsing** — The HTML-based MRM template contains annotated sections with natural-language requests (e.g., *"analyze model performance metrics for Q1 2023 vs. Q4 2022"*).
2. **Task Routing** — GPT-4 with function calling identifies the correct analytical task (performance retrieval, drift detection, segment analysis, etc.) for each request.
3. **Data Retrieval** — The system queries the MLOps platform APIs for model performance history, feature drift metrics, segment policies, and training metadata (via MLflow / Databricks).
4. **Summarization & Rendering** — Metrics are aggregated, contextualized by GPT-4 against SR 11-7 guidelines, and inserted into the final HTML report with embedded visualizations.

### Key Files

| File | Purpose |
|------|---------|
| `AutoRiskReport_FinancialServices.ipynb` | Primary notebook — user-facing entry point |
| `helperMRM.py` | Backend module — API integrations, metrics computation, report generation |
| `src/vars.yaml` | Configuration — model, deployment, and feature column definitions |

### Technologies

- **GPT-4** — Task classification via function calling; narrative summarization of quantitative metrics
- **Python** — Core language (pandas, NumPy, scikit-learn, plotly, pyecharts)
- **MLflow / Databricks** — Model registry, training metrics, artifact management
- **HTML/CSS** — Report template and rendered output format

---

## po002_hou — Home Price Prediction Engine

### Overview

A multi-modal home price prediction system that combines property imagery with structured numerical features (square footage, bedrooms, bathrooms, location) using a dual-branch neural network. The system is served through a Streamlit front end backed by a FastAPI inference API, with full experiment tracking via MLflow.

### Problem

Home valuation is traditionally subjective and time-consuming, relying heavily on manual appraisals. An automated, data-driven approach that incorporates both visual and tabular property characteristics can provide faster, more consistent, and less biased estimates.

### Solution Architecture

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1PMdTgx-37RRg6Kp-cF5EkInkurxPbMKX">
</div>

1. **Image Branch (CNN)** — A 3-layer convolutional network (16 → 32 → 64 filters) processes 64×64 property images to extract visual features.
2. **Numerical Branch (MLP)** — A 2-layer dense network processes normalized tabular features (city code, beds, baths, sqft).
3. **Fusion Layer** — Both branch outputs are concatenated and passed through a final dense layer to produce the price estimate.
4. **Comparable Homes** — The system identifies the 4 nearest properties by Euclidean distance in feature space, providing context alongside the prediction.

### Key Files

| File | Purpose |
|------|---------|
| `House Price Predictions.ipynb` | Model training notebook — CNN/MLP architecture, data preprocessing, MLflow logging |
| `app.py` | FastAPI inference API — serves model predictions |
| `str.py` | Streamlit application — user interface with image upload and comparable homes |
| `helper.py` | Utility functions — data loading, model architecture definitions, visualization |
| `src/homePrices.csv` | Training dataset — property attributes and prices |

### FastAPI

FastAPI serves as the inference backbone, handling prediction requests asynchronously. The Streamlit front end sends property data and images to a FastAPI endpoint, which invokes the dual-branch neural network and returns the price estimate. FastAPI provides automatic OpenAPI documentation, request validation, and concurrent request handling.

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1HUa6tvprIA1NXmuNfCGP4v1e9WCIRQmZ" width="1000">
</div>

### MLflow

MLflow tracks all training experiments—logging parameters (epochs, batch size), metrics (loss, val_loss), and model artifacts. This enables comparison across training runs and straightforward rollback to prior model versions.

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1KhJY0O5iFozWcyB0GAMYReffkohZyJBn" width="1000">
</div>

### Streamlit

Streamlit provides the user-facing interface for property valuation. Users input property details and upload an image; the app routes inputs to FastAPI, displays the predicted price, and surfaces four comparable homes for additional context.

<div align="center">
  <a href="https://drive.google.com/file/d/10pAADAGK6zB5AoUJNpA6uW9xLwWK6xdL/view?t=3s"><img src="https://drive.google.com/uc?export=view&id=1QaASEvndCqhkSg1O0qaSh7s_tnfiWs7Y" width="700"></a>
</div>

### Technologies

- **TensorFlow / Keras** — CNN and MLP model architectures
- **FastAPI** — High-performance async inference API
- **Streamlit** — Interactive front-end for property valuation
- **MLflow** — Experiment tracking, model versioning, artifact logging
- **OpenCV** — Image preprocessing and resizing

---

## Setup & Configuration

### Prerequisites

- Python 3.9+
- See individual project directories for specific dependency requirements

### Configuration

- **po001_mrm**: Update `src/vars.yaml` with your platform credentials, model names, and connection details before running.
- **po002_hou**: Ensure the training data (`src/homePrices.csv`) and trained model (`src/housePrices.h5`) are present in the `po002_hou/src/` directory.

---

## License

This repository is provided for portfolio and demonstration purposes.
