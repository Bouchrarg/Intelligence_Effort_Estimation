# Mining GitHub Repositories for Intelligent Effort Estimation

An end to end machine learning pipeline that estimates software development effort from GitHub repository data. Features are scraped directly from GitHub, then fed through an ensemble model to predict effort, reaching an R squared of 0.893 on the held out test set.

## What it does

- Scrapes repository level features from GitHub (`ScrappingEnsam/`)
- Builds and evaluates an ensemble regression pipeline for effort estimation (`notebooks/`)
- Serves the trained model through a small Streamlit app (`app.py`), where a GitHub repository URL can be pasted in to get an effort estimate

## Tech stack

Python, pandas, scikit learn, Streamlit, the GitHub API.

## Running locally

```
pip install streamlit pandas scikit-learn joblib requests
streamlit run app.py
```

A GitHub token is needed for the scraper to query the GitHub API without hitting rate limits, set as `GH_TOKEN` in a local `.env` file.
