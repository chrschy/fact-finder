# Fact Finder

## Getting Started

Install Dependencies:

```
pip install -e .
```

Run UI:

```
streamlit run src/fact_finder/app.py --browser.serverAddress localhost
```

Running with additional arguments (e.g. activating the normalized graph synonyms):

```
streamlit run src/fact_finder/app.py --browser.serverAddress localhost -- [args]
streamlit run src/fact_finder/app.py --browser.serverAddress localhost -- --normalized_graph
```
