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
streamlit run src/fact_finder/app.py --browser.serverAddress localhost -- --normalized_graph --use_entity_detection_preprocessing
```

The following flags are available:
```
--normalized_graph  =  Apply synonym replacement based on the normalized graph to the cypher queries before applying them to the graph. This requires the corresponding api key ($SYNONYM_API_KEY) to be set.
--use_entity_detection_preprocessing  =  Apply entity detection to the user question before generating the cypher query. The found entities will be replaced by their preferred terms and a string describing their category (e.g. "Psoriasis is a disease.") will be added to the query. This requires the corresponding api key ($SYNONYM_API_KEY) to be set. Also, the normalized graph should be used.
--use_subgraph_expansion  =  The evidence graph gets expanded through the surrounding neighborhoods.
```

## Process description

The following steps are undertaken to get from the user question to the natural language answer and the provided evidence:

1. In the first step, a language model call is used to generate a cypher query to the knowledge graph. To achieve this, the prompt template contains the schema of the graph, i.e. information about all nodes and their properties.
Additionally, the prompt template can be enriched with natural language descriptions for (some of) the relations in the graph allowing better understanding of their meaning for the language model.
In case the model decides that the user question cannot be answered by a graph with the given schema, the model is instructed to return an error message starting with the marker string "SCHEMA_ERROR". This is then detected and the error message is directly forwarded to the user.

2. In the second step, the generated cypher query is preprocessed using regular expressions.
    - First, a formatting is applied in order to make subsequent regular expressions easier to design. This includes for example removal of unnecessary whitespaces and using double quotes for all strings.
    - Next, all property values are turned to lower case. This assumes that a similar preprocessing has been done for the property values in the graph and makes the query resistant to capitalization mismatches.
    - Finally, for some node types, any names used in the query, are replaced with a synonym that is actually used in the graph. This is (for example) done by looking up synonyms for the name and checking which one actually exists in the graph.

3. In the third step, the graph is queried with the final result of the cypher preprocessing. The graph answer together with the cypher query are part of the evidence presented in the interface, allowing transparency for the user.

4. With another language model call, the final natural language answer is generated from the result of querying the graph.

5. Additionally, a subgraph is generated from the graph query and result. This serves as visual evidence for the user. The subgraph can either be generated via a rule based approach or also with help of the language model.
