# Neo4j Server with PrimeKG

This docker image downloads the PrimeKG data, imports it into a Neo4j database and runs the Neo4j service.

## Setup

Build docker image:
```
docker build --pull --rm -f "neo4j_primekg/Dockerfile" -t neo4j_primekg:latest "neo4j_primekg"
```

Start up server:
```
docker run -d --restart=always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/opensesame \
    --env NEO4J_server_databases_default__to__read__only=true \
    --env NEO4J_apoc_export_file_enabled=true \
    --env NEO4J_apoc_import_file_use__neo4j__config=true \
    --env NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    --name neo4j_primekg_service \
    neo4j_primekg:latest
```

Test via Cypher shell:
```
docker exec -it <CONTAINER_ID> cypher-shell -u neo4j -p opensesame
```
```
MATCH (disease1:disease {name: "psoriasis"})-[:parent_child]->(disease2:disease {name: "scalp disease"})
RETURN disease1, disease2;
```

Alternative start-up using an already downloaded files:
```
docker run -d --restart=always \
    --publish=7474:7474 --publish=7687:7687 \
    --volume <abs/path/to/prime/kg.csv>:/primekg_data/kg.csv:ro \
    --volume <abs/path/to/drug/features.tab>:/primekg_data/drug_features.tab:ro \
    --volume <abs/path/to/disease/features.tab>:/primekg_data/disease_features.tab:ro \
    --env NEO4J_AUTH=neo4j/opensesame \
    --env NEO4J_server_databases_default__to__read__only=true \
    --env NEO4J_apoc_export_file_enabled=true \
    --env NEO4J_apoc_import_file_enabled=true \
    --env NEO4J_apoc_import_file_use__neo4j__config=true \
    --env NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    --name neo4j_primekg_service \
    neo4j_primekg:latest
```

## Import and cleaning process for PrimeKG data

These are the steps taken when starting a new Docker container for the image build from the given Dockerfile. These steps are executed by the scripts "import_primekg.sh" and "primekg_to_neo4j_csv.py".

First, the data is downloaded from Harvard Dataverse unless it was linked into the container via volume. The data in this case is the kg.csv containing the actual graph (in form of all its edges) and the additional features for drug and disease nodes (drug_features.tab and disease_features.tab).

In the next step, the data gets loaded via pandas and some general clean up steps are performed:
- All string entries are converted to lower case in order to make queries to the graph more robust.
- In the columns that encode node or relation types ("relation", "display_relation", "x_type", "y_type") spaces and - get replaced with _ since these symbols do not work in Cypher queries.
- Similarly, / gets replaced by _ or _ because this symbol may cause problems in Cypher queries.
- For the drug and disease data, the following replacements are executed:
  - "\r" -> ""
  - "\n" -> " "
  - ",," -> ","
  - '"' -> ""

In the third step, the nodes are extracted from the graph data. Nodes get separated by type. For each node in each type the properties index, id, name and source are extracted.
In this step, the additional features for the drug and disease nodes also get extracted. For the drug nodes, these are certain properties, like the aggregate state or the molecular weight.
For the disease node, several textual descriptions get added to the graph. This includes a description of the disease, its symptoms or when to see a doctor. Note that for the disease description up to four possible candidates are available and they get prioritized as follows:
  1. orphanet_clinical_description
  2. mondo_definition
  3. umls_description
  4. orphanet_definition

In the fourth step, the relation data from kg.csv gets extracted. They can either be extracted based on the display_relation column (default) or the relation column in the data. No properties are added.

Subsequently, additional nodes and relations are extracted from the drug features. These nodes are category nodes and approval status nodes to which the drugs can be linked.

Finally, CSV files for the Neo4j import are built. For each node type a CSV file is generated where the index column is named as "index:ID(< type >)". Similarly, for each relation a CSV file is generated where start and end column are marked via ":START_ID(< node_type >)" and ":END_ID(< node_type >)". Note, that the bidirectional relations are treated as two different relations.

The script generates the Neo4j import command based on the generated files and the import gets executed.

Note that the original files (kg.csv etc.) will be deleted if and only if they were downloaded in the beginning.

## Citation

The PrimeKG data was made available by Chandak, Payal and Huang, Kexin and Zitnik, Marinka in Nature Scientific Data, 2023: [Building a knowledge graph to enable precision medicine](https://www.nature.com/articles/s41597-023-01960-3)

The data is available in [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).
