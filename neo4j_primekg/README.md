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
    --env NEO4J_server_databases_default__to__read__only=true \
    --name neo4j_primekg_service \
    neo4j_primekg:latest
```

## Citation

The PrimeKG data was made available by Chandak, Payal and Huang, Kexin and Zitnik, Marinka in Nature Scientific Data, 2023: [Building a knowledge graph to enable precision medicine](https://www.nature.com/articles/s41597-023-01960-3)

The data is available in [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).
