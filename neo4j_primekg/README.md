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
    --env NEO4J_PLUGINS=\[\"apoc\"\] \
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

Alternative start-up using an already downloaded csv file:
```
docker run -d --restart=always \
    --publish=7474:7474 --publish=7687:7687 \
    --volume <abs/path/to/prime/kg.csv>:/primekg_data/kg.csv:ro \
    --env DELETE_PRIMEKG_CSV=false \
    --env NEO4J_AUTH=neo4j/opensesame \
    --env NEO4J_PLUGINS=\[\"apoc\"\] \
    --env NEO4J_server_databases_default__to__read__only=true \
    --name neo4j_primekg_service \
    neo4j_primekg:latest
```
