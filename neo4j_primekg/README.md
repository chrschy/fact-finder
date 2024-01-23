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
    --name neo4j_primekg_service \
    neo4j_primekg:latest
```

Test via Cypher shell:
```
docker exec -it <CONTAINER_ID> cypher-shell -u neo4j -p opensesame
```
```
MATCH (disease1:disease {name: "psoriasis"})-[:disease_disease]->(disease2:disease {name: "scalp_disease"})
RETURN disease1, disease2;
```