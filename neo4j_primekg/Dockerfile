FROM neo4j:5.15.0

ENV IMPORT_DIR=primekg_data
ENV PRIMEKG_CSV=/$IMPORT_DIR/kg.csv
ENV PRIMEKG_CSVS_FOR_NEO4J=/$IMPORT_DIR/preprocessed_for_neo4j
ENV DELETE_PRIMEKG_CSV=true

COPY import_primekg.sh /startup/import_primekg.sh
COPY primekg_to_neo4j_csv.py $IMPORT_DIR/primekg_to_neo4j_csv.py

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install pandas && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p $PRIMEKG_CSVS_FOR_NEO4J && \
    chmod 777 /startup/import_primekg.sh

ENTRYPOINT ["tini", "-g", "--"]
CMD ["/startup/import_primekg.sh"]
