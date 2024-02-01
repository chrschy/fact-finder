#!/bin/bash

function import_primekg
{
    echo ">>> Downloading PrimeKG csv file..."
    wget --no-clobber -O $PRIMEKG_CSV https://dataverse.harvard.edu/api/access/datafile/6180620

    echo ">>> Processing PrimeKG csv file..."
    IMPORT_CMD=$(python3 $IMPORT_DIR/primekg_to_neo4j_csv.py)

    if [ $? -ne 0 ]; then
        echo "Error in python script. Output:"
        echo $IMPORT_CMD
        exit 1
    fi

    echo ">>> Importing PrimeKG..."
    eval "$IMPORT_CMD"

    if [ $? -ne 0 ]; then
        echo "Error while importing!"
        exit 1
    fi

    if [ "${DELETE_PRIMEKG_CSV}" == "true" ]; then
        echo ">>> Cleaning up..."
        rm -r /$IMPORT_DIR
    fi
}

SETUP_DONE_MARKER="/data/prime_kg_is_imported_to_neo4j"
if [ ! -e $SETUP_DONE_MARKER ]; then
    import_primekg
    touch $SETUP_DONE_MARKER
fi

echo ">>> Starting Neo4j..."
bash /startup/docker-entrypoint.sh neo4j
