#!/bin/bash

function download_data
{
    if [ -f "$PRIMEKG_CSV" ]; then
        declare -g PRIMEKG_CSV_EXISTED=true
    else
        echo ">>> Downloading PrimeKG csv file..."
        declare -g PRIMEKG_CSV_EXISTED=false
        wget --no-clobber -O $PRIMEKG_CSV https://dataverse.harvard.edu/api/access/datafile/6180620
    fi

    if [ -f "$PRIMEKG_DRUG_NODE_FEATURES_TSV" ]; then
        declare -g PRIMEKG_DRUG_FEATURES_EXISTED=true
    else
        echo ">>> Downloading PrimeKG drug features file..."
        declare -g PRIMEKG_DRUG_FEATURES_EXISTED=false
        wget --no-clobber -O $PRIMEKG_DRUG_NODE_FEATURES_TSV https://dataverse.harvard.edu/api/access/datafile/6180619
    fi

    if [ -f "$PRIMEKG_DISEASE_NODE_FEATURES_TSV" ]; then
        declare -g PRIMEKG_DISEASE_FEATURES_EXISTED=true
    else
        echo ">>> Downloading PrimeKG disease features file..."
        declare -g PRIMEKG_DISEASE_FEATURES_EXISTED=false
        wget --no-clobber -O $PRIMEKG_DISEASE_NODE_FEATURES_TSV https://dataverse.harvard.edu/api/access/datafile/6180618
    fi
}

function cleanup_data
{
    rm -r $PRIMEKG_CSVS_FOR_NEO4J

    if [ "${PRIMEKG_CSV_EXISTED}" == "false" ]; then
        echo ">>> Cleaning up PrimeKG csv file..."
        rm $PRIMEKG_CSV
    fi
    if [ "${PRIMEKG_DRUG_FEATURES_EXISTED}" == "false" ]; then
        echo ">>> Cleaning up PrimeKG csv file..."
        rm $PRIMEKG_DRUG_NODE_FEATURES_TSV
    fi
    if [ "${PRIMEKG_DISEASE_FEATURES_EXISTED}" == "false" ]; then
        echo ">>> Cleaning up PrimeKG csv file..."
        rm $PRIMEKG_DISEASE_NODE_FEATURES_TSV
    fi
}

function import_primekg
{
    download_data

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

    cleanup_data
}

if [ -e $DELETE_PRIMEKG_CSV ]; then
    echo "WARNING: DELETE_PRIMEKG_CSV is deprecated. This now gets handled automatically."
fi

SETUP_DONE_MARKER="/data/prime_kg_is_imported_to_neo4j"
if [ ! -e $SETUP_DONE_MARKER ]; then
    import_primekg
    touch $SETUP_DONE_MARKER
fi

echo ">>> Starting Neo4j..."
bash /startup/docker-entrypoint.sh neo4j
