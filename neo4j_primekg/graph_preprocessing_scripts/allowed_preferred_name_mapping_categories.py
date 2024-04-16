from typing import Dict, List, Set

_allowed_categories_for_each_type: Dict[str, List[str]] = {
    "gene/protein": [
        "Gene",
        # "Disease",
        # "CellLine",
        # "Drug",
        # "Organs"
    ],
    "drug": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        "Drug",
        # "Organs"
    ],
    "effect/phenotype": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "disease": [
        # "Gene",
        # "Antibody",
        "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "biological_process": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "molecular_function": [
        # "Gene",
        # "Antibody",
        # "CellLine",
        # "Disease",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "cellular_component": [
        # "Gene",
        # "Antibody",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "exposure": [
        # "Gene",
        # "Disease",
        # "CellLine",
        # "Drug",
        # "Organs"
    ],
    "pathway": [
        # "Gene",
        # "Antibody",
        # "CellLine",
        # "Disease",
        # "Cells",
        # "Drug",
        # "Organs"
    ],
    "anatomy": [
        # "Gene",
        # "Disease",
        # "CellLine",
        # "Cells",
        # "Drug",
        "Organs"
    ],
}

allowed_categories_for_each_type: Dict[str, Set[str]] = {
    k: set(v) for k, v in _allowed_categories_for_each_type.items() if v
}
