import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


_DEFAULT_ENTITY_FILENAMES = [
    "sci_gene_human_dictionary.txt",  # Human genes
    "sci_ab.dictionary.txt",  # Antibodies
    "sci_disease_dictionary.txt",  # Diseases
    "hubble_cells.dictionary.txt",  # Cells
    "sci_drug_dictionary.txt",  # Drugs
    "organ_thesarus.txt",  # Organs
    "celline.dictionary.txt",  # Celllines
]


class EntityDetector:

    def __init__(self, filenames: List[str] = _DEFAULT_ENTITY_FILENAMES):
        self.__possible_filenames = filenames
        self.__url = os.getenv("SYNONYM_API_URL")
        self.__api_key = os.getenv("SYNONYM_API_KEY")
        if self.__api_key is None or self.__url is None:
            raise ValueError("For using EntityDetector, the env variable SYNONYM_API_KEY as well as SYNONYM_API_URL must be set.")

    def __call__(self, search_text: str) -> List[Dict[str, Any]]:
        filenames_as_single_string = ", ".join(self.__possible_filenames)
        payload = json.dumps({"public_dic": filenames_as_single_string, "text": search_text})
        headers = {"x-api-key": self.__api_key, "Accept": "application/json", "Content-Type": "application/json"}
        response = requests.request("POST", self.__url, headers=headers, data=payload)
        if response.status_code == 200:
            return json.loads(response.text)["annotations"]
        response.raise_for_status()
        return []
