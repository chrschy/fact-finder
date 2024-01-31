import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()


class EntityDetector:

    def __init__(self, filenames: list[str] = None):
        self.__url = "https://api2.linguist.skgt.int.bayer.com/linnaeusannotate"
        self.__api_key = os.getenv("SYNONYM_API_KEY")
        if filenames:
            self.__possible_filenames = filenames
        else:
            self.__possible_filenames = self.__default_filenames()

    def __default_filenames(self) -> list[str]:
        return [
            "sci_gene_human_dictionary.txt",  # Human genes
            "sci_ab.dictionary.txt",  # Antibodies
            "sci_disease_dictionary.txt",  # Diseases
            "hubble_cells.dictionary.txt",  # Cells
            "sci_drug_dictionary.txt",  # Drugs
            "organ_thesarus.txt",  # Organs
            "celline.dictionary.txt",  # Celllines
        ]

    def __call__(self, search_text: str) -> dict:
        filenames_as_single_string = ", ".join(self.__possible_filenames)
        payload = json.dumps({
            "public_dic": filenames_as_single_string,
            "text": search_text
        })
        headers = {
            'x-api-key': self.__api_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.__url, headers=headers, data=payload)
        return json.loads(response.text)["annotations"]
