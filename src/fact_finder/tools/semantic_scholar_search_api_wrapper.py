import os
from typing import List

import requests


class SemanticScholarSearchApiWrapper:

    def __init__(self):
        SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY")
        self._session = requests.Session()
        self._semantic_scholar_endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
        self._header = {"x-api-key": SEMANTIC_SCHOLAR_KEY}

    def search_by_abstracts(self, keywords: str) -> List[str]:
        query_params = {"query": keywords, "limit": 5, "fields": "title,abstract"}
        response = self._session.get(self._semantic_scholar_endpoint, params=query_params, headers=self._header)
        if response.status_code != 200:
            raise ValueError(f"Semantic scholar API returned an error:\n{response}")
        papers = response.json()["data"]
        results = ["\n".join([paper["title"], paper["abstract"] or ""]) for paper in papers]
        return results

    def search_by_paper_content(self, keywords: str) -> List[str]:
        # Note that semantic scholar does only retrieve abstracts, and even they may be missing.
        # we could get the PDF links and try to access papers dynamically, run pdf extraction, etc.
        # Or: we decide on papers to download and preprocess and put in a local vector db.

        # Otherwise the idea here would be to get 1 or 2 top papers, dynamically embedd small chunks, run a retriever on that and the plug this
        # into a qa prompt.
        raise NotImplementedError


if __name__ == "__main__":
    sem = SemanticScholarSearchApiWrapper()
    print(sem.search_by_abstracts("Alternative causes, fever, malaria infections"))
