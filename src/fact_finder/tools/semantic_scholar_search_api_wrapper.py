from typing import List

import requests
from semanticscholar import SemanticScholar


class SemanticScholarSearchApiWrapper:

    def __init__(self):
        self._semantic_scholar = SemanticScholar(timeout=3)
        self._session = requests.Session()
        self._semantic_scholar_endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, keywords: str) -> List[str]:
        # papers = self._semantic_scholar.search_paper(
        #     query=keywords,
        #     limit=5
        # )
        query_params = {
            "query": keywords,
            "limit": 5,
            "fields": "title,abstract"
        }
        response = self._session.get(self._semantic_scholar_endpoint, params=query_params)
        if response.status_code != 200:
            raise ValueError(f"Semantic scholar API returned an error:\n{response}")
        papers = response.json()["data"]
        results = ['\n'.join([paper["title"], paper["abstract"] or ""]) for paper in papers]
        return results
