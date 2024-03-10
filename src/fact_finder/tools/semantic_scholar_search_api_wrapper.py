from typing import List

import requests
from semanticscholar import SemanticScholar


class SemanticScholarSearchApiWrapper:

    def __init__(self):
        self._semantic_scholar = SemanticScholar(timeout=3)
        self._session = requests.Session()
        self._semantic_scholar_endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search_by_abstracts(self, keywords: str) -> List[str]:
        # papers = self._semantic_scholar.search_paper(
        #     query=keywords,
        #     limit=5
        # )
        query_params = {"query": keywords, "limit": 5, "fields": "title,abstract"}
        response = self._session.get(self._semantic_scholar_endpoint, params=query_params)
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
