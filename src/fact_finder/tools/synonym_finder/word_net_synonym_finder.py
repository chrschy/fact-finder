import itertools
from typing import List

import nltk
from nltk.corpus import wordnet as wn

from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder


class WordNetSynonymFinder(SynonymFinder):

    def __init__(self) -> None:
        nltk.download("wordnet")

    def __call__(self, name: str) -> List[str]:
        result = list(itertools.chain(*wn.synonyms(word=name)))
        if name not in result:
            result.append(name)
        return result
