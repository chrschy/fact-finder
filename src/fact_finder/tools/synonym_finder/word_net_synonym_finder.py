from fact_finder.tools.synonym_finder.synonym_finder import SynonymFinder
from nltk.corpus import wordnet as wn


from typing import List


class WordNetSynonymFinder(SynonymFinder):

    def __call__(self, name: str) -> List[str]:
        result = wn.synonyms(word=name)[0]
        return result
