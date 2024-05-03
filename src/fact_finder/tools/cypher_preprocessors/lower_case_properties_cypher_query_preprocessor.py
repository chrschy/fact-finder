import regex as re
from fact_finder.tools.cypher_preprocessors.property_string_preprocessor import (
    PropertyStringCypherQueryPreprocessor,
)


class LowerCasePropertiesCypherQueryPreprocessor(PropertyStringCypherQueryPreprocessor):

    def _replace_match(self, matches: re.Match[str]) -> str:
        assert len(matches.groups())
        text = matches.group(0)
        for property in matches.captures(1):
            text = text.replace(property, property.lower())
        return text
