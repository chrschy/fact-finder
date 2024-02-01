import pytest
from dotenv import load_dotenv

from fact_finder.entity_detector.entity_detector import EntityDetector

load_dotenv()


@pytest.fixture()
def entity_detector():
    return EntityDetector()


def test_entity_detector(entity_detector):
    result = entity_detector("What is pink1? Does it help with epilepsy?")
    assert len(result) == 2
    result = entity_detector("What is pink1?")
    assert len(result) == 1
    result = entity_detector("atopic dermatitis")
    assert len(result) == 1
    assert result[0]["pref_term"] == "dermatitis, atopic"
