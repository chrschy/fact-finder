import os
from unittest import skipIf
from unittest.mock import Mock, patch
import pytest
from dotenv import load_dotenv

from fact_finder.tools.entity_detector import EntityDetector

load_dotenv()


@pytest.fixture()
def entity_detector():
    return EntityDetector()


@patch.dict(os.environ, {"SYNONYM_API_KEY": "dummy_key"})
@patch("requests.request")
def test_entity_detector(mock_request, entity_detector):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"annotations" : [1, 2]}'
    mock_request.return_value = mock_response

    result = entity_detector("What is pink1? Does it help with epilepsy?")
    assert len(result) == 2


@skipIf(os.getenv("SYNONYM_API_KEY") is None, "Requires SYNONYM_API_KEY to be set.")
def test_entity_detector_with_api(entity_detector):
    result = entity_detector("What is pink1? Does it help with epilepsy?")
    assert len(result) == 2
    result = entity_detector("What is pink1?")
    assert len(result) == 1
    result = entity_detector("atopic dermatitis")
    assert len(result) == 1
    assert result[0]["pref_term"] == "dermatitis, atopic"
