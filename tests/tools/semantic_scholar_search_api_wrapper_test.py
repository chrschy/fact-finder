from unittest.mock import patch, MagicMock

import pytest

from fact_finder.tools.semantic_scholar_search_api_wrapper import SemanticScholarSearchApiWrapper


@pytest.fixture
def semantic_scholar_search_api_wrapper():
    with patch("requests.Session") as mock_session:
        mock_session.return_value.get.side_effect = _mock_get
        return SemanticScholarSearchApiWrapper()


def test_search_by_abstract(semantic_scholar_search_api_wrapper):
    result = semantic_scholar_search_api_wrapper.search_by_abstracts(keywords="psoriasis, symptoms")
    assert 5 == len(result)
    assert result[0].startswith("Improve psoriasis")
    assert result[4].startswith("Prevalence and Odds")


def _mock_get(url: str, params: dict):
    assert "https://api.semanticscholar.org/graph/v1/paper/search" == url
    assert {"fields": "title,abstract", "limit": 5, "query": "psoriasis, symptoms"} == params
    response = MagicMock()
    response.status_code = 200
    response.json = lambda: {
        "data": [
            {
                "paperId": "160c6875586b9bc085ef4533daa50004d85665f8",
                "title": "Improve psoriasis symptoms with strategies to manage nutrition",
                "abstract": "Psoriasis is an inflammatory skin disease that has been linked to both genetic and environmental factors. From 0.09 to 11.43% of the world's population has this dermatosis; in industrialized nations, the prevalence is between 1.5% and 5%. Psoriasis is believed to be caused by a combination of adaptive and innate immune responses. The PASI scale measures the clinical severity of psoriasis on a scale from 0 to 100. This analysis was conducted to determine if existing nutrition interventions are effective in alleviating psoriasis symptoms. Science Direct, Google Scholar, Scopus, PubMed, and ClinicalTrials.gov were used to compile the data for this review. We used the following search terms to narrow our results: psoriasis, nutrition, diet treatment, vitamin, RCTs, and clinical trials. Ten studies were selected from the 63 articles for this review. Research designs are evaluated using the Risk of Bias 2 (RoB2), the Risk of Bias in Non-Randomized Studies of Interventions (ROBINS-I), and the Newcastle-Ottawa Scale (NOS). Studies concluded that a Mediterranean diet, vitamin D3 supplementation, the elimination of cadmium (Cd), lead (Pb), and mercury from the diet, as well as intermittent fasting and low-energy diets for weight loss in obese patients, can alleviate the symptoms of inflammatory diseases. Psoriasis patients undergoing treatment should adhere to dietary recommendations.",
            },
            {
                "paperId": "e63da5fcf63a7086a4f9041dcc7d2ffde91046ea",
                "title": "Diurnal and seasonal variation in psoriasis symptoms",
                "abstract": "The frequency of itch in people with psoriasis varies between 64-97% and may exhibit time of day differences as well as interfere with sleep. Furthermore, psoriasis flares appear to exhibit seasonal variation. Whilst a North American study, using a physician-rating scale, showed a trend for winter flaring and summer clearing; a Japanese study, using proxy measures, found no difference between hot and cold months.",
            },
            {
                "paperId": "53d1d11a83740eebe36bf660f085f1c4bb3bcaea",
                "title": "Development and Content Validation of the Psoriasis Symptoms and Impacts Measure (P-SIM) for Assessment of Plaque Psoriasis",
                "abstract": None,
            },
            {
                "paperId": "3eee2d46fc672006703f3165340549e4706c2e48",
                "title": "Improvement in Patient-Reported Outcomes (Dermatology Life Quality Index and the Psoriasis Symptoms and Signs Diary) with Guselkumab in Moderate-to-Severe Plaque Psoriasis: Results from the Phase III VOYAGE 1 and VOYAGE 2 Studies",
                "abstract": None,
            },
            {
                "paperId": "5232ae50c6bb4dddddc00c334e5da0202ba25e52",
                "title": "Prevalence and Odds of Anxiety Disorders and Anxiety Symptoms in Children and Adults with Psoriasis: Systematic Review and Meta-analysis",
                "abstract": "The magnitude of the association between psoriasis and depression has been evaluated, but not that between psoriasis and anxiety. The aim of this systematic review and meta-analysis was to examine the prevalence and odds of anxiety disorders and symptoms in patients with psoriasis. Five medical databases (Cochrane Database, EMBASE, PubMed, PsychINFO, ScienceDirect) were searched for relevant literature. A total of 101 eligible articles were identified. Meta-analysis revealed different prevalence rates depending on the type of anxiety disorder: 15% [95% confidence interval [CI] 9–21] for social anxiety disorder, 11% [9–14] for generalized anxiety disorder, and 9% [95% CI 8–10] for unspecified anxiety disorder. There were insufficient studies assessing other anxiety disorders to be able to draw any conclusions on their true prevalence. Meta-analysis also showed a high prevalence of anxiety symptoms (34% [95% CI 32–37]). Case-control studies showed a positive association between psoriasis and unspecified anxiety disorder (odds ratio 1.48 [1.18; 1.85]) and between psoriasis and anxiety symptoms (odds ratio 2.51 [2.02; 3.12]). All meta-analyses revealed an important heterogeneity, which could be explained in each case by methodological factors. The results of this study raise the necessity of screening for the presence of anxiety disorders, as previously recommended for depressive disorders, in patients with psoriasis and, if necessary, to refer such patients for evaluation by a mental health professional and appropriate treatment.",
            },
        ]
    }
    return response


@pytest.mark.skip("Not Implemented")
def test_search_by_paper_content(semantic_scholar_search_api_wrapper):
    semantic_scholar_search_api_wrapper.search_by_paper_content(keywords="psoriasis, symptoms")
