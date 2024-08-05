from unittest.mock import MagicMock, patch

import pytest
from fact_finder.chains.rag.semantic_scholar_chain import SemanticScholarChain
from fact_finder.chains.rag.text_search_qa_chain import TextSearchQAChain
from fact_finder.tools.semantic_scholar_search_api_wrapper import (
    SemanticScholarSearchApiWrapper,
)
from langchain.chains.base import Chain
from langchain_core.prompts import PromptTemplate
from tests.chains.helpers import build_llm_mock


def test_simple_question(chain: Chain):
    answer = chain.invoke({"question": "Alternative causes of fever in malaria infections?"})
    assert answer["rag_output"].startswith("Alternative causes of fever in")


@pytest.fixture
def chain(sematic_scholar_chain: SemanticScholarChain, text_qa_chain: TextSearchQAChain) -> Chain:
    with patch("requests.Session") as mock_session:
        mock_session.return_value.get.side_effect = _mock_get
        yield sematic_scholar_chain | text_qa_chain


@pytest.fixture
def text_qa_chain(
    rag_answer_generation_prompt_template: PromptTemplate, sematic_scholar_chain: SemanticScholarChain
) -> TextSearchQAChain:
    return TextSearchQAChain(
        llm=build_llm_mock("Alternative causes of fever in malaria infections are..."),
        rag_answer_generation_template=rag_answer_generation_prompt_template,
        rag_output_key=sematic_scholar_chain.output_keys[0],
    )


@pytest.fixture
def sematic_scholar_chain(keyword_prompt_template: PromptTemplate) -> SemanticScholarChain:
    return SemanticScholarChain(
        semantic_scholar_search=SemanticScholarSearchApiWrapper(),
        llm=build_llm_mock("Alternative causes, fever, malaria infections"),
        keyword_prompt_template=keyword_prompt_template,
    )


@pytest.fixture
def keyword_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. You get a user question in natural language. Please transform it into keywords that can be used in semantic scholar keyword search: Question: {question}",
    )


@pytest.fixture
def rag_answer_generation_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. You get a user question in natural language. Given the following context, please answer the given question based only on the context. Do not hallucinate. If you cannot answer based on the context, say 'The documents do not provide enough information to answer the question.'. Context: {context}, Question: {question}",
    )


def _mock_get(url: str, params: dict, headers: dict):
    assert "https://api.semanticscholar.org/graph/v1/paper/search" == url
    assert {"fields": "title,abstract", "limit": 5, "query": "Alternative causes, fever, malaria infections"} == params
    assert "x-api-key" in headers.keys()
    response = MagicMock()
    response.status_code = 200
    response.json = lambda: {
        "data": [
            {
                "paperId": "160c6875586b9bc085ef4533daa50004d85665f8",
                "title": "Improve psoriasis symptoms with strategies to manage nutrition",
                "abstract": "Classification of invasive bloodstream infections and Plasmodium falciparum malaria using autoantibodies as biomarkers\n",
            },
            {
                "paperId": "e63da5fcf63a7086a4f9041dcc7d2ffde91046ea",
                "title": "Diurnal and seasonal variation in psoriasis symptoms",
                "abstract": "Miliary and Meningeal Tuberculosis\nAcute fever associated with multi-systems illness is a common feature of several infections endemic in South-east Asia, including malaria, dengue, scrub typhus, and leptospirosis. Malaria and dengue infectionmustbe initially excluded inall patientspresented with this syndrome. The lack of sensitive and rapid methods for the laboratory confirmation of a tentative diagnosis has been an important problem. Physician in the tropic should aware thatmalaria, dengue infection, rickettsioses, and leptospirosis are major causes of acute undifferentiated fever. Travelers to endemic areas are also at risk of these infections. Early recognition and appropriate treatment reduce morbidity and mortality. Doxycycline would usually be an appropriate initial antimicrobial treatment for individual with suspected either rickettsioses or leptospirosis. Azithromycin could be considered as an alternative treatment when ever doxycycline allergy is suspected. In this presentation the outline of common causes of acute undifferentiated febrile illness among indigenous population and international travelers returned fromSouth-eastAsia, and themanagement of these patients will be discussed.",
            },
            {
                "paperId": "53d1d11a83740eebe36bf660f085f1c4bb3bcaea",
                "title": "Development and Content Validation of the Psoriasis Symptoms and Impacts Measure (P-SIM) for Assessment of Plaque Psoriasis",
                "abstract": "Neglected and emerging diseases in sub-Saharan Africa.\nGlobal strategies for controlling infections in sub-Saharan Africa have focused on the ‘big three’: AIDS, tuberculosis and malaria. Other causes of fever and infection belong to the category of neglected diseases. It is important in the 21st century to compile a comprehensive list of infections in subSaharan Africa, in particular for the two types of syndromes most frequently associated with decreases in life expectancy: respiratory and diarrhoeal infections. Regarding diarrhoeal infections, typhoid fever is the subject of a review on the management of the disease [1]. The emergence of multidrug resistance and decreased ciprofloxacin susceptibility of Salmonella enterica serovar typhi has raised the question of what should be the first line of antibiotic therapy, taking into account that a short course of treatment, quick effectiveness and prevention of relapse or chronic faecal carriage are essential. In general, ceftriaxone is useful as an alternative to ciprofloxacin. Chloramphenicol use is also regaining popularity in developing countries. However, azithromycin appears to be the best candidate for the management of typhoid fever both in developing countries with high prevalence of decreased ciprofloxacin susceptibility and in returned travellers in industrialized countries. Besides typhoid fever, the prevalence of Tropheryma whipplei in certain areas of Senegal indicates that this emergent agent probably plays a much neglected role as a causative agent of diarrhoea and should be explored in the future [2,3]. Vector-borne diseases play a particularly important part in Africa. Recent studies have shown that they are the most frequent causes of fever in both indigenes and travellers. In particular, Borrelia has been re-discovered [4,5] and the rickettsiae, including Rickettsia felis, which is the object of a review, have been reported [6]. Rickettsia felis has been retrieved in arthropods, mainly in the cat flea, Ctenocephalides felis, and has a worldwide distribution. It is interesting to note that R. felis, which is a rare cause of flea-borne spotted fever and was first described in the USA, is an unexpectedly common cause of ‘fever of unknown origin’ in sub-Saharan Africa. Indeed, two studies conducted in Senegal and Kenya by two different teams at almost the same time and published in the same issue of Emerging Infectious Diseases have reported similar data outlining the importance of R. felis infection in patients with unexplained fever and without malaria in both countries. Overall, the prevalence of R. felis DNA in blood specimens was approximately 4%. No rash was reported. More intriguing is the unanswered question concerning the complete spectrum of potential arthropod vectors associated with R. felis. There has been speculation that patients may have been exposed to fleas. However, the wide spectrum of arthropod cell lines capable of supporting R. felis growth suggests a wider spectrum of potential vectors than has been previously suspected. Among the neglected parasitic diseases, the filarioses with the recent strategy of antibiotic treatment of their Wolbachia endosymbionts, as well as trypanosomiasis, which has increased in Africa and has been a source of concern, are presented in two specific reviews [7,8]. Wuchereria bancrofti and Onchocerca volvulus, the species causing lymphatic filariosis and onchocerciasis in Africa, live in mutual symbiosis with Wolbachia endobacteria, which play a role in inducing the inflammation that leads to symptoms. The Wolbachia endobacteria are also antibiotic targets. Indeed, the standard microfilaricidal drugs ivermectin and albendazole, which are used in mass drug administration programmes aimed at interrupting transmission, have three major limitations: (i) coendemicity of Loa loa, which is a mild nuisance only, with W. bancrofti or O. volvulus presents an impediment to treatment because of the risk of encephalopathy encountered with ivermectin and Loa infection; (ii) ivermectin and albendazole do not permanently sterilize female worms; (iii) there is no clear macrofilaricidal effect at the doses currently in use; as a consequence, repeated rounds of massive drug administration in onchocerciasis for many years are required because of the extended worm lifespan. However, targeting the Wolbachia endosymbionts, which are present in W. bancrofti and O. volvulus (but not in L. loa), with doxycycline has been shown to lead to sterilization of the female worms and to have a macrofilaricidal effect.",
            },
            {
                "paperId": "3eee2d46fc672006703f3165340549e4706c2e48",
                "title": "Improvement in Patient-Reported Outcomes (Dermatology Life Quality Index and the Psoriasis Symptoms and Signs Diary) with Guselkumab in Moderate-to-Severe Plaque Psoriasis: Results from the Phase III VOYAGE 1 and VOYAGE 2 Studies",
                "abstract": "Treatable causes of fever among children under five years in a seasonal malaria transmission area in Burkina Faso\n",
            },
            {
                "paperId": "5232ae50c6bb4dddddc00c334e5da0202ba25e52",
                "title": "Prevalence and Odds of Anxiety Disorders and Anxiety Symptoms in Children and Adults with Psoriasis: Systematic Review and Meta-analysis",
                "abstract": "Prevalence of diarrhoea, acute respiratory infections, and malaria over time (1995-2017): A regional analysis of 23 countries in West and Central Africa\nBackgound The global community recognizes the urgent need to end preventable child deaths, making it an essential part of the third Sustainable Development Goal. Pneumonia, diarrhoea, and malaria still remain the leading causes of deaths among children under five years, especially in one of the poorest geographic regions of the world – West and Central Africa. This region carries a disproportionately high share of the global burden, both in terms of morbidity and mortality. The study aims to assess levels and trends of the prevalence of these three childhood diseases in West and Central Africa to better inform ongoing and future programmes to improve child survival. Methods Demographic and Health Surveys and Multiple Indicator Cluster Surveys available from 1995 to 2017 for 23 countries in West and Central Africa were analysed. We estimated the prevalence of diarrhoea, acute respiratory infections (ARI), malaria, and fever as a proxy for malaria, and split the data into three time periods to assess these trends in disease prevalence over time. Further analyses were done to assess the variations by geographic location (urban and rural) and gender (boys and girls). Results In West and Central Africa, the reduction of the prevalence rates of diarrhoea, acute respiratory infections, malaria, and fever has decelerated over time (1995-2009), and little improvements occurred between 2010 and 2017. The reduction within the region has been uneven and the prevalence rates either increased or stagnated for diarrhoea (nine countries), ARI (four countries), and fever (six countries). The proportion of affected children was high in emergency or fragile settings. Disaggregated analyses of population-based data show persistent gaps between the prevalence of diseases by geographic location and gender, albeit not significant for the latter. Conclusions Without intensified commitment to reducing the prevalence of pneumonia, malaria, and diarrhoea, many countries will not be able to meet the SDG goal to end preventable child deaths. Evidence-driven programmes that focus on improving equitable access to preventive health care information and services must be fostered, especially in complex emergency settings. This will be an opportunity to strengthen primary health care, including community health programmes, to achieve universal health coverage.",
            },
        ]
    }
    return response
