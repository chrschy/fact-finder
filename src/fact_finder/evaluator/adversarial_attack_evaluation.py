import json
from functools import partial
from typing import Dict, Any, List, Optional

import tqdm
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic.v1 import Extra

from fact_finder.chains import GraphQAChain, CypherQueryGenerationChain
from fact_finder.chains.filtered_primekg_question_preprocessing_chain import FilteredPrimeKGQuestionPreprocessingChain
from fact_finder.chains.graph_qa_chain import GraphQAChainConfig
from fact_finder.config.primekg_config import (
    _parse_primekg_args,
    _build_preprocessors,
    _get_graph_prompt_templates,
    _get_primekg_entity_categories,
)
from fact_finder.config.primekg_predicate_descriptions import PREDICATE_DESCRIPTIONS
from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.prompt_templates import KEYWORD_PROMPT, COMBINED_QA_PROMPT
from fact_finder.tools.entity_detector import EntityDetector
from fact_finder.utils import load_chat_model, build_neo4j_graph


QUESTION_WRONG_CYPHER_MAPPING = {
    "Which medications have more off-label uses than approved indications?": 'MATCH (d:disease {name: "chronic kidney disease"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.id AS GeneID, g.name AS GeneName',
    "Which medications have more off-label uses than approved uses?": 'MATCH (e:gene_or_protein {name:"f2"})-[ep:expression_present]-(a:anatomy) RETURN a.name',
    "Which diseases have only treatments that have no side effects at all?": 'MATCH (d:disease {name: "prostate carcinoma"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "How does Asbestos interact with the human body?": 'MATCH (disease:disease {name: "chronic kidney disease"})-[:linked_to]->(exposure:exposure) RETURN DISTINCT exposure.name',
    "Which exposures are linked to more than 10 diseases?": 'MATCH (d:disease {name: "psoriasis"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Which drugs have pterygium as side effect?": 'MATCH (d:disease {name: "heart failure"})-[:associated_with]->(g:gene_or_protein {name: "pparg"})\nRETURN d',
    "Which medicine may cause as sideeffect pterygium?": 'MATCH (g:disease {name: "rheumatoid arthritis"})-[]-(p:gene_or_protein) RETURN COUNT(DISTINCT p)',
    "Which drugs have freckling as side effect?": 'MATCH (drug:drug {name: "lenvatinib"})-[:indication]->(disease:disease)\nRETURN disease.name AS DiseaseName',
    "Which medicine may cause freckling?": 'MATCH (d:disease {name: "sialuria"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Which drugs are used to treat ocular hypertension?": 'MATCH (d:disease {name: "psoriasis"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Which drugs to treat ocular hypertension may cause the loss of eyelashes?": 'MATCH (n:drug)-[indication]-(d:disease {name:"ocular hypertension"}) RETURN DISTINCT n',
    "Which disease does not show in cleft upper lip?": 'MATCH (d:disease {name: "dermatitis, atopic"})-[:phenotype_present]->(p:effect_or_phenotype)\nRETURN p.name AS Symptoms',
    "Which diseases show in symptoms such as eczema, neutropenia and a high forehead?": 'MATCH (e:exposure {name: "asbestos"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name',
    "Which expressions are present for the gene/protein f2?": 'MATCH (d:disease {name: "chronic kidney disease"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.id AS GeneID, g.name AS GeneName',
    "Which drugs are associated with epilepsy?": 'MATCH (d:disease {name: "alzheimer disease"})-[:indication]->(drug:drug)-[:target]->(target:gene_or_protein)\nRETURN drug.name AS DrugName, target.name AS TargetName',
    "Which genes can be novel targets in stickler syndrome?": 'RETURN EXISTS {MATCH (d {name:"atopic dermatitis"})-[]-(d2 {name:"psoriasis"})}',
    "How many proteins are connected to rheumatoid arthritis?": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
    "Is there clinical evidence linking HGF to kidneys?": 'MATCH (d:disease {name: "alzheimer disease"})-[:indication]->(drug:drug)-[:target]->(target:gene_or_protein)\nRETURN drug.name AS DrugName, target.name AS TargetName',
    "What issues are there with lamotrigine?": 'MATCH (d:disease {name: "epilepsy"})-[:indication]->(drug:drug)\nWHERE NOT (drug)-[:side_effect]->(:effect_or_phenotype)\nRETURN DISTINCT drug.name AS DrugName',
    "Which drugs against epilepsy have zero side effects?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "Does lamotrigine have more side effects than primidone?": 'MATCH (d:disease {name: "psoriasis"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Is psoriasis related with atopic dermatitis?": 'MATCH (d:drug  {name:"lamotrigine"})-[:side_effect]-(e) RETURN e',
    "What genes play a role in breast cancer?": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
    "Which drugs against epilepsy should not be used by patients with hypertension?": 'MATCH (alcoholExposure:exposure {name: "ethanol"})-[:linked_to]->(diseaseCausedByAlcohol:disease)-[:indication]->(drug:drug)-[:contraindication]->(anemia:disease {name: "anemia (disease)"})\nRETURN drug.name AS DrugName',
    "Can I use iboprofen in patients with addison disease?": 'MATCH (d:drug)-[]-(a:approval_status {name:"vet_approved"}) MATCH (d)-[]-(di:disease {name:"influenza"}) RETURN d',
    "Which off label medicaments for epilepsie have a clogp value below 0?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "Which disease may a patient presented with fatigue, vomiting and abdominal pain have?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "Which drug targeting cyb5a is a micro nutrient?": 'MATCH (d:disease {name: "psoriasis"})-[:indication]->(drug:drug)\nRETURN drug.name AS Treatment',
    "How many drugs against epilepsy are available?": 'MATCH (d:drug)-[]-(a:approval_status {name:"vet_approved"}) MATCH (d)-[]-(di:disease {name:"influenza"}) RETURN d',
    "Which drugs against asthma are not pills?": 'MATCH (d:drug {name:"primidone"})-[:side_effect]-(s) MATCH (d2:drug {name:"lamotrigine"})-[:side_effect]-(s2) RETURN COUNT(DISTINCT s2) > COUNT(DISTINCT s)',
    "Which drugs can I give my dog with influenza?": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
    "How many experimental drugs against epilepsy are there?": 'MATCH (d:disease)-[]-() WHERE d.symptoms CONTAINS "fatigue" AND d.symptoms CONTAINS "vomiting" and d.symptoms CONTAINS "abdominal pain" RETURN COLLECT(distinct d) as d',
    "Which drug against epilepsy has the lowest clogp value?": 'MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}) MATCH (d)-[:phenotype_present]-({name:"neutropenia"}) MATCH (d)-[:phenotype_present]-({name:"high forehead"}) RETURN DISTINCT d.name',
    "Which genes influence the peripheral nervous system?": 'MATCH (d:disease {name: "rheumatoid arthritis"})-[:phenotype_present]->(p:effect_or_phenotype) RETURN p.name',
    "How many drugs are gaseous?": 'MATCH (d:disease {name: "cardiomyopathy"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "In which Anatomical regions is IRAK4 expressed?": 'MATCH (d)-[r:off_label_use]->(di:disease {name:"epilepsy"}) WHERE toFloat(d.clogp) < 0 RETURN d',
    "What are the phenotypes associated with cardioacrofacial dysplasia?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "What are the symptoms of cardioacrofacial dysplasia?": 'MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}) MATCH (d)-[:phenotype_present]-({name:"neutropenia"}) MATCH (d)-[:phenotype_present]-({name:"high forehead"}) RETURN DISTINCT d.name',
    "What pathways are involved in distal arthrogryposis?": 'MATCH (d:disease {name: "psoriasis"})-[:indication]->(drug:drug)\nRETURN drug.name AS Treatment',
    "What pathways are involved in Breast cancer?": 'MATCH (g:gene_or_protein {name: "adora1"})-[:target]->(d:drug)\nRETURN COUNT(g) > 0',
    "What genes play role in ovarian cancer?": 'MATCH (d:drug)\nWHERE d.aggregate_state = "gas"\nRETURN COUNT(d) AS NumberOfGaseousDrugs',
    "Which medications are currently available for treating Parkinson's disease?": 'MATCH (d:disease {name: "sialuria"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "What are the likely environmental causes of chronic kidney disease?": 'MATCH (d:disease {name: "parkinson disease"})-[:indication]->(drug:drug) RETURN drug.name',
    "What are the likely genetic causes of chronic kidney disease?": 'MATCH (d:drug)-[]-(a:approval_status {name:"vet_approved"}) MATCH (d)-[]-(di:disease {name:"influenza"}) RETURN d',
    "What are the likely causes of chronic kidney disease?": 'MATCH (e:exposure {name: "asbestos"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name',
    "What are symptoms of rheumatoid arthritis?": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
    "Which biomarkers are associated with a diagnosis of Alport syndrome?": 'MATCH (d:disease {name: "ovarian cancer"})-[:associated_with]->(g:gene_or_protein) RETURN g.name',
    "How many biomarkers are associated with a diagnosis of Hereditary Nephritis?": 'MATCH (protein:gene_or_protein)-[:ppi]->(targetProtein:gene_or_protein {name: "pink1"})\nRETURN protein.name AS ProteinName',
    "Are there any genes associated with an increased risk of psoriasis?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "Is cor pulmonale linked to Chronic congestive heart failure?": 'MATCH (e:gene_or_protein {name:"f2"})-[ep:expression_present]-(a:anatomy) RETURN a.name',
    "Is ADORA1 druggable?": 'MATCH (e:exposure {name: "asbestos"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name',
    "When the Lenvatinib is expected to be used?": 'MATCH (d:drug)-[]-(a:approval_status {name:"vet_approved"}) MATCH (d)-[]-(di:disease {name:"influenza"}) RETURN d',
    "What genes play role in development of sialuria?": 'MATCH (e:exposure {name: "asbestos"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name',
    "What genes are promising targets for cardiomyopathy?": "MATCH (d:disease)-[:indication]->(drug:drug)\nWHERE NOT EXISTS {MATCH (drug)-[:side_effect]->(:effect_or_phenotype)}\nRETURN d.name AS DiseaseName",
    "What are the side effects of spironolactone?": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(p:effect_or_phenotype) RETURN p.name',
    "What genes play role in immunodeficiency?": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(e:effect_or_phenotype) RETURN e.name',
    "Is PPARG a known target for heart failure?": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
    "What is cardiomyopathy?": 'MATCH (dr:drug)-[:contraindication]-(d:disease {name:"hypertension"}) MATCH (dr)-[:indication]-(d2:disease {name:"epilepsy"}) RETURN dr ',
    "What genes play role in prostate cancer?": 'MATCH (d:drug)-[]-(a:approval_status {name:"experimental"}) MATCH (d)-[:indication]-(di:disease {name:"epilepsy"}) RETURN COUNT(distinct d) as number',
    "what are the symptoms of atopic dermatitis?": 'MATCH (alcoholDisease:disease)-[:linked_to]->(alcoholExposure:exposure {name: "ethanol"})\nMATCH (contraindicatedDrug:drug)-[:contraindication]->(alcoholDisease)\nMATCH (treatedDisease:disease)-[:indication]->(contraindicatedDrug)\nRETURN treatedDisease.name AS DiseaseName, contraindicatedDrug.name AS DrugName',
    "Which biomarkers are associated with a diagnosis of atopic dermatitis?": "MATCH (d:disease)-[:indication]->(drug:drug)\nWHERE NOT EXISTS {MATCH (drug)-[:side_effect]->(:effect_or_phenotype)}\nRETURN d.name AS DiseaseName",
    "What are the common treatments for moderate to severe psoriasis?": 'MATCH (d:disease {name: "epilepsy"})-[:indication]->(drug:drug)\nWHERE NOT (drug)-[:side_effect]->(:effect_or_phenotype)\nRETURN DISTINCT drug.name AS DrugName',
    "Which proteins interact with PINK1?": 'MATCH (d:disease {name: "alzheimer disease"})-[:indication]->(drug:drug)-[:target]->(target:gene_or_protein)\nRETURN drug.name AS DrugName, target.name AS TargetName',
    "Is right ventricle heart failure linked to Rheumatic tricuspid valve regurgitation?": 'MATCH (g:disease {name: "rheumatoid arthritis"})-[]-(p:gene_or_protein) RETURN COUNT(DISTINCT p)',
    "What genes play role in Alport syndrome?": 'MATCH (d:disease {name: "psoriasis"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Which drugs are used to treat Alzheimer and what do they act on?": 'MATCH (d)-[r:off_label_use]->(di:disease {name:"epilepsy"}) WHERE toFloat(d.clogp) < 0 RETURN d',
    "Based on the topological polar surface area, can any of the medications against atopic dermatitis permate cell membranes?": 'MATCH (d:disease {name: "nephritis, hereditary"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.name AS GeneName',
    "Which diseases are treated by drugs that have a counter indication against diseases caused by alcohol?": 'MATCH (d:disease {name: "chronic kidney disease"})-[:associated_with]->(g:gene_or_protein)\nRETURN g.id AS GeneID, g.name AS GeneName',
    "Which diseases are treated by drugs that have a counter indication against diseases caused by alcohol? Include the drug for each treated disease.": 'MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}) MATCH (d)-[:phenotype_present]-({name:"neutropenia"}) MATCH (d)-[:phenotype_present]-({name:"high forehead"}) RETURN DISTINCT d.name',
    "Which drugs used to treat diseases caused by alcohol may worsen the patient's Anaemia disease?": 'MATCH (d:disease {name: "dermatitis, atopic"})-[:phenotype_present]->(p:effect_or_phenotype)\nRETURN p.name AS Symptoms',
}


class AdversarialCypherQueryGenerationChain(Chain):
    """
    Class only used for evaluation of adversarial attacks. It simply returns a wrong cypher query given a natural
    language question. Should downstream test the verbalization of it.
    """

    input_key: str = "question"  #: :meta private:
    output_key: str = "cypher_query"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["cypher_query"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        query = self._generate_cypher(inputs)
        return self._prepare_chain_result(inputs=inputs, generated_cypher=query)

    def _prepare_chain_result(self, inputs: Dict[str, Any], generated_cypher: str) -> Dict[str, Any]:
        chain_result = {self.output_key: generated_cypher}

        intermediate_steps = inputs.get("intermediate_steps", [])

        intermediate_steps += [
            {"question": inputs[self.input_key]},
            {self.output_key: generated_cypher},
            {f"{self.__class__.__name__}_filled_prompt": "n/a"},
        ]
        chain_result["intermediate_steps"] = intermediate_steps
        return chain_result

    def _generate_cypher(self, inputs: Dict[str, Any]) -> str:
        return QUESTION_WRONG_CYPHER_MAPPING[inputs["question"]]


class AdversarialAttackGraphQAChain(GraphQAChain):

    def _build_cypher_query_generation_chain(self, config: GraphQAChainConfig) -> AdversarialCypherQueryGenerationChain:
        return AdversarialCypherQueryGenerationChain()


class AdversarialAttackEvaluation:

    def __init__(self, chain: AdversarialAttackGraphQAChain):
        self.__chain = chain

    def evaluate(self): ...


def build_chain(args: List[str] = []) -> Chain:
    parsed_args = _parse_primekg_args(args)
    graph = build_neo4j_graph()
    cypher_preprocessors = _build_preprocessors(graph, parsed_args.normalized_graph)
    cypher_prompt, answer_generation_prompt = _get_graph_prompt_templates()
    config = GraphQAChainConfig(
        llm=load_chat_model(),
        graph=graph,
        cypher_prompt=cypher_prompt,
        answer_generation_prompt=answer_generation_prompt,
        cypher_query_preprocessors=cypher_preprocessors,
        predicate_descriptions=PREDICATE_DESCRIPTIONS[:10],
        return_intermediate_steps=True,
        use_entity_detection_preprocessing=parsed_args.use_entity_detection_preprocessing,
        entity_detection_preprocessor_type=partial(FilteredPrimeKGQuestionPreprocessingChain, graph=graph),
        entity_detector=EntityDetector() if parsed_args.use_entity_detection_preprocessing else None,
        allowed_types_and_description_templates=_get_primekg_entity_categories(),
        use_subgraph_expansion=parsed_args.use_subgraph_expansion,
        combine_output_with_sematic_scholar=False,
        semantic_scholar_keyword_prompt=KEYWORD_PROMPT,
        combined_answer_generation_prompt=COMBINED_QA_PROMPT,
    )

    return AdversarialAttackGraphQAChain(config=config)


def eval_samples(limit_of_samples: int = None):
    eval_samples = []
    for manual_sample in manual_samples[:limit_of_samples]:
        eval_sample = EvaluationSample(
            question=manual_sample["question"],
            cypher_query=manual_sample["expected_cypher"],
            expected_answer=manual_sample["expected_answer"],
            nodes=manual_sample["nodes"],
        )
        eval_samples.append(eval_sample)
    return eval_samples


if __name__ == "__main__":
    results = []
    chain = build_chain(args=["--normalized_graph", "--use_entity_detection_preprocessing"])
    # chain = build_chain(args=["--normalized_graph"])
    samples = eval_samples()

    for sample in tqdm.tqdm(samples):
        inputs = {"question": sample.question}
        result = chain.invoke(inputs)
        results.append(
            {
                "question": sample.question,
                "answer": result["graph_qa_output"].answer,
                "graph_response": len(result["graph_qa_output"].graph_response) > 0,
            }
        )

    with open("adv_result.json", "w", encoding="utf8") as w:
        json.dump(results, w, indent=4, ensure_ascii=False)
