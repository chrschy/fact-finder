import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from fact_finder.utils import build_neo4j_graph, load_chat_model


class EvalSampleAddition:
    """
    Class to add manually selected evaluation samples in our EvaluationSample format to a json file.
    TODO: the expected answers are rather random as of now. They may not correlate with whatever a LLM would verbalize
    """

    def __init__(self, graph: Neo4jGraph, subgraph_extractor: LLMSubGraphExtractor, path_to_json: Path):
        self._subgraph_extractor = subgraph_extractor
        self._graph = graph
        self._path_to_json = path_to_json

    def add_to_evaluation_sample_json(
        self,
        question: str,
        expected_cypher: str,
        source: str,
        expected_answer: str,
        is_answerable: Optional[bool] = True,
    ):

        subgraph_query = self._subgraph_extractor(expected_cypher)
        sub_graph = self._graph.query(query=subgraph_query)

        sample = EvaluationSample(
            question=question,
            cypher_query=expected_cypher,
            sub_graph=sub_graph,
            question_is_answerable=is_answerable,
            source=source,
            expected_answer=expected_answer,
        )
        self._persist(sample=sample)

    def _persist(self, sample: EvaluationSample):
        with open(self._path_to_json, "r", encoding="utf8") as r:
            json_content = json.load(r)
            evaluation_samples = [EvaluationSample.parse_obj(x) for x in json_content]
        if sample.question not in [x.question for x in evaluation_samples]:
            evaluation_samples.append(sample)
        with open(self._path_to_json, "w", encoding="utf8") as w:
            json.dump([x.dict() for x in evaluation_samples], w, indent=4)


def roman_script():
    load_dotenv()
    eval_sample_addition = EvalSampleAddition(
        graph=build_neo4j_graph(),
        subgraph_extractor=LLMSubGraphExtractor(
            model=load_chat_model(),
        ),
        path_to_json=Path("src/fact_finder/evaluator/evaluation_samples.json"),
    )

    manual_samples = [
        {
            "question": "Which drugs have pterygium as side effect?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
            "expected_answer": "brinzolamide",
        },
        {
            "question": "Which medicine may cause pterygium?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
            "expected_answer": "brinzolamide",
        },
        {
            "question": "Which drugs have freckling as side effect?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
            "expected_answer": "cytarabine, methoxsalen",
        },
        {
            "question": "Which medicine may cause freckling?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
            "expected_answer": "cytarabine, methoxsalen",
        },
        {
            "question": "Which drugs are used to treat ocular hypertension?",
            "expected_cypher": 'MATCH (n:drug)-[r]-(d:disease {name:"ocular hypertension"}) RETURN n',
            "expected_answer": "108 drugs in total",
        },
        {
            "question": "Which drugs to treat ocular hypertension may cause the loss of eyelashes?",
            "expected_cypher": 'MATCH (n:drug)-[r]-(d:disease {name:"ocular hypertension"}) MATCH (n)-[s:side_effect]-(e:effect_or_phenotype {name:"loss of eyelashes"}) RETURN n',
            "expected_answer": "brinzolamide, bimatoprost, travoprost, paclitaxel, tobramycin",
        },
        {
            "question": "Which disease does not show in cleft upper lip?",
            "expected_cypher": 'MATCH (d)-[:phenotype_absent]-(p:effect_or_phenotype {name:"cleft upper lip"}) RETURN d',
            "expected_answer": "ectrodactyly-cleft palate syndrome",
        },
        {
            "question": "Which diseases show in symptoms such as eczema, neutropenia and a high forehead?",
            "expected_cypher": 'MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}), MATCH (d)-[:phenotype_present]-({name:"neutropenia"}), MATCH (d)-[:phenotype_present]-({name:"high forehead"}), RETURN DISTINCT d.name',
            "expected_answer": "leigh syndrome, x-linked intellectual disability",
        },
        {
            "question": "Which expressions are present for the gene/protein f2?",
            "expected_cypher": 'MATCH (e:gene_or_protein {name:"f2"})-[ep:expression_present]-(a:anatomy) RETURN a.name',
            "expected_answer": "kidney, liver, fundus of stomach, stomach, multi-cellular organism, anatomical system, material anatomical entity, intestine, large intestine",
        },
        {
            "question": "Which drugs are associated with epilepsy?",
            "expected_cypher": 'MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = "epilepsy" RETURN d.name',
            "expected_answer": "The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam.",
        },
    ]

    for sample in manual_samples:

        eval_sample_addition.add_to_evaluation_sample_json(
            question=sample["question"],
            expected_cypher=sample["expected_cypher"],
            source="manual",
            expected_answer=sample["expected_answer"],
            is_answerable=True,
        )


if __name__ == "__main__":
    load_dotenv()
    manual_samples = [
        {
            "question": "In which Anatomical regions is IRAK4 expressed?",
            "expected_answer": "omental fat pad, cerebral cortex, adipose tissue of the abdominal region, layer, squamous epithelium, nasal cavity epithelium, mammary gland, oviduct, fallopian tube, oral cavity, esophagus, decidua, muscle, connective tissue, tonsil, bone marrow, thymus gland, adrenal gland, prostate gland, peritoneum, corpus callosum, spinal cord, subcutaneous adipose tissue, bronchus, cerebellar cortex, jejunum, duodenum, kidney, small intestine, liver, spleen, heart left ventricle, cardiac ventricle, cardiac atrium, lung, thyroid gland, substantia nigra, cerebellum, placenta, Ammon's horn, neocortex, hypothalamus, telencephalon, mesencephalon, forebrain, temporal lobe",
            "cypher_query": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
        },
        {
            "question": "What are the phenotypes associated with cardioacrofacial dysplasia?",
            "expected_answer": "mandibular prognathia, postaxial foot polydactyly, clinodactyly of the 5th finger, clubbing, hypodontia, left superior vena cava draining to coronary sinus, common atrium, recurrent patellar dislocation, postaxial hand polydactyly, narrow chest, broad forehead, prominent nasal tip, deep philtrum, tented upper lip vermilion, conical tooth, diastema, autosomal dominant inheritance, short philtrum, genu valgum, long thorax, atrioventricular canal defect, hypoplasia of the maxilla, postaxial polydactyly, accessory oral frenulum, overhanging nasal tip, midface retrusion, nail dysplasia, long face, complete atrioventricular canal defect, congenital onset, limb undergrowth",
            "cypher_query": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(p:effect_or_phenotype) RETURN p.name',
        },
        {
            "question": "What are the symptoms of cardioacrofacial dysplasia?",
            "expected_answer": "mandibular prognathia, postaxial foot polydactyly, clinodactyly of the 5th finger, clubbing, hypodontia, left superior vena cava draining to coronary sinus, common atrium, recurrent patellar dislocation, postaxial hand polydactyly, narrow chest, broad forehead, prominent nasal tip, deep philtrum, tented upper lip vermilion, conical tooth, diastema, autosomal dominant inheritance, short philtrum, genu valgum, long thorax, atrioventricular canal defect, hypoplasia of the maxilla, postaxial polydactyly, accessory oral frenulum, overhanging nasal tip, midface retrusion, nail dysplasia, long face, complete atrioventricular canal defect, congenital onset, limb undergrowth",
            "cypher_query": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(e:effect_or_phenotype) RETURN e.name',
        },
        {
            "question": "What pathways are involved in distal arthrogryposis?",
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
        {
            "question": ...,
            "expected_answer": ...,
            "cypher_query": ...,
        },
    ]
