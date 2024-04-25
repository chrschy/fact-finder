import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import tqdm
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
        nodes: List[Dict[str, Any]],
        is_answerable: Optional[bool] = True,
        subgraph_query: Optional[
            str
        ] = None,  # sometimes the subgraph query generation fails; we can set it manually then
    ):

        # TODO Note that running the subgraph query on my laptop crashes due to not enough RAM
        # subgraph_query = subgraph_query if subgraph_query else self._subgraph_extractor(expected_cypher)
        # try:
        #     sub_graph = self._graph.query(query=subgraph_query)
        # except:
        #     print(question)
        #     print(expected_cypher)
        #     print(subgraph_query)
        #     raise
        sub_graph = []
        sample = EvaluationSample(
            question=question,
            cypher_query=expected_cypher,
            sub_graph=sub_graph,
            question_is_answerable=is_answerable,
            source=source,
            expected_answer=expected_answer,
            nodes=nodes,
        )
        self._persist(sample=sample)

    def _persist(self, sample: EvaluationSample):
        with open(self._path_to_json, "r", encoding="utf8") as r:
            json_content = json.load(r)
            evaluation_samples = [EvaluationSample.model_validate(r) for r in json_content]
        evaluation_samples = [r for r in evaluation_samples if not r.question == sample.question]
        evaluation_samples.append(sample)
        with open(self._path_to_json, "w", encoding="utf8") as w:
            json.dump([r.model_dump() for r in evaluation_samples], w, indent=4)


if __name__ == "__main__":
    load_dotenv()
    manual_samples = [
        {
            "question": "Which medications have more off-label uses than approved indications?",
            "expected_cypher": "MATCH (d:disease)-[:off_label_use]->(dr:drug)\nWITH dr, COUNT(d) AS offLabelCount\nMATCH (d:disease)-[:indication]->(dr)\nWITH dr, offLabelCount, COUNT(d) AS indicationCount\nWHERE offLabelCount>indicationCount\nRETURN dr.name AS Drug, offLabelCount AS OffLabelUses, indicationCount AS ApprovedIndications",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 14016,
                    "name": "Medrysone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14017,
                    "name": "Fluorometholone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14019,
                    "name": "Betamethasone",
                    "off_label_use_count": 31,
                    "indication_count": 0
                },
                {
                    "index": 14023,
                    "name": "Triamcinolone",
                    "off_label_use_count": 30,
                    "indication_count": 0
                },
                {
                    "index": 14024,
                    "name": "Prednisone",
                    "off_label_use_count": 31,
                    "indication_count": 0
                },
                {
                    "index": 14025,
                    "name": "Mitotane",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14028,
                    "name": "Hydrocortisone",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14030,
                    "name": "Prednisolone",
                    "off_label_use_count": 39,
                    "indication_count": 0
                },
                {
                    "index": 14042,
                    "name": "Hydrocortisone acetate",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14052,
                    "name": "Vitamin A",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14061,
                    "name": "Diclofenac",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14062,
                    "name": "Diflunisal",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14106,
                    "name": "Hydroxocobalamin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14112,
                    "name": "Etretinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14115,
                    "name": "Bismuth subsalicylate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14127,
                    "name": "Octreotide",
                    "off_label_use_count": 18,
                    "indication_count": 3
                },
                {
                    "index": 14128,
                    "name": "Ascorbic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14131,
                    "name": "Icosapent",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14135,
                    "name": "Lovastatin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14140,
                    "name": "Ziprasidone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14141,
                    "name": "Phenytoin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14142,
                    "name": "Metoprolol",
                    "off_label_use_count": 15,
                    "indication_count": 7
                },
                {
                    "index": 14144,
                    "name": "Topiramate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14150,
                    "name": "Morphine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14151,
                    "name": "Desogestrel",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14152,
                    "name": "Chlorthalidone",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14153,
                    "name": "Valproic acid",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14154,
                    "name": "Acetaminophen",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14156,
                    "name": "Amitriptyline",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14158,
                    "name": "Indomethacin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14159,
                    "name": "Ipratropium",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14162,
                    "name": "Atenolol",
                    "off_label_use_count": 10,
                    "indication_count": 6
                },
                {
                    "index": 14163,
                    "name": "Diltiazem",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14164,
                    "name": "Alprazolam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14167,
                    "name": "Ampicillin",
                    "off_label_use_count": 16,
                    "indication_count": 0
                },
                {
                    "index": 14168,
                    "name": "Spironolactone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14173,
                    "name": "Imipramine",
                    "off_label_use_count": 11,
                    "indication_count": 4
                },
                {
                    "index": 14174,
                    "name": "Acitretin",
                    "off_label_use_count": 13,
                    "indication_count": 0
                },
                {
                    "index": 14175,
                    "name": "Nabumetone",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14176,
                    "name": "Fluoxetine",
                    "off_label_use_count": 10,
                    "indication_count": 9
                },
                {
                    "index": 14182,
                    "name": "Oxycodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14183,
                    "name": "Tolmetin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14184,
                    "name": "Ritonavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14185,
                    "name": "Vancomycin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14188,
                    "name": "Ciprofloxacin",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14189,
                    "name": "Nortriptyline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14190,
                    "name": "Fluorouracil",
                    "off_label_use_count": 20,
                    "indication_count": 19
                },
                {
                    "index": 14191,
                    "name": "Piroxicam",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14192,
                    "name": "Lamotrigine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14194,
                    "name": "Methotrexate",
                    "off_label_use_count": 27,
                    "indication_count": 0
                },
                {
                    "index": 14196,
                    "name": "Propranolol",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14197,
                    "name": "Fenoprofen",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14198,
                    "name": "Clonidine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14201,
                    "name": "Oxytetracycline",
                    "off_label_use_count": 12,
                    "indication_count": 0
                },
                {
                    "index": 14203,
                    "name": "Medroxyprogesterone acetate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14206,
                    "name": "Chloroquine",
                    "off_label_use_count": 13,
                    "indication_count": 11
                },
                {
                    "index": 14208,
                    "name": "Testosterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14211,
                    "name": "Estrone",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14212,
                    "name": "Verapamil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14213,
                    "name": "Tamoxifen",
                    "off_label_use_count": 25,
                    "indication_count": 3
                },
                {
                    "index": 14215,
                    "name": "Losartan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14216,
                    "name": "Warfarin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14217,
                    "name": "Furosemide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14221,
                    "name": "Norethisterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14223,
                    "name": "Risperidone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14227,
                    "name": "Etodolac",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14229,
                    "name": "Tretinoin",
                    "off_label_use_count": 11,
                    "indication_count": 4
                },
                {
                    "index": 14230,
                    "name": "Tetracycline",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 14231,
                    "name": "Irinotecan",
                    "off_label_use_count": 7,
                    "indication_count": 2
                },
                {
                    "index": 14235,
                    "name": "Estradiol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14236,
                    "name": "Mefenamic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14237,
                    "name": "Acyclovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14242,
                    "name": "Meloxicam",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14245,
                    "name": "Diazepam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14247,
                    "name": "Clofazimine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14250,
                    "name": "Terbinafine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14253,
                    "name": "Chlorhexidine",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14254,
                    "name": "Emtricitabine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14255,
                    "name": "Quinapril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14257,
                    "name": "Etacrynic acid",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14260,
                    "name": "Cyclobenzaprine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14262,
                    "name": "Salicylic acid",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 14266,
                    "name": "Fexofenadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14267,
                    "name": "Isoniazid",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14268,
                    "name": "Norgestimate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14269,
                    "name": "Methylprednisolone",
                    "off_label_use_count": 34,
                    "indication_count": 0
                },
                {
                    "index": 14270,
                    "name": "Ethinylestradiol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14271,
                    "name": "Isotretinoin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14273,
                    "name": "Azathioprine",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 14274,
                    "name": "Auranofin",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14275,
                    "name": "Doxorubicin",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 14276,
                    "name": "Hydrochlorothiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14277,
                    "name": "Letrozole",
                    "off_label_use_count": 21,
                    "indication_count": 0
                },
                {
                    "index": 14278,
                    "name": "Ketoprofen",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14279,
                    "name": "Sulfamethoxazole",
                    "off_label_use_count": 21,
                    "indication_count": 9
                },
                {
                    "index": 14281,
                    "name": "Ketoconazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14282,
                    "name": "Irbesartan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14285,
                    "name": "Gatifloxacin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14286,
                    "name": "Rifampicin",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 14288,
                    "name": "Benzylpenicillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 14291,
                    "name": "Amoxicillin",
                    "off_label_use_count": 14,
                    "indication_count": 12
                },
                {
                    "index": 14294,
                    "name": "Clonazepam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14299,
                    "name": "Sertraline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14300,
                    "name": "Miconazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14301,
                    "name": "Nifedipine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14302,
                    "name": "Amiodarone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14308,
                    "name": "Carvedilol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14309,
                    "name": "Levofloxacin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14311,
                    "name": "Cloxacillin",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 14313,
                    "name": "Arsenic trioxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14317,
                    "name": "Captopril",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14318,
                    "name": "Ceftriaxone",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14320,
                    "name": "Dexamethasone",
                    "off_label_use_count": 33,
                    "indication_count": 0
                },
                {
                    "index": 14321,
                    "name": "Levodopa",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14324,
                    "name": "Gemfibrozil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14325,
                    "name": "Clomipramine",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14326,
                    "name": "Darunavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14331,
                    "name": "Polythiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14335,
                    "name": "Cefotetan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14340,
                    "name": "Magnesium salicylate",
                    "off_label_use_count": 10,
                    "indication_count": 8
                },
                {
                    "index": 14346,
                    "name": "Lopinavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14347,
                    "name": "Hydroxychloroquine",
                    "off_label_use_count": 13,
                    "indication_count": 11
                },
                {
                    "index": 14369,
                    "name": "Nebivolol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14374,
                    "name": "Iodine",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14378,
                    "name": "Rufinamide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14381,
                    "name": "Levocetirizine",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14387,
                    "name": "Methyltestosterone",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14415,
                    "name": "Chlortetracycline",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14466,
                    "name": "Favipiravir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14483,
                    "name": "Estradiol valerate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14484,
                    "name": "Tenofovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14490,
                    "name": "Zinc chloride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14509,
                    "name": "Meclofenamic acid",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14510,
                    "name": "Heparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14511,
                    "name": "Dantrolene",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 14527,
                    "name": "Fluoxymesterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14528,
                    "name": "Danazol",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 14530,
                    "name": "Stanolone",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14538,
                    "name": "Disopyramide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14539,
                    "name": "Prazosin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14542,
                    "name": "Maprotiline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14543,
                    "name": "Desipramine",
                    "off_label_use_count": 8,
                    "indication_count": 3
                },
                {
                    "index": 14544,
                    "name": "Bupropion",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14545,
                    "name": "Clindamycin",
                    "off_label_use_count": 17,
                    "indication_count": 16
                },
                {
                    "index": 14559,
                    "name": "Calcitriol",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14560,
                    "name": "Ergocalciferol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14561,
                    "name": "Cholecalciferol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14579,
                    "name": "Nitazoxanide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14582,
                    "name": "Nadolol",
                    "off_label_use_count": 8,
                    "indication_count": 2
                },
                {
                    "index": 14589,
                    "name": "Polyethylene glycol 400",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14597,
                    "name": "Dapsone",
                    "off_label_use_count": 14,
                    "indication_count": 3
                },
                {
                    "index": 14603,
                    "name": "Chlorpropamide",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14604,
                    "name": "Etoposide",
                    "off_label_use_count": 59,
                    "indication_count": 8
                },
                {
                    "index": 14605,
                    "name": "Candesartan cilexetil",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14606,
                    "name": "Thalidomide",
                    "off_label_use_count": 7,
                    "indication_count": 1
                },
                {
                    "index": 14608,
                    "name": "Ifosfamide",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14610,
                    "name": "Ketamine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14628,
                    "name": "Tioguanine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14635,
                    "name": "Nitroglycerin",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14662,
                    "name": "Ribavirin",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14668,
                    "name": "Bumetanide",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14670,
                    "name": "Drospirenone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14678,
                    "name": "Folic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14685,
                    "name": "Progesterone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14686,
                    "name": "Clomifene",
                    "off_label_use_count": 14,
                    "indication_count": 1
                },
                {
                    "index": 14691,
                    "name": "Amiloride",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14696,
                    "name": "Pentamidine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14698,
                    "name": "Pyrantel",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14703,
                    "name": "Methantheline",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14704,
                    "name": "Calcifediol",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14706,
                    "name": "Cyanocobalamin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14709,
                    "name": "Gabapentin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14715,
                    "name": "Coenzyme M",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14719,
                    "name": "Epirubicin",
                    "off_label_use_count": 21,
                    "indication_count": 2
                },
                {
                    "index": 14729,
                    "name": "Cimetidine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14730,
                    "name": "Metyrapone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14733,
                    "name": "Levomefolic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14736,
                    "name": "Oxaliplatin",
                    "off_label_use_count": 12,
                    "indication_count": 1
                },
                {
                    "index": 14737,
                    "name": "Carboplatin",
                    "off_label_use_count": 42,
                    "indication_count": 21
                },
                {
                    "index": 14740,
                    "name": "Chlorambucil",
                    "off_label_use_count": 10,
                    "indication_count": 9
                },
                {
                    "index": 14751,
                    "name": "Zinc sulfate",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14758,
                    "name": "Capecitabine",
                    "off_label_use_count": 22,
                    "indication_count": 6
                },
                {
                    "index": 14764,
                    "name": "Mercaptopurine",
                    "off_label_use_count": 23,
                    "indication_count": 5
                },
                {
                    "index": 14768,
                    "name": "Dalteparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14770,
                    "name": "Bivalirudin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14772,
                    "name": "Aminosalicylic acid",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14774,
                    "name": "Enoxaparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14775,
                    "name": "Calcipotriol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14780,
                    "name": "Cytarabine",
                    "off_label_use_count": 13,
                    "indication_count": 0
                },
                {
                    "index": 14782,
                    "name": "Biotin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14783,
                    "name": "Caffeine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14789,
                    "name": "Daunorubicin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14790,
                    "name": "Primaquine",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14791,
                    "name": "Mitoxantrone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14792,
                    "name": "Paclitaxel",
                    "off_label_use_count": 24,
                    "indication_count": 0
                },
                {
                    "index": 14793,
                    "name": "Docetaxel",
                    "off_label_use_count": 25,
                    "indication_count": 14
                },
                {
                    "index": 14797,
                    "name": "Imipenem",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14807,
                    "name": "Epinephrine",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 14819,
                    "name": "Econazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14828,
                    "name": "Cycloserine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14830,
                    "name": "Amantadine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14833,
                    "name": "Riboflavin",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14835,
                    "name": "Benazepril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14836,
                    "name": "Ramipril",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14843,
                    "name": "Nizatidine",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14846,
                    "name": "Perindopril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14848,
                    "name": "Terbutaline",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14851,
                    "name": "Cyclopentolate",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14880,
                    "name": "Nicotine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14881,
                    "name": "Phenylephrine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14884,
                    "name": "Phenelzine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14898,
                    "name": "Phenylpropanolamine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14911,
                    "name": "Allopurinol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14914,
                    "name": "Procarbazine",
                    "off_label_use_count": 45,
                    "indication_count": 5
                },
                {
                    "index": 14918,
                    "name": "Adenosine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14934,
                    "name": "Cyclosporine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14935,
                    "name": "Fluconazole",
                    "off_label_use_count": 15,
                    "indication_count": 7
                },
                {
                    "index": 14936,
                    "name": "Erythromycin",
                    "off_label_use_count": 30,
                    "indication_count": 22
                },
                {
                    "index": 14940,
                    "name": "Lidocaine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14942,
                    "name": "Terfenadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14943,
                    "name": "Levonorgestrel",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 14948,
                    "name": "Teniposide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14949,
                    "name": "Chloramphenicol",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14950,
                    "name": "Loratadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14951,
                    "name": "Quinine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14952,
                    "name": "Haloperidol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14955,
                    "name": "Vincristine",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 14956,
                    "name": "Carbamazepine",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14958,
                    "name": "Cisapride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14959,
                    "name": "Nicardipine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14960,
                    "name": "Astemizole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14962,
                    "name": "Trazodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14963,
                    "name": "Midazolam",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14975,
                    "name": "Sirolimus",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14977,
                    "name": "Ondansetron",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14980,
                    "name": "Metronidazole",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14984,
                    "name": "Felodipine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14987,
                    "name": "Fluvastatin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14989,
                    "name": "Quinacrine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14992,
                    "name": "Itraconazole",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14993,
                    "name": "Phenobarbital",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14995,
                    "name": "Clarithromycin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14997,
                    "name": "Anastrozole",
                    "off_label_use_count": 21,
                    "indication_count": 0
                },
                {
                    "index": 15001,
                    "name": "Paliperidone",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15003,
                    "name": "Cortisone acetate",
                    "off_label_use_count": 39,
                    "indication_count": 0
                },
                {
                    "index": 15026,
                    "name": "Hydroxyprogesterone caproate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15034,
                    "name": "Cobicistat",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15059,
                    "name": "Troleandomycin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 15061,
                    "name": "Umifenovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15073,
                    "name": "Fluvoxamine",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15076,
                    "name": "Esmolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15078,
                    "name": "Tramadol",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15082,
                    "name": "Citalopram",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15084,
                    "name": "Clotrimazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15087,
                    "name": "Venlafaxine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15089,
                    "name": "Codeine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15092,
                    "name": "Vinorelbine",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 15093,
                    "name": "Clozapine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15094,
                    "name": "Mirtazapine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15095,
                    "name": "Timolol",
                    "off_label_use_count": 7,
                    "indication_count": 4
                },
                {
                    "index": 15107,
                    "name": "Vinblastine",
                    "off_label_use_count": 10,
                    "indication_count": 0
                },
                {
                    "index": 15113,
                    "name": "Galantamine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15116,
                    "name": "Paroxetine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15117,
                    "name": "Trimipramine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15121,
                    "name": "Methimazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15126,
                    "name": "Donepezil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15134,
                    "name": "Hydroxyurea",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15145,
                    "name": "Proguanil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15146,
                    "name": "Nefazodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15148,
                    "name": "Escitalopram",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15153,
                    "name": "Acebutolol",
                    "off_label_use_count": 9,
                    "indication_count": 2
                },
                {
                    "index": 15156,
                    "name": "Metipranolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15159,
                    "name": "Lisdexamfetamine",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15164,
                    "name": "Yohimbine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15167,
                    "name": "Antipyrine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15179,
                    "name": "Nicotinamide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15180,
                    "name": "Fusidic acid",
                    "off_label_use_count": 25,
                    "indication_count": 0
                },
                {
                    "index": 15211,
                    "name": "Desvenlafaxine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15218,
                    "name": "Phenylbutyric acid",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15267,
                    "name": "Remdesivir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15271,
                    "name": "Mitomycin",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 15289,
                    "name": "Moxifloxacin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15295,
                    "name": "Triamterene",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15298,
                    "name": "Streptozocin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15299,
                    "name": "Trimethoprim",
                    "off_label_use_count": 22,
                    "indication_count": 7
                },
                {
                    "index": 15300,
                    "name": "Enoxacin",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15303,
                    "name": "Methoxsalen",
                    "off_label_use_count": 7,
                    "indication_count": 3
                },
                {
                    "index": 15307,
                    "name": "Thiabendazole",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15311,
                    "name": "Primidone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15312,
                    "name": "Pentoxifylline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15322,
                    "name": "Norfloxacin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 15326,
                    "name": "Ofloxacin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 15363,
                    "name": "Bendamustine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15413,
                    "name": "Gemcitabine",
                    "off_label_use_count": 16,
                    "indication_count": 0
                },
                {
                    "index": 15416,
                    "name": "Lorazepam",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 15419,
                    "name": "Azithromycin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 15420,
                    "name": "Pantoprazole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15425,
                    "name": "Doxycycline",
                    "off_label_use_count": 18,
                    "indication_count": 0
                },
                {
                    "index": 15426,
                    "name": "Isradipine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15430,
                    "name": "Pentobarbital",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15435,
                    "name": "Megestrol acetate",
                    "off_label_use_count": 21,
                    "indication_count": 4
                },
                {
                    "index": 15438,
                    "name": "Sulfadiazine",
                    "off_label_use_count": 12,
                    "indication_count": 10
                },
                {
                    "index": 15447,
                    "name": "Valdecoxib",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15451,
                    "name": "Bisoprolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15452,
                    "name": "Rifabutin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15456,
                    "name": "Amphotericin B",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 15461,
                    "name": "Imiquimod",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15465,
                    "name": "Phenylbutazone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15466,
                    "name": "Acetazolamide",
                    "off_label_use_count": 12,
                    "indication_count": 6
                },
                {
                    "index": 15467,
                    "name": "Ethynodiol diacetate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15479,
                    "name": "Salbutamol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15481,
                    "name": "Topotecan",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15485,
                    "name": "Atovaquone",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15493,
                    "name": "Bromocriptine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15496,
                    "name": "Rifaximin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15521,
                    "name": "Rutin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15549,
                    "name": "Dexlansoprazole",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 15563,
                    "name": "Tocilizumab",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15582,
                    "name": "Difluprednate",
                    "off_label_use_count": 11,
                    "indication_count": 0
                },
                {
                    "index": 15597,
                    "name": "Ruxolitinib",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15621,
                    "name": "Dienogest",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15646,
                    "name": "Oleandomycin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 15666,
                    "name": "Baricitinib",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15730,
                    "name": "Baloxavir marboxil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15761,
                    "name": "Oseltamivir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15762,
                    "name": "Trandolapril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 15763,
                    "name": "Benzocaine",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 15769,
                    "name": "Valsartan",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 15771,
                    "name": "Sulfisoxazole",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15777,
                    "name": "Oxandrolone",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15780,
                    "name": "Sulfapyridine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 15804,
                    "name": "Cladribine",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 15834,
                    "name": "Thiopental",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15836,
                    "name": "Telmisartan",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15856,
                    "name": "Naltrexone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15860,
                    "name": "Sodium aurothiomalate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15867,
                    "name": "Mebendazole",
                    "off_label_use_count": 9,
                    "indication_count": 3
                },
                {
                    "index": 15870,
                    "name": "Amifostine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15876,
                    "name": "Mesalazine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15877,
                    "name": "Cyproheptadine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15885,
                    "name": "Inositol nicotinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15886,
                    "name": "Sodium fluoride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15893,
                    "name": "Bleomycin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15913,
                    "name": "Citrulline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15914,
                    "name": "Minocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 15982,
                    "name": "Sulfasalazine",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 15984,
                    "name": "Oxaprozin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16036,
                    "name": "Acetylcysteine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16057,
                    "name": "Hyoscyamine",
                    "off_label_use_count": 6,
                    "indication_count": 1
                },
                {
                    "index": 16059,
                    "name": "Tridihexethyl",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16060,
                    "name": "Anisotropine methylbromide",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16063,
                    "name": "Scopolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16074,
                    "name": "Mepenzolate",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16087,
                    "name": "Proflavine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16210,
                    "name": "Eprosartan",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16217,
                    "name": "Serine",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 16219,
                    "name": "Alprostadil",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16221,
                    "name": "Misoprostol",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 16281,
                    "name": "Cysteine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16305,
                    "name": "Potassium nitrate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16313,
                    "name": "Propantheline",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 16347,
                    "name": "Ferrous fumarate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16350,
                    "name": "Citric acid",
                    "off_label_use_count": 5,
                    "indication_count": 2
                },
                {
                    "index": 16357,
                    "name": "Isometheptene",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16377,
                    "name": "Phenolphthalein",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16443,
                    "name": "Stanozolol",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16550,
                    "name": "Dipyridamole",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 16572,
                    "name": "Leuprolide",
                    "off_label_use_count": 21,
                    "indication_count": 5
                },
                {
                    "index": 16588,
                    "name": "Metformin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16635,
                    "name": "Fondaparinux",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 16672,
                    "name": "Baclofen",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16687,
                    "name": "Fosinopril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16689,
                    "name": "Moexipril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16690,
                    "name": "Lisinopril",
                    "off_label_use_count": 10,
                    "indication_count": 6
                },
                {
                    "index": 16788,
                    "name": "Methyclothiazide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16793,
                    "name": "Chlorothiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16802,
                    "name": "Aurothioglucose",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16825,
                    "name": "Ardeparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 16837,
                    "name": "Urea",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16842,
                    "name": "Pramipexole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16844,
                    "name": "Phentolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16847,
                    "name": "Apraclonidine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16855,
                    "name": "Methylphenidate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16867,
                    "name": "Demeclocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 17062,
                    "name": "Butalbital",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17124,
                    "name": "Phylloquinone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 17149,
                    "name": "Butorphanol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17173,
                    "name": "Physostigmine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 17174,
                    "name": "Rivastigmine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17215,
                    "name": "Salmon calcitonin",
                    "off_label_use_count": 7,
                    "indication_count": 4
                },
                {
                    "index": 17229,
                    "name": "Cetirizine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 17238,
                    "name": "Dimenhydrinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17273,
                    "name": "Amisulpride",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17305,
                    "name": "Protriptyline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17338,
                    "name": "Metaraminol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17376,
                    "name": "Cefotaxime",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 17411,
                    "name": "Naftifine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17524,
                    "name": "Carmustine",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 17727,
                    "name": "Pamidronic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17728,
                    "name": "Zoledronic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17739,
                    "name": "Dihydrotachysterol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17829,
                    "name": "Lithium citrate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17831,
                    "name": "Lithium carbonate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17886,
                    "name": "Tetracosactide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17999,
                    "name": "Flucytosine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18036,
                    "name": "Bacitracin",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 18160,
                    "name": "Emapalumab",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18179,
                    "name": "Sucralfate",
                    "off_label_use_count": 6,
                    "indication_count": 1
                },
                {
                    "index": 18312,
                    "name": "Phenoxymethylpenicillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 18313,
                    "name": "Oxacillin",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 18687,
                    "name": "Cilastatin",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 18755,
                    "name": "Cromoglicic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18884,
                    "name": "Lactose",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 19207,
                    "name": "Carbidopa",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 19250,
                    "name": "Pentostatin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 19863,
                    "name": "Anthralin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20129,
                    "name": "Chlordiazepoxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20146,
                    "name": "Magnesium carbonate",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 20148,
                    "name": "Synephrine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20156,
                    "name": "Dactinomycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20157,
                    "name": "Guaifenesin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20169,
                    "name": "Magnesium sulfate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20172,
                    "name": "Silicon dioxide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20203,
                    "name": "Ceftazidime",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 20206,
                    "name": "Cyclacillin",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 20212,
                    "name": "Melphalan",
                    "off_label_use_count": 10,
                    "indication_count": 0
                },
                {
                    "index": 20214,
                    "name": "Potassium bicarbonate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20218,
                    "name": "Sodium citrate",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 20239,
                    "name": "Penicillamine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20254,
                    "name": "Polyethylene glycol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20263,
                    "name": "Ganciclovir",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20279,
                    "name": "Tyramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20289,
                    "name": "Clavulanic acid",
                    "off_label_use_count": 11,
                    "indication_count": 10
                },
                {
                    "index": 20293,
                    "name": "Inositol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20296,
                    "name": "Polyvinyl alcohol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20298,
                    "name": "Potassium citrate",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 20302,
                    "name": "Potassium Iodide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20303,
                    "name": "Trolnitrate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20306,
                    "name": "Zinc gluconate",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 20313,
                    "name": "Polyethylene glycol 300",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20314,
                    "name": "Polyethylene glycol 3500",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20318,
                    "name": "Sulfacetamide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20320,
                    "name": "Cobalamin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20321,
                    "name": "Methscopolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20322,
                    "name": "Zinc oxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20324,
                    "name": "Triethylenetetramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20332,
                    "name": "Adomiparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20333,
                    "name": "Parnaparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20334,
                    "name": "Heparin, bovine",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20336,
                    "name": "Temozolomide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20337,
                    "name": "Monopotassium phosphate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20340,
                    "name": "Ethionamide",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 20351,
                    "name": "Meropenem",
                    "off_label_use_count": 12,
                    "indication_count": 7
                },
                {
                    "index": 20352,
                    "name": "Povidone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20353,
                    "name": "Povidone K30",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20376,
                    "name": "Magnesium hydroxide",
                    "off_label_use_count": 13,
                    "indication_count": 6
                },
                {
                    "index": 20397,
                    "name": "Pantothenic acid",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20400,
                    "name": "Dipotassium phosphate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20401,
                    "name": "Paromomycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20402,
                    "name": "Ethambutol",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 20403,
                    "name": "Almasilate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20404,
                    "name": "Sulbactam",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20406,
                    "name": "Meticillin",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 20407,
                    "name": "Kanamycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20410,
                    "name": "Mezlocillin",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 20411,
                    "name": "Bacampicillin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 20412,
                    "name": "Amikacin",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 20413,
                    "name": "Azlocillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 20415,
                    "name": "Ticarcillin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 20418,
                    "name": "Tobramycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20422,
                    "name": "Meclocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 20424,
                    "name": "Magnesium trisilicate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20427,
                    "name": "Cholestyramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20429,
                    "name": "Colistin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20431,
                    "name": "Diiodohydroxyquinoline",
                    "off_label_use_count": 7,
                    "indication_count": 2
                },
                {
                    "index": 20432,
                    "name": "Nandrolone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 20436,
                    "name": "Dexpanthenol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20454,
                    "name": "Mupirocin",
                    "off_label_use_count": 9,
                    "indication_count": 2
                },
                {
                    "index": 20555,
                    "name": "Lodoxamide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20566,
                    "name": "Polysorbate 80",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20567,
                    "name": "Propylene glycol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20583,
                    "name": "Xylitol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20584,
                    "name": "Stannous fluoride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                }
            ]
        },
        {
            "question": "Which medications have more off-label uses than approved uses?",
            "expected_cypher": "MATCH (d:disease)-[:off_label_use]->(dr:drug)\nWITH dr, COUNT(d) AS offLabelCount\nMATCH (d:disease)-[:indication]->(dr)\nWITH dr, offLabelCount, COUNT(d) AS approvedCount\nWHERE offLabelCount>approvedCount\nRETURN dr.name AS DrugName, offLabelCount AS OffLabelUses, approvedCount AS ApprovedUses",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 14016,
                    "name": "Medrysone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14017,
                    "name": "Fluorometholone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14019,
                    "name": "Betamethasone",
                    "off_label_use_count": 31,
                    "indication_count": 0
                },
                {
                    "index": 14023,
                    "name": "Triamcinolone",
                    "off_label_use_count": 30,
                    "indication_count": 0
                },
                {
                    "index": 14024,
                    "name": "Prednisone",
                    "off_label_use_count": 31,
                    "indication_count": 0
                },
                {
                    "index": 14025,
                    "name": "Mitotane",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14028,
                    "name": "Hydrocortisone",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14030,
                    "name": "Prednisolone",
                    "off_label_use_count": 39,
                    "indication_count": 0
                },
                {
                    "index": 14042,
                    "name": "Hydrocortisone acetate",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14052,
                    "name": "Vitamin A",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14061,
                    "name": "Diclofenac",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14062,
                    "name": "Diflunisal",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14106,
                    "name": "Hydroxocobalamin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14112,
                    "name": "Etretinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14115,
                    "name": "Bismuth subsalicylate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14127,
                    "name": "Octreotide",
                    "off_label_use_count": 18,
                    "indication_count": 3
                },
                {
                    "index": 14128,
                    "name": "Ascorbic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14131,
                    "name": "Icosapent",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14135,
                    "name": "Lovastatin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14140,
                    "name": "Ziprasidone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14141,
                    "name": "Phenytoin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14142,
                    "name": "Metoprolol",
                    "off_label_use_count": 15,
                    "indication_count": 7
                },
                {
                    "index": 14144,
                    "name": "Topiramate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14150,
                    "name": "Morphine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14151,
                    "name": "Desogestrel",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14152,
                    "name": "Chlorthalidone",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14153,
                    "name": "Valproic acid",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14154,
                    "name": "Acetaminophen",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14156,
                    "name": "Amitriptyline",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14158,
                    "name": "Indomethacin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14159,
                    "name": "Ipratropium",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14162,
                    "name": "Atenolol",
                    "off_label_use_count": 10,
                    "indication_count": 6
                },
                {
                    "index": 14163,
                    "name": "Diltiazem",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14164,
                    "name": "Alprazolam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14167,
                    "name": "Ampicillin",
                    "off_label_use_count": 16,
                    "indication_count": 0
                },
                {
                    "index": 14168,
                    "name": "Spironolactone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14173,
                    "name": "Imipramine",
                    "off_label_use_count": 11,
                    "indication_count": 4
                },
                {
                    "index": 14174,
                    "name": "Acitretin",
                    "off_label_use_count": 13,
                    "indication_count": 0
                },
                {
                    "index": 14175,
                    "name": "Nabumetone",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14176,
                    "name": "Fluoxetine",
                    "off_label_use_count": 10,
                    "indication_count": 9
                },
                {
                    "index": 14182,
                    "name": "Oxycodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14183,
                    "name": "Tolmetin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14184,
                    "name": "Ritonavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14185,
                    "name": "Vancomycin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14188,
                    "name": "Ciprofloxacin",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14189,
                    "name": "Nortriptyline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14190,
                    "name": "Fluorouracil",
                    "off_label_use_count": 20,
                    "indication_count": 19
                },
                {
                    "index": 14191,
                    "name": "Piroxicam",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14192,
                    "name": "Lamotrigine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14194,
                    "name": "Methotrexate",
                    "off_label_use_count": 27,
                    "indication_count": 0
                },
                {
                    "index": 14196,
                    "name": "Propranolol",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14197,
                    "name": "Fenoprofen",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14198,
                    "name": "Clonidine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14201,
                    "name": "Oxytetracycline",
                    "off_label_use_count": 12,
                    "indication_count": 0
                },
                {
                    "index": 14203,
                    "name": "Medroxyprogesterone acetate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14206,
                    "name": "Chloroquine",
                    "off_label_use_count": 13,
                    "indication_count": 11
                },
                {
                    "index": 14208,
                    "name": "Testosterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14211,
                    "name": "Estrone",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14212,
                    "name": "Verapamil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14213,
                    "name": "Tamoxifen",
                    "off_label_use_count": 25,
                    "indication_count": 3
                },
                {
                    "index": 14215,
                    "name": "Losartan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14216,
                    "name": "Warfarin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14217,
                    "name": "Furosemide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14221,
                    "name": "Norethisterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14223,
                    "name": "Risperidone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14227,
                    "name": "Etodolac",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14229,
                    "name": "Tretinoin",
                    "off_label_use_count": 11,
                    "indication_count": 4
                },
                {
                    "index": 14230,
                    "name": "Tetracycline",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 14231,
                    "name": "Irinotecan",
                    "off_label_use_count": 7,
                    "indication_count": 2
                },
                {
                    "index": 14235,
                    "name": "Estradiol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14236,
                    "name": "Mefenamic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14237,
                    "name": "Acyclovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14242,
                    "name": "Meloxicam",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14245,
                    "name": "Diazepam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14247,
                    "name": "Clofazimine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14250,
                    "name": "Terbinafine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14253,
                    "name": "Chlorhexidine",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14254,
                    "name": "Emtricitabine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14255,
                    "name": "Quinapril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14257,
                    "name": "Etacrynic acid",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14260,
                    "name": "Cyclobenzaprine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14262,
                    "name": "Salicylic acid",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 14266,
                    "name": "Fexofenadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14267,
                    "name": "Isoniazid",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14268,
                    "name": "Norgestimate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14269,
                    "name": "Methylprednisolone",
                    "off_label_use_count": 34,
                    "indication_count": 0
                },
                {
                    "index": 14270,
                    "name": "Ethinylestradiol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14271,
                    "name": "Isotretinoin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14273,
                    "name": "Azathioprine",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 14274,
                    "name": "Auranofin",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14275,
                    "name": "Doxorubicin",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 14276,
                    "name": "Hydrochlorothiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14277,
                    "name": "Letrozole",
                    "off_label_use_count": 21,
                    "indication_count": 0
                },
                {
                    "index": 14278,
                    "name": "Ketoprofen",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14279,
                    "name": "Sulfamethoxazole",
                    "off_label_use_count": 21,
                    "indication_count": 9
                },
                {
                    "index": 14281,
                    "name": "Ketoconazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14282,
                    "name": "Irbesartan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14285,
                    "name": "Gatifloxacin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14286,
                    "name": "Rifampicin",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 14288,
                    "name": "Benzylpenicillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 14291,
                    "name": "Amoxicillin",
                    "off_label_use_count": 14,
                    "indication_count": 12
                },
                {
                    "index": 14294,
                    "name": "Clonazepam",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14299,
                    "name": "Sertraline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14300,
                    "name": "Miconazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14301,
                    "name": "Nifedipine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14302,
                    "name": "Amiodarone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14308,
                    "name": "Carvedilol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14309,
                    "name": "Levofloxacin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14311,
                    "name": "Cloxacillin",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 14313,
                    "name": "Arsenic trioxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14317,
                    "name": "Captopril",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14318,
                    "name": "Ceftriaxone",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14320,
                    "name": "Dexamethasone",
                    "off_label_use_count": 33,
                    "indication_count": 0
                },
                {
                    "index": 14321,
                    "name": "Levodopa",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14324,
                    "name": "Gemfibrozil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14325,
                    "name": "Clomipramine",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14326,
                    "name": "Darunavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14331,
                    "name": "Polythiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14335,
                    "name": "Cefotetan",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14340,
                    "name": "Magnesium salicylate",
                    "off_label_use_count": 10,
                    "indication_count": 8
                },
                {
                    "index": 14346,
                    "name": "Lopinavir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14347,
                    "name": "Hydroxychloroquine",
                    "off_label_use_count": 13,
                    "indication_count": 11
                },
                {
                    "index": 14369,
                    "name": "Nebivolol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14374,
                    "name": "Iodine",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14378,
                    "name": "Rufinamide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14381,
                    "name": "Levocetirizine",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14387,
                    "name": "Methyltestosterone",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 14415,
                    "name": "Chlortetracycline",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14466,
                    "name": "Favipiravir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14483,
                    "name": "Estradiol valerate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14484,
                    "name": "Tenofovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14490,
                    "name": "Zinc chloride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14509,
                    "name": "Meclofenamic acid",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14510,
                    "name": "Heparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14511,
                    "name": "Dantrolene",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 14527,
                    "name": "Fluoxymesterone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14528,
                    "name": "Danazol",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 14530,
                    "name": "Stanolone",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14538,
                    "name": "Disopyramide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14539,
                    "name": "Prazosin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14542,
                    "name": "Maprotiline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14543,
                    "name": "Desipramine",
                    "off_label_use_count": 8,
                    "indication_count": 3
                },
                {
                    "index": 14544,
                    "name": "Bupropion",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14545,
                    "name": "Clindamycin",
                    "off_label_use_count": 17,
                    "indication_count": 16
                },
                {
                    "index": 14559,
                    "name": "Calcitriol",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14560,
                    "name": "Ergocalciferol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14561,
                    "name": "Cholecalciferol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14579,
                    "name": "Nitazoxanide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14582,
                    "name": "Nadolol",
                    "off_label_use_count": 8,
                    "indication_count": 2
                },
                {
                    "index": 14589,
                    "name": "Polyethylene glycol 400",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14597,
                    "name": "Dapsone",
                    "off_label_use_count": 14,
                    "indication_count": 3
                },
                {
                    "index": 14603,
                    "name": "Chlorpropamide",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14604,
                    "name": "Etoposide",
                    "off_label_use_count": 59,
                    "indication_count": 8
                },
                {
                    "index": 14605,
                    "name": "Candesartan cilexetil",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14606,
                    "name": "Thalidomide",
                    "off_label_use_count": 7,
                    "indication_count": 1
                },
                {
                    "index": 14608,
                    "name": "Ifosfamide",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14610,
                    "name": "Ketamine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14628,
                    "name": "Tioguanine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14635,
                    "name": "Nitroglycerin",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14662,
                    "name": "Ribavirin",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 14668,
                    "name": "Bumetanide",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 14670,
                    "name": "Drospirenone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14678,
                    "name": "Folic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14685,
                    "name": "Progesterone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14686,
                    "name": "Clomifene",
                    "off_label_use_count": 14,
                    "indication_count": 1
                },
                {
                    "index": 14691,
                    "name": "Amiloride",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 14696,
                    "name": "Pentamidine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14698,
                    "name": "Pyrantel",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14703,
                    "name": "Methantheline",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14704,
                    "name": "Calcifediol",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14706,
                    "name": "Cyanocobalamin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14709,
                    "name": "Gabapentin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14715,
                    "name": "Coenzyme M",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14719,
                    "name": "Epirubicin",
                    "off_label_use_count": 21,
                    "indication_count": 2
                },
                {
                    "index": 14729,
                    "name": "Cimetidine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14730,
                    "name": "Metyrapone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14733,
                    "name": "Levomefolic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14736,
                    "name": "Oxaliplatin",
                    "off_label_use_count": 12,
                    "indication_count": 1
                },
                {
                    "index": 14737,
                    "name": "Carboplatin",
                    "off_label_use_count": 42,
                    "indication_count": 21
                },
                {
                    "index": 14740,
                    "name": "Chlorambucil",
                    "off_label_use_count": 10,
                    "indication_count": 9
                },
                {
                    "index": 14751,
                    "name": "Zinc sulfate",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14758,
                    "name": "Capecitabine",
                    "off_label_use_count": 22,
                    "indication_count": 6
                },
                {
                    "index": 14764,
                    "name": "Mercaptopurine",
                    "off_label_use_count": 23,
                    "indication_count": 5
                },
                {
                    "index": 14768,
                    "name": "Dalteparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14770,
                    "name": "Bivalirudin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14772,
                    "name": "Aminosalicylic acid",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14774,
                    "name": "Enoxaparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 14775,
                    "name": "Calcipotriol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14780,
                    "name": "Cytarabine",
                    "off_label_use_count": 13,
                    "indication_count": 0
                },
                {
                    "index": 14782,
                    "name": "Biotin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14783,
                    "name": "Caffeine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14789,
                    "name": "Daunorubicin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14790,
                    "name": "Primaquine",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 14791,
                    "name": "Mitoxantrone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14792,
                    "name": "Paclitaxel",
                    "off_label_use_count": 24,
                    "indication_count": 0
                },
                {
                    "index": 14793,
                    "name": "Docetaxel",
                    "off_label_use_count": 25,
                    "indication_count": 14
                },
                {
                    "index": 14797,
                    "name": "Imipenem",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14807,
                    "name": "Epinephrine",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 14819,
                    "name": "Econazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14828,
                    "name": "Cycloserine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14830,
                    "name": "Amantadine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14833,
                    "name": "Riboflavin",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14835,
                    "name": "Benazepril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14836,
                    "name": "Ramipril",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14843,
                    "name": "Nizatidine",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 14846,
                    "name": "Perindopril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 14848,
                    "name": "Terbutaline",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14851,
                    "name": "Cyclopentolate",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14880,
                    "name": "Nicotine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14881,
                    "name": "Phenylephrine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14884,
                    "name": "Phenelzine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14898,
                    "name": "Phenylpropanolamine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14911,
                    "name": "Allopurinol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14914,
                    "name": "Procarbazine",
                    "off_label_use_count": 45,
                    "indication_count": 5
                },
                {
                    "index": 14918,
                    "name": "Adenosine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14934,
                    "name": "Cyclosporine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14935,
                    "name": "Fluconazole",
                    "off_label_use_count": 15,
                    "indication_count": 7
                },
                {
                    "index": 14936,
                    "name": "Erythromycin",
                    "off_label_use_count": 30,
                    "indication_count": 22
                },
                {
                    "index": 14940,
                    "name": "Lidocaine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14942,
                    "name": "Terfenadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14943,
                    "name": "Levonorgestrel",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 14948,
                    "name": "Teniposide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 14949,
                    "name": "Chloramphenicol",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14950,
                    "name": "Loratadine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14951,
                    "name": "Quinine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14952,
                    "name": "Haloperidol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14955,
                    "name": "Vincristine",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 14956,
                    "name": "Carbamazepine",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 14958,
                    "name": "Cisapride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14959,
                    "name": "Nicardipine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14960,
                    "name": "Astemizole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14962,
                    "name": "Trazodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14963,
                    "name": "Midazolam",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 14975,
                    "name": "Sirolimus",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14977,
                    "name": "Ondansetron",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 14980,
                    "name": "Metronidazole",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14984,
                    "name": "Felodipine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 14987,
                    "name": "Fluvastatin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 14989,
                    "name": "Quinacrine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 14992,
                    "name": "Itraconazole",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14993,
                    "name": "Phenobarbital",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 14995,
                    "name": "Clarithromycin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 14997,
                    "name": "Anastrozole",
                    "off_label_use_count": 21,
                    "indication_count": 0
                },
                {
                    "index": 15001,
                    "name": "Paliperidone",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15003,
                    "name": "Cortisone acetate",
                    "off_label_use_count": 39,
                    "indication_count": 0
                },
                {
                    "index": 15026,
                    "name": "Hydroxyprogesterone caproate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15034,
                    "name": "Cobicistat",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15059,
                    "name": "Troleandomycin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 15061,
                    "name": "Umifenovir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15073,
                    "name": "Fluvoxamine",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15076,
                    "name": "Esmolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15078,
                    "name": "Tramadol",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15082,
                    "name": "Citalopram",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15084,
                    "name": "Clotrimazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15087,
                    "name": "Venlafaxine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15089,
                    "name": "Codeine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15092,
                    "name": "Vinorelbine",
                    "off_label_use_count": 4,
                    "indication_count": 2
                },
                {
                    "index": 15093,
                    "name": "Clozapine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15094,
                    "name": "Mirtazapine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15095,
                    "name": "Timolol",
                    "off_label_use_count": 7,
                    "indication_count": 4
                },
                {
                    "index": 15107,
                    "name": "Vinblastine",
                    "off_label_use_count": 10,
                    "indication_count": 0
                },
                {
                    "index": 15113,
                    "name": "Galantamine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15116,
                    "name": "Paroxetine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15117,
                    "name": "Trimipramine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15121,
                    "name": "Methimazole",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15126,
                    "name": "Donepezil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15134,
                    "name": "Hydroxyurea",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15145,
                    "name": "Proguanil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15146,
                    "name": "Nefazodone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15148,
                    "name": "Escitalopram",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15153,
                    "name": "Acebutolol",
                    "off_label_use_count": 9,
                    "indication_count": 2
                },
                {
                    "index": 15156,
                    "name": "Metipranolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15159,
                    "name": "Lisdexamfetamine",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15164,
                    "name": "Yohimbine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15167,
                    "name": "Antipyrine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15179,
                    "name": "Nicotinamide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15180,
                    "name": "Fusidic acid",
                    "off_label_use_count": 25,
                    "indication_count": 0
                },
                {
                    "index": 15211,
                    "name": "Desvenlafaxine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15218,
                    "name": "Phenylbutyric acid",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15267,
                    "name": "Remdesivir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15271,
                    "name": "Mitomycin",
                    "off_label_use_count": 10,
                    "indication_count": 1
                },
                {
                    "index": 15289,
                    "name": "Moxifloxacin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15295,
                    "name": "Triamterene",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15298,
                    "name": "Streptozocin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15299,
                    "name": "Trimethoprim",
                    "off_label_use_count": 22,
                    "indication_count": 7
                },
                {
                    "index": 15300,
                    "name": "Enoxacin",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15303,
                    "name": "Methoxsalen",
                    "off_label_use_count": 7,
                    "indication_count": 3
                },
                {
                    "index": 15307,
                    "name": "Thiabendazole",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15311,
                    "name": "Primidone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15312,
                    "name": "Pentoxifylline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15322,
                    "name": "Norfloxacin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 15326,
                    "name": "Ofloxacin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 15363,
                    "name": "Bendamustine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15413,
                    "name": "Gemcitabine",
                    "off_label_use_count": 16,
                    "indication_count": 0
                },
                {
                    "index": 15416,
                    "name": "Lorazepam",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 15419,
                    "name": "Azithromycin",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 15420,
                    "name": "Pantoprazole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15425,
                    "name": "Doxycycline",
                    "off_label_use_count": 18,
                    "indication_count": 0
                },
                {
                    "index": 15426,
                    "name": "Isradipine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15430,
                    "name": "Pentobarbital",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15435,
                    "name": "Megestrol acetate",
                    "off_label_use_count": 21,
                    "indication_count": 4
                },
                {
                    "index": 15438,
                    "name": "Sulfadiazine",
                    "off_label_use_count": 12,
                    "indication_count": 10
                },
                {
                    "index": 15447,
                    "name": "Valdecoxib",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 15451,
                    "name": "Bisoprolol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15452,
                    "name": "Rifabutin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15456,
                    "name": "Amphotericin B",
                    "off_label_use_count": 15,
                    "indication_count": 0
                },
                {
                    "index": 15461,
                    "name": "Imiquimod",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15465,
                    "name": "Phenylbutazone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15466,
                    "name": "Acetazolamide",
                    "off_label_use_count": 12,
                    "indication_count": 6
                },
                {
                    "index": 15467,
                    "name": "Ethynodiol diacetate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15479,
                    "name": "Salbutamol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15481,
                    "name": "Topotecan",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15485,
                    "name": "Atovaquone",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 15493,
                    "name": "Bromocriptine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15496,
                    "name": "Rifaximin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15521,
                    "name": "Rutin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15549,
                    "name": "Dexlansoprazole",
                    "off_label_use_count": 5,
                    "indication_count": 4
                },
                {
                    "index": 15563,
                    "name": "Tocilizumab",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15582,
                    "name": "Difluprednate",
                    "off_label_use_count": 11,
                    "indication_count": 0
                },
                {
                    "index": 15597,
                    "name": "Ruxolitinib",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15621,
                    "name": "Dienogest",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15646,
                    "name": "Oleandomycin",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 15666,
                    "name": "Baricitinib",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15730,
                    "name": "Baloxavir marboxil",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15761,
                    "name": "Oseltamivir",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15762,
                    "name": "Trandolapril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 15763,
                    "name": "Benzocaine",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 15769,
                    "name": "Valsartan",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 15771,
                    "name": "Sulfisoxazole",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15777,
                    "name": "Oxandrolone",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15780,
                    "name": "Sulfapyridine",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 15804,
                    "name": "Cladribine",
                    "off_label_use_count": 6,
                    "indication_count": 3
                },
                {
                    "index": 15834,
                    "name": "Thiopental",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15836,
                    "name": "Telmisartan",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15856,
                    "name": "Naltrexone",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15860,
                    "name": "Sodium aurothiomalate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15867,
                    "name": "Mebendazole",
                    "off_label_use_count": 9,
                    "indication_count": 3
                },
                {
                    "index": 15870,
                    "name": "Amifostine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15876,
                    "name": "Mesalazine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15877,
                    "name": "Cyproheptadine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15885,
                    "name": "Inositol nicotinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 15886,
                    "name": "Sodium fluoride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 15893,
                    "name": "Bleomycin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 15913,
                    "name": "Citrulline",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 15914,
                    "name": "Minocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 15982,
                    "name": "Sulfasalazine",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 15984,
                    "name": "Oxaprozin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16036,
                    "name": "Acetylcysteine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16057,
                    "name": "Hyoscyamine",
                    "off_label_use_count": 6,
                    "indication_count": 1
                },
                {
                    "index": 16059,
                    "name": "Tridihexethyl",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16060,
                    "name": "Anisotropine methylbromide",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16063,
                    "name": "Scopolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16074,
                    "name": "Mepenzolate",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16087,
                    "name": "Proflavine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16210,
                    "name": "Eprosartan",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16217,
                    "name": "Serine",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 16219,
                    "name": "Alprostadil",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16221,
                    "name": "Misoprostol",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 16281,
                    "name": "Cysteine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16305,
                    "name": "Potassium nitrate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16313,
                    "name": "Propantheline",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 16347,
                    "name": "Ferrous fumarate",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16350,
                    "name": "Citric acid",
                    "off_label_use_count": 5,
                    "indication_count": 2
                },
                {
                    "index": 16357,
                    "name": "Isometheptene",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16377,
                    "name": "Phenolphthalein",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16443,
                    "name": "Stanozolol",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16550,
                    "name": "Dipyridamole",
                    "off_label_use_count": 3,
                    "indication_count": 1
                },
                {
                    "index": 16572,
                    "name": "Leuprolide",
                    "off_label_use_count": 21,
                    "indication_count": 5
                },
                {
                    "index": 16588,
                    "name": "Metformin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16635,
                    "name": "Fondaparinux",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 16672,
                    "name": "Baclofen",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16687,
                    "name": "Fosinopril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16689,
                    "name": "Moexipril",
                    "off_label_use_count": 3,
                    "indication_count": 2
                },
                {
                    "index": 16690,
                    "name": "Lisinopril",
                    "off_label_use_count": 10,
                    "indication_count": 6
                },
                {
                    "index": 16788,
                    "name": "Methyclothiazide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16793,
                    "name": "Chlorothiazide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16802,
                    "name": "Aurothioglucose",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16825,
                    "name": "Ardeparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 16837,
                    "name": "Urea",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 16842,
                    "name": "Pramipexole",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 16844,
                    "name": "Phentolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16847,
                    "name": "Apraclonidine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 16855,
                    "name": "Methylphenidate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 16867,
                    "name": "Demeclocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 17062,
                    "name": "Butalbital",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17124,
                    "name": "Phylloquinone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 17149,
                    "name": "Butorphanol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17173,
                    "name": "Physostigmine",
                    "off_label_use_count": 2,
                    "indication_count": 1
                },
                {
                    "index": 17174,
                    "name": "Rivastigmine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17215,
                    "name": "Salmon calcitonin",
                    "off_label_use_count": 7,
                    "indication_count": 4
                },
                {
                    "index": 17229,
                    "name": "Cetirizine",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 17238,
                    "name": "Dimenhydrinate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17273,
                    "name": "Amisulpride",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17305,
                    "name": "Protriptyline",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17338,
                    "name": "Metaraminol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17376,
                    "name": "Cefotaxime",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 17411,
                    "name": "Naftifine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17524,
                    "name": "Carmustine",
                    "off_label_use_count": 7,
                    "indication_count": 0
                },
                {
                    "index": 17727,
                    "name": "Pamidronic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17728,
                    "name": "Zoledronic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17739,
                    "name": "Dihydrotachysterol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 17829,
                    "name": "Lithium citrate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17831,
                    "name": "Lithium carbonate",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17886,
                    "name": "Tetracosactide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 17999,
                    "name": "Flucytosine",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18036,
                    "name": "Bacitracin",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 18160,
                    "name": "Emapalumab",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18179,
                    "name": "Sucralfate",
                    "off_label_use_count": 6,
                    "indication_count": 1
                },
                {
                    "index": 18312,
                    "name": "Phenoxymethylpenicillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 18313,
                    "name": "Oxacillin",
                    "off_label_use_count": 6,
                    "indication_count": 5
                },
                {
                    "index": 18687,
                    "name": "Cilastatin",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 18755,
                    "name": "Cromoglicic acid",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 18884,
                    "name": "Lactose",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 19207,
                    "name": "Carbidopa",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 19250,
                    "name": "Pentostatin",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 19863,
                    "name": "Anthralin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20129,
                    "name": "Chlordiazepoxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20146,
                    "name": "Magnesium carbonate",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 20148,
                    "name": "Synephrine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20156,
                    "name": "Dactinomycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20157,
                    "name": "Guaifenesin",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20169,
                    "name": "Magnesium sulfate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20172,
                    "name": "Silicon dioxide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20203,
                    "name": "Ceftazidime",
                    "off_label_use_count": 6,
                    "indication_count": 0
                },
                {
                    "index": 20206,
                    "name": "Cyclacillin",
                    "off_label_use_count": 5,
                    "indication_count": 3
                },
                {
                    "index": 20212,
                    "name": "Melphalan",
                    "off_label_use_count": 10,
                    "indication_count": 0
                },
                {
                    "index": 20214,
                    "name": "Potassium bicarbonate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20218,
                    "name": "Sodium citrate",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 20239,
                    "name": "Penicillamine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20254,
                    "name": "Polyethylene glycol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20263,
                    "name": "Ganciclovir",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20279,
                    "name": "Tyramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20289,
                    "name": "Clavulanic acid",
                    "off_label_use_count": 11,
                    "indication_count": 10
                },
                {
                    "index": 20293,
                    "name": "Inositol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20296,
                    "name": "Polyvinyl alcohol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20298,
                    "name": "Potassium citrate",
                    "off_label_use_count": 4,
                    "indication_count": 3
                },
                {
                    "index": 20302,
                    "name": "Potassium Iodide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20303,
                    "name": "Trolnitrate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20306,
                    "name": "Zinc gluconate",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 20313,
                    "name": "Polyethylene glycol 300",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20314,
                    "name": "Polyethylene glycol 3500",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20318,
                    "name": "Sulfacetamide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20320,
                    "name": "Cobalamin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20321,
                    "name": "Methscopolamine",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20322,
                    "name": "Zinc oxide",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20324,
                    "name": "Triethylenetetramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20332,
                    "name": "Adomiparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20333,
                    "name": "Parnaparin",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20334,
                    "name": "Heparin, bovine",
                    "off_label_use_count": 6,
                    "indication_count": 4
                },
                {
                    "index": 20336,
                    "name": "Temozolomide",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20337,
                    "name": "Monopotassium phosphate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20340,
                    "name": "Ethionamide",
                    "off_label_use_count": 5,
                    "indication_count": 1
                },
                {
                    "index": 20351,
                    "name": "Meropenem",
                    "off_label_use_count": 12,
                    "indication_count": 7
                },
                {
                    "index": 20352,
                    "name": "Povidone",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20353,
                    "name": "Povidone K30",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20376,
                    "name": "Magnesium hydroxide",
                    "off_label_use_count": 13,
                    "indication_count": 6
                },
                {
                    "index": 20397,
                    "name": "Pantothenic acid",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20400,
                    "name": "Dipotassium phosphate",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20401,
                    "name": "Paromomycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20402,
                    "name": "Ethambutol",
                    "off_label_use_count": 4,
                    "indication_count": 1
                },
                {
                    "index": 20403,
                    "name": "Almasilate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20404,
                    "name": "Sulbactam",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20406,
                    "name": "Meticillin",
                    "off_label_use_count": 6,
                    "indication_count": 2
                },
                {
                    "index": 20407,
                    "name": "Kanamycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20410,
                    "name": "Mezlocillin",
                    "off_label_use_count": 8,
                    "indication_count": 6
                },
                {
                    "index": 20411,
                    "name": "Bacampicillin",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 20412,
                    "name": "Amikacin",
                    "off_label_use_count": 8,
                    "indication_count": 0
                },
                {
                    "index": 20413,
                    "name": "Azlocillin",
                    "off_label_use_count": 5,
                    "indication_count": 0
                },
                {
                    "index": 20415,
                    "name": "Ticarcillin",
                    "off_label_use_count": 9,
                    "indication_count": 0
                },
                {
                    "index": 20418,
                    "name": "Tobramycin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20422,
                    "name": "Meclocycline",
                    "off_label_use_count": 14,
                    "indication_count": 0
                },
                {
                    "index": 20424,
                    "name": "Magnesium trisilicate",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20427,
                    "name": "Cholestyramine",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20429,
                    "name": "Colistin",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20431,
                    "name": "Diiodohydroxyquinoline",
                    "off_label_use_count": 7,
                    "indication_count": 2
                },
                {
                    "index": 20432,
                    "name": "Nandrolone",
                    "off_label_use_count": 4,
                    "indication_count": 0
                },
                {
                    "index": 20436,
                    "name": "Dexpanthenol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20454,
                    "name": "Mupirocin",
                    "off_label_use_count": 9,
                    "indication_count": 2
                },
                {
                    "index": 20555,
                    "name": "Lodoxamide",
                    "off_label_use_count": 3,
                    "indication_count": 0
                },
                {
                    "index": 20566,
                    "name": "Polysorbate 80",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20567,
                    "name": "Propylene glycol",
                    "off_label_use_count": 2,
                    "indication_count": 0
                },
                {
                    "index": 20583,
                    "name": "Xylitol",
                    "off_label_use_count": 1,
                    "indication_count": 0
                },
                {
                    "index": 20584,
                    "name": "Stannous fluoride",
                    "off_label_use_count": 1,
                    "indication_count": 0
                }
            ]
        },
        {
            "question": "Which diseases have only treatments that have no side effects at all?",
            "expected_cypher": "MATCH (d:disease)-[:indication]->(dr:drug)\nWHERE NOT EXISTS ((dr)-[:side_effect]->(:effect_or_phenotype))\nRETURN d.name AS DiseaseName",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 16687,
                    "name": "Fosinopril"
                },
                {
                    "index": 20297,
                    "name": "Imidapril"
                },
                {
                    "index": 16693,
                    "name": "Cilazapril"
                },
                {
                    "index": 14944,
                    "name": "Amlodipine"
                },
                {
                    "index": 15762,
                    "name": "Trandolapril"
                },
                {
                    "index": 14836,
                    "name": "Ramipril"
                },
                {
                    "index": 14421,
                    "name": "Aranidipine"
                },
                {
                    "index": 14835,
                    "name": "Benazepril"
                },
                {
                    "index": 16214,
                    "name": "Azilsartan medoxomil"
                },
                {
                    "index": 14327,
                    "name": "Hydralazine"
                },
                {
                    "index": 14255,
                    "name": "Quinapril"
                },
                {
                    "index": 16694,
                    "name": "Spirapril"
                },
                {
                    "index": 20301,
                    "name": "Delapril"
                },
                {
                    "index": 14846,
                    "name": "Perindopril"
                },
                {
                    "index": 16699,
                    "name": "Zofenopril"
                },
                {
                    "index": 14152,
                    "name": "Chlorthalidone"
                },
                {
                    "index": 16690,
                    "name": "Lisinopril"
                },
                {
                    "index": 16698,
                    "name": "Enalaprilat"
                },
                {
                    "index": 14200,
                    "name": "Enalapril"
                },
                {
                    "index": 15426,
                    "name": "Isradipine"
                },
                {
                    "index": 15108,
                    "name": "Doxazosin"
                },
                {
                    "index": 17278,
                    "name": "Ketanserin"
                },
                {
                    "index": 14762,
                    "name": "Bendroflumethiazide"
                },
                {
                    "index": 14308,
                    "name": "Carvedilol"
                },
                {
                    "index": 14984,
                    "name": "Felodipine"
                },
                {
                    "index": 15451,
                    "name": "Bisoprolol"
                },
                {
                    "index": 15175,
                    "name": "Oxprenolol"
                },
                {
                    "index": 14946,
                    "name": "Nisoldipine"
                },
                {
                    "index": 15079,
                    "name": "Betaxolol"
                },
                {
                    "index": 16303,
                    "name": "Terazosin"
                },
                {
                    "index": 14317,
                    "name": "Captopril"
                },
                {
                    "index": 15580,
                    "name": "Pinacidil"
                },
                {
                    "index": 17647,
                    "name": "Mecamylamine"
                },
                {
                    "index": 16793,
                    "name": "Chlorothiazide"
                },
                {
                    "index": 14276,
                    "name": "Hydrochlorothiazide"
                },
                {
                    "index": 15245,
                    "name": "Talinolol"
                },
                {
                    "index": 15186,
                    "name": "Celiprolol"
                },
                {
                    "index": 14133,
                    "name": "Torasemide"
                },
                {
                    "index": 14959,
                    "name": "Nicardipine"
                },
                {
                    "index": 17344,
                    "name": "Piretanide"
                },
                {
                    "index": 19223,
                    "name": "Pargyline"
                },
                {
                    "index": 14809,
                    "name": "Methyldopa"
                },
                {
                    "index": 17308,
                    "name": "Guanethidine"
                },
                {
                    "index": 16854,
                    "name": "Rilmenidine"
                },
                {
                    "index": 14217,
                    "name": "Furosemide"
                },
                {
                    "index": 14196,
                    "name": "Propranolol"
                },
                {
                    "index": 14212,
                    "name": "Verapamil"
                },
                {
                    "index": 14168,
                    "name": "Spironolactone"
                },
                {
                    "index": 15105,
                    "name": "Carteolol"
                },
                {
                    "index": 14142,
                    "name": "Metoprolol"
                },
                {
                    "index": 15304,
                    "name": "Guanabenz"
                },
                {
                    "index": 14938,
                    "name": "Reserpine"
                },
                {
                    "index": 14271,
                    "name": "Isotretinoin"
                },
                {
                    "index": 14163,
                    "name": "Diltiazem"
                },
                {
                    "index": 14582,
                    "name": "Nadolol"
                },
                {
                    "index": 14198,
                    "name": "Clonidine"
                },
                {
                    "index": 17304,
                    "name": "Guanadrel"
                },
                {
                    "index": 15789,
                    "name": "Tienilic acid"
                },
                {
                    "index": 20307,
                    "name": "Endralazine"
                },
                {
                    "index": 15295,
                    "name": "Triamterene"
                },
                {
                    "index": 14547,
                    "name": "Penbutolol"
                },
                {
                    "index": 15855,
                    "name": "Minoxidil"
                },
                {
                    "index": 15153,
                    "name": "Acebutolol"
                },
                {
                    "index": 14202,
                    "name": "Labetalol"
                },
                {
                    "index": 20310,
                    "name": "Trimazosin"
                },
                {
                    "index": 14331,
                    "name": "Polythiazide"
                },
                {
                    "index": 14792,
                    "name": "Paclitaxel"
                },
                {
                    "index": 14162,
                    "name": "Atenolol"
                },
                {
                    "index": 15480,
                    "name": "Guanfacine"
                },
                {
                    "index": 15095,
                    "name": "Timolol"
                },
                {
                    "index": 14240,
                    "name": "Indapamide"
                },
                {
                    "index": 14691,
                    "name": "Amiloride"
                },
                {
                    "index": 16215,
                    "name": "Fimasartan"
                },
                {
                    "index": 14301,
                    "name": "Nifedipine"
                },
                {
                    "index": 14539,
                    "name": "Prazosin"
                },
                {
                    "index": 20316,
                    "name": "Epitizide"
                },
                {
                    "index": 17148,
                    "name": "Metolazone"
                },
                {
                    "index": 15609,
                    "name": "Aliskiren"
                },
                {
                    "index": 14869,
                    "name": "Clevidipine"
                },
                {
                    "index": 15505,
                    "name": "Tasosartan"
                },
                {
                    "index": 15221,
                    "name": "Bupranolol"
                },
                {
                    "index": 14605,
                    "name": "Candesartan cilexetil"
                },
                {
                    "index": 15836,
                    "name": "Telmisartan"
                },
                {
                    "index": 20319,
                    "name": "Xipamide"
                },
                {
                    "index": 14282,
                    "name": "Irbesartan"
                },
                {
                    "index": 15769,
                    "name": "Valsartan"
                },
                {
                    "index": 15133,
                    "name": "Pindolol"
                },
                {
                    "index": 14233,
                    "name": "Hydroflumethiazide"
                },
                {
                    "index": 16788,
                    "name": "Methyclothiazide"
                },
                {
                    "index": 16210,
                    "name": "Eprosartan"
                },
                {
                    "index": 14763,
                    "name": "Trichlormethiazide"
                },
                {
                    "index": 15629,
                    "name": "Azelnidipine"
                },
                {
                    "index": 14369,
                    "name": "Nebivolol"
                },
                {
                    "index": 14731,
                    "name": "Mibefradil"
                },
                {
                    "index": 14215,
                    "name": "Losartan"
                },
                {
                    "index": 16697,
                    "name": "Temocapril"
                },
                {
                    "index": 15632,
                    "name": "Efonidipine"
                },
                {
                    "index": 20325,
                    "name": "Meticrane"
                },
                {
                    "index": 14965,
                    "name": "Eplerenone"
                },
                {
                    "index": 16689,
                    "name": "Moexipril"
                },
                {
                    "index": 14953,
                    "name": "Lercanidipine"
                },
                {
                    "index": 15193,
                    "name": "Rotigotine"
                },
                {
                    "index": 15291,
                    "name": "Ropinirole"
                },
                {
                    "index": 14709,
                    "name": "Gabapentin"
                },
                {
                    "index": 17559,
                    "name": "Gabapentin enacarbil"
                },
                {
                    "index": 16842,
                    "name": "Pramipexole"
                },
                {
                    "index": 15812,
                    "name": "Opicapone"
                },
                {
                    "index": 14139,
                    "name": "Benzatropine"
                },
                {
                    "index": 16315,
                    "name": "Cycrimine"
                },
                {
                    "index": 16056,
                    "name": "Procyclidine"
                },
                {
                    "index": 14830,
                    "name": "Amantadine"
                },
                {
                    "index": 17270,
                    "name": "Pimavanserin"
                },
                {
                    "index": 14321,
                    "name": "Levodopa"
                },
                {
                    "index": 14220,
                    "name": "Apomorphine"
                },
                {
                    "index": 15124,
                    "name": "Biperiden"
                },
                {
                    "index": 19207,
                    "name": "Carbidopa"
                },
                {
                    "index": 15493,
                    "name": "Bromocriptine"
                },
                {
                    "index": 14831,
                    "name": "Droxidopa"
                },
                {
                    "index": 14452,
                    "name": "Istradefylline"
                },
                {
                    "index": 15138,
                    "name": "Selegiline"
                },
                {
                    "index": 16054,
                    "name": "Trihexyphenidyl"
                },
                {
                    "index": 15333,
                    "name": "Rasagiline"
                },
                {
                    "index": 15772,
                    "name": "Tolcapone"
                },
                {
                    "index": 14904,
                    "name": "Safinamide"
                },
                {
                    "index": 14806,
                    "name": "Entacapone"
                },
                {
                    "index": 15081,
                    "name": "Midodrine"
                },
                {
                    "index": 14153,
                    "name": "Valproic acid"
                },
                {
                    "index": 14330,
                    "name": "Fosphenytoin"
                },
                {
                    "index": 20354,
                    "name": "Phensuximide"
                },
                {
                    "index": 14192,
                    "name": "Lamotrigine"
                },
                {
                    "index": 15837,
                    "name": "Methsuximide"
                },
                {
                    "index": 15834,
                    "name": "Thiopental"
                },
                {
                    "index": 15430,
                    "name": "Pentobarbital"
                },
                {
                    "index": 15297,
                    "name": "Secobarbital"
                },
                {
                    "index": 16475,
                    "name": "Phenacemide"
                },
                {
                    "index": 17066,
                    "name": "Amobarbital"
                },
                {
                    "index": 14141,
                    "name": "Phenytoin"
                },
                {
                    "index": 20341,
                    "name": "Ethadione"
                },
                {
                    "index": 14505,
                    "name": "Mephenytoin"
                },
                {
                    "index": 17065,
                    "name": "Metharbital"
                },
                {
                    "index": 14993,
                    "name": "Phenobarbital"
                },
                {
                    "index": 14245,
                    "name": "Diazepam"
                },
                {
                    "index": 14522,
                    "name": "Neocitrullamon"
                },
                {
                    "index": 14956,
                    "name": "Carbamazepine"
                },
                {
                    "index": 15475,
                    "name": "Felbamate"
                },
                {
                    "index": 15434,
                    "name": "Trimethadione"
                },
                {
                    "index": 15311,
                    "name": "Primidone"
                },
                {
                    "index": 15453,
                    "name": "Paramethadione"
                },
                {
                    "index": 15835,
                    "name": "Methylphenobarbital"
                },
                {
                    "index": 15421,
                    "name": "Temazepam"
                },
                {
                    "index": 15099,
                    "name": "Zolpidem"
                },
                {
                    "index": 20363,
                    "name": "Methylpentynol"
                },
                {
                    "index": 20364,
                    "name": "Carbromal"
                },
                {
                    "index": 17067,
                    "name": "Butobarbital"
                },
                {
                    "index": 15841,
                    "name": "Glutethimide"
                },
                {
                    "index": 15506,
                    "name": "Aprobarbital"
                },
                {
                    "index": 14653,
                    "name": "Choline"
                },
                {
                    "index": 20138,
                    "name": "Loprazolam"
                },
                {
                    "index": 14607,
                    "name": "Diphenhydramine"
                },
                {
                    "index": 15513,
                    "name": "Barbital"
                },
                {
                    "index": 15570,
                    "name": "Clomethiazole"
                },
                {
                    "index": 14609,
                    "name": "Zopiclone"
                },
                {
                    "index": 20139,
                    "name": "Doxefazepam"
                },
                {
                    "index": 15518,
                    "name": "Quazepam"
                },
                {
                    "index": 20343,
                    "name": "Chloral hydrate"
                },
                {
                    "index": 15517,
                    "name": "Prazepam"
                },
                {
                    "index": 14976,
                    "name": "Triazolam"
                },
                {
                    "index": 15457,
                    "name": "Flurazepam"
                },
                {
                    "index": 14310,
                    "name": "Doxepin"
                },
                {
                    "index": 15519,
                    "name": "Nitrazepam"
                },
                {
                    "index": 15441,
                    "name": "Eszopiclone"
                },
                {
                    "index": 15052,
                    "name": "Lemborexant"
                },
                {
                    "index": 17244,
                    "name": "Aceprometazine"
                },
                {
                    "index": 16794,
                    "name": "Ethinamate"
                },
                {
                    "index": 17782,
                    "name": "Methyprylon"
                },
                {
                    "index": 14137,
                    "name": "Butabarbital"
                },
                {
                    "index": 17063,
                    "name": "Talbutal"
                },
                {
                    "index": 20128,
                    "name": "Ethchlorvynol"
                },
                {
                    "index": 15611,
                    "name": "Suvorexant"
                },
                {
                    "index": 20365,
                    "name": "Sertaconazole"
                },
                {
                    "index": 15763,
                    "name": "Benzocaine"
                },
                {
                    "index": 14250,
                    "name": "Terbinafine"
                },
                {
                    "index": 20366,
                    "name": "Haloprogin"
                },
                {
                    "index": 14262,
                    "name": "Salicylic acid"
                },
                {
                    "index": 17411,
                    "name": "Naftifine"
                },
                {
                    "index": 14281,
                    "name": "Ketoconazole"
                },
                {
                    "index": 15422,
                    "name": "Oxiconazole"
                },
                {
                    "index": 15219,
                    "name": "Sulconazole"
                },
                {
                    "index": 14028,
                    "name": "Hydrocortisone"
                },
                {
                    "index": 15003,
                    "name": "Cortisone acetate"
                },
                {
                    "index": 14019,
                    "name": "Betamethasone"
                },
                {
                    "index": 17628,
                    "name": "Ciclopirox"
                },
                {
                    "index": 14819,
                    "name": "Econazole"
                },
                {
                    "index": 20367,
                    "name": "Tolnaftate"
                },
                {
                    "index": 15084,
                    "name": "Clotrimazole"
                },
                {
                    "index": 14300,
                    "name": "Miconazole"
                },
                {
                    "index": 15604,
                    "name": "Luliconazole"
                },
                {
                    "index": 20193,
                    "name": "Pramocaine"
                },
                {
                    "index": 20322,
                    "name": "Zinc oxide"
                },
                {
                    "index": 20368,
                    "name": "Clioquinol"
                },
                {
                    "index": 15296,
                    "name": "Griseofulvin"
                },
                {
                    "index": 20369,
                    "name": "Undecylenic acid"
                },
                {
                    "index": 17412,
                    "name": "Butenafine"
                },
                {
                    "index": 20370,
                    "name": "Oxitropium"
                },
                {
                    "index": 15797,
                    "name": "Olodaterol"
                },
                {
                    "index": 14458,
                    "name": "Revefenacin"
                },
                {
                    "index": 15160,
                    "name": "Arformoterol"
                },
                {
                    "index": 14159,
                    "name": "Ipratropium"
                },
                {
                    "index": 15614,
                    "name": "Vilanterol"
                },
                {
                    "index": 17631,
                    "name": "Almitrine"
                },
                {
                    "index": 20371,
                    "name": "Neltenexine"
                },
                {
                    "index": 15956,
                    "name": "Dyphylline"
                },
                {
                    "index": 15166,
                    "name": "Tiotropium"
                },
                {
                    "index": 20372,
                    "name": "Vonoprazan"
                },
                {
                    "index": 16060,
                    "name": "Anisotropine methylbromide"
                },
                {
                    "index": 16340,
                    "name": "Roxatidine acetate"
                },
                {
                    "index": 14843,
                    "name": "Nizatidine"
                },
                {
                    "index": 15316,
                    "name": "Famotidine"
                },
                {
                    "index": 14785,
                    "name": "Omeprazole"
                },
                {
                    "index": 14702,
                    "name": "Ranitidine"
                },
                {
                    "index": 15815,
                    "name": "Sofalcone"
                },
                {
                    "index": 16059,
                    "name": "Tridihexethyl"
                },
                {
                    "index": 14703,
                    "name": "Methantheline"
                },
                {
                    "index": 20373,
                    "name": "Chlorbenzoxamine"
                },
                {
                    "index": 14729,
                    "name": "Cimetidine"
                },
                {
                    "index": 20374,
                    "name": "Troxipide"
                },
                {
                    "index": 16074,
                    "name": "Mepenzolate"
                },
                {
                    "index": 20375,
                    "name": "Aluminum hydroxide"
                },
                {
                    "index": 15839,
                    "name": "Dexrabeprazole"
                },
                {
                    "index": 14786,
                    "name": "Lansoprazole"
                },
                {
                    "index": 15461,
                    "name": "Imiquimod"
                },
                {
                    "index": 20378,
                    "name": "Trioxsalen"
                },
                {
                    "index": 15288,
                    "name": "Interferon alfa-2b"
                },
                {
                    "index": 20380,
                    "name": "Trichloroacetate"
                },
                {
                    "index": 16478,
                    "name": "Podofilox"
                },
                {
                    "index": 20381,
                    "name": "Cianidanol"
                },
                {
                    "index": 20382,
                    "name": "Benzoin"
                },
                {
                    "index": 14269,
                    "name": "Methylprednisolone"
                },
                {
                    "index": 14024,
                    "name": "Prednisone"
                },
                {
                    "index": 14030,
                    "name": "Prednisolone"
                },
                {
                    "index": 14042,
                    "name": "Hydrocortisone acetate"
                },
                {
                    "index": 14023,
                    "name": "Triamcinolone"
                },
                {
                    "index": 14190,
                    "name": "Fluorouracil"
                },
                {
                    "index": 18680,
                    "name": "Ingenol mebutate"
                },
                {
                    "index": 14655,
                    "name": "Masoprocol"
                },
                {
                    "index": 14061,
                    "name": "Diclofenac"
                },
                {
                    "index": 16183,
                    "name": "Aminolevulinic acid"
                },
                {
                    "index": 14299,
                    "name": "Sertraline"
                },
                {
                    "index": 15117,
                    "name": "Trimipramine"
                },
                {
                    "index": 14189,
                    "name": "Nortriptyline"
                },
                {
                    "index": 14885,
                    "name": "Moclobemide"
                },
                {
                    "index": 20384,
                    "name": "Mebanazine"
                },
                {
                    "index": 15116,
                    "name": "Paroxetine"
                },
                {
                    "index": 14905,
                    "name": "Pirlindole"
                },
                {
                    "index": 15082,
                    "name": "Citalopram"
                },
                {
                    "index": 17310,
                    "name": "Amineptine"
                },
                {
                    "index": 20147,
                    "name": "Setiptiline"
                },
                {
                    "index": 14176,
                    "name": "Fluoxetine"
                },
                {
                    "index": 14893,
                    "name": "Iproniazid"
                },
                {
                    "index": 14884,
                    "name": "Phenelzine"
                },
                {
                    "index": 14173,
                    "name": "Imipramine"
                },
                {
                    "index": 14156,
                    "name": "Amitriptyline"
                },
                {
                    "index": 14543,
                    "name": "Desipramine"
                },
                {
                    "index": 17305,
                    "name": "Protriptyline"
                },
                {
                    "index": 17257,
                    "name": "Butriptyline"
                },
                {
                    "index": 20385,
                    "name": "Iproclozide"
                },
                {
                    "index": 14544,
                    "name": "Bupropion"
                },
                {
                    "index": 15667,
                    "name": "Esketamine"
                },
                {
                    "index": 20386,
                    "name": "Opipramol"
                },
                {
                    "index": 14906,
                    "name": "Toloxatone"
                },
                {
                    "index": 20133,
                    "name": "Ethyl loflazepate"
                },
                {
                    "index": 20387,
                    "name": "Demexiptiline"
                },
                {
                    "index": 20388,
                    "name": "Lofepramine"
                },
                {
                    "index": 20389,
                    "name": "Caroxazone"
                },
                {
                    "index": 14540,
                    "name": "Amoxapine"
                },
                {
                    "index": 14542,
                    "name": "Maprotiline"
                },
                {
                    "index": 14144,
                    "name": "Topiramate"
                },
                {
                    "index": 14508,
                    "name": "Ethotoin"
                },
                {
                    "index": 15218,
                    "name": "Phenylbutyric acid"
                },
                {
                    "index": 15590,
                    "name": "Lurasidone"
                },
                {
                    "index": 14161,
                    "name": "Olanzapine"
                },
                {
                    "index": 14999,
                    "name": "Quetiapine"
                },
                {
                    "index": 17829,
                    "name": "Lithium citrate"
                },
                {
                    "index": 17831,
                    "name": "Lithium carbonate"
                },
                {
                    "index": 16011,
                    "name": "Loxapine"
                },
                {
                    "index": 15194,
                    "name": "Cariprazine"
                },
                {
                    "index": 14380,
                    "name": "Asenapine"
                },
                {
                    "index": 15098,
                    "name": "Promazine"
                },
                {
                    "index": 14178,
                    "name": "Chlorpromazine"
                },
                {
                    "index": 14140,
                    "name": "Ziprasidone"
                },
                {
                    "index": 14323,
                    "name": "Aripiprazole"
                },
                {
                    "index": 14223,
                    "name": "Risperidone"
                },
                {
                    "index": 14234,
                    "name": "Oxcarbazepine"
                },
                {
                    "index": 15472,
                    "name": "Tiagabine"
                },
                {
                    "index": 18277,
                    "name": "Levetiracetam"
                },
                {
                    "index": 15799,
                    "name": "Avatrombopag"
                },
                {
                    "index": 17141,
                    "name": "Oprelvekin"
                },
                {
                    "index": 14699,
                    "name": "Lusutrombopag"
                },
                {
                    "index": 15416,
                    "name": "Lorazepam"
                },
                {
                    "index": 15231,
                    "name": "Stiripentol"
                },
                {
                    "index": 14294,
                    "name": "Clonazepam"
                },
                {
                    "index": 15449,
                    "name": "Ethosuximide"
                },
                {
                    "index": 15466,
                    "name": "Acetazolamide"
                },
                {
                    "index": 20377,
                    "name": "Benzoyl peroxide"
                },
                {
                    "index": 15064,
                    "name": "Mometasone furoate"
                },
                {
                    "index": 14020,
                    "name": "Fluticasone propionate"
                },
                {
                    "index": 15486,
                    "name": "Prednicarbate"
                },
                {
                    "index": 14021,
                    "name": "Fluocinolone acetonide"
                },
                {
                    "index": 15500,
                    "name": "Desonide"
                },
                {
                    "index": 16837,
                    "name": "Urea"
                },
                {
                    "index": 14680,
                    "name": "Zinc acetate"
                },
                {
                    "index": 20437,
                    "name": "Ammonium lactate"
                },
                {
                    "index": 14941,
                    "name": "Amcinonide"
                },
                {
                    "index": 14320,
                    "name": "Dexamethasone"
                },
                {
                    "index": 17776,
                    "name": "Desoximetasone"
                },
                {
                    "index": 14033,
                    "name": "Fluocinonide"
                },
                {
                    "index": 15583,
                    "name": "Halcinonide"
                },
                {
                    "index": 14026,
                    "name": "Flumethasone"
                },
                {
                    "index": 14029,
                    "name": "Flurandrenolide"
                },
                {
                    "index": 20438,
                    "name": "Ferric oxide"
                },
                {
                    "index": 20339,
                    "name": "Mannitol"
                },
                {
                    "index": 14257,
                    "name": "Etacrynic acid"
                },
                {
                    "index": 14668,
                    "name": "Bumetanide"
                },
                {
                    "index": 14170,
                    "name": "Trifluridine"
                },
                {
                    "index": 20439,
                    "name": "Idoxuridine"
                },
                {
                    "index": 14208,
                    "name": "Testosterone"
                },
                {
                    "index": 20440,
                    "name": "Neridronic Acid"
                },
                {
                    "index": 17727,
                    "name": "Pamidronic acid"
                },
                {
                    "index": 17428,
                    "name": "Etidronic acid"
                },
                {
                    "index": 17735,
                    "name": "Incadronic acid"
                },
                {
                    "index": 17728,
                    "name": "Zoledronic acid"
                },
                {
                    "index": 14502,
                    "name": "Asparaginase Escherichia coli"
                },
                {
                    "index": 14194,
                    "name": "Methotrexate"
                },
                {
                    "index": 14955,
                    "name": "Vincristine"
                },
                {
                    "index": 15429,
                    "name": "Vindesine"
                },
                {
                    "index": 14764,
                    "name": "Mercaptopurine"
                },
                {
                    "index": 14948,
                    "name": "Teniposide"
                },
                {
                    "index": 14275,
                    "name": "Doxorubicin"
                },
                {
                    "index": 14789,
                    "name": "Daunorubicin"
                },
                {
                    "index": 14780,
                    "name": "Cytarabine"
                },
                {
                    "index": 14514,
                    "name": "Asparaginase Erwinia chrysanthemi"
                },
                {
                    "index": 14503,
                    "name": "Pegaspargase"
                },
                {
                    "index": 15805,
                    "name": "Clofarabine"
                },
                {
                    "index": 17886,
                    "name": "Tetracosactide"
                },
                {
                    "index": 14059,
                    "name": "Liothyronine"
                },
                {
                    "index": 14060,
                    "name": "Levothyroxine"
                },
                {
                    "index": 17422,
                    "name": "Vedolizumab"
                },
                {
                    "index": 14034,
                    "name": "Budesonide"
                },
                {
                    "index": 15887,
                    "name": "Certolizumab pegol"
                },
                {
                    "index": 17584,
                    "name": "Adalimumab"
                },
                {
                    "index": 17417,
                    "name": "Natalizumab"
                },
                {
                    "index": 17585,
                    "name": "Infliximab"
                },
                {
                    "index": 18314,
                    "name": "Benzoic acid"
                },
                {
                    "index": 20423,
                    "name": "Aminobenzoic acid"
                },
                {
                    "index": 15450,
                    "name": "Ivermectin"
                },
                {
                    "index": 15307,
                    "name": "Thiabendazole"
                },
                {
                    "index": 17243,
                    "name": "Alimemazine"
                },
                {
                    "index": 17235,
                    "name": "Methdilazine"
                },
                {
                    "index": 14158,
                    "name": "Indomethacin"
                },
                {
                    "index": 14204,
                    "name": "Sulindac"
                },
                {
                    "index": 14238,
                    "name": "Naproxen"
                },
                {
                    "index": 14340,
                    "name": "Magnesium salicylate"
                },
                {
                    "index": 20442,
                    "name": "Rebamipide"
                },
                {
                    "index": 15767,
                    "name": "Homatropine"
                },
                {
                    "index": 16062,
                    "name": "Homatropine methylbromide"
                },
                {
                    "index": 20172,
                    "name": "Silicon dioxide"
                },
                {
                    "index": 16071,
                    "name": "Isopropamide"
                },
                {
                    "index": 20403,
                    "name": "Almasilate"
                },
                {
                    "index": 20218,
                    "name": "Sodium citrate"
                },
                {
                    "index": 20443,
                    "name": "Poldine"
                },
                {
                    "index": 20129,
                    "name": "Chlordiazepoxide"
                },
                {
                    "index": 16977,
                    "name": "Carbenoxolone"
                },
                {
                    "index": 20146,
                    "name": "Magnesium carbonate"
                },
                {
                    "index": 16063,
                    "name": "Scopolamine"
                },
                {
                    "index": 19331,
                    "name": "Ecabet"
                },
                {
                    "index": 16313,
                    "name": "Propantheline"
                },
                {
                    "index": 20214,
                    "name": "Potassium bicarbonate"
                },
                {
                    "index": 16311,
                    "name": "Pirenzepine"
                },
                {
                    "index": 20444,
                    "name": "Penthienate"
                },
                {
                    "index": 20445,
                    "name": "Ilaprazole"
                },
                {
                    "index": 20424,
                    "name": "Magnesium trisilicate"
                },
                {
                    "index": 15549,
                    "name": "Dexlansoprazole"
                },
                {
                    "index": 20376,
                    "name": "Magnesium hydroxide"
                },
                {
                    "index": 16055,
                    "name": "Oxyphencyclimine"
                },
                {
                    "index": 15462,
                    "name": "Esomeprazole"
                },
                {
                    "index": 15144,
                    "name": "Rabeprazole"
                },
                {
                    "index": 14734,
                    "name": "Polaprezinc"
                },
                {
                    "index": 15420,
                    "name": "Pantoprazole"
                },
                {
                    "index": 16057,
                    "name": "Hyoscyamine"
                },
                {
                    "index": 18230,
                    "name": "Ustekinumab"
                },
                {
                    "index": 15324,
                    "name": "Leflunomide"
                },
                {
                    "index": 15349,
                    "name": "Apremilast"
                },
                {
                    "index": 17601,
                    "name": "Golimumab"
                },
                {
                    "index": 17583,
                    "name": "Etanercept"
                },
                {
                    "index": 15067,
                    "name": "Methylprednisolone hemisuccinate"
                },
                {
                    "index": 20330,
                    "name": "Dichlorophen"
                },
                {
                    "index": 15239,
                    "name": "Levobetaxolol"
                },
                {
                    "index": 15483,
                    "name": "Pilocarpine"
                },
                {
                    "index": 17436,
                    "name": "Latanoprostene bunod"
                },
                {
                    "index": 20448,
                    "name": "Aceclidine"
                },
                {
                    "index": 14290,
                    "name": "Echothiophate"
                },
                {
                    "index": 16847,
                    "name": "Apraclonidine"
                },
                {
                    "index": 15849,
                    "name": "Brimonidine"
                },
                {
                    "index": 17173,
                    "name": "Physostigmine"
                },
                {
                    "index": 16217,
                    "name": "Serine"
                },
                {
                    "index": 16792,
                    "name": "Methazolamide"
                },
                {
                    "index": 16843,
                    "name": "Dipivefrin"
                },
                {
                    "index": 14807,
                    "name": "Epinephrine"
                },
                {
                    "index": 15155,
                    "name": "Levobunolol"
                },
                {
                    "index": 15225,
                    "name": "Befunolol"
                },
                {
                    "index": 15156,
                    "name": "Metipranolol"
                },
                {
                    "index": 14677,
                    "name": "Tafluprost"
                },
                {
                    "index": 17434,
                    "name": "Travoprost"
                },
                {
                    "index": 14978,
                    "name": "Bimatoprost"
                },
                {
                    "index": 15491,
                    "name": "Brinzolamide"
                },
                {
                    "index": 17435,
                    "name": "Latanoprost"
                },
                {
                    "index": 17317,
                    "name": "Netarsudil"
                },
                {
                    "index": 15779,
                    "name": "Dorzolamide"
                },
                {
                    "index": 16795,
                    "name": "Diclofenamide"
                },
                {
                    "index": 15794,
                    "name": "Aceclofenac"
                },
                {
                    "index": 14242,
                    "name": "Meloxicam"
                },
                {
                    "index": 15816,
                    "name": "Loxoprofen"
                },
                {
                    "index": 20449,
                    "name": "Flunoxaprofen"
                },
                {
                    "index": 15987,
                    "name": "Choline magnesium trisalicylate"
                },
                {
                    "index": 14509,
                    "name": "Meclofenamic acid"
                },
                {
                    "index": 15982,
                    "name": "Sulfasalazine"
                },
                {
                    "index": 14934,
                    "name": "Cyclosporine"
                },
                {
                    "index": 20450,
                    "name": "Proglumetacin"
                },
                {
                    "index": 15986,
                    "name": "Salsalate"
                },
                {
                    "index": 14206,
                    "name": "Chloroquine"
                },
                {
                    "index": 20239,
                    "name": "Penicillamine"
                },
                {
                    "index": 20451,
                    "name": "Benoxaprofen"
                },
                {
                    "index": 14419,
                    "name": "Dexibuprofen"
                },
                {
                    "index": 14219,
                    "name": "Flurbiprofen"
                },
                {
                    "index": 15781,
                    "name": "Acetylsalicylic acid"
                },
                {
                    "index": 14407,
                    "name": "Tofacitinib"
                },
                {
                    "index": 14273,
                    "name": "Azathioprine"
                },
                {
                    "index": 14175,
                    "name": "Nabumetone"
                },
                {
                    "index": 14227,
                    "name": "Etodolac"
                },
                {
                    "index": 15563,
                    "name": "Tocilizumab"
                },
                {
                    "index": 14191,
                    "name": "Piroxicam"
                },
                {
                    "index": 14274,
                    "name": "Auranofin"
                },
                {
                    "index": 17883,
                    "name": "Abatacept"
                },
                {
                    "index": 15989,
                    "name": "Tiaprofenic acid"
                },
                {
                    "index": 14197,
                    "name": "Fenoprofen"
                },
                {
                    "index": 14183,
                    "name": "Tolmetin"
                },
                {
                    "index": 14062,
                    "name": "Diflunisal"
                },
                {
                    "index": 16758,
                    "name": "Alclofenac"
                },
                {
                    "index": 14278,
                    "name": "Ketoprofen"
                },
                {
                    "index": 15984,
                    "name": "Oxaprozin"
                },
                {
                    "index": 16226,
                    "name": "Rituximab"
                },
                {
                    "index": 15101,
                    "name": "Celecoxib"
                },
                {
                    "index": 14600,
                    "name": "Rofecoxib"
                },
                {
                    "index": 14287,
                    "name": "Ibuprofen"
                },
                {
                    "index": 17398,
                    "name": "Anakinra"
                },
                {
                    "index": 14390,
                    "name": "Azapropazone"
                },
                {
                    "index": 15269,
                    "name": "Upadacitinib"
                },
                {
                    "index": 20452,
                    "name": "Iguratimod"
                },
                {
                    "index": 15860,
                    "name": "Sodium aurothiomalate"
                },
                {
                    "index": 16802,
                    "name": "Aurothioglucose"
                },
                {
                    "index": 20453,
                    "name": "Tenidap"
                },
                {
                    "index": 15662,
                    "name": "Sarilumab"
                },
                {
                    "index": 15666,
                    "name": "Baricitinib"
                },
                {
                    "index": 14347,
                    "name": "Hydroxychloroquine"
                },
                {
                    "index": 15791,
                    "name": "Crisaborole"
                },
                {
                    "index": 14193,
                    "name": "Hydroxyzine"
                },
                {
                    "index": 15586,
                    "name": "Tranilast"
                },
                {
                    "index": 14032,
                    "name": "Clobetasol propionate"
                },
                {
                    "index": 15432,
                    "name": "Pimecrolimus"
                },
                {
                    "index": 19330,
                    "name": "Dupilumab"
                },
                {
                    "index": 14251,
                    "name": "Tacrolimus"
                },
                {
                    "index": 17124,
                    "name": "Phylloquinone"
                },
                {
                    "index": 20396,
                    "name": "Calcium lactate"
                },
                {
                    "index": 14561,
                    "name": "Cholecalciferol"
                },
                {
                    "index": 19671,
                    "name": "Denosumab"
                },
                {
                    "index": 14560,
                    "name": "Ergocalciferol"
                },
                {
                    "index": 20417,
                    "name": "Magnesium citrate"
                },
                {
                    "index": 20311,
                    "name": "Calcium gluconate"
                },
                {
                    "index": 17736,
                    "name": "Minodronic acid"
                },
                {
                    "index": 20362,
                    "name": "Magnesium oxide"
                },
                {
                    "index": 14667,
                    "name": "Risedronic acid"
                },
                {
                    "index": 17746,
                    "name": "Eldecalcitol"
                },
                {
                    "index": 14623,
                    "name": "Prasterone"
                },
                {
                    "index": 18283,
                    "name": "Belimumab"
                },
                {
                    "index": 20455,
                    "name": "Clopenthixol"
                },
                {
                    "index": 15110,
                    "name": "Fluphenazine"
                },
                {
                    "index": 14952,
                    "name": "Haloperidol"
                },
                {
                    "index": 16434,
                    "name": "Acetophenazine"
                },
                {
                    "index": 14841,
                    "name": "Triflupromazine"
                },
                {
                    "index": 15001,
                    "name": "Paliperidone"
                },
                {
                    "index": 20457,
                    "name": "Ozagrel"
                },
                {
                    "index": 20458,
                    "name": "Pemirolast"
                },
                {
                    "index": 16323,
                    "name": "Nedocromil"
                },
                {
                    "index": 15399,
                    "name": "Acefylline"
                },
                {
                    "index": 17560,
                    "name": "Orciprenaline"
                },
                {
                    "index": 14784,
                    "name": "Theophylline"
                },
                {
                    "index": 14018,
                    "name": "Beclomethasone dipropionate"
                },
                {
                    "index": 20459,
                    "name": "Reproterol"
                },
                {
                    "index": 16352,
                    "name": "Norepinephrine"
                },
                {
                    "index": 17224,
                    "name": "Ibudilast"
                },
                {
                    "index": 17568,
                    "name": "Tulobuterol"
                },
                {
                    "index": 14038,
                    "name": "Fluticasone furoate"
                },
                {
                    "index": 16532,
                    "name": "Pirbuterol"
                },
                {
                    "index": 14014,
                    "name": "Flunisolide"
                },
                {
                    "index": 17561,
                    "name": "Bitolterol"
                },
                {
                    "index": 15971,
                    "name": "Omalizumab"
                },
                {
                    "index": 14848,
                    "name": "Terbutaline"
                },
                {
                    "index": 18755,
                    "name": "Cromoglicic acid"
                },
                {
                    "index": 14036,
                    "name": "Ciclesonide"
                },
                {
                    "index": 14601,
                    "name": "Zafirlukast"
                },
                {
                    "index": 15511,
                    "name": "Pranlukast"
                },
                {
                    "index": 20298,
                    "name": "Potassium citrate"
                },
                {
                    "index": 16085,
                    "name": "Potassium chloride"
                },
                {
                    "index": 14921,
                    "name": "Aspartic acid"
                },
                {
                    "index": 14652,
                    "name": "Potassium gluconate"
                },
                {
                    "index": 17634,
                    "name": "Potassium acetate"
                },
                {
                    "index": 17987,
                    "name": "Calcium chloride"
                },
                {
                    "index": 15826,
                    "name": "Dalfampridine"
                },
                {
                    "index": 15366,
                    "name": "Teriflunomide"
                },
                {
                    "index": 14590,
                    "name": "Ozanimod"
                },
                {
                    "index": 15882,
                    "name": "Acemetacin"
                },
                {
                    "index": 20161,
                    "name": "Ixekizumab"
                },
                {
                    "index": 14383,
                    "name": "Oxymetholone"
                },
                {
                    "index": 15107,
                    "name": "Vinblastine"
                },
                {
                    "index": 14849,
                    "name": "Mechlorethamine"
                },
                {
                    "index": 14954,
                    "name": "Cyclophosphamide"
                },
                {
                    "index": 20194,
                    "name": "Mogamulizumab"
                },
                {
                    "index": 20462,
                    "name": "Acetoxolone"
                },
                {
                    "index": 14597,
                    "name": "Dapsone"
                },
                {
                    "index": 15780,
                    "name": "Sulfapyridine"
                },
                {
                    "index": 14520,
                    "name": "Allantoin"
                },
                {
                    "index": 14585,
                    "name": "Ammonia"
                },
                {
                    "index": 14112,
                    "name": "Etretinate"
                },
                {
                    "index": 20166,
                    "name": "Risankizumab"
                },
                {
                    "index": 15809,
                    "name": "Tazarotene"
                },
                {
                    "index": 16496,
                    "name": "Fumaric Acid"
                },
                {
                    "index": 18416,
                    "name": "Canakinumab"
                },
                {
                    "index": 14283,
                    "name": "Probenecid"
                },
                {
                    "index": 14911,
                    "name": "Allopurinol"
                },
                {
                    "index": 20157,
                    "name": "Guaifenesin"
                },
                {
                    "index": 20294,
                    "name": "Phenyltoloxamine"
                },
                {
                    "index": 15210,
                    "name": "Mepyramine"
                },
                {
                    "index": 15122,
                    "name": "Tripelennamine"
                },
                {
                    "index": 20295,
                    "name": "Methapyrilene"
                },
                {
                    "index": 17252,
                    "name": "Antazoline"
                },
                {
                    "index": 15660,
                    "name": "Ebastine"
                },
                {
                    "index": 14249,
                    "name": "Pseudoephedrine"
                },
                {
                    "index": 17258,
                    "name": "Acrivastine"
                },
                {
                    "index": 16066,
                    "name": "Brompheniramine"
                },
                {
                    "index": 16490,
                    "name": "Pheniramine"
                },
                {
                    "index": 17229,
                    "name": "Cetirizine"
                },
                {
                    "index": 14266,
                    "name": "Fexofenadine"
                },
                {
                    "index": 17256,
                    "name": "Chlorcyclizine"
                },
                {
                    "index": 15783,
                    "name": "Cyclizine"
                },
                {
                    "index": 17245,
                    "name": "Phenindamine"
                },
                {
                    "index": 14950,
                    "name": "Loratadine"
                },
                {
                    "index": 17246,
                    "name": "Clofedanol"
                },
                {
                    "index": 16306,
                    "name": "Pentoxyverine"
                },
                {
                    "index": 20302,
                    "name": "Potassium Iodide"
                },
                {
                    "index": 15089,
                    "name": "Codeine"
                },
                {
                    "index": 14960,
                    "name": "Astemizole"
                },
                {
                    "index": 14881,
                    "name": "Phenylephrine"
                },
                {
                    "index": 14983,
                    "name": "Azelastine"
                },
                {
                    "index": 14783,
                    "name": "Caffeine"
                },
                {
                    "index": 14591,
                    "name": "Glycine"
                },
                {
                    "index": 20463,
                    "name": "Mebhydrolin"
                },
                {
                    "index": 14942,
                    "name": "Terfenadine"
                },
                {
                    "index": 17233,
                    "name": "Carbinoxamine"
                },
                {
                    "index": 17232,
                    "name": "Triprolidine"
                },
                {
                    "index": 17230,
                    "name": "Doxylamine"
                },
                {
                    "index": 15460,
                    "name": "Azatadine"
                },
                {
                    "index": 16331,
                    "name": "Epoprostenol"
                },
                {
                    "index": 20464,
                    "name": "Isoflupredone acetate"
                },
                {
                    "index": 14336,
                    "name": "Ephedrine"
                },
                {
                    "index": 15260,
                    "name": "Dexchlorpheniramine"
                },
                {
                    "index": 17259,
                    "name": "Bilastine"
                },
                {
                    "index": 14599,
                    "name": "Montelukast"
                },
                {
                    "index": 15241,
                    "name": "Rupatadine"
                },
                {
                    "index": 15086,
                    "name": "Clemastine"
                },
                {
                    "index": 14898,
                    "name": "Phenylpropanolamine"
                },
                {
                    "index": 17241,
                    "name": "Diphenylpyraline"
                },
                {
                    "index": 20321,
                    "name": "Methscopolamine"
                },
                {
                    "index": 14990,
                    "name": "Chlorpheniramine"
                },
                {
                    "index": 17231,
                    "name": "Dexbrompheniramine"
                },
                {
                    "index": 14381,
                    "name": "Levocetirizine"
                },
                {
                    "index": 15877,
                    "name": "Cyproheptadine"
                },
                {
                    "index": 15104,
                    "name": "Dextromethorphan"
                },
                {
                    "index": 15132,
                    "name": "Hydrocodone"
                },
                {
                    "index": 15171,
                    "name": "Dihydrocodeine"
                },
                {
                    "index": 19700,
                    "name": "Ramatroban"
                },
                {
                    "index": 14232,
                    "name": "Olopatadine"
                },
                {
                    "index": 15251,
                    "name": "Mizolastine"
                },
                {
                    "index": 20326,
                    "name": "Tramazoline"
                },
                {
                    "index": 20465,
                    "name": "Quifenadine"
                },
                {
                    "index": 14154,
                    "name": "Acetaminophen"
                },
                {
                    "index": 17237,
                    "name": "Desloratadine"
                },
                {
                    "index": 14237,
                    "name": "Acyclovir"
                },
                {
                    "index": 15594,
                    "name": "Brentuximab vedotin"
                },
                {
                    "index": 14914,
                    "name": "Procarbazine"
                },
                {
                    "index": 14868,
                    "name": "Thiotepa"
                },
                {
                    "index": 15315,
                    "name": "Dacarbazine"
                },
                {
                    "index": 14740,
                    "name": "Chlorambucil"
                },
                {
                    "index": 17524,
                    "name": "Carmustine"
                },
                {
                    "index": 15154,
                    "name": "Lomustine"
                },
                {
                    "index": 15893,
                    "name": "Bleomycin"
                },
                {
                    "index": 20466,
                    "name": "Lucinactant"
                },
                {
                    "index": 20467,
                    "name": "Colfosceril palmitate"
                },
                {
                    "index": 20468,
                    "name": "Cetyl alcohol"
                },
                {
                    "index": 16714,
                    "name": "Tyloxapol"
                },
                {
                    "index": 20469,
                    "name": "Calfactant"
                },
                {
                    "index": 20470,
                    "name": "Poractant alfa"
                },
                {
                    "index": 20471,
                    "name": "Beractant"
                },
                {
                    "index": 14435,
                    "name": "Insulin degludec"
                },
                {
                    "index": 15332,
                    "name": "Insulin glulisine"
                },
                {
                    "index": 14329,
                    "name": "Insulin detemir"
                },
                {
                    "index": 15331,
                    "name": "Insulin aspart"
                },
                {
                    "index": 20473,
                    "name": "Uracil mustard"
                },
                {
                    "index": 19250,
                    "name": "Pentostatin"
                },
                {
                    "index": 15804,
                    "name": "Cladribine"
                },
                {
                    "index": 16559,
                    "name": "Moxetumomab Pasudotox"
                },
                {
                    "index": 15134,
                    "name": "Hydroxyurea"
                },
                {
                    "index": 20474,
                    "name": "Talimogene laherparepvec"
                },
                {
                    "index": 15033,
                    "name": "Idelalisib"
                },
                {
                    "index": 16227,
                    "name": "Ibritumomab tiuxetan"
                },
                {
                    "index": 15363,
                    "name": "Bendamustine"
                },
                {
                    "index": 15671,
                    "name": "Duvelisib"
                },
                {
                    "index": 14467,
                    "name": "Copanlisib"
                },
                {
                    "index": 20475,
                    "name": "Beclabuvir"
                },
                {
                    "index": 14382,
                    "name": "Simeprevir"
                },
                {
                    "index": 20476,
                    "name": "Vaniprevir"
                },
                {
                    "index": 15043,
                    "name": "Asunaprevir"
                },
                {
                    "index": 14375,
                    "name": "Telaprevir"
                },
                {
                    "index": 15028,
                    "name": "Boceprevir"
                },
                {
                    "index": 14662,
                    "name": "Ribavirin"
                },
                {
                    "index": 15072,
                    "name": "Peginterferon alfa-2b"
                },
                {
                    "index": 15277,
                    "name": "Peginterferon alfa-2a"
                },
                {
                    "index": 14184,
                    "name": "Ritonavir"
                },
                {
                    "index": 15392,
                    "name": "Voxilaprevir"
                },
                {
                    "index": 15655,
                    "name": "Velpatasvir"
                },
                {
                    "index": 15042,
                    "name": "Elbasvir"
                },
                {
                    "index": 15725,
                    "name": "Glecaprevir"
                },
                {
                    "index": 15864,
                    "name": "Pibrentasvir"
                },
                {
                    "index": 14443,
                    "name": "Grazoprevir"
                },
                {
                    "index": 15811,
                    "name": "Ombitasvir"
                },
                {
                    "index": 20253,
                    "name": "Ledipasvir"
                },
                {
                    "index": 15040,
                    "name": "Paritaprevir"
                },
                {
                    "index": 14714,
                    "name": "Sofosbuvir"
                },
                {
                    "index": 15235,
                    "name": "Dasabuvir"
                },
                {
                    "index": 15038,
                    "name": "Daclatasvir"
                },
                {
                    "index": 20416,
                    "name": "Telbivudine"
                },
                {
                    "index": 15889,
                    "name": "Tenofovir disoproxil"
                },
                {
                    "index": 20477,
                    "name": "Clevudine"
                },
                {
                    "index": 14484,
                    "name": "Tenofovir"
                },
                {
                    "index": 15817,
                    "name": "Adefovir dipivoxil"
                },
                {
                    "index": 20478,
                    "name": "Entecavir"
                },
                {
                    "index": 14218,
                    "name": "Lamivudine"
                },
                {
                    "index": 14996,
                    "name": "Finasteride"
                },
                {
                    "index": 15433,
                    "name": "Alfuzosin"
                },
                {
                    "index": 15677,
                    "name": "Naftopidil"
                },
                {
                    "index": 14971,
                    "name": "Tadalafil"
                },
                {
                    "index": 14305,
                    "name": "Dutasteride"
                },
                {
                    "index": 15557,
                    "name": "Silodosin"
                },
                {
                    "index": 14541,
                    "name": "Tamsulosin"
                },
                {
                    "index": 14188,
                    "name": "Ciprofloxacin"
                },
                {
                    "index": 15326,
                    "name": "Ofloxacin"
                },
                {
                    "index": 15322,
                    "name": "Norfloxacin"
                },
                {
                    "index": 14949,
                    "name": "Chloramphenicol"
                },
                {
                    "index": 15318,
                    "name": "Lomefloxacin"
                },
                {
                    "index": 20479,
                    "name": "Mandelic acid"
                },
                {
                    "index": 20204,
                    "name": "Cefixime"
                },
                {
                    "index": 14201,
                    "name": "Oxytetracycline"
                },
                {
                    "index": 20398,
                    "name": "Aztreonam"
                },
                {
                    "index": 20248,
                    "name": "Loracarbef"
                },
                {
                    "index": 20290,
                    "name": "Sodium phosphate, monobasic"
                },
                {
                    "index": 15300,
                    "name": "Enoxacin"
                },
                {
                    "index": 15299,
                    "name": "Trimethoprim"
                },
                {
                    "index": 15180,
                    "name": "Fusidic acid"
                },
                {
                    "index": 14167,
                    "name": "Ampicillin"
                },
                {
                    "index": 20480,
                    "name": "Ceftolozane"
                },
                {
                    "index": 20250,
                    "name": "Ceftizoxime"
                },
                {
                    "index": 14333,
                    "name": "Cefonicid"
                },
                {
                    "index": 14230,
                    "name": "Tetracycline"
                },
                {
                    "index": 20407,
                    "name": "Kanamycin"
                },
                {
                    "index": 20289,
                    "name": "Clavulanic acid"
                },
                {
                    "index": 14145,
                    "name": "Cefmetazole"
                },
                {
                    "index": 18773,
                    "name": "Acetohydroxamic acid"
                },
                {
                    "index": 14773,
                    "name": "Cefaclor"
                },
                {
                    "index": 20410,
                    "name": "Mezlocillin"
                },
                {
                    "index": 20411,
                    "name": "Bacampicillin"
                },
                {
                    "index": 20207,
                    "name": "Cefadroxil"
                },
                {
                    "index": 14720,
                    "name": "Carbenicillin"
                },
                {
                    "index": 15504,
                    "name": "Cefradine"
                },
                {
                    "index": 20413,
                    "name": "Azlocillin"
                },
                {
                    "index": 20206,
                    "name": "Cyclacillin"
                },
                {
                    "index": 20415,
                    "name": "Ticarcillin"
                },
                {
                    "index": 20235,
                    "name": "Cefamandole"
                },
                {
                    "index": 15314,
                    "name": "Cinoxacin"
                },
                {
                    "index": 14291,
                    "name": "Amoxicillin"
                },
                {
                    "index": 20421,
                    "name": "Fosfomycin"
                },
                {
                    "index": 20481,
                    "name": "Cefapirin"
                },
                {
                    "index": 20422,
                    "name": "Meclocycline"
                },
                {
                    "index": 20482,
                    "name": "Furazidin"
                },
                {
                    "index": 14172,
                    "name": "Cefalotin"
                },
                {
                    "index": 15305,
                    "name": "Trovafloxacin"
                },
                {
                    "index": 20483,
                    "name": "Alatrofloxacin"
                },
                {
                    "index": 15456,
                    "name": "Amphotericin B"
                },
                {
                    "index": 15771,
                    "name": "Sulfisoxazole"
                },
                {
                    "index": 16867,
                    "name": "Demeclocycline"
                },
                {
                    "index": 14495,
                    "name": "Cefiderocol"
                },
                {
                    "index": 20237,
                    "name": "Relebactam"
                },
                {
                    "index": 20284,
                    "name": "Plazomicin"
                },
                {
                    "index": 20484,
                    "name": "Nifurtoinol"
                },
                {
                    "index": 15914,
                    "name": "Minocycline"
                },
                {
                    "index": 20328,
                    "name": "Methenamine"
                },
                {
                    "index": 20485,
                    "name": "Cefprozil"
                },
                {
                    "index": 15476,
                    "name": "Dirithromycin"
                },
                {
                    "index": 16036,
                    "name": "Acetylcysteine"
                },
                {
                    "index": 15425,
                    "name": "Doxycycline"
                },
                {
                    "index": 16281,
                    "name": "Cysteine"
                },
                {
                    "index": 15653,
                    "name": "Lesinurad"
                },
                {
                    "index": 14352,
                    "name": "Topiroxostat"
                },
                {
                    "index": 14574,
                    "name": "Sulfinpyrazone"
                },
                {
                    "index": 15838,
                    "name": "Benzbromarone"
                },
                {
                    "index": 19245,
                    "name": "Febuxostat"
                },
                {
                    "index": 20486,
                    "name": "Rasburicase"
                },
                {
                    "index": 15076,
                    "name": "Esmolol"
                },
                {
                    "index": 15200,
                    "name": "Vernakalant"
                },
                {
                    "index": 16550,
                    "name": "Dipyridamole"
                },
                {
                    "index": 14258,
                    "name": "Quinidine"
                },
                {
                    "index": 14316,
                    "name": "Flecainide"
                },
                {
                    "index": 15150,
                    "name": "Propafenone"
                },
                {
                    "index": 15102,
                    "name": "Sotalol"
                },
                {
                    "index": 15840,
                    "name": "Digoxin"
                },
                {
                    "index": 14368,
                    "name": "Dronedarone"
                },
                {
                    "index": 15418,
                    "name": "Dofetilide"
                },
                {
                    "index": 17624,
                    "name": "Acetyldigitoxin"
                },
                {
                    "index": 19350,
                    "name": "Ecallantide"
                },
                {
                    "index": 19351,
                    "name": "Lanadelumab"
                },
                {
                    "index": 18576,
                    "name": "Icatibant"
                },
                {
                    "index": 16443,
                    "name": "Stanozolol"
                },
                {
                    "index": 17338,
                    "name": "Metaraminol"
                },
                {
                    "index": 14641,
                    "name": "Dopamine"
                },
                {
                    "index": 17339,
                    "name": "Methoxamine"
                },
                {
                    "index": 15417,
                    "name": "Phentermine"
                },
                {
                    "index": 20151,
                    "name": "Mephentermine"
                },
                {
                    "index": 14562,
                    "name": "Alfacalcidol"
                },
                {
                    "index": 20401,
                    "name": "Paromomycin"
                },
                {
                    "index": 20488,
                    "name": "Pegloticase"
                },
                {
                    "index": 14338,
                    "name": "Colchicine"
                },
                {
                    "index": 20489,
                    "name": "Cinchophen"
                },
                {
                    "index": 14816,
                    "name": "Edetic acid"
                },
                {
                    "index": 17730,
                    "name": "Ibandronate"
                },
                {
                    "index": 14910,
                    "name": "Pyrazinamide"
                },
                {
                    "index": 20490,
                    "name": "Tiocarlide"
                },
                {
                    "index": 15600,
                    "name": "Bedaquiline"
                },
                {
                    "index": 20402,
                    "name": "Ethambutol"
                },
                {
                    "index": 14828,
                    "name": "Cycloserine"
                },
                {
                    "index": 14772,
                    "name": "Aminosalicylic acid"
                },
                {
                    "index": 15494,
                    "name": "Rifapentine"
                },
                {
                    "index": 18590,
                    "name": "Streptomycin"
                },
                {
                    "index": 14267,
                    "name": "Isoniazid"
                },
                {
                    "index": 20340,
                    "name": "Ethionamide"
                },
                {
                    "index": 20491,
                    "name": "Enviomycin"
                },
                {
                    "index": 14286,
                    "name": "Rifampicin"
                },
                {
                    "index": 20428,
                    "name": "Capreomycin"
                },
                {
                    "index": 20492,
                    "name": "Amithiozone"
                },
                {
                    "index": 14972,
                    "name": "Disulfiram"
                },
                {
                    "index": 18756,
                    "name": "Acamprosate"
                },
                {
                    "index": 15879,
                    "name": "Nalmefene"
                },
                {
                    "index": 15856,
                    "name": "Naltrexone"
                },
                {
                    "index": 15179,
                    "name": "Nicotinamide"
                },
                {
                    "index": 15696,
                    "name": "Trifarotene"
                },
                {
                    "index": 14221,
                    "name": "Norethisterone"
                },
                {
                    "index": 14670,
                    "name": "Drospirenone"
                },
                {
                    "index": 14270,
                    "name": "Ethinylestradiol"
                },
                {
                    "index": 20494,
                    "name": "Motretinide"
                },
                {
                    "index": 14211,
                    "name": "Estrone"
                },
                {
                    "index": 14235,
                    "name": "Estradiol"
                },
                {
                    "index": 14268,
                    "name": "Norgestimate"
                },
                {
                    "index": 14733,
                    "name": "Levomefolic acid"
                },
                {
                    "index": 14229,
                    "name": "Tretinoin"
                },
                {
                    "index": 20495,
                    "name": "Ozenoxacin"
                },
                {
                    "index": 14545,
                    "name": "Clindamycin"
                },
                {
                    "index": 20318,
                    "name": "Sulfacetamide"
                },
                {
                    "index": 16347,
                    "name": "Ferrous fumarate"
                },
                {
                    "index": 20496,
                    "name": "Nadifloxacin"
                },
                {
                    "index": 16741,
                    "name": "Azelaic acid"
                },
                {
                    "index": 14936,
                    "name": "Erythromycin"
                },
                {
                    "index": 17345,
                    "name": "Resorcinol"
                },
                {
                    "index": 16276,
                    "name": "Adapalene"
                },
                {
                    "index": 20257,
                    "name": "Sarecycline"
                },
                {
                    "index": 20293,
                    "name": "Inositol"
                },
                {
                    "index": 15885,
                    "name": "Inositol nicotinate"
                },
                {
                    "index": 14880,
                    "name": "Nicotine"
                },
                {
                    "index": 20497,
                    "name": "Nicotinyl alcohol"
                },
                {
                    "index": 14399,
                    "name": "Ticagrelor"
                },
                {
                    "index": 18695,
                    "name": "Eptifibatide"
                },
                {
                    "index": 14379,
                    "name": "Prasugrel"
                },
                {
                    "index": 17226,
                    "name": "Tirofiban"
                },
                {
                    "index": 14969,
                    "name": "Clopidogrel"
                },
                {
                    "index": 15747,
                    "name": "Voxelotor"
                },
                {
                    "index": 14424,
                    "name": "Methylene blue"
                },
                {
                    "index": 20500,
                    "name": "Succinimide"
                },
                {
                    "index": 20501,
                    "name": "Icosapent ethyl"
                },
                {
                    "index": 17407,
                    "name": "Pravastatin"
                },
                {
                    "index": 14961,
                    "name": "Simvastatin"
                },
                {
                    "index": 14135,
                    "name": "Lovastatin"
                },
                {
                    "index": 14506,
                    "name": "Niacin"
                },
                {
                    "index": 16264,
                    "name": "Ciprofibrate"
                },
                {
                    "index": 14284,
                    "name": "Fenofibrate"
                },
                {
                    "index": 16268,
                    "name": "Fenofibric acid"
                },
                {
                    "index": 14658,
                    "name": "Omega-3 fatty acids"
                },
                {
                    "index": 14298,
                    "name": "Rosuvastatin"
                },
                {
                    "index": 14324,
                    "name": "Gemfibrozil"
                },
                {
                    "index": 14297,
                    "name": "Atorvastatin"
                },
                {
                    "index": 17739,
                    "name": "Dihydrotachysterol"
                },
                {
                    "index": 20405,
                    "name": "Calcium acetate"
                },
                {
                    "index": 20503,
                    "name": "Calcium carbonate"
                },
                {
                    "index": 14559,
                    "name": "Calcitriol"
                },
                {
                    "index": 20419,
                    "name": "Calcium glucoheptonate"
                },
                {
                    "index": 20420,
                    "name": "Calcium glycerophosphate"
                },
                {
                    "index": 14704,
                    "name": "Calcifediol"
                },
                {
                    "index": 20504,
                    "name": "Calcium lactate gluconate"
                },
                {
                    "index": 19175,
                    "name": "Testolactone"
                },
                {
                    "index": 15601,
                    "name": "Formestane"
                },
                {
                    "index": 14213,
                    "name": "Tamoxifen"
                },
                {
                    "index": 15431,
                    "name": "Dihydroergotamine"
                },
                {
                    "index": 15244,
                    "name": "Lasmiditan"
                },
                {
                    "index": 20505,
                    "name": "Oxetorone"
                },
                {
                    "index": 15448,
                    "name": "Lisuride"
                },
                {
                    "index": 19287,
                    "name": "Fremanezumab"
                },
                {
                    "index": 19286,
                    "name": "Eptinezumab"
                },
                {
                    "index": 15320,
                    "name": "Frovatriptan"
                },
                {
                    "index": 20176,
                    "name": "Erenumab"
                },
                {
                    "index": 19288,
                    "name": "Galcanezumab"
                },
                {
                    "index": 14900,
                    "name": "Almotriptan"
                },
                {
                    "index": 14902,
                    "name": "Rizatriptan"
                },
                {
                    "index": 14596,
                    "name": "Eletriptan"
                },
                {
                    "index": 14897,
                    "name": "Zolmitriptan"
                },
                {
                    "index": 14908,
                    "name": "Ubrogepant"
                },
                {
                    "index": 15687,
                    "name": "Rimegepant"
                },
                {
                    "index": 14901,
                    "name": "Naratriptan"
                },
                {
                    "index": 15458,
                    "name": "Ergotamine"
                },
                {
                    "index": 14899,
                    "name": "Sumatriptan"
                },
                {
                    "index": 14483,
                    "name": "Estradiol valerate"
                },
                {
                    "index": 14481,
                    "name": "Estradiol cypionate"
                },
                {
                    "index": 16376,
                    "name": "Quinestrol"
                },
                {
                    "index": 20506,
                    "name": "Colestilan chloride"
                },
                {
                    "index": 14507,
                    "name": "Clofibrate"
                },
                {
                    "index": 20279,
                    "name": "Tyramine"
                },
                {
                    "index": 16879,
                    "name": "Dextrothyroxine"
                },
                {
                    "index": 15510,
                    "name": "Bezafibrate"
                },
                {
                    "index": 16360,
                    "name": "Probucol"
                },
                {
                    "index": 15477,
                    "name": "Ezetimibe"
                },
                {
                    "index": 20427,
                    "name": "Cholestyramine"
                },
                {
                    "index": 14721,
                    "name": "Orlistat"
                },
                {
                    "index": 17307,
                    "name": "Diethylpropion"
                },
                {
                    "index": 17309,
                    "name": "Phendimetrazine"
                },
                {
                    "index": 15188,
                    "name": "Lorcaserin"
                },
                {
                    "index": 15174,
                    "name": "Metamfetamine"
                },
                {
                    "index": 20305,
                    "name": "Chlorphentermine"
                },
                {
                    "index": 14957,
                    "name": "Fenfluramine"
                },
                {
                    "index": 15151,
                    "name": "Dexfenfluramine"
                },
                {
                    "index": 15074,
                    "name": "Amphetamine"
                },
                {
                    "index": 17548,
                    "name": "Cetilistat"
                },
                {
                    "index": 16353,
                    "name": "Mazindol"
                },
                {
                    "index": 20507,
                    "name": "Mefenorex"
                },
                {
                    "index": 15555,
                    "name": "Rimonabant"
                },
                {
                    "index": 20508,
                    "name": "Fenproporex"
                },
                {
                    "index": 15274,
                    "name": "Benzphetamine"
                },
                {
                    "index": 17306,
                    "name": "Phenmetrazine"
                },
                {
                    "index": 15484,
                    "name": "Sibutramine"
                },
                {
                    "index": 15413,
                    "name": "Gemcitabine"
                },
                {
                    "index": 15036,
                    "name": "Olaparib"
                },
                {
                    "index": 20434,
                    "name": "Altretamine"
                },
                {
                    "index": 14875,
                    "name": "Tretamine"
                },
                {
                    "index": 14737,
                    "name": "Carboplatin"
                },
                {
                    "index": 15057,
                    "name": "Rucaparib"
                },
                {
                    "index": 20509,
                    "name": "Belotecan"
                },
                {
                    "index": 14186,
                    "name": "Cisplatin"
                },
                {
                    "index": 20212,
                    "name": "Melphalan"
                },
                {
                    "index": 15481,
                    "name": "Topotecan"
                },
                {
                    "index": 14612,
                    "name": "Trabectedin"
                },
                {
                    "index": 15492,
                    "name": "Estramustine"
                },
                {
                    "index": 14787,
                    "name": "Flutamide"
                },
                {
                    "index": 20510,
                    "name": "Acipimox"
                },
                {
                    "index": 15247,
                    "name": "Elagolix"
                },
                {
                    "index": 16575,
                    "name": "Nafarelin"
                },
                {
                    "index": 15507,
                    "name": "Mestranol"
                },
                {
                    "index": 14203,
                    "name": "Medroxyprogesterone acetate"
                },
                {
                    "index": 15638,
                    "name": "Norethynodrel"
                },
                {
                    "index": 16468,
                    "name": "Goserelin"
                },
                {
                    "index": 14685,
                    "name": "Progesterone"
                },
                {
                    "index": 16572,
                    "name": "Leuprolide"
                },
                {
                    "index": 14528,
                    "name": "Danazol"
                },
                {
                    "index": 16593,
                    "name": "Tranexamic acid"
                },
                {
                    "index": 15435,
                    "name": "Megestrol acetate"
                },
                {
                    "index": 15482,
                    "name": "Ergoloid mesylate"
                },
                {
                    "index": 15679,
                    "name": "Vinpocetine"
                },
                {
                    "index": 15640,
                    "name": "Dihydroergocornine"
                },
                {
                    "index": 15610,
                    "name": "Vorapaxar"
                },
                {
                    "index": 16825,
                    "name": "Ardeparin"
                },
                {
                    "index": 14510,
                    "name": "Heparin"
                },
                {
                    "index": 14774,
                    "name": "Enoxaparin"
                },
                {
                    "index": 20332,
                    "name": "Adomiparin"
                },
                {
                    "index": 14768,
                    "name": "Dalteparin"
                },
                {
                    "index": 20333,
                    "name": "Parnaparin"
                },
                {
                    "index": 20334,
                    "name": "Heparin, bovine"
                },
                {
                    "index": 14216,
                    "name": "Warfarin"
                },
                {
                    "index": 16635,
                    "name": "Fondaparinux"
                },
                {
                    "index": 20511,
                    "name": "Givosiran"
                },
                {
                    "index": 14143,
                    "name": "Dicoumarol"
                },
                {
                    "index": 14265,
                    "name": "Phenprocoumon"
                },
                {
                    "index": 14231,
                    "name": "Irinotecan"
                },
                {
                    "index": 16517,
                    "name": "Aflibercept"
                },
                {
                    "index": 15599,
                    "name": "Regorafenib"
                },
                {
                    "index": 14736,
                    "name": "Oxaliplatin"
                },
                {
                    "index": 14758,
                    "name": "Capecitabine"
                },
                {
                    "index": 14604,
                    "name": "Etoposide"
                },
                {
                    "index": 20336,
                    "name": "Temozolomide"
                },
                {
                    "index": 17153,
                    "name": "Diphenoxylate"
                },
                {
                    "index": 14150,
                    "name": "Morphine"
                },
                {
                    "index": 15125,
                    "name": "Loperamide"
                },
                {
                    "index": 20312,
                    "name": "Difenoxin"
                },
                {
                    "index": 14115,
                    "name": "Bismuth subsalicylate"
                },
                {
                    "index": 16282,
                    "name": "Telotristat ethyl"
                },
                {
                    "index": 17042,
                    "name": "Ramucirumab"
                },
                {
                    "index": 19603,
                    "name": "Nivolumab"
                },
                {
                    "index": 17718,
                    "name": "Necitumumab"
                },
                {
                    "index": 17335,
                    "name": "Porfimer sodium"
                },
                {
                    "index": 15027,
                    "name": "Crizotinib"
                },
                {
                    "index": 15092,
                    "name": "Vinorelbine"
                },
                {
                    "index": 14413,
                    "name": "Nintedanib"
                },
                {
                    "index": 15046,
                    "name": "Icotinib"
                },
                {
                    "index": 14155,
                    "name": "Gefitinib"
                },
                {
                    "index": 14187,
                    "name": "Erlotinib"
                },
                {
                    "index": 15054,
                    "name": "Lorlatinib"
                },
                {
                    "index": 19604,
                    "name": "Pembrolizumab"
                },
                {
                    "index": 14802,
                    "name": "Pemetrexed"
                },
                {
                    "index": 17720,
                    "name": "Olmutinib"
                },
                {
                    "index": 17030,
                    "name": "Alectinib"
                },
                {
                    "index": 15683,
                    "name": "Brigatinib"
                },
                {
                    "index": 14793,
                    "name": "Docetaxel"
                },
                {
                    "index": 15613,
                    "name": "Ceritinib"
                },
                {
                    "index": 20156,
                    "name": "Dactinomycin"
                },
                {
                    "index": 15271,
                    "name": "Mitomycin"
                },
                {
                    "index": 15595,
                    "name": "Fidaxomicin"
                },
                {
                    "index": 14185,
                    "name": "Vancomycin"
                },
                {
                    "index": 20251,
                    "name": "Ceftibuten"
                },
                {
                    "index": 18312,
                    "name": "Phenoxymethylpenicillin"
                },
                {
                    "index": 14995,
                    "name": "Clarithromycin"
                },
                {
                    "index": 14279,
                    "name": "Sulfamethoxazole"
                },
                {
                    "index": 14288,
                    "name": "Benzylpenicillin"
                },
                {
                    "index": 20249,
                    "name": "Cefuroxime"
                },
                {
                    "index": 20409,
                    "name": "Procaine benzylpenicillin"
                },
                {
                    "index": 14195,
                    "name": "Cephalexin"
                },
                {
                    "index": 14199,
                    "name": "Sulfamethizole"
                },
                {
                    "index": 18574,
                    "name": "Cefdinir"
                },
                {
                    "index": 20208,
                    "name": "Cefepime"
                },
                {
                    "index": 15419,
                    "name": "Azithromycin"
                },
                {
                    "index": 14318,
                    "name": "Ceftriaxone"
                },
                {
                    "index": 20203,
                    "name": "Ceftazidime"
                },
                {
                    "index": 14335,
                    "name": "Cefotetan"
                },
                {
                    "index": 20414,
                    "name": "Cefoxitin"
                },
                {
                    "index": 15478,
                    "name": "Telithromycin"
                },
                {
                    "index": 14882,
                    "name": "Linezolid"
                },
                {
                    "index": 15289,
                    "name": "Moxifloxacin"
                },
                {
                    "index": 15508,
                    "name": "Quinupristin"
                },
                {
                    "index": 15522,
                    "name": "Dalfopristin"
                },
                {
                    "index": 14309,
                    "name": "Levofloxacin"
                },
                {
                    "index": 14813,
                    "name": "Troglitazone"
                },
                {
                    "index": 14166,
                    "name": "Acetohexamide"
                },
                {
                    "index": 14603,
                    "name": "Chlorpropamide"
                },
                {
                    "index": 15770,
                    "name": "Glimepiride"
                },
                {
                    "index": 17950,
                    "name": "Dulaglutide"
                },
                {
                    "index": 20513,
                    "name": "Gemigliptin"
                },
                {
                    "index": 14475,
                    "name": "Semaglutide"
                },
                {
                    "index": 20514,
                    "name": "Tofogliflozin"
                },
                {
                    "index": 20515,
                    "name": "Luseogliflozin"
                },
                {
                    "index": 20516,
                    "name": "Trelagliptin"
                },
                {
                    "index": 15847,
                    "name": "Empagliflozin"
                },
                {
                    "index": 15198,
                    "name": "Alogliptin"
                },
                {
                    "index": 14551,
                    "name": "Canagliflozin"
                },
                {
                    "index": 17833,
                    "name": "Voglibose"
                },
                {
                    "index": 17949,
                    "name": "Albiglutide"
                },
                {
                    "index": 20517,
                    "name": "Ipragliflozin"
                },
                {
                    "index": 20518,
                    "name": "Teneligliptin"
                },
                {
                    "index": 20519,
                    "name": "Anagliptin"
                },
                {
                    "index": 15753,
                    "name": "Miglitol"
                },
                {
                    "index": 20278,
                    "name": "Buformin"
                },
                {
                    "index": 15598,
                    "name": "Linagliptin"
                },
                {
                    "index": 16588,
                    "name": "Metformin"
                },
                {
                    "index": 14304,
                    "name": "Tolbutamide"
                },
                {
                    "index": 15373,
                    "name": "Lobeglitazone"
                },
                {
                    "index": 17832,
                    "name": "Acarbose"
                },
                {
                    "index": 15501,
                    "name": "Sitagliptin"
                },
                {
                    "index": 15201,
                    "name": "Dapagliflozin"
                },
                {
                    "index": 15013,
                    "name": "Saxagliptin"
                },
                {
                    "index": 15801,
                    "name": "Carbutamide"
                },
                {
                    "index": 15784,
                    "name": "Gliquidone"
                },
                {
                    "index": 17951,
                    "name": "Lixisenatide"
                },
                {
                    "index": 14293,
                    "name": "Glipizide"
                },
                {
                    "index": 17903,
                    "name": "Vildagliptin"
                },
                {
                    "index": 15796,
                    "name": "Glibornuride"
                },
                {
                    "index": 15785,
                    "name": "Glisoxepide"
                },
                {
                    "index": 14303,
                    "name": "Gliclazide"
                },
                {
                    "index": 14385,
                    "name": "Liraglutide"
                },
                {
                    "index": 15283,
                    "name": "Insulin glargine"
                },
                {
                    "index": 15878,
                    "name": "Mitiglinide"
                },
                {
                    "index": 14259,
                    "name": "Repaglinide"
                },
                {
                    "index": 15282,
                    "name": "Insulin lispro"
                },
                {
                    "index": 20520,
                    "name": "Omarigliptin"
                },
                {
                    "index": 20521,
                    "name": "Evogliptin"
                },
                {
                    "index": 14455,
                    "name": "Ertugliflozin"
                },
                {
                    "index": 15778,
                    "name": "Tolazamide"
                },
                {
                    "index": 14306,
                    "name": "Pioglitazone"
                },
                {
                    "index": 14799,
                    "name": "Insulin human"
                },
                {
                    "index": 14222,
                    "name": "Nateglinide"
                },
                {
                    "index": 14280,
                    "name": "Glyburide"
                },
                {
                    "index": 16719,
                    "name": "Isosorbide"
                },
                {
                    "index": 18311,
                    "name": "Ripasudil"
                },
                {
                    "index": 20522,
                    "name": "Omidenepag isopropyl"
                },
                {
                    "index": 15090,
                    "name": "Clobazam"
                },
                {
                    "index": 14656,
                    "name": "Cannabidiol"
                },
                {
                    "index": 14378,
                    "name": "Rufinamide"
                },
                {
                    "index": 20523,
                    "name": "Pramiracetam"
                },
                {
                    "index": 16070,
                    "name": "Diphenidol"
                },
                {
                    "index": 14224,
                    "name": "Meclizine"
                },
                {
                    "index": 17238,
                    "name": "Dimenhydrinate"
                },
                {
                    "index": 15106,
                    "name": "Cinnarizine"
                },
                {
                    "index": 20335,
                    "name": "Racementhol"
                },
                {
                    "index": 20524,
                    "name": "Chloroxine"
                },
                {
                    "index": 20525,
                    "name": "Selenium Sulfide"
                },
                {
                    "index": 14439,
                    "name": "Butamben"
                },
                {
                    "index": 17652,
                    "name": "Tetracaine"
                },
                {
                    "index": 14842,
                    "name": "Cinchocaine"
                },
                {
                    "index": 16544,
                    "name": "Dyclonine"
                },
                {
                    "index": 15088,
                    "name": "Bupivacaine"
                },
                {
                    "index": 14489,
                    "name": "Aluminum acetate"
                },
                {
                    "index": 14940,
                    "name": "Lidocaine"
                },
                {
                    "index": 17377,
                    "name": "Phenol"
                },
                {
                    "index": 14638,
                    "name": "Benzyl alcohol"
                },
                {
                    "index": 20526,
                    "name": "Climbazole"
                },
                {
                    "index": 20529,
                    "name": "Acetarsol"
                },
                {
                    "index": 20431,
                    "name": "Diiodohydroxyquinoline"
                },
                {
                    "index": 20530,
                    "name": "Quinfamide"
                },
                {
                    "index": 20531,
                    "name": "Diloxanide furoate"
                },
                {
                    "index": 19712,
                    "name": "Doxapram"
                },
                {
                    "index": 20532,
                    "name": "Sotagliflozin"
                },
                {
                    "index": 14854,
                    "name": "Ambenonium"
                },
                {
                    "index": 14861,
                    "name": "Neostigmine"
                },
                {
                    "index": 20533,
                    "name": "Distigmine"
                },
                {
                    "index": 17170,
                    "name": "Pyridostigmine"
                },
                {
                    "index": 14511,
                    "name": "Dantrolene"
                },
                {
                    "index": 15222,
                    "name": "Levomilnacipran"
                },
                {
                    "index": 15087,
                    "name": "Venlafaxine"
                },
                {
                    "index": 15211,
                    "name": "Desvenlafaxine"
                },
                {
                    "index": 15232,
                    "name": "Brexpiprazole"
                },
                {
                    "index": 15094,
                    "name": "Mirtazapine"
                },
                {
                    "index": 20534,
                    "name": "Gepirone"
                },
                {
                    "index": 15146,
                    "name": "Nefazodone"
                },
                {
                    "index": 15083,
                    "name": "Reboxetine"
                },
                {
                    "index": 19222,
                    "name": "Isocarboxazid"
                },
                {
                    "index": 15035,
                    "name": "Vortioxetine"
                },
                {
                    "index": 15197,
                    "name": "Mianserin"
                },
                {
                    "index": 14962,
                    "name": "Trazodone"
                },
                {
                    "index": 15209,
                    "name": "Vilazodone"
                },
                {
                    "index": 15119,
                    "name": "Tranylcypromine"
                },
                {
                    "index": 15360,
                    "name": "Agomelatine"
                },
                {
                    "index": 15148,
                    "name": "Escitalopram"
                },
                {
                    "index": 14177,
                    "name": "Duloxetine"
                },
                {
                    "index": 15233,
                    "name": "Dosulepin"
                },
                {
                    "index": 20535,
                    "name": "Clortermine"
                },
                {
                    "index": 14833,
                    "name": "Riboflavin"
                },
                {
                    "index": 14707,
                    "name": "Ferrous sulfate anhydrous"
                },
                {
                    "index": 15886,
                    "name": "Sodium fluoride"
                },
                {
                    "index": 14052,
                    "name": "Vitamin A"
                },
                {
                    "index": 14706,
                    "name": "Cyanocobalamin"
                },
                {
                    "index": 14678,
                    "name": "Folic acid"
                },
                {
                    "index": 14054,
                    "name": "Vitamin E"
                },
                {
                    "index": 20265,
                    "name": "Tazobactam"
                },
                {
                    "index": 18687,
                    "name": "Cilastatin"
                },
                {
                    "index": 14797,
                    "name": "Imipenem"
                },
                {
                    "index": 17376,
                    "name": "Cefotaxime"
                },
                {
                    "index": 14465,
                    "name": "Omadacycline"
                },
                {
                    "index": 20412,
                    "name": "Amikacin"
                },
                {
                    "index": 20308,
                    "name": "Telavancin"
                },
                {
                    "index": 14332,
                    "name": "Cefazolin"
                },
                {
                    "index": 15353,
                    "name": "Garenoxacin"
                },
                {
                    "index": 15325,
                    "name": "Gemifloxacin"
                },
                {
                    "index": 20360,
                    "name": "Ertapenem"
                },
                {
                    "index": 14285,
                    "name": "Gatifloxacin"
                },
                {
                    "index": 15697,
                    "name": "Lefamulin"
                },
                {
                    "index": 20351,
                    "name": "Meropenem"
                },
                {
                    "index": 20536,
                    "name": "Raxibacumab"
                },
                {
                    "index": 20537,
                    "name": "Obiltoxaximab"
                },
                {
                    "index": 20404,
                    "name": "Sulbactam"
                },
                {
                    "index": 14980,
                    "name": "Metronidazole"
                },
                {
                    "index": 20418,
                    "name": "Tobramycin"
                },
                {
                    "index": 17437,
                    "name": "Metyrosine"
                },
                {
                    "index": 16845,
                    "name": "Phenoxybenzamine"
                },
                {
                    "index": 16844,
                    "name": "Phentolamine"
                },
                {
                    "index": 15622,
                    "name": "Sonidegib"
                },
                {
                    "index": 14401,
                    "name": "Vismodegib"
                },
                {
                    "index": 15668,
                    "name": "Apalutamide"
                },
                {
                    "index": 14408,
                    "name": "Enzalutamide"
                },
                {
                    "index": 20538,
                    "name": "Radium Ra 223 dichloride"
                },
                {
                    "index": 20539,
                    "name": "Iron sucrose"
                },
                {
                    "index": 14790,
                    "name": "Primaquine"
                },
                {
                    "index": 14426,
                    "name": "Artesunate"
                },
                {
                    "index": 20540,
                    "name": "Pyronaridine"
                },
                {
                    "index": 14998,
                    "name": "Halofantrine"
                },
                {
                    "index": 14701,
                    "name": "Mefloquine"
                },
                {
                    "index": 15438,
                    "name": "Sulfadiazine"
                },
                {
                    "index": 15808,
                    "name": "Pyrimethamine"
                },
                {
                    "index": 15485,
                    "name": "Atovaquone"
                },
                {
                    "index": 15213,
                    "name": "Lumefantrine"
                },
                {
                    "index": 15873,
                    "name": "Artemisinin"
                },
                {
                    "index": 15145,
                    "name": "Proguanil"
                },
                {
                    "index": 14951,
                    "name": "Quinine"
                },
                {
                    "index": 14682,
                    "name": "Trimetrexate"
                },
                {
                    "index": 14696,
                    "name": "Pentamidine"
                },
                {
                    "index": 16495,
                    "name": "Quizartinib"
                },
                {
                    "index": 14791,
                    "name": "Mitoxantrone"
                },
                {
                    "index": 15149,
                    "name": "Idarubicin"
                },
                {
                    "index": 15412,
                    "name": "Histamine"
                },
                {
                    "index": 15414,
                    "name": "Azacitidine"
                },
                {
                    "index": 14713,
                    "name": "Decitabine"
                },
                {
                    "index": 17545,
                    "name": "Gemtuzumab ozogamicin"
                },
                {
                    "index": 14628,
                    "name": "Tioguanine"
                },
                {
                    "index": 15739,
                    "name": "Ivosidenib"
                },
                {
                    "index": 15263,
                    "name": "Enasidenib"
                },
                {
                    "index": 14463,
                    "name": "Gilteritinib"
                },
                {
                    "index": 15018,
                    "name": "Midostaurin"
                },
                {
                    "index": 14663,
                    "name": "Aldesleukin"
                },
                {
                    "index": 14461,
                    "name": "Glasdegib"
                },
                {
                    "index": 20541,
                    "name": "Plicamycin"
                },
                {
                    "index": 17654,
                    "name": "Valrubicin"
                },
                {
                    "index": 18165,
                    "name": "Tasonermin"
                },
                {
                    "index": 15205,
                    "name": "Pazopanib"
                },
                {
                    "index": 17893,
                    "name": "Olaratumab"
                },
                {
                    "index": 16230,
                    "name": "Obinutuzumab"
                },
                {
                    "index": 14411,
                    "name": "Ibrutinib"
                },
                {
                    "index": 16229,
                    "name": "Ofatumumab"
                },
                {
                    "index": 15654,
                    "name": "Venetoclax"
                },
                {
                    "index": 14945,
                    "name": "Sorafenib"
                },
                {
                    "index": 14765,
                    "name": "Cobimetinib"
                },
                {
                    "index": 15602,
                    "name": "Trametinib"
                },
                {
                    "index": 15391,
                    "name": "Binimetinib"
                },
                {
                    "index": 17991,
                    "name": "Ipilimumab"
                },
                {
                    "index": 15243,
                    "name": "Encorafenib"
                },
                {
                    "index": 15603,
                    "name": "Dabrafenib"
                },
                {
                    "index": 20158,
                    "name": "Daratumumab"
                },
                {
                    "index": 19641,
                    "name": "Elotuzumab"
                },
                {
                    "index": 19281,
                    "name": "Carfilzomib"
                },
                {
                    "index": 14606,
                    "name": "Thalidomide"
                },
                {
                    "index": 15206,
                    "name": "Panobinostat"
                },
                {
                    "index": 16753,
                    "name": "Lenalidomide"
                },
                {
                    "index": 15367,
                    "name": "Pomalidomide"
                },
                {
                    "index": 15077,
                    "name": "Bortezomib"
                },
                {
                    "index": 20159,
                    "name": "Isatuximab"
                },
                {
                    "index": 14493,
                    "name": "Dexamethasone acetate"
                },
                {
                    "index": 18102,
                    "name": "Plerixafor"
                },
                {
                    "index": 15240,
                    "name": "Ixazomib"
                },
                {
                    "index": 14113,
                    "name": "Tamibarotene"
                },
                {
                    "index": 14313,
                    "name": "Arsenic trioxide"
                },
                {
                    "index": 15876,
                    "name": "Mesalazine"
                },
                {
                    "index": 15985,
                    "name": "Balsalazide"
                },
                {
                    "index": 15030,
                    "name": "Ponatinib"
                },
                {
                    "index": 15970,
                    "name": "Radotinib"
                },
                {
                    "index": 19886,
                    "name": "Omacetaxine mepesuccinate"
                },
                {
                    "index": 15597,
                    "name": "Ruxolitinib"
                },
                {
                    "index": 20542,
                    "name": "Sodium phosphate P 32"
                },
                {
                    "index": 15409,
                    "name": "Ropeginterferon alfa-2b"
                },
                {
                    "index": 15250,
                    "name": "Fedratinib"
                },
                {
                    "index": 15496,
                    "name": "Rifaximin"
                },
                {
                    "index": 20472,
                    "name": "Lactulose"
                },
                {
                    "index": 15121,
                    "name": "Methimazole"
                },
                {
                    "index": 14640,
                    "name": "Propylthiouracil"
                },
                {
                    "index": 17538,
                    "name": "Dibromotyrosine"
                },
                {
                    "index": 14815,
                    "name": "Carbimazole"
                },
                {
                    "index": 20436,
                    "name": "Dexpanthenol"
                },
                {
                    "index": 14958,
                    "name": "Cisapride"
                },
                {
                    "index": 14546,
                    "name": "Metoclopramide"
                },
                {
                    "index": 15254,
                    "name": "Levosalbutamol"
                },
                {
                    "index": 15479,
                    "name": "Salbutamol"
                },
                {
                    "index": 16341,
                    "name": "Iron"
                },
                {
                    "index": 20446,
                    "name": "Hematin"
                },
                {
                    "index": 20447,
                    "name": "Docusate"
                },
                {
                    "index": 15312,
                    "name": "Pentoxifylline"
                },
                {
                    "index": 17319,
                    "name": "Papaverine"
                },
                {
                    "index": 20543,
                    "name": "Azapetine"
                },
                {
                    "index": 14754,
                    "name": "Loteprednol etabonate"
                },
                {
                    "index": 14050,
                    "name": "Prednisolone acetate"
                },
                {
                    "index": 14031,
                    "name": "Rimexolone"
                },
                {
                    "index": 14017,
                    "name": "Fluorometholone"
                },
                {
                    "index": 14016,
                    "name": "Medrysone"
                },
                {
                    "index": 15755,
                    "name": "Beta-D-Glucose"
                },
                {
                    "index": 18410,
                    "name": "Caplacizumab"
                },
                {
                    "index": 15579,
                    "name": "Ambroxol"
                },
                {
                    "index": 14578,
                    "name": "Cholic Acid"
                },
                {
                    "index": 20544,
                    "name": "Dehydrocholic acid"
                },
                {
                    "index": 20291,
                    "name": "Sodium phosphate, dibasic"
                },
                {
                    "index": 20169,
                    "name": "Magnesium sulfate"
                },
                {
                    "index": 20357,
                    "name": "Bisacodyl"
                },
                {
                    "index": 14712,
                    "name": "Glycerin"
                },
                {
                    "index": 20254,
                    "name": "Polyethylene glycol"
                },
                {
                    "index": 14589,
                    "name": "Polyethylene glycol 400"
                },
                {
                    "index": 20313,
                    "name": "Polyethylene glycol 300"
                },
                {
                    "index": 20314,
                    "name": "Polyethylene glycol 3500"
                },
                {
                    "index": 17638,
                    "name": "Magnesium lactate"
                },
                {
                    "index": 14384,
                    "name": "Prucalopride"
                },
                {
                    "index": 20546,
                    "name": "Oxyphenisatin"
                },
                {
                    "index": 14366,
                    "name": "Dantron"
                },
                {
                    "index": 20547,
                    "name": "Oxyphenisatin acetate"
                },
                {
                    "index": 16350,
                    "name": "Citric acid"
                },
                {
                    "index": 20189,
                    "name": "Romosozumab"
                },
                {
                    "index": 14179,
                    "name": "Raloxifene"
                },
                {
                    "index": 14943,
                    "name": "Levonorgestrel"
                },
                {
                    "index": 17729,
                    "name": "Alendronic acid"
                },
                {
                    "index": 14624,
                    "name": "Tibolone"
                },
                {
                    "index": 19663,
                    "name": "Teriparatide"
                },
                {
                    "index": 17215,
                    "name": "Salmon calcitonin"
                },
                {
                    "index": 20548,
                    "name": "Ipriflavone"
                },
                {
                    "index": 15881,
                    "name": "Bazedoxifene"
                },
                {
                    "index": 16321,
                    "name": "Lasofoxifene"
                },
                {
                    "index": 20549,
                    "name": "Strontium ranelate"
                },
                {
                    "index": 14987,
                    "name": "Fluvastatin"
                },
                {
                    "index": 20148,
                    "name": "Synephrine"
                },
                {
                    "index": 17378,
                    "name": "Guaiacol"
                },
                {
                    "index": 17341,
                    "name": "Tetryzoline"
                },
                {
                    "index": 16851,
                    "name": "Xylometazoline"
                },
                {
                    "index": 20550,
                    "name": "Indanazoline"
                },
                {
                    "index": 14745,
                    "name": "Levmetamfetamine"
                },
                {
                    "index": 20551,
                    "name": "Metizoline"
                },
                {
                    "index": 16846,
                    "name": "Oxymetazoline"
                },
                {
                    "index": 20323,
                    "name": "Ammonium chloride"
                },
                {
                    "index": 17242,
                    "name": "Bromodiphenhydramine"
                },
                {
                    "index": 14402,
                    "name": "Pitavastatin"
                },
                {
                    "index": 14947,
                    "name": "Cerivastatin"
                },
                {
                    "index": 20552,
                    "name": "Oteracil"
                },
                {
                    "index": 14516,
                    "name": "Tegafur"
                },
                {
                    "index": 18195,
                    "name": "Gimeracil"
                },
                {
                    "index": 14427,
                    "name": "Lumacaftor"
                },
                {
                    "index": 14400,
                    "name": "Ivacaftor"
                },
                {
                    "index": 14632,
                    "name": "Adenosine phosphate"
                },
                {
                    "index": 15045,
                    "name": "Tezacaftor"
                },
                {
                    "index": 15748,
                    "name": "Zanubrutinib"
                },
                {
                    "index": 15012,
                    "name": "Temsirolimus"
                },
                {
                    "index": 14450,
                    "name": "Acalabrutinib"
                },
                {
                    "index": 14853,
                    "name": "Trimethaphan"
                },
                {
                    "index": 15606,
                    "name": "Fendiline"
                },
                {
                    "index": 15073,
                    "name": "Fluvoxamine"
                },
                {
                    "index": 15558,
                    "name": "Tolvaptan"
                },
                {
                    "index": 20553,
                    "name": "Bifemelane"
                },
                {
                    "index": 16337,
                    "name": "Tolazoline"
                },
                {
                    "index": 20487,
                    "name": "Isoxsuprine"
                },
                {
                    "index": 20554,
                    "name": "Naftidrofuryl"
                },
                {
                    "index": 15712,
                    "name": "Bencyclane"
                },
                {
                    "index": 14643,
                    "name": "Capsaicin"
                },
                {
                    "index": 14180,
                    "name": "Buspirone"
                },
                {
                    "index": 14164,
                    "name": "Alprazolam"
                },
                {
                    "index": 17558,
                    "name": "Pregabalin"
                },
                {
                    "index": 14295,
                    "name": "Promethazine"
                },
                {
                    "index": 14325,
                    "name": "Clomipramine"
                },
                {
                    "index": 14263,
                    "name": "Salmeterol"
                },
                {
                    "index": 14272,
                    "name": "Formoterol"
                },
                {
                    "index": 17239,
                    "name": "Emedastine"
                },
                {
                    "index": 15118,
                    "name": "Epinastine"
                },
                {
                    "index": 17240,
                    "name": "Levocabastine"
                },
                {
                    "index": 19551,
                    "name": "Tromethamine"
                },
                {
                    "index": 15774,
                    "name": "Ketorolac"
                },
                {
                    "index": 17236,
                    "name": "Ketotifen"
                },
                {
                    "index": 17251,
                    "name": "Alcaftadine"
                },
                {
                    "index": 17247,
                    "name": "Bepotastine"
                },
                {
                    "index": 15868,
                    "name": "Isoprenaline"
                },
                {
                    "index": 14157,
                    "name": "Hydromorphone"
                },
                {
                    "index": 14246,
                    "name": "Oxazepam"
                },
                {
                    "index": 14595,
                    "name": "Desmopressin"
                },
                {
                    "index": 15902,
                    "name": "Vonicog Alfa"
                },
                {
                    "index": 15903,
                    "name": "Von Willebrand Factor Human"
                },
                {
                    "index": 20304,
                    "name": "Salicylamide"
                },
                {
                    "index": 16182,
                    "name": "Nafamostat"
                },
                {
                    "index": 14499,
                    "name": "Elexacaftor"
                },
                {
                    "index": 20556,
                    "name": "Dornase alfa"
                },
                {
                    "index": 20557,
                    "name": "Alipogene tiparvovec"
                },
                {
                    "index": 15049,
                    "name": "Osilodrostat"
                },
                {
                    "index": 18313,
                    "name": "Oxacillin"
                },
                {
                    "index": 14311,
                    "name": "Cloxacillin"
                },
                {
                    "index": 15442,
                    "name": "Dicloxacillin"
                },
                {
                    "index": 14205,
                    "name": "Nafcillin"
                },
                {
                    "index": 15542,
                    "name": "Milnacipran"
                },
                {
                    "index": 16419,
                    "name": "Amlexanox"
                },
                {
                    "index": 20558,
                    "name": "Omoconazole"
                },
                {
                    "index": 20238,
                    "name": "Nystatin"
                },
                {
                    "index": 18036,
                    "name": "Bacitracin"
                },
                {
                    "index": 14711,
                    "name": "Chenodeoxycholic acid"
                },
                {
                    "index": 15822,
                    "name": "Ursodeoxycholic acid"
                },
                {
                    "index": 14798,
                    "name": "Doripenem"
                },
                {
                    "index": 20559,
                    "name": "Bamipine"
                },
                {
                    "index": 15646,
                    "name": "Oleandomycin"
                },
                {
                    "index": 15059,
                    "name": "Troleandomycin"
                },
                {
                    "index": 20344,
                    "name": "Tigecycline"
                },
                {
                    "index": 15293,
                    "name": "Grepafloxacin"
                },
                {
                    "index": 15327,
                    "name": "Sparfloxacin"
                },
                {
                    "index": 15053,
                    "name": "Letermovir"
                },
                {
                    "index": 18417,
                    "name": "Rilonacept"
                },
                {
                    "index": 14261,
                    "name": "Metacycline"
                },
                {
                    "index": 20560,
                    "name": "Mosapramine"
                },
                {
                    "index": 20300,
                    "name": "Perazine"
                },
                {
                    "index": 16317,
                    "name": "Molindone"
                },
                {
                    "index": 15127,
                    "name": "Perphenazine"
                },
                {
                    "index": 15100,
                    "name": "Prochlorperazine"
                },
                {
                    "index": 15131,
                    "name": "Mesoridazine"
                },
                {
                    "index": 15177,
                    "name": "Zuclopenthixol"
                },
                {
                    "index": 14351,
                    "name": "Thiothixene"
                },
                {
                    "index": 14633,
                    "name": "Lumateperone"
                },
                {
                    "index": 14829,
                    "name": "Flupentixol"
                },
                {
                    "index": 15538,
                    "name": "Fluspirilene"
                },
                {
                    "index": 14839,
                    "name": "Sulpiride"
                },
                {
                    "index": 20561,
                    "name": "Oxypertine"
                },
                {
                    "index": 15223,
                    "name": "Perospirone"
                },
                {
                    "index": 15006,
                    "name": "Iloperidone"
                },
                {
                    "index": 14420,
                    "name": "Blonanserin"
                },
                {
                    "index": 14485,
                    "name": "Aripiprazole lauroxil"
                },
                {
                    "index": 14913,
                    "name": "Trifluoperazine"
                },
                {
                    "index": 16014,
                    "name": "Chlorprothixene"
                },
                {
                    "index": 15196,
                    "name": "Sertindole"
                },
                {
                    "index": 14377,
                    "name": "Cenobamate"
                },
                {
                    "index": 15029,
                    "name": "Perampanel"
                },
                {
                    "index": 15551,
                    "name": "Brivaracetam"
                },
                {
                    "index": 15620,
                    "name": "Eslicarbazepine acetate"
                },
                {
                    "index": 20562,
                    "name": "Tavaborole"
                },
                {
                    "index": 20563,
                    "name": "Efinaconazole"
                },
                {
                    "index": 15502,
                    "name": "Posaconazole"
                },
                {
                    "index": 14686,
                    "name": "Clomifene"
                },
                {
                    "index": 15890,
                    "name": "Tinzaparin"
                },
                {
                    "index": 20564,
                    "name": "Reviparin"
                },
                {
                    "index": 14801,
                    "name": "Urokinase"
                },
                {
                    "index": 16667,
                    "name": "Edoxaban"
                },
                {
                    "index": 14341,
                    "name": "Acenocoumarol"
                },
                {
                    "index": 16589,
                    "name": "Alteplase"
                },
                {
                    "index": 20565,
                    "name": "Elosulfase alfa"
                },
                {
                    "index": 15674,
                    "name": "Fostamatinib"
                },
                {
                    "index": 14924,
                    "name": "Diazoxide"
                },
                {
                    "index": 20566,
                    "name": "Polysorbate 80"
                },
                {
                    "index": 20352,
                    "name": "Povidone"
                },
                {
                    "index": 20353,
                    "name": "Povidone K30"
                },
                {
                    "index": 20296,
                    "name": "Polyvinyl alcohol"
                },
                {
                    "index": 20567,
                    "name": "Propylene glycol"
                },
                {
                    "index": 14129,
                    "name": "Lutein"
                },
                {
                    "index": 15798,
                    "name": "Lifitegrast"
                },
                {
                    "index": 16511,
                    "name": "Ranibizumab"
                },
                {
                    "index": 20568,
                    "name": "Polidocanol"
                },
                {
                    "index": 17060,
                    "name": "Sodium tetradecyl sulfate"
                },
                {
                    "index": 15782,
                    "name": "Vigabatrin"
                },
                {
                    "index": 16023,
                    "name": "Collagenase clostridium histolyticum"
                },
                {
                    "index": 14844,
                    "name": "Diethylcarbamazine"
                },
                {
                    "index": 18798,
                    "name": "Reslizumab"
                },
                {
                    "index": 18799,
                    "name": "Mepolizumab"
                },
                {
                    "index": 19177,
                    "name": "Benralizumab"
                },
                {
                    "index": 20569,
                    "name": "Moxidectin"
                },
                {
                    "index": 15446,
                    "name": "Bosentan"
                },
                {
                    "index": 14634,
                    "name": "Nitric Oxide"
                },
                {
                    "index": 20022,
                    "name": "Gluconolactone"
                },
                {
                    "index": 14307,
                    "name": "Tiludronic acid"
                },
                {
                    "index": 19198,
                    "name": "Felbinac"
                },
                {
                    "index": 20571,
                    "name": "Imrecoxib"
                },
                {
                    "index": 20572,
                    "name": "Oxaceprol"
                },
                {
                    "index": 14515,
                    "name": "Etofenamate"
                },
                {
                    "index": 20573,
                    "name": "Polmacoxib"
                },
                {
                    "index": 15078,
                    "name": "Tramadol"
                },
                {
                    "index": 15329,
                    "name": "Lumiracoxib"
                },
                {
                    "index": 14935,
                    "name": "Fluconazole"
                },
                {
                    "index": 14226,
                    "name": "Zileuton"
                },
                {
                    "index": 20270,
                    "name": "Edaravone"
                },
                {
                    "index": 20574,
                    "name": "Butylphthalide"
                },
                {
                    "index": 15308,
                    "name": "Riluzole"
                },
                {
                    "index": 20185,
                    "name": "Epicriptine"
                },
                {
                    "index": 20211,
                    "name": "Acetylcarnitine"
                },
                {
                    "index": 20575,
                    "name": "Ipidacrine"
                },
                {
                    "index": 15113,
                    "name": "Galantamine"
                },
                {
                    "index": 15294,
                    "name": "Tacrine"
                },
                {
                    "index": 17174,
                    "name": "Rivastigmine"
                },
                {
                    "index": 15126,
                    "name": "Donepezil"
                },
                {
                    "index": 18226,
                    "name": "Velaglucerase alfa"
                },
                {
                    "index": 20576,
                    "name": "Taliglucerase alfa"
                },
                {
                    "index": 20577,
                    "name": "Alglucerase"
                },
                {
                    "index": 15810,
                    "name": "Beraprost"
                },
                {
                    "index": 15773,
                    "name": "Treprostinil"
                },
                {
                    "index": 16222,
                    "name": "Iloprost"
                },
                {
                    "index": 14409,
                    "name": "Riociguat"
                },
                {
                    "index": 15642,
                    "name": "Selexipag"
                },
                {
                    "index": 14410,
                    "name": "Macitentan"
                },
                {
                    "index": 15709,
                    "name": "Terguride"
                },
                {
                    "index": 15014,
                    "name": "Ambrisentan"
                },
                {
                    "index": 14937,
                    "name": "Sildenafil"
                },
                {
                    "index": 18179,
                    "name": "Sucralfate"
                },
                {
                    "index": 18988,
                    "name": "Camostat"
                },
                {
                    "index": 20578,
                    "name": "Tandospirone"
                },
                {
                    "index": 14769,
                    "name": "Ataluren"
                },
                {
                    "index": 15051,
                    "name": "Deflazacort"
                },
                {
                    "index": 20196,
                    "name": "Golodirsen"
                },
                {
                    "index": 20579,
                    "name": "Eteplirsen"
                },
                {
                    "index": 20580,
                    "name": "Bezlotoxumab"
                },
                {
                    "index": 20581,
                    "name": "Spectinomycin"
                },
                {
                    "index": 20582,
                    "name": "Evocalcet"
                },
                {
                    "index": 17350,
                    "name": "Etelcalcetide"
                },
                {
                    "index": 14253,
                    "name": "Chlorhexidine"
                },
                {
                    "index": 14851,
                    "name": "Cyclopentolate"
                },
                {
                    "index": 16065,
                    "name": "Tropicamide"
                },
                {
                    "index": 19650,
                    "name": "Linaclotide"
                },
                {
                    "index": 16314,
                    "name": "Dicyclomine"
                },
                {
                    "index": 16312,
                    "name": "Clidinium"
                },
                {
                    "index": 20585,
                    "name": "Mebeverine"
                },
                {
                    "index": 15862,
                    "name": "Tecovirimat"
                },
                {
                    "index": 14527,
                    "name": "Fluoxymesterone"
                },
                {
                    "index": 14779,
                    "name": "Tiopronin"
                },
                {
                    "index": 20586,
                    "name": "Dinutuximab"
                },
                {
                    "index": 20587,
                    "name": "Fenticonazole"
                },
                {
                    "index": 20588,
                    "name": "Terconazole"
                },
                {
                    "index": 14818,
                    "name": "Tioconazole"
                },
                {
                    "index": 15085,
                    "name": "Sulfanilamide"
                },
                {
                    "index": 14876,
                    "name": "Miltefosine"
                },
                {
                    "index": 14992,
                    "name": "Itraconazole"
                },
                {
                    "index": 14810,
                    "name": "Micafungin"
                },
                {
                    "index": 20433,
                    "name": "Anidulafungin"
                },
                {
                    "index": 15443,
                    "name": "Caspofungin"
                },
                {
                    "index": 14602,
                    "name": "Voriconazole"
                },
                {
                    "index": 20589,
                    "name": "Menatetrenone"
                },
                {
                    "index": 15895,
                    "name": "Mirodenafil"
                },
                {
                    "index": 16219,
                    "name": "Alprostadil"
                },
                {
                    "index": 14874,
                    "name": "Moxisylyte"
                },
                {
                    "index": 15561,
                    "name": "Avanafil"
                },
                {
                    "index": 15011,
                    "name": "Udenafil"
                },
                {
                    "index": 14974,
                    "name": "Vardenafil"
                },
                {
                    "index": 20590,
                    "name": "Anecortave acetate"
                },
                {
                    "index": 16512,
                    "name": "Pegaptanib"
                },
                {
                    "index": 16519,
                    "name": "Brolucizumab"
                },
                {
                    "index": 20393,
                    "name": "Verteporfin"
                },
                {
                    "index": 17493,
                    "name": "Sermorelin"
                },
                {
                    "index": 16272,
                    "name": "Somatrem"
                },
                {
                    "index": 16271,
                    "name": "Somatotropin"
                },
                {
                    "index": 15055,
                    "name": "Deutetrabenazine"
                },
                {
                    "index": 15050,
                    "name": "Valbenazine"
                },
                {
                    "index": 15147,
                    "name": "Orphenadrine"
                },
                {
                    "index": 20591,
                    "name": "Melevodopa"
                },
                {
                    "index": 15490,
                    "name": "Pergolide"
                },
                {
                    "index": 20153,
                    "name": "Metixene"
                },
                {
                    "index": 17414,
                    "name": "Piribedil"
                },
                {
                    "index": 20592,
                    "name": "Sacrosidase"
                },
                {
                    "index": 15364,
                    "name": "Besifloxacin"
                },
                {
                    "index": 20252,
                    "name": "Gramicidin D"
                },
                {
                    "index": 14451,
                    "name": "Rifamycin"
                },
                {
                    "index": 20429,
                    "name": "Colistin"
                },
                {
                    "index": 20228,
                    "name": "Acetic acid"
                },
                {
                    "index": 14895,
                    "name": "Tedizolid phosphate"
                },
                {
                    "index": 14459,
                    "name": "Delafloxacin"
                },
                {
                    "index": 20593,
                    "name": "Dalbavancin"
                },
                {
                    "index": 14370,
                    "name": "Oritavancin"
                },
                {
                    "index": 20594,
                    "name": "Faropenem medoxomil"
                },
                {
                    "index": 20595,
                    "name": "Faropenem"
                },
                {
                    "index": 19611,
                    "name": "Romiplostim"
                },
                {
                    "index": 20596,
                    "name": "Bunazosin"
                },
                {
                    "index": 16853,
                    "name": "Moxonidine"
                },
                {
                    "index": 15596,
                    "name": "Cabozantinib"
                },
                {
                    "index": 15002,
                    "name": "Sunitinib"
                },
                {
                    "index": 14975,
                    "name": "Sirolimus"
                },
                {
                    "index": 20597,
                    "name": "Tivozanib"
                },
                {
                    "index": 15230,
                    "name": "Lenvatinib"
                },
                {
                    "index": 15020,
                    "name": "Axitinib"
                },
                {
                    "index": 15176,
                    "name": "Everolimus"
                },
                {
                    "index": 20324,
                    "name": "Triethylenetetramine"
                },
                {
                    "index": 15573,
                    "name": "Isavuconazonium"
                },
                {
                    "index": 20598,
                    "name": "Ibopamine"
                },
                {
                    "index": 20599,
                    "name": "Chlormezanone"
                },
                {
                    "index": 14963,
                    "name": "Midazolam"
                },
                {
                    "index": 17064,
                    "name": "Meprobamate"
                },
                {
                    "index": 15623,
                    "name": "Etizolam"
                },
                {
                    "index": 20600,
                    "name": "Oxaflozane"
                },
                {
                    "index": 20130,
                    "name": "Halazepam"
                },
                {
                    "index": 20601,
                    "name": "Mebicar"
                },
                {
                    "index": 19360,
                    "name": "Afamelanotide"
                },
                {
                    "index": 20602,
                    "name": "Hemin"
                },
                {
                    "index": 14973,
                    "name": "Mifepristone"
                },
                {
                    "index": 19699,
                    "name": "Dinoprost"
                },
                {
                    "index": 14827,
                    "name": "Oxytocin"
                },
                {
                    "index": 16220,
                    "name": "Dinoprostone"
                },
                {
                    "index": 14445,
                    "name": "Delamanid"
                },
                {
                    "index": 20603,
                    "name": "Amenamevir"
                },
                {
                    "index": 15848,
                    "name": "Famciclovir"
                },
                {
                    "index": 15867,
                    "name": "Mebendazole"
                },
                {
                    "index": 14698,
                    "name": "Pyrantel"
                },
                {
                    "index": 15109,
                    "name": "Piperazine"
                },
                {
                    "index": 15424,
                    "name": "Cabergoline"
                },
                {
                    "index": 18028,
                    "name": "Levamisole"
                },
                {
                    "index": 15191,
                    "name": "Ospemifene"
                },
                {
                    "index": 14147,
                    "name": "Conjugated estrogens"
                },
                {
                    "index": 14479,
                    "name": "Estradiol acetate"
                },
                {
                    "index": 16309,
                    "name": "Buclizine"
                },
                {
                    "index": 15303,
                    "name": "Methoxsalen"
                },
                {
                    "index": 17533,
                    "name": "Denileukin diftitox"
                },
                {
                    "index": 15428,
                    "name": "Bexarotene"
                },
                {
                    "index": 18959,
                    "name": "Vorinostat"
                },
                {
                    "index": 15009,
                    "name": "Romidepsin"
                },
                {
                    "index": 15958,
                    "name": "Defibrotide"
                },
                {
                    "index": 17303,
                    "name": "Miglustat"
                },
                {
                    "index": 15226,
                    "name": "Eliglustat"
                },
                {
                    "index": 20604,
                    "name": "Imiglucerase"
                },
                {
                    "index": 14584,
                    "name": "L-Glutamine"
                },
                {
                    "index": 19197,
                    "name": "Teduglutide"
                },
                {
                    "index": 14106,
                    "name": "Hydroxocobalamin"
                },
                {
                    "index": 20320,
                    "name": "Cobalamin"
                },
                {
                    "index": 14127,
                    "name": "Octreotide"
                },
                {
                    "index": 14695,
                    "name": "Pegvisomant"
                },
                {
                    "index": 15584,
                    "name": "Lanreotide"
                },
                {
                    "index": 18769,
                    "name": "Crofelemer"
                },
                {
                    "index": 14148,
                    "name": "Atomoxetine"
                },
                {
                    "index": 15159,
                    "name": "Lisdexamfetamine"
                },
                {
                    "index": 15173,
                    "name": "Dextroamphetamine"
                },
                {
                    "index": 17315,
                    "name": "Dexmethylphenidate"
                },
                {
                    "index": 20605,
                    "name": "Pemoline"
                },
                {
                    "index": 16855,
                    "name": "Methylphenidate"
                },
                {
                    "index": 14989,
                    "name": "Quinacrine"
                },
                {
                    "index": 14579,
                    "name": "Nitazoxanide"
                },
                {
                    "index": 20606,
                    "name": "Secnidazole"
                },
                {
                    "index": 15473,
                    "name": "Tinidazole"
                },
                {
                    "index": 20607,
                    "name": "Idebenone"
                },
                {
                    "index": 18309,
                    "name": "Fasudil"
                },
                {
                    "index": 20272,
                    "name": "Avibactam"
                },
                {
                    "index": 20406,
                    "name": "Meticillin"
                },
                {
                    "index": 20263,
                    "name": "Ganciclovir"
                },
                {
                    "index": 14918,
                    "name": "Adenosine"
                },
                {
                    "index": 20608,
                    "name": "Epalrestat"
                },
                {
                    "index": 14302,
                    "name": "Amiodarone"
                },
                {
                    "index": 16012,
                    "name": "Fenoldopam"
                },
                {
                    "index": 20609,
                    "name": "Racecadotril"
                },
                {
                    "index": 19247,
                    "name": "Roxadustat"
                },
                {
                    "index": 15574,
                    "name": "Pasireotide"
                },
                {
                    "index": 16191,
                    "name": "Trilostane"
                },
                {
                    "index": 20274,
                    "name": "Leucovorin"
                },
                {
                    "index": 16180,
                    "name": "Conestat alfa"
                },
                {
                    "index": 14636,
                    "name": "Amyl Nitrite"
                },
                {
                    "index": 18183,
                    "name": "Pyrethrum extract"
                },
                {
                    "index": 20610,
                    "name": "Tetramethrin"
                },
                {
                    "index": 16193,
                    "name": "Lindane"
                },
                {
                    "index": 15872,
                    "name": "Permethrin"
                },
                {
                    "index": 20527,
                    "name": "Piperonyl butoxide"
                },
                {
                    "index": 20611,
                    "name": "Spinosad"
                },
                {
                    "index": 15309,
                    "name": "Malathion"
                },
                {
                    "index": 14247,
                    "name": "Clofazimine"
                },
                {
                    "index": 15591,
                    "name": "Lomitapide"
                },
                {
                    "index": 15901,
                    "name": "Evolocumab"
                },
                {
                    "index": 14264,
                    "name": "Zalcitabine"
                },
                {
                    "index": 15619,
                    "name": "Elvitegravir"
                },
                {
                    "index": 14256,
                    "name": "Didanosine"
                },
                {
                    "index": 18042,
                    "name": "Ibalizumab"
                },
                {
                    "index": 15859,
                    "name": "Raltegravir"
                },
                {
                    "index": 14403,
                    "name": "Rilpivirine"
                },
                {
                    "index": 14428,
                    "name": "Tenofovir alafenamide"
                },
                {
                    "index": 14210,
                    "name": "Stavudine"
                },
                {
                    "index": 14181,
                    "name": "Zidovudine"
                },
                {
                    "index": 15566,
                    "name": "Etravirine"
                },
                {
                    "index": 15503,
                    "name": "Fosamprenavir"
                },
                {
                    "index": 14326,
                    "name": "Darunavir"
                },
                {
                    "index": 14296,
                    "name": "Atazanavir"
                },
                {
                    "index": 14346,
                    "name": "Lopinavir"
                },
                {
                    "index": 15130,
                    "name": "Tipranavir"
                },
                {
                    "index": 14966,
                    "name": "Amprenavir"
                },
                {
                    "index": 15664,
                    "name": "Bictegravir"
                },
                {
                    "index": 14134,
                    "name": "Nelfinavir"
                },
                {
                    "index": 15820,
                    "name": "Enfuvirtide"
                },
                {
                    "index": 14209,
                    "name": "Efavirenz"
                },
                {
                    "index": 14939,
                    "name": "Indinavir"
                },
                {
                    "index": 14254,
                    "name": "Emtricitabine"
                },
                {
                    "index": 14967,
                    "name": "Delavirdine"
                },
                {
                    "index": 15853,
                    "name": "Abacavir"
                },
                {
                    "index": 15056,
                    "name": "Doravirine"
                },
                {
                    "index": 14138,
                    "name": "Nevirapine"
                },
                {
                    "index": 14319,
                    "name": "Saquinavir"
                },
                {
                    "index": 15846,
                    "name": "Dolutegravir"
                },
                {
                    "index": 15034,
                    "name": "Cobicistat"
                },
                {
                    "index": 16574,
                    "name": "Gonadorelin"
                },
                {
                    "index": 14504,
                    "name": "Floxuridine"
                },
                {
                    "index": 20397,
                    "name": "Pantothenic acid"
                },
                {
                    "index": 14128,
                    "name": "Ascorbic acid"
                },
                {
                    "index": 20306,
                    "name": "Zinc gluconate"
                },
                {
                    "index": 15521,
                    "name": "Rutin"
                },
                {
                    "index": 14415,
                    "name": "Chlortetracycline"
                },
                {
                    "index": 14373,
                    "name": "Vandetanib"
                },
                {
                    "index": 14664,
                    "name": "Sapropterin"
                },
                {
                    "index": 20612,
                    "name": "Phenylmercuric nitrate"
                },
                {
                    "index": 14207,
                    "name": "Imatinib"
                },
                {
                    "index": 20499,
                    "name": "Nitrogen"
                },
                {
                    "index": 16344,
                    "name": "Ferrous gluconate"
                },
                {
                    "index": 20200,
                    "name": "Ferumoxytol"
                },
                {
                    "index": 15892,
                    "name": "Ferric maltol"
                },
                {
                    "index": 14646,
                    "name": "Succinic acid"
                },
                {
                    "index": 16580,
                    "name": "Histrelin"
                },
                {
                    "index": 15469,
                    "name": "Natamycin"
                },
                {
                    "index": 14108,
                    "name": "Ubidecarenone"
                },
                {
                    "index": 20613,
                    "name": "Azosemide"
                },
                {
                    "index": 17221,
                    "name": "Levosimendan"
                },
                {
                    "index": 16510,
                    "name": "Bevacizumab"
                },
                {
                    "index": 14761,
                    "name": "Niraparib"
                },
                {
                    "index": 14476,
                    "name": "Testosterone cypionate"
                },
                {
                    "index": 14387,
                    "name": "Methyltestosterone"
                },
                {
                    "index": 14342,
                    "name": "Testosterone propionate"
                },
                {
                    "index": 14477,
                    "name": "Testosterone enanthate"
                },
                {
                    "index": 20614,
                    "name": "Padeliporfin"
                },
                {
                    "index": 18412,
                    "name": "Susoctocog alfa"
                },
                {
                    "index": 15750,
                    "name": "Avapritinib"
                },
                {
                    "index": 16223,
                    "name": "Limaprost"
                },
                {
                    "index": 16740,
                    "name": "Forodesine"
                },
                {
                    "index": 14727,
                    "name": "Pralatrexate"
                },
                {
                    "index": 16818,
                    "name": "Vintafolide"
                },
                {
                    "index": 14991,
                    "name": "Cilostazol"
                },
                {
                    "index": 14726,
                    "name": "Melatonin"
                },
                {
                    "index": 14982,
                    "name": "Zaleplon"
                },
                {
                    "index": 16570,
                    "name": "Prilocaine"
                },
                {
                    "index": 15189,
                    "name": "Dapoxetine"
                },
                {
                    "index": 14822,
                    "name": "Sulfathiazole"
                },
                {
                    "index": 20394,
                    "name": "Sulfabenzamide"
                },
                {
                    "index": 15242,
                    "name": "Artenimol"
                },
                {
                    "index": 15726,
                    "name": "Piperaquine"
                },
                {
                    "index": 15207,
                    "name": "Tafenoquine"
                },
                {
                    "index": 14988,
                    "name": "Pimozide"
                },
                {
                    "index": 17128,
                    "name": "Eflornithine"
                },
                {
                    "index": 20616,
                    "name": "Melarsoprol"
                },
                {
                    "index": 14722,
                    "name": "Suramin"
                },
                {
                    "index": 20617,
                    "name": "Carmofur"
                },
                {
                    "index": 17605,
                    "name": "Cetuximab"
                },
                {
                    "index": 19217,
                    "name": "Tipiracil"
                },
                {
                    "index": 20347,
                    "name": "Lincomycin"
                },
                {
                    "index": 20618,
                    "name": "Teicoplanin"
                },
                {
                    "index": 14456,
                    "name": "Neratinib"
                },
                {
                    "index": 17355,
                    "name": "Pertuzumab"
                },
                {
                    "index": 15000,
                    "name": "Lapatinib"
                },
                {
                    "index": 17351,
                    "name": "Trastuzumab"
                },
                {
                    "index": 15008,
                    "name": "Trastuzumab emtansine"
                },
                {
                    "index": 20619,
                    "name": "Trepibutone"
                },
                {
                    "index": 14796,
                    "name": "Triclabendazole"
                },
                {
                    "index": 14968,
                    "name": "Modafinil"
                },
                {
                    "index": 14644,
                    "name": "Sodium oxybate"
                },
                {
                    "index": 14447,
                    "name": "Pitolisant"
                },
                {
                    "index": 17318,
                    "name": "Solriamfetol"
                },
                {
                    "index": 15015,
                    "name": "Armodafinil"
                },
                {
                    "index": 14554,
                    "name": "Mecasermin"
                },
                {
                    "index": 16041,
                    "name": "Mecasermin rinfabate"
                },
                {
                    "index": 15290,
                    "name": "Anagrelide"
                },
                {
                    "index": 14063,
                    "name": "Dimethyl sulfoxide"
                },
                {
                    "index": 16975,
                    "name": "Pentosan polysulfate"
                },
                {
                    "index": 20620,
                    "name": "Docosanol"
                },
                {
                    "index": 20621,
                    "name": "Penciclovir"
                },
                {
                    "index": 15440,
                    "name": "Nimodipine"
                },
                {
                    "index": 14371,
                    "name": "Pirfenidone"
                },
                {
                    "index": 20126,
                    "name": "Dimethyl fumarate"
                },
                {
                    "index": 16285,
                    "name": "Alemtuzumab"
                },
                {
                    "index": 15829,
                    "name": "Fingolimod"
                },
                {
                    "index": 17535,
                    "name": "Daclizumab"
                },
                {
                    "index": 15284,
                    "name": "Interferon beta-1a"
                },
                {
                    "index": 15285,
                    "name": "Interferon beta-1b"
                },
                {
                    "index": 15371,
                    "name": "Peginterferon beta-1a"
                },
                {
                    "index": 14132,
                    "name": "Pyridoxine"
                },
                {
                    "index": 15499,
                    "name": "Retapamulin"
                },
                {
                    "index": 20454,
                    "name": "Mupirocin"
                },
                {
                    "index": 15875,
                    "name": "Amifampridine"
                },
                {
                    "index": 16465,
                    "name": "Guanidine"
                },
                {
                    "index": 15187,
                    "name": "Nilotinib"
                },
                {
                    "index": 14794,
                    "name": "Dasatinib"
                },
                {
                    "index": 16232,
                    "name": "Ocrelizumab"
                },
                {
                    "index": 18884,
                    "name": "Lactose"
                },
                {
                    "index": 17023,
                    "name": "Tafamidis"
                },
                {
                    "index": 14725,
                    "name": "Glycerol phenylbutyrate"
                },
                {
                    "index": 15369,
                    "name": "Tasimelteon"
                },
                {
                    "index": 14449,
                    "name": "Selumetinib"
                },
                {
                    "index": 20178,
                    "name": "Catridecacog"
                },
                {
                    "index": 14248,
                    "name": "Cysteamine"
                },
                {
                    "index": 15775,
                    "name": "Tenoxicam"
                },
                {
                    "index": 14478,
                    "name": "Testosterone undecanoate"
                },
                {
                    "index": 14811,
                    "name": "Nylidrin"
                },
                {
                    "index": 18611,
                    "name": "Lonidamine"
                },
                {
                    "index": 20622,
                    "name": "Talaporfin"
                },
                {
                    "index": 20528,
                    "name": "Crotamiton"
                },
                {
                    "index": 20625,
                    "name": "Phenothrin"
                },
                {
                    "index": 14782,
                    "name": "Biotin"
                },
                {
                    "index": 15185,
                    "name": "Tetrabenazine"
                },
                {
                    "index": 14719,
                    "name": "Epirubicin"
                },
                {
                    "index": 15539,
                    "name": "Ixabepilone"
                },
                {
                    "index": 15898,
                    "name": "Betahistine"
                },
                {
                    "index": 14292,
                    "name": "Oxybutynin"
                },
                {
                    "index": 15212,
                    "name": "Fesoterodine"
                },
                {
                    "index": 14583,
                    "name": "Solifenacin"
                },
                {
                    "index": 14406,
                    "name": "Mirabegron"
                },
                {
                    "index": 14425,
                    "name": "Imidafenacin"
                },
                {
                    "index": 20317,
                    "name": "Butacaine"
                },
                {
                    "index": 15103,
                    "name": "Darifenacin"
                },
                {
                    "index": 15137,
                    "name": "Tolterodine"
                },
                {
                    "index": 15746,
                    "name": "Vibegron"
                },
                {
                    "index": 17945,
                    "name": "Eculizumab"
                },
                {
                    "index": 17946,
                    "name": "Ravulizumab"
                },
                {
                    "index": 20626,
                    "name": "Maxacalcitol"
                },
                {
                    "index": 17948,
                    "name": "Galsulfase"
                },
                {
                    "index": 14986,
                    "name": "Praziquantel"
                },
                {
                    "index": 15142,
                    "name": "Oxamniquine"
                },
                {
                    "index": 20627,
                    "name": "Stibophen"
                },
                {
                    "index": 17197,
                    "name": "Itopride"
                },
                {
                    "index": 15302,
                    "name": "Albendazole"
                },
                {
                    "index": 14735,
                    "name": "Brexanolone"
                },
                {
                    "index": 15372,
                    "name": "Viloxazine"
                },
                {
                    "index": 15612,
                    "name": "Siltuximab"
                },
                {
                    "index": 20555,
                    "name": "Lodoxamide"
                },
                {
                    "index": 20628,
                    "name": "Pancrelipase"
                },
                {
                    "index": 14981,
                    "name": "Buprenorphine"
                },
                {
                    "index": 17724,
                    "name": "Methadyl acetate"
                },
                {
                    "index": 14314,
                    "name": "Naloxone"
                },
                {
                    "index": 15497,
                    "name": "Levacetylmethadol"
                },
                {
                    "index": 20629,
                    "name": "Levomethadone"
                },
                {
                    "index": 20630,
                    "name": "Drostanolone propionate"
                },
                {
                    "index": 17947,
                    "name": "Idursulfase"
                },
                {
                    "index": 20631,
                    "name": "Burosumab"
                },
                {
                    "index": 20632,
                    "name": "Formaldehyde"
                },
                {
                    "index": 20633,
                    "name": "Moroxydine"
                },
                {
                    "index": 20634,
                    "name": "Peramivir"
                },
                {
                    "index": 14466,
                    "name": "Favipiravir"
                },
                {
                    "index": 20635,
                    "name": "Laninamivir octanoate"
                },
                {
                    "index": 15730,
                    "name": "Baloxavir marboxil"
                },
                {
                    "index": 15761,
                    "name": "Oseltamivir"
                },
                {
                    "index": 19447,
                    "name": "Zanamivir"
                },
                {
                    "index": 15061,
                    "name": "Umifenovir"
                },
                {
                    "index": 15354,
                    "name": "Eltrombopag"
                },
                {
                    "index": 16508,
                    "name": "Captodiame"
                },
                {
                    "index": 17112,
                    "name": "Filgrastim"
                },
                {
                    "index": 20636,
                    "name": "Tenonitrozole"
                },
                {
                    "index": 20637,
                    "name": "Hachimycin"
                },
                {
                    "index": 17282,
                    "name": "Lutetium Lu 177 dotatate"
                },
                {
                    "index": 20077,
                    "name": "Alglucosidase alfa"
                },
                {
                    "index": 20638,
                    "name": "Uridine triacetate"
                },
                {
                    "index": 20639,
                    "name": "Hexamidine"
                },
                {
                    "index": 15572,
                    "name": "Bosutinib"
                },
                {
                    "index": 17881,
                    "name": "Carbetocin"
                },
                {
                    "index": 17607,
                    "name": "Palivizumab"
                },
                {
                    "index": 20640,
                    "name": "Sulbentine"
                },
                {
                    "index": 20641,
                    "name": "Urapidil"
                },
                {
                    "index": 17711,
                    "name": "Panitumumab"
                },
                {
                    "index": 15891,
                    "name": "Deferiprone"
                },
                {
                    "index": 15338,
                    "name": "Deferasirox"
                },
                {
                    "index": 20642,
                    "name": "Luspatercept"
                },
                {
                    "index": 19605,
                    "name": "Cemiplimab"
                },
                {
                    "index": 20643,
                    "name": "Temoporfin"
                },
                {
                    "index": 14622,
                    "name": "Moroctocog alfa"
                },
                {
                    "index": 16671,
                    "name": "Emicizumab"
                },
                {
                    "index": 14619,
                    "name": "Antihemophilic factor, human recombinant"
                },
                {
                    "index": 14621,
                    "name": "Lonoctocog alfa"
                },
                {
                    "index": 14557,
                    "name": "Turoctocog alfa pegol"
                },
                {
                    "index": 18413,
                    "name": "Efmoroctocog alfa"
                },
                {
                    "index": 18411,
                    "name": "Simoctocog alfa"
                },
                {
                    "index": 16179,
                    "name": "Turoctocog alfa"
                },
                {
                    "index": 16634,
                    "name": "Coagulation factor VIIa Recombinant Human"
                },
                {
                    "index": 16028,
                    "name": "Eribulin"
                },
                {
                    "index": 20180,
                    "name": "Dibotermin alfa"
                },
                {
                    "index": 20644,
                    "name": "Monoxerutin"
                },
                {
                    "index": 14454,
                    "name": "Nifurtimox"
                },
                {
                    "index": 20645,
                    "name": "Benznidazole"
                },
                {
                    "index": 15975,
                    "name": "Coagulation Factor IX (Recombinant)"
                },
                {
                    "index": 15979,
                    "name": "Coagulation Factor IX Human"
                },
                {
                    "index": 14594,
                    "name": "Albutrepenonacog alfa"
                },
                {
                    "index": 20646,
                    "name": "Eftrenonacog alfa"
                },
                {
                    "index": 20647,
                    "name": "Laronidase"
                },
                {
                    "index": 17666,
                    "name": "Finafloxacin"
                },
                {
                    "index": 20648,
                    "name": "Benperidol"
                },
                {
                    "index": 18406,
                    "name": "Tagraxofusp"
                },
                {
                    "index": 16592,
                    "name": "Tenecteplase"
                },
                {
                    "index": 16590,
                    "name": "Reteplase"
                },
                {
                    "index": 17888,
                    "name": "Bremelanotide"
                },
                {
                    "index": 15543,
                    "name": "Flibanserin"
                },
                {
                    "index": 19504,
                    "name": "Metreleptin"
                },
                {
                    "index": 18409,
                    "name": "Cenegermin"
                },
                {
                    "index": 20210,
                    "name": "Azidocillin"
                },
                {
                    "index": 18160,
                    "name": "Emapalumab"
                },
                {
                    "index": 20649,
                    "name": "Vestronidase alfa"
                },
                {
                    "index": 14446,
                    "name": "Vinflunine"
                },
                {
                    "index": 19615,
                    "name": "Durvalumab"
                },
                {
                    "index": 19614,
                    "name": "Atezolizumab"
                },
                {
                    "index": 18216,
                    "name": "Sodium stibogluconate"
                },
                {
                    "index": 19616,
                    "name": "Avelumab"
                },
                {
                    "index": 19718,
                    "name": "Cerliponase alfa"
                },
                {
                    "index": 20650,
                    "name": "Prussian blue"
                },
                {
                    "index": 15698,
                    "name": "Tazemetostat"
                },
                {
                    "index": 15488,
                    "name": "Candicidin"
                },
                {
                    "index": 18098,
                    "name": "Pegademase"
                },
                {
                    "index": 20651,
                    "name": "Sebelipase alfa"
                },
                {
                    "index": 20652,
                    "name": "Sivelestat"
                },
                {
                    "index": 20653,
                    "name": "Nusinersen"
                },
                {
                    "index": 20162,
                    "name": "Alirocumab"
                },
                {
                    "index": 14919,
                    "name": "Nelarabine"
                },
                {
                    "index": 20164,
                    "name": "Asfotase alfa"
                },
                {
                    "index": 15858,
                    "name": "Migalastat"
                },
                {
                    "index": 17286,
                    "name": "Nitisinone"
                },
                {
                    "index": 17955,
                    "name": "Teprotumumab"
                },
                {
                    "index": 14471,
                    "name": "Pexidartinib"
                }
            ]
        },
        {
            "question": "How does Asbestos interact with the human body?",
            "expected_cypher": "MATCH (e:exposure {name: \"asbestos\"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 39999,
                    "name": "inflammatory response"
                },
                {
                    "index": 106754,
                    "name": "cytokine production involved in inflammatory response"
                },
                {
                    "index": 106758,
                    "name": "interleukin-6 production"
                },
                {
                    "index": 44674,
                    "name": "gene expression"
                },
                {
                    "index": 44217,
                    "name": "cellular response to oxidative stress"
                }
            ]
        },
        {
            "question": "Which exposures are linked to more than 10 diseases?",
            "expected_cypher": "MATCH (e:exposure)-[:linked_to]->(d:disease)\nWITH e, COUNT(d) AS diseaseCount\nWHERE diseaseCount>10\nRETURN e.name",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 61848,
                    "name": "2,4,5-Trichlorophenoxyacetic Acid"
                },
                {
                    "index": 61701,
                    "name": "Agent Orange"
                },
                {
                    "index": 61702,
                    "name": "Air Pollutants"
                },
                {
                    "index": 61876,
                    "name": "alachlor"
                },
                {
                    "index": 61707,
                    "name": "Arsenic"
                },
                {
                    "index": 61709,
                    "name": "Asbestos"
                },
                {
                    "index": 61711,
                    "name": "Atrazine"
                },
                {
                    "index": 61712,
                    "name": "Benzene"
                },
                {
                    "index": 61715,
                    "name": "bisphenol A"
                },
                {
                    "index": 61717,
                    "name": "Cadmium"
                },
                {
                    "index": 61927,
                    "name": "Carbaryl"
                },
                {
                    "index": 61719,
                    "name": "Carbon Monoxide"
                },
                {
                    "index": 61944,
                    "name": "Chloroform"
                },
                {
                    "index": 61722,
                    "name": "Chlorpyrifos"
                },
                {
                    "index": 61964,
                    "name": "cyanazine"
                },
                {
                    "index": 61726,
                    "name": "DDT"
                },
                {
                    "index": 61974,
                    "name": "Diazinon"
                },
                {
                    "index": 61729,
                    "name": "Dichlorodiphenyl Dichloroethylene"
                },
                {
                    "index": 61748,
                    "name": "glyphosate"
                },
                {
                    "index": 62042,
                    "name": "Hydrocarbons, Chlorinated"
                },
                {
                    "index": 61754,
                    "name": "Insecticides"
                },
                {
                    "index": 61757,
                    "name": "Lead"
                },
                {
                    "index": 62057,
                    "name": "Malathion"
                },
                {
                    "index": 61759,
                    "name": "Manganese"
                },
                {
                    "index": 61760,
                    "name": "Mercury"
                },
                {
                    "index": 62073,
                    "name": "metolachlor"
                },
                {
                    "index": 62088,
                    "name": "Nickel"
                },
                {
                    "index": 61774,
                    "name": "Nitrates"
                },
                {
                    "index": 61775,
                    "name": "Nitric Oxide"
                },
                {
                    "index": 61776,
                    "name": "Nitrogen Dioxide"
                },
                {
                    "index": 61780,
                    "name": "Organophosphates"
                },
                {
                    "index": 61782,
                    "name": "Ozone"
                },
                {
                    "index": 61785,
                    "name": "Particulate Matter"
                },
                {
                    "index": 62098,
                    "name": "pendimethalin"
                },
                {
                    "index": 61793,
                    "name": "perfluorooctane sulfonic acid"
                },
                {
                    "index": 61794,
                    "name": "perfluorooctanoic acid"
                },
                {
                    "index": 62104,
                    "name": "Permethrin"
                },
                {
                    "index": 61796,
                    "name": "Pesticides"
                },
                {
                    "index": 62106,
                    "name": "Petroleum"
                },
                {
                    "index": 61802,
                    "name": "Polychlorinated Biphenyls"
                },
                {
                    "index": 62132,
                    "name": "Selenium"
                },
                {
                    "index": 61809,
                    "name": "Solvents"
                },
                {
                    "index": 62141,
                    "name": "Styrene"
                },
                {
                    "index": 61812,
                    "name": "Sulfur Dioxide"
                },
                {
                    "index": 61813,
                    "name": "Tetrachlorodibenzodioxin"
                },
                {
                    "index": 62150,
                    "name": "Tetrachloroethylene"
                },
                {
                    "index": 61815,
                    "name": "Tobacco Smoke Pollution"
                },
                {
                    "index": 61818,
                    "name": "Trihalomethanes"
                },
                {
                    "index": 61821,
                    "name": "Vehicle Emissions"
                },
                {
                    "index": 61823,
                    "name": "Water Pollutants, Chemical"
                },
                {
                    "index": 61825,
                    "name": "Zinc"
                }
            ]
        },
        {
            "question": "How does Asbestos interact with the human body?",
            "expected_cypher": "\nMATCH (e:exposure {name: \"asbestos\"})-[:interacts_with]->(bp:biological_process)\nRETURN e.name, bp.name\n",
            "expected_answer": "",
            "nodes": [
                {
                    "index": 39999,
                    "name": "inflammatory response"
                },
                {
                    "index": 106754,
                    "name": "cytokine production involved in inflammatory response"
                },
                {
                    "index": 106758,
                    "name": "interleukin-6 production"
                },
                {
                    "index": 44674,
                    "name": "gene expression"
                },
                {
                    "index": 44217,
                    "name": "cellular response to oxidative stress"
                }
            ]
        },
        {
            "question": "Which drugs have pterygium as side effect?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
            "expected_answer": "brinzolamide",
            "subgraph_query": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN n, g, s',
            "nodes": [{"index": 15491}],
        },
        {
            "question": "Which medicine may cause pterygium?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN distinct n',
            "expected_answer": "brinzolamide",
            "subgraph_query": 'MATCH (n:drug)-[s:side_effect]-(g {name:"pterygium"}) RETURN n, g, s',
            "nodes": [{"index": 15491}],
        },
        {
            "question": "Which drugs have freckling as side effect?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
            "expected_answer": "cytarabine, methoxsalen",
            "nodes": [{"index": 14780}, {"index": 15303}],
        },
        {
            "question": "Which medicine may cause freckling?",
            "expected_cypher": 'MATCH (n:drug)-[s:side_effect]-(g {name:"freckling"}) RETURN distinct n',
            "expected_answer": "cytarabine, methoxsalen",
            "nodes": [{"index": 14780}, {"index": 15303}],
        },
        {
            "question": "Which drugs are used to treat ocular hypertension?",
            "expected_cypher": 'MATCH (n:drug)-[r]-(d:disease {name:"ocular hypertension"}) RETURN n',
            "expected_answer": "108 drugs in total",
            "nodes": [
                {"index": 14198},
                {"index": 16847},
                {"index": 14198},
                {"index": 16847},
                {"index": 20522},
                {"index": 15779},
                {"index": 17317},
                {"index": 17435},
                {"index": 15491},
                {"index": 14978},
                {"index": 17434},
                {"index": 14677},
                {"index": 18311},
                {"index": 15156},
                {"index": 15095},
                {"index": 14792},
                {"index": 15225},
                {"index": 15155},
                {"index": 15105},
                {"index": 15849},
                {"index": 15079},
                {"index": 16719},
                {"index": 20339},
                {"index": 17436},
                {"index": 15483},
                {"index": 15239},
                {"index": 20522},
                {"index": 15779},
                {"index": 17317},
                {"index": 17435},
                {"index": 15491},
                {"index": 14978},
                {"index": 17434},
                {"index": 14677},
                {"index": 18311},
                {"index": 15156},
                {"index": 15095},
                {"index": 14792},
                {"index": 15225},
                {"index": 15155},
                {"index": 15105},
                {"index": 15849},
                {"index": 15079},
                {"index": 16719},
                {"index": 20339},
                {"index": 17436},
                {"index": 15483},
                {"index": 15239},
                {"index": 16057},
                {"index": 14542},
                {"index": 20327},
                {"index": 15406},
                {"index": 20323},
                {"index": 14023},
                {"index": 15171},
                {"index": 15132},
                {"index": 15104},
                {"index": 14036},
                {"index": 15877},
                {"index": 17231},
                {"index": 14990},
                {"index": 20321},
                {"index": 14540},
                {"index": 18036},
                {"index": 20318},
                {"index": 17241},
                {"index": 14898},
                {"index": 16065},
                {"index": 15086},
                {"index": 14287},
                {"index": 15479},
                {"index": 14340},
                {"index": 15338},
                {"index": 15582},
                {"index": 15260},
                {"index": 14956},
                {"index": 14336},
                {"index": 15074},
                {"index": 14838},
                {"index": 20418},
                {"index": 16220},
                {"index": 14038},
                {"index": 14017},
                {"index": 17305},
                {"index": 17230},
                {"index": 17232},
                {"index": 17233},
                {"index": 14320},
                {"index": 14042},
                {"index": 15003},
                {"index": 14028},
                {"index": 14030},
                {"index": 14173},
                {"index": 16063},
                {"index": 15412},
                {"index": 14034},
                {"index": 14050},
                {"index": 17238},
                {"index": 19699},
                {"index": 14018},
                {"index": 14783},
                {"index": 14881},
                {"index": 14159},
                {"index": 15500},
                {"index": 20304},
                {"index": 14610},
                {"index": 20218},
                {"index": 15180},
                {"index": 20303},
                {"index": 14272},
                {"index": 15117},
                {"index": 15089},
                {"index": 16306},
                {"index": 17246},
                {"index": 14020},
                {"index": 15783},
                {"index": 17256},
                {"index": 14754},
                {"index": 15064},
                {"index": 16490},
                {"index": 16066},
                {"index": 17378},
                {"index": 14249},
                {"index": 17252},
                {"index": 20295},
                {"index": 15122},
                {"index": 15210},
                {"index": 20294},
                {"index": 20157},
                {"index": 15211},
                {"index": 15087},
                {"index": 14144},
                {"index": 16057},
                {"index": 14542},
                {"index": 20327},
                {"index": 15406},
                {"index": 20323},
                {"index": 14023},
                {"index": 15171},
                {"index": 15132},
                {"index": 15104},
                {"index": 14036},
                {"index": 15877},
                {"index": 17231},
                {"index": 14990},
                {"index": 20321},
                {"index": 14540},
                {"index": 18036},
                {"index": 20318},
                {"index": 17241},
                {"index": 14898},
                {"index": 16065},
                {"index": 15086},
                {"index": 14287},
                {"index": 15479},
                {"index": 14340},
                {"index": 15338},
                {"index": 15582},
                {"index": 15260},
                {"index": 14956},
                {"index": 14336},
                {"index": 15074},
                {"index": 14838},
                {"index": 20418},
                {"index": 16220},
                {"index": 14038},
                {"index": 14017},
                {"index": 17305},
                {"index": 17230},
                {"index": 17232},
                {"index": 17233},
                {"index": 14320},
                {"index": 14042},
                {"index": 15003},
                {"index": 14028},
                {"index": 14030},
                {"index": 14173},
                {"index": 16063},
                {"index": 15412},
                {"index": 14034},
                {"index": 14050},
                {"index": 17238},
                {"index": 19699},
                {"index": 14018},
                {"index": 14783},
                {"index": 14881},
                {"index": 14159},
                {"index": 15500},
                {"index": 20304},
                {"index": 14610},
                {"index": 20218},
                {"index": 15180},
                {"index": 20303},
                {"index": 14272},
                {"index": 15117},
                {"index": 15089},
                {"index": 16306},
                {"index": 17246},
                {"index": 14020},
                {"index": 15783},
                {"index": 17256},
                {"index": 14754},
                {"index": 15064},
                {"index": 16490},
                {"index": 16066},
                {"index": 17378},
                {"index": 14249},
                {"index": 17252},
                {"index": 20295},
                {"index": 15122},
                {"index": 15210},
                {"index": 20294},
                {"index": 20157},
                {"index": 15211},
                {"index": 15087},
                {"index": 14144},
            ],
        },
        {
            "question": "Which drugs to treat ocular hypertension may cause the loss of eyelashes?",
            "expected_cypher": 'MATCH (n:drug)-[r]-(d:disease {name:"ocular hypertension"}) MATCH (n)-[s:side_effect]-(e:effect_or_phenotype {name:"loss of eyelashes"}) RETURN n',
            "expected_answer": "brinzolamide, bimatoprost, travoprost, paclitaxel, tobramycin",
            "nodes": [{"index": 15491}, {"index": 14978}, {"index": 17434}, {"index": 14792}, {"index": 20418}],
        },
        {
            "question": "Which disease does not show in cleft upper lip?",
            "expected_cypher": 'MATCH (d)-[:phenotype_absent]-(p:effect_or_phenotype {name:"cleft upper lip"}) RETURN d',
            "expected_answer": "ectrodactyly-cleft palate syndrome",
            "nodes": [{"index": 27588}],
        },
        {
            "question": "Which diseases show in symptoms such as eczema, neutropenia and a high forehead?",
            "expected_cypher": 'MATCH (d:disease)-[:phenotype_present]-({name:"eczema"}) MATCH (d)-[:phenotype_present]-({name:"neutropenia"}) MATCH (d)-[:phenotype_present]-({name:"high forehead"}) RETURN DISTINCT d.name',
            "expected_answer": "leigh syndrome, x-linked intellectual disability",
            "nodes": [{"index": 29857}, {"index": 27376}],
        },
        {
            "question": "Which expressions are present for the gene/protein f2?",
            "expected_cypher": 'MATCH (e:gene_or_protein {name:"f2"})-[ep:expression_present]-(a:anatomy) RETURN a.name',
            "expected_answer": "kidney, liver, fundus of stomach, stomach, multi-cellular organism, anatomical system, material anatomical entity, intestine, large intestine",
            "nodes": [
                {"index": 64551},
                {"index": 64545},
                {"index": 63609},
                {"index": 63423},
                {"index": 129373},
                {"index": 63372},
                {"index": 63370},
                {"index": 63223},
                {"index": 63164},
            ],
        },
        {
            "question": "Which drugs are associated with epilepsy?",
            "expected_cypher": 'MATCH (d:drug)-[:indication]->(dis:disease) WHERE dis.name = "epilepsy" RETURN d.name',
            "expected_answer": "The drugs associated with epilepsy are phenytoin, valproic acid, lamotrigine, diazepam, clonazepam, fosphenytoin, mephenytoin, neocitrullamon, carbamazepine, phenobarbital, secobarbital, primidone, and lorazepam.",
            "nodes": [
                {"index": 14141},
                {"index": 14153},
                {"index": 14192},
                {"index": 14245},
                {"index": 14330},
                {"index": 14505},
                {"index": 14522},
                {"index": 14956},
                {"index": 14993},
                {"index": 15297},
                {"index": 15311},
                {"index": 15430},
                {"index": 15434},
                {"index": 15453},
                {"index": 15475},
                {"index": 15834},
                {"index": 15835},
                {"index": 15837},
                {"index": 16475},
                {"index": 20341},
                {"index": 20354},
                {"index": 17065},
                {"index": 17066},
            ],
        },
        {
            "question": "Which genes can be novel targets in stickler syndrome?",
            "expected_cypher": 'MATCH (g:gene_or_protein)-[:associated_with]-(d {name: "stickler syndrome"}) RETURN g',
            "expected_answer": "Genes associated with stickler syndrome are col11a1, col9a1, col9a2, loxl3, col2a1 and col9a3.",
            "nodes": [
                {"index": 1905},
                {"index": 3698},
                {"index": 4556},
                {"index": 5659},
                {"index": 7277},
                {"index": 12573},
            ],
        },
        {
            "question": "How many proteins are connected to rheumatoid arthritis?",
            "expected_cypher": 'MATCH (g:disease {name: "rheumatoid arthritis"})-[]-(p:gene_or_protein) RETURN COUNT(DISTINCT p)',
            "expected_answer": "There are 174 genes or proteins associated with rheumatoid arthritis",
            "nodes": [
                {"index": 22094},
                {"index": 34085},
                {"index": 12200},
                {"index": 34816},
                {"index": 13983},
                {"index": 34815},
                {"index": 34814},
                {"index": 34813},
                {"index": 34812},
                {"index": 13396},
                {"index": 34811},
                {"index": 9502},
                {"index": 9363},
                {"index": 34810},
                {"index": 4154},
                {"index": 4496},
                {"index": 7807},
                {"index": 9012},
                {"index": 34809},
                {"index": 557},
                {"index": 11014},
                {"index": 7573},
                {"index": 4579},
                {"index": 12881},
                {"index": 5792},
                {"index": 8877},
                {"index": 4142},
                {"index": 8042},
                {"index": 2837},
                {"index": 6048},
                {"index": 3901},
                {"index": 13740},
                {"index": 11806},
                {"index": 3183},
                {"index": 1480},
                {"index": 8160},
                {"index": 4510},
                {"index": 9376},
                {"index": 974},
                {"index": 8707},
                {"index": 2328},
                {"index": 13259},
                {"index": 10120},
                {"index": 4469},
                {"index": 3576},
                {"index": 7076},
                {"index": 6088},
                {"index": 749},
                {"index": 6428},
                {"index": 429},
                {"index": 843},
                {"index": 3647},
                {"index": 13382},
                {"index": 2329},
                {"index": 1508},
                {"index": 1770},
                {"index": 2730},
                {"index": 8748},
                {"index": 795},
                {"index": 4959},
                {"index": 7902},
                {"index": 5385},
                {"index": 12663},
                {"index": 13245},
                {"index": 1912},
                {"index": 10565},
                {"index": 1064},
                {"index": 430},
                {"index": 8086},
                {"index": 4162},
                {"index": 484},
                {"index": 4425},
                {"index": 9031},
                {"index": 5205},
                {"index": 1705},
                {"index": 34808},
                {"index": 12120},
                {"index": 8597},
                {"index": 4152},
                {"index": 11846},
                {"index": 1390},
                {"index": 3403},
                {"index": 3830},
                {"index": 6175},
                {"index": 4818},
                {"index": 8069},
                {"index": 6874},
                {"index": 2151},
                {"index": 5692},
                {"index": 1239},
                {"index": 3595},
                {"index": 7865},
                {"index": 1037},
                {"index": 4582},
                {"index": 1553},
                {"index": 3646},
                {"index": 5437},
                {"index": 2618},
                {"index": 3552},
                {"index": 2983},
                {"index": 2978},
                {"index": 1533},
                {"index": 9234},
                {"index": 1567},
                {"index": 4322},
                {"index": 7059},
                {"index": 4941},
                {"index": 1004},
                {"index": 862},
                {"index": 8676},
                {"index": 4098},
                {"index": 3495},
                {"index": 2075},
                {"index": 9181},
                {"index": 6664},
                {"index": 2625},
                {"index": 12946},
                {"index": 7083},
                {"index": 13169},
                {"index": 8432},
                {"index": 4499},
                {"index": 6836},
                {"index": 3950},
                {"index": 8579},
                {"index": 1559},
                {"index": 7090},
                {"index": 4850},
                {"index": 1836},
                {"index": 1990},
                {"index": 29},
                {"index": 3434},
                {"index": 3533},
                {"index": 6250},
                {"index": 7134},
                {"index": 1039},
                {"index": 4610},
                {"index": 3722},
                {"index": 2579},
                {"index": 151},
                {"index": 3231},
                {"index": 10317},
                {"index": 2384},
                {"index": 4184},
                {"index": 7277},
                {"index": 3181},
                {"index": 12740},
                {"index": 802},
                {"index": 5098},
                {"index": 1609},
                {"index": 4111},
                {"index": 1977},
                {"index": 3745},
                {"index": 3438},
                {"index": 2027},
                {"index": 4487},
                {"index": 2797},
                {"index": 2214},
                {"index": 10121},
                {"index": 3104},
                {"index": 388},
                {"index": 2874},
                {"index": 6873},
                {"index": 4186},
                {"index": 5631},
                {"index": 1691},
                {"index": 6558},
                {"index": 4676},
                {"index": 6095},
                {"index": 3631},
                {"index": 5230},
                {"index": 1694},
                {"index": 5784},
                {"index": 2959},
                {"index": 4969},
            ],
        },
        {
            "question": "Is there clinical evidence linking HGF to kidneys?",
            "expected_cypher": 'MATCH (g {name: "hgf"})-[]-(g2 {name: "kidney"}) RETURN COUNT(g)>0',
            "expected_answer": "Yes.",
            "subgraph_query": 'MATCH (g {name: "hgf"})-[r]-(g2 {name: "kidney"}) RETURN g,r, g2',
            "nodes": [],  # TODO what would the index values be for a boolean query?
        },
        {
            "question": "What issues are there with lamotrigine?",
            "expected_cypher": 'MATCH (d:drug  {name:"lamotrigine"})-[:side_effect]-(e) RETURN e',
            "expected_answer": "THere are many known side effects, among which are edema, pain and cardiac arrest.",
            "nodes": [
                {"index": 23327},
                {"index": 22323},
                {"index": 22998},
                {"index": 26938},
                {"index": 22811},
                {"index": 23097},
                {"index": 23214},
                {"index": 24285},
                {"index": 94295},
                {"index": 22791},
                {"index": 22613},
                {"index": 24326},
                {"index": 89315},
                {"index": 23003},
                {"index": 23043},
                {"index": 84340},
                {"index": 84420},
                {"index": 23544},
                {"index": 85848},
                {"index": 94161},
                {"index": 84767},
                {"index": 89076},
                {"index": 26096},
                {"index": 24475},
                {"index": 33733},
                {"index": 84526},
                {"index": 26607},
                {"index": 89307},
                {"index": 84469},
                {"index": 84920},
                {"index": 84697},
                {"index": 84693},
                {"index": 84704},
                {"index": 84694},
                {"index": 24197},
                {"index": 24296},
                {"index": 84392},
                {"index": 84549},
                {"index": 22786},
                {"index": 23157},
                {"index": 23956},
                {"index": 25311},
                {"index": 93440},
                {"index": 23101},
                {"index": 24568},
                {"index": 23183},
                {"index": 22934},
                {"index": 23833},
                {"index": 25672},
                {"index": 84374},
                {"index": 22594},
                {"index": 88529},
                {"index": 25332},
                {"index": 26206},
                {"index": 22454},
                {"index": 22571},
                {"index": 22250},
                {"index": 23094},
                {"index": 84452},
                {"index": 23000},
                {"index": 23083},
                {"index": 22939},
                {"index": 25945},
                {"index": 84364},
                {"index": 33751},
                {"index": 84771},
                {"index": 33737},
                {"index": 23153},
                {"index": 25644},
                {"index": 22938},
                {"index": 23950},
                {"index": 85015},
                {"index": 22982},
                {"index": 84485},
                {"index": 22457},
                {"index": 84855},
                {"index": 22458},
                {"index": 84409},
                {"index": 84429},
                {"index": 22658},
                {"index": 23545},
                {"index": 22272},
                {"index": 22676},
                {"index": 22450},
                {"index": 84519},
                {"index": 85105},
                {"index": 23983},
                {"index": 33730},
                {"index": 22923},
                {"index": 84401},
                {"index": 85002},
                {"index": 84512},
                {"index": 24952},
                {"index": 22604},
                {"index": 23244},
                {"index": 22588},
                {"index": 22484},
                {"index": 23139},
                {"index": 23863},
                {"index": 25526},
                {"index": 84416},
                {"index": 23072},
                {"index": 23844},
                {"index": 25562},
                {"index": 23284},
                {"index": 22916},
                {"index": 85104},
                {"index": 22859},
                {"index": 22828},
                {"index": 23002},
                {"index": 23113},
                {"index": 26337},
                {"index": 84544},
                {"index": 84555},
                {"index": 84398},
                {"index": 84553},
                {"index": 23079},
                {"index": 24337},
                {"index": 22581},
                {"index": 22952},
                {"index": 25620},
                {"index": 25608},
                {"index": 22692},
                {"index": 22539},
                {"index": 23764},
                {"index": 26322},
                {"index": 22262},
                {"index": 84773},
                {"index": 22550},
                {"index": 84774},
                {"index": 90663},
                {"index": 93442},
                {"index": 23864},
                {"index": 23011},
                {"index": 84362},
                {"index": 23096},
                {"index": 23135},
                {"index": 84853},
                {"index": 25201},
                {"index": 24015},
                {"index": 84675},
                {"index": 24338},
                {"index": 25527},
                {"index": 22999},
                {"index": 22304},
                {"index": 93464},
                {"index": 26160},
                {"index": 26162},
                {"index": 33727},
                {"index": 26336},
                {"index": 90710},
                {"index": 23254},
                {"index": 25900},
                {"index": 86984},
                {"index": 23469},
                {"index": 25276},
                {"index": 22718},
                {"index": 23901},
                {"index": 22421},
                {"index": 23266},
                {"index": 23125},
                {"index": 22447},
                {"index": 85849},
                {"index": 84361},
                {"index": 22928},
                {"index": 85001},
                {"index": 23799},
                {"index": 23073},
                {"index": 25569},
                {"index": 23158},
            ],
        },
        {
            "question": "Which drug against epilepsy has the least side effects?",
            "expected_cypher": 'MATCH (d:drug)-[:indication]-(e:disease {name:"epilepsy"}) MATCH (d)-[:side_effect]-(s) RETURN d, COUNT(DISTINCT s) as side_effects ORDER BY side_effects ASC LIMIT 1',
            "expected_answer": "The drug primidone",
            "subgraph_query": 'MATCH (d:drug)-[i:indication]-(e:disease {name:"epilepsy"}), (d)-[se:side_effect]-(s) RETURN d, i, e, se, s',
            "nodes": [{"index": 15311}],
        },
        {
            "question": "Does lamotrigine have more side effects than primidone?",
            "expected_cypher": 'MATCH (d:drug {name:"primidone"})-[:side_effect]-(s) MATCH (d2:drug {name:"lamotrigine"})-[:side_effect]-(s2) RETURN COUNT(DISTINCT s2) > COUNT(DISTINCT s)',
            "expected_answer": "Yes",
            "nodes": [],  # TODO what would the index values be for a boolean query?
        },
        {
            "question": "Is psoriasis related with atopic dermatitis?",
            "expected_cypher": 'RETURN EXISTS {MATCH (d {name:"atopic dermatitis"})-[]-(d2 {name:"psoriasis"})}',
            "expected_answer": "No",
            "subgraph_query": 'MATCH (d {name:"atopic dermatitis"})-[r]-(d2 {name:"psoriasis"}) RETURN d, r, d2',
            "nodes": [],  # TODO what would the index values be for a boolean query?
        },
        {
            "question": "What genes play a role in breast cancer?",
            "expected_cypher": 'MATCH (d:disease {name:"breast cancer"})-[]-(g:gene_or_protein) RETURN DISTINCT g',
            "expected_answer": "More than 300 genes or proteins are associated with breast cancer.",
            "nodes": [
                {"index": 22075},
                {"index": 34242},
                {"index": 34241},
                {"index": 34240},
                {"index": 34239},
                {"index": 34238},
                {"index": 34237},
                {"index": 34236},
                {"index": 34235},
                {"index": 34234},
                {"index": 34233},
                {"index": 34232},
                {"index": 34231},
                {"index": 34230},
                {"index": 33958},
                {"index": 34229},
                {"index": 34228},
                {"index": 33859},
                {"index": 34227},
                {"index": 34226},
                {"index": 34225},
                {"index": 34224},
                {"index": 33885},
                {"index": 22058},
                {"index": 34223},
                {"index": 34222},
                {"index": 33954},
                {"index": 34221},
                {"index": 34220},
                {"index": 33921},
                {"index": 34219},
                {"index": 34218},
                {"index": 34217},
                {"index": 34216},
                {"index": 34215},
                {"index": 34214},
                {"index": 34213},
                {"index": 34212},
                {"index": 34211},
                {"index": 4570},
                {"index": 34210},
                {"index": 34209},
                {"index": 34208},
                {"index": 34207},
                {"index": 34206},
                {"index": 34205},
                {"index": 34204},
                {"index": 3934},
                {"index": 34203},
                {"index": 34202},
                {"index": 34201},
                {"index": 34200},
                {"index": 34199},
                {"index": 34198},
                {"index": 34197},
                {"index": 34196},
                {"index": 34195},
                {"index": 13074},
                {"index": 13121},
                {"index": 34194},
                {"index": 34193},
                {"index": 11120},
                {"index": 13021},
                {"index": 34192},
                {"index": 34191},
                {"index": 34190},
                {"index": 34189},
                {"index": 12105},
                {"index": 33841},
                {"index": 34188},
                {"index": 34187},
                {"index": 34186},
                {"index": 12608},
                {"index": 11311},
                {"index": 34185},
                {"index": 5079},
                {"index": 34184},
                {"index": 13227},
                {"index": 34183},
                {"index": 34182},
                {"index": 33837},
                {"index": 34181},
                {"index": 12148},
                {"index": 7086},
                {"index": 34180},
                {"index": 12072},
                {"index": 34179},
                {"index": 34178},
                {"index": 7375},
                {"index": 34177},
                {"index": 9319},
                {"index": 34176},
                {"index": 34175},
                {"index": 34174},
                {"index": 34173},
                {"index": 12749},
                {"index": 34172},
                {"index": 3376},
                {"index": 34171},
                {"index": 12467},
                {"index": 34170},
                {"index": 11296},
                {"index": 9840},
                {"index": 4436},
                {"index": 34169},
                {"index": 34168},
                {"index": 34167},
                {"index": 13463},
                {"index": 9365},
                {"index": 12843},
                {"index": 34166},
                {"index": 34165},
                {"index": 5758},
                {"index": 34164},
                {"index": 34163},
                {"index": 786},
                {"index": 34162},
                {"index": 22089},
                {"index": 12128},
                {"index": 34076},
                {"index": 34161},
                {"index": 34160},
                {"index": 34159},
                {"index": 34158},
                {"index": 6950},
                {"index": 34157},
                {"index": 11966},
                {"index": 34156},
                {"index": 34155},
                {"index": 34154},
                {"index": 34153},
                {"index": 34152},
                {"index": 34151},
                {"index": 34150},
                {"index": 34149},
                {"index": 13165},
                {"index": 34074},
                {"index": 13993},
                {"index": 34148},
                {"index": 34147},
                {"index": 776},
                {"index": 34146},
                {"index": 34145},
                {"index": 12145},
                {"index": 34144},
                {"index": 8722},
                {"index": 34143},
                {"index": 10512},
                {"index": 34142},
                {"index": 34141},
                {"index": 9103},
                {"index": 21975},
                {"index": 11248},
                {"index": 34140},
                {"index": 12235},
                {"index": 9739},
                {"index": 13988},
                {"index": 34139},
                {"index": 34138},
                {"index": 34137},
                {"index": 13905},
                {"index": 7592},
                {"index": 12490},
                {"index": 34136},
                {"index": 13029},
                {"index": 10086},
                {"index": 3670},
                {"index": 34135},
                {"index": 13573},
                {"index": 11102},
                {"index": 5208},
                {"index": 6512},
                {"index": 380},
                {"index": 3494},
                {"index": 13555},
                {"index": 13258},
                {"index": 12790},
                {"index": 34134},
                {"index": 7431},
                {"index": 7577},
                {"index": 34133},
                {"index": 3395},
                {"index": 6809},
                {"index": 9918},
                {"index": 5108},
                {"index": 13619},
                {"index": 9602},
                {"index": 34132},
                {"index": 34131},
                {"index": 11938},
                {"index": 34130},
                {"index": 5760},
                {"index": 10802},
                {"index": 10861},
                {"index": 34129},
                {"index": 34128},
                {"index": 12621},
                {"index": 34127},
                {"index": 7476},
                {"index": 2916},
                {"index": 4642},
                {"index": 4395},
                {"index": 6733},
                {"index": 13364},
                {"index": 5506},
                {"index": 5126},
                {"index": 8480},
                {"index": 8401},
                {"index": 812},
                {"index": 12140},
                {"index": 11413},
                {"index": 7877},
                {"index": 5746},
                {"index": 7915},
                {"index": 9993},
                {"index": 1228},
                {"index": 7317},
                {"index": 10035},
                {"index": 13},
                {"index": 165},
                {"index": 11623},
                {"index": 5221},
                {"index": 34126},
                {"index": 12355},
                {"index": 11456},
                {"index": 8673},
                {"index": 10276},
                {"index": 34125},
                {"index": 5652},
                {"index": 8732},
                {"index": 12947},
                {"index": 12347},
                {"index": 6697},
                {"index": 11867},
                {"index": 34124},
                {"index": 9849},
                {"index": 34123},
                {"index": 12652},
                {"index": 3806},
                {"index": 12571},
                {"index": 34122},
                {"index": 12493},
                {"index": 2520},
                {"index": 7487},
                {"index": 9109},
                {"index": 22004},
                {"index": 10683},
                {"index": 6361},
                {"index": 11988},
                {"index": 8533},
                {"index": 34121},
                {"index": 9643},
                {"index": 11567},
                {"index": 5138},
                {"index": 34120},
                {"index": 12420},
                {"index": 34119},
                {"index": 10994},
                {"index": 34118},
                {"index": 9742},
                {"index": 7848},
                {"index": 9401},
                {"index": 13967},
                {"index": 4816},
                {"index": 12414},
                {"index": 6682},
                {"index": 8201},
                {"index": 6017},
                {"index": 7401},
                {"index": 6111},
                {"index": 4924},
                {"index": 12126},
                {"index": 33796},
                {"index": 34117},
                {"index": 11556},
                {"index": 10508},
                {"index": 5445},
                {"index": 10456},
                {"index": 10636},
                {"index": 469},
                {"index": 5444},
                {"index": 12702},
                {"index": 5797},
                {"index": 12647},
                {"index": 9134},
                {"index": 9252},
                {"index": 12997},
                {"index": 6307},
                {"index": 2287},
                {"index": 4174},
                {"index": 22013},
                {"index": 7473},
                {"index": 11232},
                {"index": 9562},
                {"index": 602},
                {"index": 7380},
                {"index": 7596},
                {"index": 3610},
                {"index": 733},
                {"index": 6605},
                {"index": 10831},
                {"index": 11090},
                {"index": 4270},
                {"index": 4516},
                {"index": 11758},
                {"index": 1617},
                {"index": 8688},
                {"index": 34116},
                {"index": 6014},
                {"index": 9476},
                {"index": 34115},
                {"index": 11994},
                {"index": 2315},
                {"index": 6478},
                {"index": 5351},
                {"index": 4579},
                {"index": 13939},
                {"index": 12008},
                {"index": 1673},
                {"index": 7943},
                {"index": 2050},
                {"index": 1536},
                {"index": 34114},
                {"index": 5699},
                {"index": 6357},
                {"index": 4246},
                {"index": 8017},
                {"index": 11192},
                {"index": 34113},
                {"index": 9101},
                {"index": 7528},
                {"index": 12319},
                {"index": 34035},
                {"index": 7510},
                {"index": 9033},
                {"index": 34112},
                {"index": 1595},
                {"index": 5708},
                {"index": 11997},
                {"index": 9871},
                {"index": 7672},
                {"index": 10058},
                {"index": 34111},
                {"index": 13425},
                {"index": 8658},
                {"index": 3969},
                {"index": 2687},
                {"index": 2334},
                {"index": 3802},
                {"index": 5151},
                {"index": 626},
                {"index": 9464},
                {"index": 7028},
                {"index": 5537},
                {"index": 9584},
                {"index": 4525},
                {"index": 9968},
                {"index": 4562},
                {"index": 586},
                {"index": 2407},
                {"index": 8085},
                {"index": 6669},
                {"index": 7027},
                {"index": 4384},
                {"index": 6727},
                {"index": 8694},
                {"index": 10304},
                {"index": 6453},
                {"index": 11878},
                {"index": 9304},
                {"index": 7033},
                {"index": 5624},
                {"index": 6181},
                {"index": 10938},
                {"index": 13736},
                {"index": 11800},
                {"index": 10629},
                {"index": 3868},
                {"index": 34110},
                {"index": 8979},
                {"index": 4096},
                {"index": 11262},
                {"index": 4895},
                {"index": 4087},
                {"index": 13291},
                {"index": 10639},
                {"index": 9527},
                {"index": 34109},
                {"index": 134},
                {"index": 11044},
                {"index": 34108},
                {"index": 3172},
                {"index": 7140},
                {"index": 12287},
                {"index": 4946},
                {"index": 10340},
                {"index": 8720},
                {"index": 5460},
                {"index": 34107},
                {"index": 34106},
                {"index": 34105},
                {"index": 6966},
                {"index": 7945},
                {"index": 8392},
                {"index": 5058},
                {"index": 10292},
                {"index": 2309},
                {"index": 7810},
                {"index": 12077},
                {"index": 3428},
                {"index": 3702},
                {"index": 12182},
                {"index": 9437},
                {"index": 4568},
                {"index": 5311},
                {"index": 1171},
                {"index": 5560},
                {"index": 4567},
                {"index": 10647},
                {"index": 8612},
                {"index": 10560},
                {"index": 4466},
                {"index": 7662},
                {"index": 1115},
                {"index": 4691},
                {"index": 993},
                {"index": 5930},
                {"index": 10506},
                {"index": 6251},
                {"index": 1943},
                {"index": 10221},
                {"index": 2840},
                {"index": 10535},
                {"index": 1205},
                {"index": 13937},
                {"index": 7776},
                {"index": 3478},
                {"index": 6937},
                {"index": 11320},
                {"index": 9874},
                {"index": 206},
                {"index": 13779},
                {"index": 10491},
                {"index": 242},
                {"index": 34104},
                {"index": 11155},
                {"index": 13285},
                {"index": 6497},
                {"index": 3247},
                {"index": 34103},
                {"index": 2504},
                {"index": 11149},
                {"index": 6444},
                {"index": 2203},
                {"index": 34102},
                {"index": 3398},
                {"index": 12059},
                {"index": 7268},
                {"index": 728},
                {"index": 1072},
                {"index": 7630},
                {"index": 472},
                {"index": 2248},
                {"index": 10996},
                {"index": 8250},
                {"index": 9784},
                {"index": 12193},
                {"index": 6575},
                {"index": 11362},
                {"index": 4784},
                {"index": 7754},
                {"index": 4077},
                {"index": 6630},
                {"index": 4287},
                {"index": 4962},
                {"index": 7588},
                {"index": 8333},
                {"index": 9689},
                {"index": 13171},
                {"index": 10800},
                {"index": 5706},
                {"index": 3183},
                {"index": 9636},
                {"index": 12325},
                {"index": 2366},
                {"index": 432},
                {"index": 10238},
                {"index": 3935},
                {"index": 7100},
                {"index": 9488},
                {"index": 34101},
                {"index": 9933},
                {"index": 6833},
                {"index": 7870},
                {"index": 7066},
                {"index": 10492},
                {"index": 26},
                {"index": 13515},
                {"index": 3609},
                {"index": 11659},
            ],
        },
        {
            "question": "Which drugs against epilepsy should not be used by patients with hypertension?",
            "expected_cypher": 'MATCH (dr:drug)-[:contraindication]-(d:disease {name:"hypertension"}) MATCH (dr)-[:indication]-(d2:disease {name:"epilepsy"}) RETURN dr ',
            "expected_answer": "Carbamazepine, phenobarbital, amobarbital, secobarbital and pentobarbital should not be used. ",
            "nodes": [{"index": 14956}, {"index": 14993}, {"index": 17066}, {"index": 15297}, {"index": 15430}],
        },
        {
            "question": "Can I use iboprofen in patients with addison disease?",
            "expected_cypher": 'RETURN NOT EXISTS {MATCH (dr:drug {name:"ibuprofen"})-[:contraindication]-(c {name:"addison disease"}) }',
            "expected_answer": "No",
            "subgraph_query": 'MATCH (dr:drug {name:"ibuprofen"})-[c:contraindication]-(a {name:"addison disease"}) RETURN dr, c, a',
            "nodes": [{"index": 14287}, {"index": 30813}],  # TODO what would the index values be for a boolean query?
        },
        {
            "question": "Which off label medicaments for epilepsie have a clogp value below 0?",
            "expected_cypher": 'MATCH (d)-[r:off_label_use]->(di:disease {name:"epilepsy"}) WHERE toFloat(d.clogp) < 0 RETURN d',
            "expected_answer": "Acetazolamide.",
            "nodes": [{"index": 15466}],
        },
        {
            "question": "Which disease may a patient presented with fatigue, vomiting and abdominal pain have?",
            "expected_cypher": 'MATCH (d:disease)-[]-() WHERE d.symptoms CONTAINS "fatigue" AND d.symptoms CONTAINS "vomiting" and d.symptoms CONTAINS "abdominal pain" RETURN COLLECT(distinct d) as d',
            "expected_answer": "X-linked intellectual disability with hypopituitarism, panhypopituitarism, xr-linked, hypopituitarism, adrenal insufficiency, addison disease, corpus callosum dysgenesis hypopituitarism, arthrogryposis, distal, with hypopituitarism, intellectual disability, and facial anomalies, midline malformations, multiple, with limb abnormalities and hypopituitarism, or choroideremia-hypopituitarism",
            "nodes": [
                {"index": 31341},
                {"index": 31665},
                {"index": 32217},
                {"index": 32384},
                {"index": 30813},
                {"index": 99307},
                {"index": 28641},
                {"index": 31006},
                {"index": 97972},
            ],
        },
        {
            "question": "Which drug targeting cyb5a is a micro nutrient?",
            "expected_cypher": 'MATCH (c:category {name: "Micronutrients"})-[:is_a]-(d:drug)-[:target]-(t:gene_or_protein {name:"cyb5a"}) RETURN d',
            "expected_answer": "Chromium",
            "nodes": [{"index": 14117}],
        },
        {
            "question": "How many drugs against epilepsy are on the market?",
            "expected_cypher": 'MATCH (d:drug)-[:indication]-(di:disease {name:"epilepsy"}) RETURN COUNT(DISTINCT(d)) as n_drugs',
            "expected_answer": "23",
            "nodes": [
                {"index": 15835},
                {"index": 15453},
                {"index": 15311},
                {"index": 15434},
                {"index": 15475},
                {"index": 14956},
                {"index": 14522},
                {"index": 14245},
                {"index": 14993},
                {"index": 17065},
                {"index": 14505},
                {"index": 20341},
                {"index": 14141},
                {"index": 17066},
                {"index": 16475},
                {"index": 15297},
                {"index": 15430},
                {"index": 15834},
                {"index": 15837},
                {"index": 14192},
                {"index": 20354},
                {"index": 14330},
                {"index": 14153},
            ],
        },
        {
            "question": "Which drugs against asthma are not pills?",
            "expected_cypher": 'MATCH (d:disease {name:"asthma"})-[]-(dr:drug) WHERE dr.aggregate_state <> "solid" RETURN distinct dr',
            "expected_answer": "Oleandomycin, tulobuterol, hydrocortisone acetate, reproterol, acefylline, ozagrel, chlorpheniramine, dexchlorpheniramine, amphetamine, trolnitrate, brompheniramine, secretin human, iocetamic acid, edrophonium, estradiol cypionate, regadenoson, dinoprost, misoprostol and estradiol valerate",
            "nodes": [
                {"index": 15646},
                {"index": 17568},
                {"index": 14042},
                {"index": 20459},
                {"index": 15399},
                {"index": 20457},
                {"index": 14990},
                {"index": 15260},
                {"index": 15074},
                {"index": 20303},
                {"index": 16066},
                {"index": 16892},
                {"index": 20346},
                {"index": 17175},
                {"index": 14481},
                {"index": 18123},
                {"index": 19699},
                {"index": 16221},
                {"index": 14483},
            ],
        },
        {
            "question": "Which drugs can I give my dog with influenza?",
            "expected_cypher": 'MATCH (d:drug)-[]-(a:approval_status {name:"vet_approved"}) MATCH (d)-[]-(di:disease {name:"influenza"}) RETURN d',
            "expected_answer": "Formaldehyde",
            "nodes": [{"index": 20632}],
        },
        {
            "question": "How many experimental drugs against epilepsy are there?",
            "expected_cypher": 'MATCH (d:drug)-[]-(a:approval_status {name:"experimental"}) MATCH (d)-[]-(di:disease {name:"epilepsy"}) RETURN COUNT(distinct d) as number',
            "expected_answer": "13",
            "nodes": [
                {"index": 14522},
                {"index": 20341},
                {"index": 17377},
                {"index": 16014},
                {"index": 20350},
                {"index": 14484},
                {"index": 14340},
                {"index": 20314},
                {"index": 15260},
                {"index": 14159},
                {"index": 20303},
                {"index": 20335},
                {"index": 20353},
            ],
        },
        {
            "question": "Which drug against epilepsy has the lowest clogp value?",
            "expected_cypher": 'MATCH (d)-[r:off_label_use]->(di:disease {name:"epilepsy"}) RETURN d ORDER BY toFloat(d.clogp) ASC LIMIT 1',
            "expected_answer": "Acetazolamide",
            "nodes": [{"index": 15466}],
        },
        {
            "question": "Which genes influence the peripheral nervous system?",
            "expected_cypher": 'MATCH (a:anatomy {name:"peripheral nervous system"})-[e:expression_present]-(b) RETURN DISTINCT b',
            "expected_answer": "aqp1, vamp2, mbp, ahcyl1, rpl13, tsc22d4, ndrg2, aplp1, itm2c, glul, rpl3, ddx17, znf428, gfap, bsdc1, calm3, slc22a17, csrp1, gnas, sparcl1, dst, plp1, tnfrsf1a, tram1, mapk8ip1, gpm6b, rhbdd2",
            "nodes": [
                {"index": 118},
                {"index": 1581},
                {"index": 4480},
                {"index": 3125},
                {"index": 885},
                {"index": 5760},
                {"index": 12027},
                {"index": 7377},
                {"index": 10447},
                {"index": 6418},
                {"index": 1454},
                {"index": 7892},
                {"index": 10889},
                {"index": 1863},
                {"index": 7448},
                {"index": 941},
                {"index": 56628},
                {"index": 11537},
                {"index": 3176},
                {"index": 13628},
                {"index": 3795},
                {"index": 560},
                {"index": 665},
                {"index": 2777},
                {"index": 6347},
                {"index": 10407},
                {"index": 3777},
            ],
        },
        {
            "question": "How many drugs are gaseous?",
            "expected_cypher": 'MATCH (d:drug)-[]-(c:category) WHERE c.name ="Gases" RETURN COUNT(DISTINCT d)',
            "expected_answer": "16",
            "nodes": [
                {"index": 39894},
                {"index": 21417},
                {"index": 21231},
                {"index": 20664},
                {"index": 20571},
                {"index": 20499},
                {"index": 20064},
                {"index": 19502},
                {"index": 19495},
                {"index": 19159},
                {"index": 16035},
                {"index": 14972},
                {"index": 14928},
                {"index": 14634},
                {"index": 14585},
                {"index": 14013},
            ],
        },
        {
            "question": "In which Anatomical regions is IRAK4 expressed?",
            "expected_answer": "omental fat pad, cerebral cortex, adipose tissue of the abdominal region, layer, squamous epithelium, nasal cavity epithelium, mammary gland, oviduct, fallopian tube, oral cavity, esophagus, decidua, muscle, connective tissue, tonsil, bone marrow, thymus gland, adrenal gland, prostate gland, peritoneum, corpus callosum, spinal cord, subcutaneous adipose tissue, bronchus, cerebellar cortex, jejunum, duodenum, kidney, small intestine, liver, spleen, heart left ventricle, cardiac ventricle, cardiac atrium, lung, thyroid gland, substantia nigra, cerebellum, placenta, Ammon's horn, neocortex, hypothalamus, telencephalon, mesencephalon, forebrain, temporal lobe",
            "expected_cypher": 'MATCH (g:gene_or_protein {name: "irak4"})-[:expression_present]->(a:anatomy) RETURN a.name',
            "nodes": [
                {"index": 69864},
                {"index": 69442},
                {"index": 68729},
                {"index": 68618},
                {"index": 68316},
                {"index": 67302},
                {"index": 67143},
                {"index": 66826},
                {"index": 66050},
                {"index": 65990},
                {"index": 65557},
                {"index": 64894},
                {"index": 64876},
                {"index": 64814},
                {"index": 64813},
                {"index": 64801},
                {"index": 64800},
                {"index": 64799},
                {"index": 64798},
                {"index": 64796},
                {"index": 64787},
                {"index": 64766},
                {"index": 64675},
                {"index": 64626},
                {"index": 64621},
                {"index": 64565},
                {"index": 64553},
                {"index": 64552},
                {"index": 64551},
                {"index": 64546},
                {"index": 64545},
                {"index": 64544},
                {"index": 64523},
                {"index": 64521},
                {"index": 64520},
                {"index": 64487},
                {"index": 64485},
                {"index": 64477},
                {"index": 64476},
                {"index": 64470},
                {"index": 64427},
                {"index": 64416},
                {"index": 64395},
                {"index": 64391},
                {"index": 64352},
                {"index": 64339},
                {"index": 64334},
                {"index": 64332},
                {"index": 64331},
                {"index": 64312},
                {"index": 64311},
                {"index": 64269},
                {"index": 64267},
                {"index": 64253},
                {"index": 64065},
                {"index": 63830},
                {"index": 63772},
                {"index": 63745},
                {"index": 63744},
                {"index": 63713},
                {"index": 63704},
                {"index": 63684},
                {"index": 63674},
                {"index": 63609},
                {"index": 63604},
                {"index": 63603},
                {"index": 63602},
                {"index": 63584},
                {"index": 63501},
                {"index": 63500},
                {"index": 63481},
                {"index": 63466},
                {"index": 63465},
                {"index": 63464},
                {"index": 129374},
                {"index": 63431},
                {"index": 63430},
                {"index": 63426},
                {"index": 63425},
                {"index": 63423},
                {"index": 63377},
                {"index": 63376},
                {"index": 129373},
                {"index": 63372},
                {"index": 63370},
                {"index": 63359},
                {"index": 63273},
                {"index": 63240},
                {"index": 63223},
                {"index": 63181},
                {"index": 63164},
                {"index": 63148},
                {"index": 63136},
                {"index": 63123},
                {"index": 63117},
                {"index": 63116},
                {"index": 63112},
            ],
        },
        {
            "question": "What are the phenotypes associated with cardioacrofacial dysplasia?",
            "expected_answer": "mandibular prognathia, postaxial foot polydactyly, clinodactyly of the 5th finger, clubbing, hypodontia, left superior vena cava draining to coronary sinus, common atrium, recurrent patellar dislocation, postaxial hand polydactyly, narrow chest, broad forehead, prominent nasal tip, deep philtrum, tented upper lip vermilion, conical tooth, diastema, autosomal dominant inheritance, short philtrum, genu valgum, long thorax, atrioventricular canal defect, hypoplasia of the maxilla, postaxial polydactyly, accessory oral frenulum, overhanging nasal tip, midface retrusion, nail dysplasia, long face, complete atrioventricular canal defect, congenital onset, limb undergrowth",
            "expected_cypher": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(p:effect_or_phenotype) RETURN p.name',
            "nodes": [
                {"index": 84579},
                {"index": 24072},
                {"index": 24065},
                {"index": 26961},
                {"index": 84683},
                {"index": 88879},
                {"index": 88810},
                {"index": 86071},
                {"index": 24042},
                {"index": 84713},
                {"index": 84589},
                {"index": 86195},
                {"index": 85031},
                {"index": 88415},
                {"index": 25325},
                {"index": 24152},
                {"index": 22759},
                {"index": 84586},
                {"index": 26936},
                {"index": 93559},
                {"index": 22838},
                {"index": 23993},
                {"index": 22623},
                {"index": 84537},
                {"index": 88974},
                {"index": 88949},
                {"index": 22808},
                {"index": 84569},
                {"index": 84937},
                {"index": 85558},
                {"index": 23319},
            ],
        },
        {
            "question": "What are the symptoms of cardioacrofacial dysplasia?",
            "expected_answer": "mandibular prognathia, postaxial foot polydactyly, clinodactyly of the 5th finger, clubbing, hypodontia, left superior vena cava draining to coronary sinus, common atrium, recurrent patellar dislocation, postaxial hand polydactyly, narrow chest, broad forehead, prominent nasal tip, deep philtrum, tented upper lip vermilion, conical tooth, diastema, autosomal dominant inheritance, short philtrum, genu valgum, long thorax, atrioventricular canal defect, hypoplasia of the maxilla, postaxial polydactyly, accessory oral frenulum, overhanging nasal tip, midface retrusion, nail dysplasia, long face, complete atrioventricular canal defect, congenital onset, limb undergrowth",
            "expected_cypher": 'MATCH (d:disease {name: "cardioacrofacial dysplasia"})-[:phenotype_present]->(e:effect_or_phenotype) RETURN e.name',
            "nodes": [
                {"index": 84579},
                {"index": 24072},
                {"index": 24065},
                {"index": 26961},
                {"index": 84683},
                {"index": 88879},
                {"index": 88810},
                {"index": 86071},
                {"index": 24042},
                {"index": 84713},
                {"index": 84589},
                {"index": 86195},
                {"index": 85031},
                {"index": 88415},
                {"index": 25325},
                {"index": 24152},
                {"index": 22759},
                {"index": 84586},
                {"index": 26936},
                {"index": 93559},
                {"index": 22838},
                {"index": 23993},
                {"index": 22623},
                {"index": 84537},
                {"index": 88974},
                {"index": 88949},
                {"index": 22808},
                {"index": 84569},
                {"index": 84937},
                {"index": 85558},
                {"index": 23319},
            ],
        },
        {
            "question": "What pathways are involved in distal arthrogryposis?",
            "expected_answer": "The pathways involved in distal arthrogryposis are lgi-adam interactions, mitochondrial translation termination, alkbh3 mediated reversal of alkylation damage, prevention of phagosomal-lysosomal fusion, copi-independent golgi-to-er retrograde traffic, striated muscle contraction, platelet degranulation, smooth muscle contraction, striated muscle contraction, striated muscle contraction, striated muscle contraction, snrnp assembly, clathrin-mediated endocytosis, cargo recognition for clathrin-mediated endocytosis, acetylcholine neurotransmitter release cycle, rhov gtpase cycle, rhob gtpase cycle, rhoa gtpase cycle, striated muscle contraction, signal transduction by l1, interaction between l1 and ankyrins, recycling pathway of l1, l1cam interactions, basigin interactions, rna polymerase i transcription initiation, gap-filling dna repair synthesis and ligation in tc-ner, dual incision in tc-ner, transcription-coupled nucleotide excision repair (tc-ner), formation of tc-ner pre-incision complex, b-wich complex positively regulates rrna expression, ercc6 (csb) and ehmt2 (g9a) positively regulate rrna expression, eml4 and nudc in mitotic spindle formation, aggrephagy, hcmv early events, aurka activation by tpx2, mitotic prometaphase, copi-independent golgi-to-er retrograde traffic, copi-mediated anterograde transport, neutrophil degranulation, rho gtpases activate formins, anchoring of the basal body to the plasma membrane, recruitment of numa to mitotic centrosomes, loss of proteins required for interphase microtubule organization from the centrosome, recruitment of mitotic centrosome proteins and complexes, loss of nlp from mitotic centrosomes, hsp90 chaperone cycle for steroid hormone receptors (shr), regulation of plk1 activity at g2_or_m transition, resolution of sister chromatid cohesion, separation of sister chromatids, mhc class ii antigen presentation, amplification  of signal from unattached  kinetochores via a mad2  inhibitory signal, highly sodium permeable postsynaptic acetylcholine nicotinic receptors, potential therapeutics for sars, ion transport by p-type atpases, ion homeostasis, meiotic synapsis, lgi-adam interactions, mitochondrial translation termination, alkbh3 mediated reversal of alkylation damage, prevention of phagosomal-lysosomal fusion, copi-independent golgi-to-er retrograde traffic, striated muscle contraction, platelet degranulation, smooth muscle contraction, striated muscle contraction, striated muscle contraction, striated muscle contraction, snrnp assembly, clathrin-mediated endocytosis, cargo recognition for clathrin-mediated endocytosis, acetylcholine neurotransmitter release cycle, rhov gtpase cycle, rhob gtpase cycle, rhoa gtpase cycle, striated muscle contraction, signal transduction by l1, interaction between l1 and ankyrins, recycling pathway of l1, l1cam interactions, basigin interactions, rna polymerase i transcription initiation, gap-filling dna repair synthesis and ligation in tc-ner, dual incision in tc-ner, transcription-coupled nucleotide excision repair (tc-ner), formation of tc-ner pre-incision complex, b-wich complex positively regulates rrna expression, ercc6 (csb) and ehmt2 (g9a) positively regulate rrna expression, eml4 and nudc in mitotic spindle formation, aggrephagy, hcmv early events, aurka activation by tpx2, mitotic prometaphase, copi-independent golgi-to-er retrograde traffic, copi-mediated anterograde transport, neutrophil degranulation, rho gtpases activate formins, anchoring of the basal body to the plasma membrane, recruitment of numa to mitotic centrosomes, loss of proteins required for interphase microtubule organization from the centrosome, recruitment of mitotic centrosome proteins and complexes, loss of nlp from mitotic centrosomes, hsp90 chaperone cycle for steroid hormone receptors (shr), regulation of plk1 activity at g2_or_m transition, resolution of sister chromatid cohesion, separation of sister chromatids, mhc class ii antigen presentation, amplification  of signal from unattached  kinetochores via a mad2  inhibitory signal, highly sodium permeable postsynaptic acetylcholine nicotinic receptors, potential therapeutics for sars, ion transport by p-type atpases, and ion homeostasis.",
            "expected_cypher": 'MATCH (d:disease {name: "arthrogryposis"})-[:associated_with]->(gp:gene_or_protein)-[:interacts_with]->(p:pathway) RETURN p.name',
            "nodes": [
                {"index": 127690},
                {"index": 128542},
                {"index": 128960},
                {"index": 129230},
                {"index": 129022},
                {"index": 128385},
                {"index": 128990},
                {"index": 128386},
                {"index": 128385},
                {"index": 128385},
                {"index": 128385},
                {"index": 128013},
                {"index": 62931},
                {"index": 129023},
                {"index": 127635},
                {"index": 129149},
                {"index": 129137},
                {"index": 129136},
                {"index": 128385},
                {"index": 128293},
                {"index": 128292},
                {"index": 128291},
                {"index": 62655},
                {"index": 128076},
                {"index": 128939},
                {"index": 128812},
                {"index": 128811},
                {"index": 62828},
                {"index": 128810},
                {"index": 128529},
                {"index": 128528},
                {"index": 128866},
                {"index": 129262},
                {"index": 129209},
                {"index": 128887},
                {"index": 62845},
                {"index": 129022},
                {"index": 128052},
                {"index": 127886},
                {"index": 128023},
                {"index": 128623},
                {"index": 128865},
                {"index": 62664},
                {"index": 128338},
                {"index": 128337},
                {"index": 128156},
                {"index": 128884},
                {"index": 128863},
                {"index": 128867},
                {"index": 127696},
                {"index": 127721},
                {"index": 128807},
                {"index": 129310},
                {"index": 129370},
                {"index": 128559},
                {"index": 127779},
                {"index": 127690},
                {"index": 128542},
                {"index": 128960},
                {"index": 129230},
                {"index": 129022},
                {"index": 128385},
                {"index": 128990},
                {"index": 128386},
                {"index": 128385},
                {"index": 128385},
                {"index": 128385},
                {"index": 128013},
                {"index": 62931},
                {"index": 129023},
                {"index": 127635},
                {"index": 129149},
                {"index": 129137},
                {"index": 129136},
                {"index": 128385},
                {"index": 128293},
                {"index": 128292},
                {"index": 128291},
                {"index": 62655},
                {"index": 128076},
                {"index": 128939},
                {"index": 128812},
                {"index": 128811},
                {"index": 62828},
                {"index": 128810},
                {"index": 128529},
                {"index": 128528},
                {"index": 128866},
                {"index": 129262},
                {"index": 129209},
                {"index": 128887},
                {"index": 62845},
                {"index": 129022},
                {"index": 128052},
                {"index": 127886},
                {"index": 128023},
                {"index": 128623},
                {"index": 128865},
                {"index": 62664},
                {"index": 128338},
                {"index": 128337},
                {"index": 128156},
                {"index": 128884},
                {"index": 128863},
                {"index": 128867},
                {"index": 127696},
                {"index": 127721},
                {"index": 128807},
                {"index": 129310},
                {"index": 129370},
                {"index": 128559},
            ],
        },
        {
            "question": "What pathways are involved in Breast cancer?",
            "expected_answer": "The following pathways are involved in breast cancer: runx1 regulates expression of components of tight junctions,apoptotic cleavage of cell adhesion  proteins,trna modification in the nucleus and cytosol,transport of bile salts and organic acids, metal ions and amine compounds,antigen processing: ubiquitination & proteasome degradation,trna modification in the nucleus and cytosol,formation of the editosome,mrna editing: c to u conversion,rac1 gtpase cycle,post-translational modification: synthesis of gpi-anchored proteins,platelet degranulation,regulation of runx2 expression and activity,transcriptional activation of mitochondrial biogenesis,ppara activates gene expression,pyruvate metabolism,trna modification in the nucleus and cytosol,trna modification in the nucleus and cytosol,amyloid fiber formation,defective pyroptosis,inhibition of dna recombination at telomere,transcriptional regulation of granulopoiesis,hcmv late events,hcmv early events,meiotic recombination,estrogen-dependent gene expression,runx1 regulates transcription of genes involved in differentiation of hscs,runx1 regulates genes involved in megakaryocyte differentiation and platelet function,e3 ubiquitin ligases ubiquitinate target proteins,rna polymerase i promoter escape,rna polymerase i promoter opening,g2_or_m dna damage checkpoint,deposition of new cenpa-containing nucleosomes at the centromere,processing of dna double-strand break ends,nonhomologous end-joining (nhej),recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,ub-specific processing proteases,activated pkn1 stimulates transcription of ar (androgen receptor) regulated genes klk2 and klk3,activation of anterior hox genes in hindbrain development during early embryogenesis,transcriptional regulation by small rnas,dna methylation,b-wich complex positively regulates rrna expression,norc negatively regulates rrna expression,ercc6 (csb) and ehmt2 (g9a) positively regulate rrna expression,sirt1 negatively regulates rrna expression,hats acetylate histones,hdacs deacetylate histones,dna damage_or_telomere stress induced senescence,senescence-associated secretory phenotype (sasp),oxidative stress induced senescence,condensation of prophase chromosomes,prc2 methylates histones and dna,formation of the beta-catenin:tcf transactivating complex,pre-notch transcription and translation,packaging of telomere ends,meiotic synapsis,cleavage of the damaged purine,recognition and association of dna glycosylase with site containing an affected purine,cleavage of the damaged pyrimidine,recognition and association of dna glycosylase with site containing an affected pyrimidine,runx1 regulates transcription of genes involved in wnt signaling,regulation of fzd by ubiquitination,g2_or_m dna damage checkpoint,regulation of tp53 activity through phosphorylation,presynaptic phase of homologous dna pairing and strand exchange,processing of dna double-strand break ends,homologous dna pairing and strand exchange,resolution of d-loop structures through holliday junction intermediates,resolution of d-loop structures through synthesis-dependent strand annealing (sdsa),hdr through homologous recombination (hrr),hdr through single strand annealing (ssa),cytosolic iron-sulfur cluster assembly,macroautophagy,defective slc2a10 causes arterial tortuosity syndrome (ats),cellular hexose transport,transcriptional regulation of white adipocyte differentiation,ppara activates gene expression,homologous dna pairing and strand exchange,resolution of d-loop structures through holliday junction intermediates,resolution of d-loop structures through synthesis-dependent strand annealing (sdsa),hdr through homologous recombination (hrr),intraflagellar transport,hedgehog 'off' state,reversal of alkylation damage by dna dioxygenases,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,the role of gtse1 in g2_or_m progression after g2 checkpoint,trna modification in the nucleus and cytosol,regulation of fzd by ubiquitination,runx1 interacts with co-factors whose precise effect on runx1 targets is not known,rmts methylate histone arginines,o-linked glycosylation of mucins,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,mitochondrial protein import,antigen processing: ubiquitination & proteasome degradation,negative regulation of notch4 signaling,neddylation,regulation of runx2 expression and activity,association of tric_or_cct with target proteins during biosynthesis,constitutive signaling by notch1 hd+pest domain mutants,loss of function of fbxw7 in cancer and notch1 signaling,constitutive signaling by notch1 pest domain mutants,notch1 intracellular domain regulates transcription,runx1 interacts with co-factors whose precise effect on runx1 targets is not known,rmts methylate histone arginines,hats acetylate histones,notch4 activation and transmission of signal to the nucleus,notch3 activation and transmission of signal to the nucleus,notch2 activation and transmission of signal to the nucleus,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 hd domain mutants,constitutive signaling by notch1 t(7;9)(notch1:m1580_k2555) translocation mutant,constitutive signaling by notch1 pest domain mutants,activated notch1 transmits signal to the nucleus,activation of the tfap2 (ap-2) family of transcription factors,negative regulation of activity of tfap2 (ap-2) family transcription factors,nuclear signaling by erbb4,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,rho gtpases activate formins,signaling by robo receptors,generation of second messenger molecules,transcriptional regulation of granulopoiesis,runx3 regulates wnt signaling,transcriptional regulation by ventx,repression of wnt target genes,binding of tcf_or_lef:ctnnb1 to target gene promoters,ca2+ pathway,deactivation of the beta-catenin transactivating complex,formation of the beta-catenin:tcf transactivating complex,assembly of active lpl and lipc lipase complexes,transcriptional regulation of white adipocyte differentiation,ppara activates gene expression,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,runx1 regulates transcription of genes involved in wnt signaling,runx1 and foxp3 control the development of regulatory t lymphocytes (tregs),clathrin-mediated endocytosis,cargo recognition for clathrin-mediated endocytosis,pkmts methylate histone lysines,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,mecp2 regulates transcription of neuronal ligands,notch3 activation and transmission of signal to the nucleus,notch2 activation and transmission of signal to the nucleus,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 hd domain mutants,constitutive signaling by notch1 t(7;9)(notch1:m1580_k2555) translocation mutant,constitutive signaling by notch1 pest domain mutants,activated notch1 transmits signal to the nucleus,antigen processing: ubiquitination & proteasome degradation,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),regulation of expression of slits and robos,translation initiation complex formation,auf1 (hnrnp d0) binds and destabilizes mrna,deadenylation of mrna,l13a-mediated translational silencing of ceruloplasmin expression,notch3 intracellular domain regulates transcription,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 pest domain mutants,notch1 intracellular domain regulates transcription,serine biosynthesis,zinc influx into cells by the slc39 gene family,nectin_or_necl  trans heterodimerization,adherens junctions interactions,notch4 intracellular domain regulates transcription,notch3 intracellular domain regulates transcription,runx2 regulates osteoblast differentiation,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 pest domain mutants,notch1 intracellular domain regulates transcription,notch4 intracellular domain regulates transcription,notch3 intracellular domain regulates transcription,runx2 regulates osteoblast differentiation,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 pest domain mutants,notch1 intracellular domain regulates transcription,heme signaling,regulation of foxo transcriptional activity by acetylation,sirt1 negatively regulates rrna expression,circadian clock,regulation of hsf1-mediated heat shock response,neutrophil degranulation,meiotic synapsis,hdms demethylate histones,sensory processing of sound by outer hair cells of the cochlea,sensory processing of sound by inner hair cells of the cochlea,neurexins and neuroligins,cooperation of pdcl (phlp1) and tric_or_cct in g-protein beta folding,bbsome-mediated cargo-targeting to cilium,association of tric_or_cct with target proteins during biosynthesis,folding of actin by cct_or_tric,formation of tubulin folding intermediates by cct_or_tric,prefoldin mediated transfer of substrate  to cct_or_tric,signaling by lrp5 mutants,negative regulation of tcf-dependent signaling by wnt ligand antagonists,tcf dependent signaling in response to wnt,chk1_or_chk2(cds1) mediated inactivation of cyclin b:cdk1 complex,ubiquitin mediated degradation of phosphorylated cdc25a,stabilization of p53,g2_or_m dna damage checkpoint,regulation of tp53 activity through methylation,regulation of tp53 degradation,regulation of tp53 activity through phosphorylation,recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,rnd2 gtpase cycle,rnd3 gtpase cycle,rac1 gtpase cycle,butyrophilin (btn) family interactions,antigen processing: ubiquitination & proteasome degradation,aberrant regulation of mitotic exit in cancer due to rb1 defects,synthesis of active ubiquitin: roles of e1 and e2 enzymes,transcriptional regulation by ventx,cdk-mediated phosphorylation and removal of cdc6,senescence-associated secretory phenotype (sasp),separation of sister chromatids,apc-cdc20 mediated degradation of nek2a,phosphorylation of the apc_or_c,apc_or_c:cdc20 mediated degradation of mitotic proteins,regulation of apc_or_c activators between g1_or_s and early anaphase,conversion from apc_or_c:cdc20 to apc_or_c:cdh1 in late anaphase,cdc20:phospho-apc_or_c mediated degradation of cyclin a,apc_or_c:cdh1 mediated degradation of cdc20 and other apc_or_c:cdh1 targeted proteins in late mitosis_or_early g1,apc_or_c:cdc20 mediated degradation of securin,autodegradation of cdh1 by cdh1:apc_or_c,apc_or_c:cdc20 mediated degradation of cyclin b,inactivation of apc_or_c via direct inhibition of the apc_or_c complex,o-linked glycosylation,defective large causes mddga6 and mddgb6,keratan sulfate biosynthesis,interleukin-20 family signaling,neutrophil degranulation,hs-gag degradation,transport of organic anions,defective slco1b1 causes hyperbilirubinemia, rotor type (hblrr),heme degradation,recycling of bile acids and salts,amino acids regulate mtorc1,regulation of pten gene transcription,tp53 regulates metabolic genes,energy dependent regulation of mtor by lkb1-ampk,mtorc1-mediated signalling,mtor signalling,macroautophagy,neddylation,heme signaling,cytoprotection by hmox1,estrogen-dependent gene expression,activated pkn1 stimulates transcription of ar (androgen receptor) regulated genes klk2 and klk3,circadian clock,regulation of lipid metabolism by pparalpha,sumoylation of transcription cofactors,transcriptional regulation of white adipocyte differentiation,hats acetylate histones,activation of gene expression by srebf (srebp),transcriptional activation of mitochondrial biogenesis,endogenous sterols,ppara activates gene expression,synthesis of bile acids and bile salts via 27-hydroxycholesterol,synthesis of bile acids and bile salts via 7alpha-hydroxycholesterol,synthesis of bile acids and bile salts,recycling of bile acids and salts,bmal1:clock,npas2 activates circadian gene expression,rora activates gene expression,antagonism of activin by follistatin,ptk6 promotes hif1a stabilization,egr2 and sox10-mediated initiation of schwann cell myelination,runx3 regulates yap1-mediated transcription,runx2 regulates osteoblast differentiation,runx1 regulates transcription of genes involved in differentiation of hscs,yap1- and wwtr1 (taz)-stimulated gene expression,signaling by hippo,nuclear signaling by erbb4,tp53 regulates transcription of several additional cell death genes whose specific roles in p53-dependent apoptosis remain uncertain,kinesins,sealing of the nuclear envelope (ne) by escrt-iii,eml4 and nudc in mitotic spindle formation,aggrephagy,activation of ampk downstream of nmdars,assembly and cell surface presentation of nmda receptors,hcmv early events,carboxyterminal post-translational modifications of tubulin,the role of gtse1 in g2_or_m progression after g2 checkpoint,mitotic prometaphase,copi-independent golgi-to-er retrograde traffic,copi-dependent golgi-to-er retrograde traffic,copi-mediated anterograde transport,rho gtpases activate formins,rho gtpases activate iqgaps,intraflagellar transport,cilium assembly,hedgehog 'off' state,recycling pathway of l1,post-chaperonin tubulin folding pathway,formation of tubulin folding intermediates by cct_or_tric,prefoldin mediated transfer of substrate  to cct_or_tric,recruitment of numa to mitotic centrosomes,hsp90 chaperone cycle for steroid hormone receptors (shr),resolution of sister chromatid cohesion,separation of sister chromatids,mhc class ii antigen presentation,gap junction assembly,microtubule-dependent trafficking of connexons from golgi to the plasma membrane,translocation of slc2a4 (glut4) to the plasma membrane,ovarian tumor domain proteases,processing of capped intron-containing pre-mrna,mrna splicing - major pathway,activation of irf3_or_irf7 mediated by tbk1_or_ikk epsilon,traf6 mediated irf7 activation,ticam1-dependent activation of irf3_or_irf7,transcriptional regulation of white adipocyte differentiation,generic transcription pathway,ppara activates gene expression,rna polymerase ii transcription termination,mrna 3'-end processing,transport of mature mrna derived from an intron-containing transcript,macroautophagy,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,cdc42 gtpase cycle,rhob gtpase cycle,rhoa gtpase cycle,generic transcription pathway,heme signaling,cytoprotection by hmox1,nr1h2 & nr1h3 regulate gene expression to control bile acid homeostasis,hcmv early events,nr1h3 & nr1h2 regulate gene expression linked to cholesterol transport and efflux,regulation of mecp2 expression and activity,loss of mecp2 binding ability to the ncor_or_smrt complex,activation of anterior hox genes in hindbrain development during early embryogenesis,circadian clock,regulation of lipid metabolism by pparalpha,nuclear receptor transcription pathway,transcriptional regulation of white adipocyte differentiation,notch-hlh transcription pathway,hdacs deacetylate histones,constitutive signaling by notch1 hd+pest domain mutants,constitutive signaling by notch1 pest domain mutants,downregulation of smad2_or_3:smad4 transcriptional activity,transcriptional activation of mitochondrial biogenesis,notch1 intracellular domain regulates transcription,ppara activates gene expression,nr1d1 (rev-erba) represses gene expression,nuclear signaling by erbb4,rhof gtpase cycle,rhod gtpase cycle,formation of the editosome,mrna editing: c to u conversion,o-glycosylation of tsr domain-containing proteins,defective b3galtl causes peters-plus syndrome (pps),degradation of the extracellular matrix,iron uptake and transport,abacavir transmembrane transport,heme degradation,heme biosynthesis,clathrin-mediated endocytosis,cargo recognition for clathrin-mediated endocytosis,g2_or_m dna damage checkpoint,regulation of tp53 activity through phosphorylation,presynaptic phase of homologous dna pairing and strand exchange,processing of dna double-strand break ends,homologous dna pairing and strand exchange,resolution of d-loop structures through holliday junction intermediates,resolution of d-loop structures through synthesis-dependent strand annealing (sdsa),hdr through homologous recombination (hrr),hdr through single strand annealing (ssa),mismatch repair (mmr) directed by msh2:msh3 (mutsbeta),mismatch repair (mmr) directed by msh2:msh6 (mutsalpha),transport of nucleosides and free purine and pyrimidine bases across the plasma membrane,pyruvate metabolism,proton-coupled monocarboxylate transport,basigin interactions,tight junction interactions,rho gtpases activate cit,ret signaling,raf_or_map kinase cascade,ncam1 interactions,antigen processing: ubiquitination & proteasome degradation,g2_or_m dna damage checkpoint,processing of dna double-strand break ends,nonhomologous end-joining (nhej),recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,sumoylation of dna damage response and repair proteins,response of eif2ak1 (hri) to heme deficiency,response of eif2ak4 (gcn2) to amino acid deficiency,recycling of eif2:gdp,gtp hydrolysis and joining of the 60s ribosomal subunit,ribosomal scanning and start codon recognition,formation of the ternary complex, and subsequently, the 43s complex,translation initiation complex formation,abc-family proteins mediated transport,perk regulates gene expression,l13a-mediated translational silencing of ceruloplasmin expression,clathrin-mediated endocytosis,synthesis of pips at the plasma membrane,synthesis of ip2, ip, and ins in the cytosol,synthesis of pips at the early endosome membrane,synthesis of pips at the plasma membrane,trail  signaling,dimerization of procaspase-8,casp8 activity is inhibited,ripk1-mediated regulated necrosis,regulation by c-flip,caspase activation via death receptors in the presence of ligand,heme signaling,cytoprotection by hmox1,nr1h2 & nr1h3 regulate gene expression to control bile acid homeostasis,nr1h3 & nr1h2 regulate gene expression linked to cholesterol transport and efflux,estrogen-dependent gene expression,circadian clock,regulation of lipid metabolism by pparalpha,sumoylation of transcription cofactors,transcriptional regulation of white adipocyte differentiation,hats acetylate histones,activation of gene expression by srebf (srebp),transcriptional activation of mitochondrial biogenesis,endogenous sterols,ppara activates gene expression,synthesis of bile acids and bile salts via 27-hydroxycholesterol,synthesis of bile acids and bile salts via 7alpha-hydroxycholesterol,synthesis of bile acids and bile salts,recycling of bile acids and salts,bmal1:clock,npas2 activates circadian gene expression,rora activates gene expression,post-translational protein phosphorylation,regulation of insulin-like growth factor (igf) transport and uptake by insulin-like growth factor binding proteins (igfbps),hats acetylate histones,transcriptional regulation by runx2,amyloid fiber formation,defective pyroptosis,inhibition of dna recombination at telomere,transcriptional regulation of granulopoiesis,hcmv late events,hcmv early events,meiotic recombination,estrogen-dependent gene expression,runx1 regulates transcription of genes involved in differentiation of hscs,runx1 regulates genes involved in megakaryocyte differentiation and platelet function,e3 ubiquitin ligases ubiquitinate target proteins,rna polymerase i promoter escape,rna polymerase i promoter opening,g2_or_m dna damage checkpoint,deposition of new cenpa-containing nucleosomes at the centromere,processing of dna double-strand break ends,nonhomologous end-joining (nhej),recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,ub-specific processing proteases,activated pkn1 stimulates transcription of ar (androgen receptor) regulated genes klk2 and klk3,activation of anterior hox genes in hindbrain development during early embryogenesis,transcriptional regulation by small rnas,dna methylation,b-wich complex positively regulates rrna expression,norc negatively regulates rrna expression,ercc6 (csb) and ehmt2 (g9a) positively regulate rrna expression,sirt1 negatively regulates rrna expression,hats acetylate histones,hdacs deacetylate histones,dna damage_or_telomere stress induced senescence,senescence-associated secretory phenotype (sasp),oxidative stress induced senescence,condensation of prophase chromosomes,prc2 methylates histones and dna,formation of the beta-catenin:tcf transactivating complex,pre-notch transcription and translation,packaging of telomere ends,meiotic synapsis,cleavage of the damaged purine,recognition and association of dna glycosylase with site containing an affected purine,cleavage of the damaged pyrimidine,recognition and association of dna glycosylase with site containing an affected pyrimidine,recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,uch proteinases,runx1 interacts with co-factors whose precise effect on runx1 targets is not known,rmts methylate histone arginines,heme signaling,nr1h2 & nr1h3 regulate gene expression linked to gluconeogenesis,nr1h2 & nr1h3 regulate gene expression linked to lipogenesis,estrogen-dependent gene expression,circadian clock,sumoylation of transcription cofactors,estrogen-dependent gene expression,mapk6_or_mapk4 signaling,activation of anterior hox genes in hindbrain development during early embryogenesis,transcriptional regulation of white adipocyte differentiation,ppara activates gene expression,runx1 regulates genes involved in megakaryocyte differentiation and platelet function,activation of anterior hox genes in hindbrain development during early embryogenesis,deactivation of the beta-catenin transactivating complex,pkmts methylate histone lysines,formation of the beta-catenin:tcf transactivating complex,antigen processing: ubiquitination & proteasome degradation,inactivation of csf3 (g-csf) signaling,neddylation,downregulation of erbb2 signaling,vif-mediated degradation of apobec3g,transcriptional regulation of granulopoiesis,transcriptional regulation by the ap-2 (tfap2) family of transcription factors,b-wich complex positively regulates rrna expression,g alpha (i) signalling events,chemokine receptors bind chemokines,signaling by robo receptors,binding and entry of hiv virion,mitochondrial translation termination,mitochondrial translation elongation,mitochondrial translation initiation,homologous dna pairing and strand exchange,resolution of d-loop structures through holliday junction intermediates,resolution of d-loop structures through synthesis-dependent strand annealing (sdsa),hdr through homologous recombination (hrr),presynaptic phase of homologous dna pairing and strand exchange,homologous dna pairing and strand exchange,resolution of d-loop structures through holliday junction intermediates,resolution of d-loop structures through synthesis-dependent strand annealing (sdsa),hdr through homologous recombination (hrr),transcriptional regulation of testis differentiation,transcriptional regulation of white adipocyte differentiation,class b_or_2 (secretin family receptors),wnt ligand biogenesis and trafficking,g2_or_m dna damage checkpoint,processing of dna double-strand break ends,nonhomologous end-joining (nhej),recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,pkmts methylate histone lysines,aggrephagy,late endosomal microautophagy,chaperone mediated autophagy,rhobtb1 gtpase cycle,interleukin-4 and interleukin-13 signaling,striated muscle contraction,caspase-mediated cleavage of cytoskeletal proteins,vegf binds to vegfr leading to receptor dimerization,vegf ligand-receptor interactions,platelet degranulation,vegf binds to vegfr leading to receptor dimerization,vegf ligand-receptor interactions,platelet degranulation,sumoylation of intracellular receptors,nuclear receptor transcription pathway,vitamin d (calciferol) metabolism,pyrimidine biosynthesis,g1_or_s-specific transcription,interconversion of nucleotide di- and triphosphates,purinergic signaling in leishmaniasis infection,regulation of foxo transcriptional activity by acetylation,the nlrp3 inflammasome,protein repair,tp53 regulates metabolic genes,interconversion of nucleotide di- and triphosphates,detoxification of reactive oxygen species,oxidative stress induced senescence,runx1 regulates transcription of genes involved in differentiation of hscs,regulation of tp53 activity through association with co-factors,tp53 regulates transcription of death receptors and ligands,tp53 regulates transcription of caspase activators and caspases,tp53 regulates transcription of several additional cell death genes whose specific roles in p53-dependent apoptosis remain uncertain,tp53 regulates transcription of genes involved in cytochrome c release,activation of puma and translocation to mitochondria,regulation of tp53 activity through association with co-factors,tp53 regulates transcription of death receptors and ligands,tp53 regulates transcription of several additional cell death genes whose specific roles in p53-dependent apoptosis remain uncertain,tp53 regulates transcription of genes involved in cytochrome c release,activation of puma and translocation to mitochondria,g2_or_m dna damage checkpoint,processing of dna double-strand break ends,nonhomologous end-joining (nhej),recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,sumoylation of transcription factors,factors involved in megakaryocyte development and platelet production,regulation of pten gene transcription,runx3 regulates cdkn1a transcription,transcriptional regulation by ventx,the role of gtse1 in g2_or_m progression after g2 checkpoint,transcriptional  activation of  cell cycle inhibitor p21,stabilization of p53,g2_or_m checkpoints,g2_or_m dna damage checkpoint,pi5p regulates tp53 acetylation,regulation of tp53 activity through methylation,regulation of tp53 activity through association with co-factors,regulation of tp53 activity through acetylation,regulation of tp53 degradation,regulation of tp53 activity through phosphorylation,regulation of tp53 expression,tp53 regulates transcription of genes involved in g1 cell cycle arrest,tp53 regulates transcription of additional cell cycle genes whose exact role in the p53 pathway remain uncertain,tp53 regulates transcription of genes involved in g2 cell cycle arrest,tp53 regulates transcription of death receptors and ligands,tp53 regulates transcription of caspase activators and caspases,tp53 regulates transcription of several additional cell death genes whose specific roles in p53-dependent apoptosis remain uncertain,tp53 regulates transcription of genes involved in cytochrome c release,tp53 regulates transcription of dna repair genes,interleukin-4 and interleukin-13 signaling,recruitment and atm-mediated phosphorylation of repair and signaling proteins at dna double strand breaks,ovarian tumor domain proteases,ub-specific processing proteases,tp53 regulates metabolic genes,pyroptosis,association of tric_or_cct with target proteins during biosynthesis,autodegradation of the e3 ubiquitin ligase cop1,sumoylation of transcription factors,dna damage_or_telomere stress induced senescence,oncogene induced senescence,formation of senescence-associated heterochromatin foci (sahf),oxidative stress induced senescence,pre-notch transcription and translation,activation of puma and translocation to mitochondria,activation of noxa and translocation to mitochondria,sumoylation of dna replication proteins,transcription of e2f targets under negative control by dream complex,tnf signaling,interleukin-4 and interleukin-13 signaling,interleukin-10 signaling,tnfr2 non-canonical nf-kb pathway,tnfr1-mediated ceramide production,tnfr1-induced nfkappab signaling pathway,regulation of tnfr1 signaling,tnfr1-induced proapoptotic signaling,transcriptional regulation of white adipocyte differentiation,estrogen-dependent gene expression,repression of wnt target genes,deactivation of the beta-catenin transactivating complex,notch1 intracellular domain regulates transcription,formation of the beta-catenin:tcf transactivating complex,integrin cell surface interactions,immunoregulatory interactions between a lymphoid and a non-lymphoid cell,runx1 regulates genes involved in megakaryocyte differentiation and platelet function,o-glycosylation of tsr domain-containing proteins,defective b3galtl causes peters-plus syndrome (pps),syndecan interactions,integrin cell surface interactions,signaling by pdgf,platelet degranulation,rnd1 gtpase cycle,rnd2 gtpase cycle,transferrin endocytosis and recycling,rac3 gtpase cycle,rhoj gtpase cycle,rhog gtpase cycle,rhoh gtpase cycle,rhoq gtpase cycle,rac2 gtpase cycle,rac1 gtpase cycle,cdc42 gtpase cycle,rhoc gtpase cycle,rhob gtpase cycle,rhoa gtpase cycle,clathrin-mediated endocytosis,cargo recognition for clathrin-mediated endocytosis,golgi associated vesicle biogenesis,nuclear receptor transcription pathway,tfap2a acts as a transcriptional repressor during retinoic acid induced cell differentiation,tfap2 (ap-2) family regulates transcription of cell cycle factors,tfap2 (ap-2) family regulates transcription of growth factors and their receptors,activation of the tfap2 (ap-2) family of transcription factors,tfap2 (ap-2) family regulates transcription of other transcription factors,negative regulation of activity of tfap2 (ap-2) family transcription factors,transcriptional regulation by the ap-2 (tfap2) family of transcription factors,sumoylation of transcription factors,formation of the beta-catenin:tcf transactivating complex,telomere extension by telomerase,interleukin-4 and interleukin-13 signaling,cytosolic sulfonation of small molecules,interaction between phlda1 and aurka,aurka activation by tpx2,fbxl7 down-regulates aurka during mitotic entry and in early mitosis,regulation of tp53 activity through phosphorylation,tp53 regulates transcription of genes involved in g2 cell cycle arrest,sumoylation of dna replication proteins,regulation of plk1 activity at g2_or_m transition,apc_or_c:cdh1 mediated degradation of cdc20 and other apc_or_c:cdh1 targeted proteins in late mitosis_or_early g1,growth hormone receptor signaling,inactivation of csf3 (g-csf) signaling,signaling by flt3 fusion proteins,stat5 activation downstream of flt3 itd mutants,signaling by csf3 (g-csf),signaling by phosphorylated juxtamembrane, extracellular and kinase domain kit mutants,stat5 activation,erythropoietin activates stat5,interleukin-21 signaling,interleukin-2 signaling,interleukin-9 signaling,interleukin-15 signaling,interleukin-20 family signaling,interleukin-3, interleukin-5 and gm-csf signaling,signaling by leptin,downstream signal transduction,signaling by cytosolic fgfr1 fusion mutants,signaling by scf-kit,interleukin-7 signaling,nuclear signaling by erbb4,prolactin receptor signaling,growth hormone receptor signaling,cytoprotection by hmox1,inactivation of csf3 (g-csf) signaling,signaling by csf3 (g-csf),signaling by pdgfra extracellular domain mutants,signaling by pdgfra transmembrane, juxtamembrane and kinase domain mutants,signaling by phosphorylated juxtamembrane, extracellular and kinase domain kit mutants,transcriptional regulation of granulopoiesis,interleukin-21 signaling,interleukin-27 signaling,interleukin-23 signaling,interleukin-37 signaling,interleukin-9 signaling,interleukin-35 signalling,interleukin-15 signaling,met activates stat3,interleukin-20 family signaling,ptk6 activates stat3,interleukin-4 and interleukin-13 signaling,interleukin-10 signaling,transcriptional regulation of pluripotent stem cells,association of tric_or_cct with target proteins during biosynthesis,pou5f1 (oct4), sox2, nanog activate genes related to proliferation,signaling by leptin,senescence-associated secretory phenotype (sasp),signalling to stat3,downstream signal transduction,signaling by cytosolic fgfr1 fusion mutants,signaling by scf-kit,interleukin-7 signaling,bh3-only proteins associate with and inactivate anti-apoptotic bcl-2 members,interleukin-6 signaling,egr2 and sox10-mediated initiation of schwann cell myelination,transcriptional regulation of white adipocyte differentiation,activation of gene expression by srebf (srebp),ppara activates gene expression,regulation of cholesterol biosynthesis by srebp (srebf),signaling by phosphorylated juxtamembrane, extracellular and kinase domain kit mutants,fcgr3a-mediated phagocytosis,fcgr3a-mediated il10 synthesis,signaling by raf1 mutants,signaling downstream of ras mutants,long-term potentiation,activated ntrk3 signals through pi3k,activated ntrk2 signals through fyn,extra-nuclear estrogen signaling,regulation of runx3 expression and activity,runx2 regulates osteoblast differentiation,receptor mediated mitophagy,regulation of runx1 expression and activity,inla-mediated entry of listeria monocytogenes into host cells,met activates ptk2 signaling,ret signaling,cyclin d associated events in g1,pi5p, pp2a and ier3 regulate pi3k_or_akt signaling,paradoxical activation of raf signaling by kinase inactive braf,signaling by braf and raf fusions,signaling by high-kinase activity braf mutants,signaling by moderate kinase activity braf mutants,map2k and mapk activation,raf activation,rho gtpases activate formins,clec7a (dectin-1) signaling,vegfr2 mediated cell proliferation,thrombin signalling through proteinase activated receptors (pars),vegfa-vegfr2 pathway,recycling pathway of l1,gp1b-ix-v activation signalling,regulation of commissural axon pathfinding by slit and robo,netrin mediated repulsion signals,dcc mediated attractive signaling,g alpha (i) signalling events,adp signalling through p2y purinoceptor 1,g alpha (s) signalling events,eph-ephrin mediated repulsion of cells,ephrin signaling,epha-mediated growth cone collapse,ephb-mediated forward signaling,signal regulatory protein family interactions,ctla4 inhibitory signaling,cd28 co-stimulation,ncam signaling for neurite out-growth,p130cas linkage to mapk signaling for integrins,grb2:sos provides linkage to mapk signaling for integrins,integrin signaling,eph-ephrin signaling,constitutive signaling by aberrant pi3k in cancer,pecam1 interactions,fcgr activation,regulation of gap junction activity,downstream signal transduction,gab1 signalosome,signaling by egfr,p38mapk events,regulation of kit signaling,signaling by scf-kit,spry regulation of fgf signaling,pip3 activates akt signaling,downregulation of erbb4 signaling,nuclear signaling by erbb4,signaling by erbb2,post-translational protein phosphorylation,runx3 regulates immune response and cell migration,regulation of insulin-like growth factor (igf) transport and uptake by insulin-like growth factor binding proteins (igfbps),integrin cell surface interactions,signaling by pdgf,degradation of the extracellular matrix,foxo-mediated transcription of oxidative stress, metabolic and neuronal genes,gene and protein expression by jak-stat signaling after interleukin-12 stimulation,deregulated cdk5 triggers multiple neurodegenerative pathways in alzheimer's disease models,detoxification of reactive oxygen species,transcriptional activation of mitochondrial biogenesis,regulation of pten gene transcription,runx1 interacts with co-factors whose precise effect on runx1 targets is not known,rmts methylate histone arginines,regulation of pten gene transcription,defective slc5a5 causes thyroid dyshormonogenesis 1 (tdh1),organic anion transporters,thyroxine biosynthesis,intestinal hexose absorption,neutrophil degranulation,intestinal hexose absorption,defective slc2a2 causes fanconi-bickel syndrome (fbs),regulation of insulin secretion,regulation of gene expression in beta cells,cellular hexose transport,lactose synthesis,defective slc2a1 causes glut1 deficiency syndrome 1 (glut1ds1),regulation of insulin secretion,vitamin c (ascorbate) metabolism,cellular hexose transport,carnitine synthesis,metabolism of folate and pterines,negative regulation of tcf-dependent signaling by wnt ligand antagonists,tcf dependent signaling in response to wnt,negative regulation of tcf-dependent signaling by wnt ligand antagonists,tcf dependent signaling in response to wnt,map3k8 (tpl2)-dependent mapk1_or_3 activation,uptake and function of anthrax toxins,jnk (c-jun kinases) phosphorylation and  activation mediated by activated human tak1,fceri mediated mapk activation,oxidative stress induced senescence,estrogen-dependent gene expression,g alpha (i) signalling events,chemokine receptors bind chemokines,signaling by robo receptors,nuclear signaling by erbb4,interleukin-10 signaling,g alpha (i) signalling events,chemokine receptors bind chemokines,nr1h2 & nr1h3 regulate gene expression linked to gluconeogenesis,nr1h2 & nr1h3 regulate gene expression to control bile acid homeostasis,nr1h2 & nr1h3 regulate gene expression linked to triglyceride lipolysis in adipose,nr1h2 & nr1h3 regulate gene expression to limit cholesterol uptake,nr1h3 & nr1h2 regulate gene expression linked to cholesterol transport and efflux,nr1h2 & nr1h3 regulate gene expression linked to lipogenesis,signaling by retinoic acid,nuclear receptor transcription pathway,ppara activates gene expression,ngf-stimulated transcription,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,ribosomal scanning and start codon recognition,formation of the ternary complex, and subsequently, the 43s complex,formation of a pool of free 40s subunits,translation initiation complex formation,major pathway of rrna processing in the nucleolus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,ribosomal scanning and start codon recognition,formation of the ternary complex, and subsequently, the 43s complex,formation of a pool of free 40s subunits,translation initiation complex formation,major pathway of rrna processing in the nucleolus and cytosol,rrna modification in the nucleus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,constitutive signaling by akt1 e17k in cancer,akt phosphorylates targets in the nucleus,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,ribosomal scanning and start codon recognition,formation of the ternary complex, and subsequently, the 43s complex,formation of a pool of free 40s subunits,translation initiation complex formation,major pathway of rrna processing in the nucleolus and cytosol,rrna modification in the nucleus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,mtorc1-mediated signalling,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,ribosomal scanning and start codon recognition,formation of the ternary complex, and subsequently, the 43s complex,formation of a pool of free 40s subunits,translation initiation complex formation,major pathway of rrna processing in the nucleolus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,formation of a pool of free 40s subunits,major pathway of rrna processing in the nucleolus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,formation of a pool of free 40s subunits,major pathway of rrna processing in the nucleolus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,nonsense mediated decay (nmd) enhanced by the exon junction complex (ejc),nonsense mediated decay (nmd) independent of the exon junction complex (ejc),response of eif2ak4 (gcn2) to amino acid deficiency,regulation of expression of slits and robos,eukaryotic translation termination,gtp hydrolysis and joining of the 60s ribosomal subunit,formation of a pool of free 40s subunits,major pathway of rrna processing in the nucleolus and cytosol,selenocysteine synthesis,viral mrna translation,srp-dependent cotranslational protein targeting to membrane,peptide chain elongation,l13a-mediated translational silencing of ceruloplasmin expression,g alpha (q) signalling events,purinergic signaling in leishmaniasis infection,traf6 mediated nf-kb activation,interleukin-1 signaling,transcriptional regulation by ventx,the nlrp3 inflammasome,clec7a_or_inflammasome pathway,cd209 (dc-sign) signaling,clec7a (dectin-1) signaling,dectin-1 mediated noncanonical nf-kb signaling,ikba variant leads to eda-id,sumoylation of immune response proteins,interleukin-1 processing,tak1 activates nfkb by phosphorylation and activation of ikks complex,transcriptional regulation of white adipocyte differentiation,pkmts methylate histone lysines,dex_or_h-box helicases activate type i ifn and inflammatory cytokines production,fceri mediated nf-kb activation,senescence-associated secretory phenotype (sasp),nf-kb is activated and signals survival,downstream tcr signaling,regulated proteolysis of p75ntr,rip-mediated nfkb activation via zbp1,activation of nf-kappab in b cells,replication of the sars-cov-2 genome,aberrant regulation of mitotic exit in cancer due to rb1 defects,replication of the sars-cov-1 genome,defective translocation of rb1 mutants to the nucleus,defective binding of rb1 mutants to e2f1,(e2f2, e2f3),runx2 regulates osteoblast differentiation.",
            "expected_cypher": 'MATCH (d:disease {name: "breast carcinoma"})-[:associated_with]->(gp:gene_or_protein)-[:interacts_with]->(p:pathway) RETURN p.name',
            "nodes": [
                {"index": 129055},
                {"index": 127617},
                {"index": 62829},
                {"index": 62707},
                {"index": 129365},
                {"index": 62829},
                {"index": 128923},
                {"index": 62883},
                {"index": 129140},
                {"index": 62445},
                {"index": 128990},
                {"index": 129049},
                {"index": 127797},
                {"index": 128393},
                {"index": 62876},
                {"index": 62829},
                {"index": 62829},
                {"index": 128381},
                {"index": 129238},
                {"index": 127794},
                {"index": 127691},
                {"index": 129210},
                {"index": 129209},
                {"index": 127780},
                {"index": 129069},
                {"index": 129057},
                {"index": 129056},
                {"index": 129020},
                {"index": 128940},
                {"index": 128938},
                {"index": 62865},
                {"index": 129006},
                {"index": 128789},
                {"index": 128784},
                {"index": 128792},
                {"index": 128781},
                {"index": 128716},
                {"index": 128709},
                {"index": 128096},
                {"index": 128113},
                {"index": 128529},
                {"index": 128531},
                {"index": 128528},
                {"index": 128530},
                {"index": 128226},
                {"index": 128223},
                {"index": 62615},
                {"index": 128176},
                {"index": 128175},
                {"index": 128861},
                {"index": 128111},
                {"index": 62571},
                {"index": 127995},
                {"index": 127793},
                {"index": 127779},
                {"index": 128952},
                {"index": 128951},
                {"index": 128954},
                {"index": 128953},
                {"index": 129063},
                {"index": 128063},
                {"index": 62865},
                {"index": 128720},
                {"index": 128790},
                {"index": 128789},
                {"index": 62819},
                {"index": 128786},
                {"index": 128785},
                {"index": 62812},
                {"index": 128788},
                {"index": 127723},
                {"index": 62448},
                {"index": 128666},
                {"index": 128438},
                {"index": 127688},
                {"index": 128393},
                {"index": 62819},
                {"index": 128786},
                {"index": 128785},
                {"index": 62812},
                {"index": 128624},
                {"index": 62770},
                {"index": 62905},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 128885},
                {"index": 62829},
                {"index": 128063},
                {"index": 129059},
                {"index": 128227},
                {"index": 62998},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 129206},
                {"index": 129365},
                {"index": 129157},
                {"index": 128799},
                {"index": 129049},
                {"index": 128360},
                {"index": 128188},
                {"index": 128180},
                {"index": 128179},
                {"index": 128041},
                {"index": 129059},
                {"index": 128227},
                {"index": 128226},
                {"index": 129156},
                {"index": 62980},
                {"index": 128044},
                {"index": 128188},
                {"index": 128187},
                {"index": 128181},
                {"index": 128179},
                {"index": 128042},
                {"index": 129028},
                {"index": 129026},
                {"index": 127680},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 128023},
                {"index": 62659},
                {"index": 128074},
                {"index": 127691},
                {"index": 129044},
                {"index": 128117},
                {"index": 128015},
                {"index": 128065},
                {"index": 128348},
                {"index": 128058},
                {"index": 62571},
                {"index": 129090},
                {"index": 127688},
                {"index": 128393},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 129063},
                {"index": 129052},
                {"index": 62931},
                {"index": 129023},
                {"index": 128224},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 129109},
                {"index": 62980},
                {"index": 128044},
                {"index": 128188},
                {"index": 128187},
                {"index": 128181},
                {"index": 128179},
                {"index": 128042},
                {"index": 129365},
                {"index": 129194},
                {"index": 129193},
                {"index": 128315},
                {"index": 128930},
                {"index": 128490},
                {"index": 128442},
                {"index": 128929},
                {"index": 129135},
                {"index": 128188},
                {"index": 128179},
                {"index": 128041},
                {"index": 128913},
                {"index": 128446},
                {"index": 128412},
                {"index": 62703},
                {"index": 129155},
                {"index": 129135},
                {"index": 129070},
                {"index": 128188},
                {"index": 128179},
                {"index": 128041},
                {"index": 129155},
                {"index": 129135},
                {"index": 129070},
                {"index": 128188},
                {"index": 128179},
                {"index": 128041},
                {"index": 128158},
                {"index": 129216},
                {"index": 128530},
                {"index": 62692},
                {"index": 128247},
                {"index": 127886},
                {"index": 127779},
                {"index": 128225},
                {"index": 63024},
                {"index": 129246},
                {"index": 128824},
                {"index": 128361},
                {"index": 128711},
                {"index": 128360},
                {"index": 128359},
                {"index": 128358},
                {"index": 128357},
                {"index": 128507},
                {"index": 128059},
                {"index": 62570},
                {"index": 128889},
                {"index": 128894},
                {"index": 62867},
                {"index": 62865},
                {"index": 128722},
                {"index": 128842},
                {"index": 128720},
                {"index": 128792},
                {"index": 129152},
                {"index": 129151},
                {"index": 129140},
                {"index": 127698},
                {"index": 129365},
                {"index": 129321},
                {"index": 129019},
                {"index": 128117},
                {"index": 128871},
                {"index": 128176},
                {"index": 128867},
                {"index": 127954},
                {"index": 127943},
                {"index": 62514},
                {"index": 62512},
                {"index": 127931},
                {"index": 127953},
                {"index": 127930},
                {"index": 127942},
                {"index": 127929},
                {"index": 127941},
                {"index": 127720},
                {"index": 62745},
                {"index": 128363},
                {"index": 127839},
                {"index": 128484},
                {"index": 127886},
                {"index": 127843},
                {"index": 128437},
                {"index": 128693},
                {"index": 127981},
                {"index": 128010},
                {"index": 129354},
                {"index": 128852},
                {"index": 128274},
                {"index": 127855},
                {"index": 127854},
                {"index": 62460},
                {"index": 62448},
                {"index": 128799},
                {"index": 128158},
                {"index": 63064},
                {"index": 129069},
                {"index": 128716},
                {"index": 62692},
                {"index": 62691},
                {"index": 128202},
                {"index": 127688},
                {"index": 128226},
                {"index": 127856},
                {"index": 127797},
                {"index": 62588},
                {"index": 128393},
                {"index": 128001},
                {"index": 127999},
                {"index": 62543},
                {"index": 128010},
                {"index": 128396},
                {"index": 128395},
                {"index": 127783},
                {"index": 129016},
                {"index": 129303},
                {"index": 129045},
                {"index": 129070},
                {"index": 129057},
                {"index": 128114},
                {"index": 127814},
                {"index": 127680},
                {"index": 128724},
                {"index": 129367},
                {"index": 128197},
                {"index": 128866},
                {"index": 129262},
                {"index": 128450},
                {"index": 128457},
                {"index": 129209},
                {"index": 128800},
                {"index": 128885},
                {"index": 62845},
                {"index": 129022},
                {"index": 129021},
                {"index": 128052},
                {"index": 128023},
                {"index": 128018},
                {"index": 128624},
                {"index": 62771},
                {"index": 62770},
                {"index": 128291},
                {"index": 128374},
                {"index": 128358},
                {"index": 128357},
                {"index": 128865},
                {"index": 128156},
                {"index": 128863},
                {"index": 128867},
                {"index": 127696},
                {"index": 62538},
                {"index": 127992},
                {"index": 128053},
                {"index": 128782},
                {"index": 62885},
                {"index": 128921},
                {"index": 129196},
                {"index": 127920},
                {"index": 127884},
                {"index": 127688},
                {"index": 62504},
                {"index": 128393},
                {"index": 128944},
                {"index": 128925},
                {"index": 128924},
                {"index": 62448},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 129139},
                {"index": 129137},
                {"index": 129136},
                {"index": 62504},
                {"index": 128158},
                {"index": 63064},
                {"index": 129172},
                {"index": 129209},
                {"index": 129169},
                {"index": 129107},
                {"index": 129114},
                {"index": 128709},
                {"index": 62692},
                {"index": 62691},
                {"index": 128116},
                {"index": 127688},
                {"index": 128115},
                {"index": 128223},
                {"index": 128188},
                {"index": 128179},
                {"index": 128138},
                {"index": 127797},
                {"index": 128041},
                {"index": 128393},
                {"index": 128394},
                {"index": 127680},
                {"index": 129150},
                {"index": 129142},
                {"index": 128923},
                {"index": 62883},
                {"index": 128523},
                {"index": 128369},
                {"index": 62404},
                {"index": 62999},
                {"index": 128129},
                {"index": 127981},
                {"index": 127980},
                {"index": 62931},
                {"index": 129023},
                {"index": 62865},
                {"index": 128720},
                {"index": 128790},
                {"index": 128789},
                {"index": 62819},
                {"index": 128786},
                {"index": 128785},
                {"index": 62812},
                {"index": 128788},
                {"index": 128538},
                {"index": 128537},
                {"index": 128436},
                {"index": 62876},
                {"index": 128422},
                {"index": 128076},
                {"index": 128414},
                {"index": 128016},
                {"index": 128418},
                {"index": 62807},
                {"index": 128297},
                {"index": 129365},
                {"index": 62865},
                {"index": 128789},
                {"index": 128784},
                {"index": 128792},
                {"index": 128199},
                {"index": 128157},
                {"index": 129353},
                {"index": 128935},
                {"index": 128934},
                {"index": 128933},
                {"index": 128932},
                {"index": 128930},
                {"index": 62672},
                {"index": 62668},
                {"index": 128929},
                {"index": 62931},
                {"index": 127768},
                {"index": 127758},
                {"index": 127773},
                {"index": 127768},
                {"index": 128948},
                {"index": 127716},
                {"index": 128770},
                {"index": 62747},
                {"index": 127715},
                {"index": 62394},
                {"index": 128158},
                {"index": 63064},
                {"index": 129172},
                {"index": 129169},
                {"index": 129069},
                {"index": 62692},
                {"index": 62691},
                {"index": 128202},
                {"index": 127688},
                {"index": 128226},
                {"index": 127856},
                {"index": 127797},
                {"index": 62588},
                {"index": 128393},
                {"index": 128001},
                {"index": 127999},
                {"index": 62543},
                {"index": 128010},
                {"index": 128396},
                {"index": 128395},
                {"index": 128801},
                {"index": 128378},
                {"index": 128226},
                {"index": 62940},
                {"index": 128381},
                {"index": 129238},
                {"index": 127794},
                {"index": 127691},
                {"index": 129210},
                {"index": 129209},
                {"index": 127780},
                {"index": 129069},
                {"index": 129057},
                {"index": 129056},
                {"index": 129020},
                {"index": 128940},
                {"index": 128938},
                {"index": 62865},
                {"index": 129006},
                {"index": 128789},
                {"index": 128784},
                {"index": 128792},
                {"index": 128781},
                {"index": 128716},
                {"index": 128709},
                {"index": 128096},
                {"index": 128113},
                {"index": 128529},
                {"index": 128531},
                {"index": 128528},
                {"index": 128530},
                {"index": 128226},
                {"index": 128223},
                {"index": 62615},
                {"index": 128176},
                {"index": 128175},
                {"index": 128861},
                {"index": 128111},
                {"index": 62571},
                {"index": 127995},
                {"index": 127793},
                {"index": 127779},
                {"index": 128952},
                {"index": 128951},
                {"index": 128954},
                {"index": 128953},
                {"index": 128792},
                {"index": 128779},
                {"index": 129059},
                {"index": 128227},
                {"index": 128158},
                {"index": 129173},
                {"index": 129168},
                {"index": 129069},
                {"index": 62692},
                {"index": 128202},
                {"index": 129069},
                {"index": 128771},
                {"index": 128709},
                {"index": 127688},
                {"index": 128393},
                {"index": 129056},
                {"index": 128709},
                {"index": 128058},
                {"index": 128224},
                {"index": 62571},
                {"index": 129365},
                {"index": 129302},
                {"index": 128799},
                {"index": 62933},
                {"index": 127827},
                {"index": 127691},
                {"index": 62934},
                {"index": 128529},
                {"index": 62702},
                {"index": 128299},
                {"index": 62659},
                {"index": 127822},
                {"index": 128542},
                {"index": 128541},
                {"index": 128540},
                {"index": 62819},
                {"index": 128786},
                {"index": 128785},
                {"index": 62812},
                {"index": 128790},
                {"index": 62819},
                {"index": 128786},
                {"index": 128785},
                {"index": 62812},
                {"index": 127692},
                {"index": 127688},
                {"index": 62652},
                {"index": 128026},
                {"index": 62865},
                {"index": 128789},
                {"index": 128784},
                {"index": 128792},
                {"index": 128224},
                {"index": 129262},
                {"index": 129212},
                {"index": 129211},
                {"index": 129349},
                {"index": 128483},
                {"index": 128385},
                {"index": 127616},
                {"index": 128012},
                {"index": 62549},
                {"index": 128990},
                {"index": 128012},
                {"index": 62549},
                {"index": 128990},
                {"index": 128204},
                {"index": 128116},
                {"index": 129082},
                {"index": 129078},
                {"index": 128877},
                {"index": 127796},
                {"index": 129266},
                {"index": 129216},
                {"index": 128804},
                {"index": 128379},
                {"index": 128274},
                {"index": 127796},
                {"index": 129355},
                {"index": 128175},
                {"index": 129057},
                {"index": 128721},
                {"index": 128726},
                {"index": 128725},
                {"index": 128724},
                {"index": 128723},
                {"index": 127647},
                {"index": 128721},
                {"index": 128726},
                {"index": 128724},
                {"index": 128723},
                {"index": 127647},
                {"index": 62865},
                {"index": 128789},
                {"index": 128784},
                {"index": 128792},
                {"index": 128200},
                {"index": 63076},
                {"index": 128852},
                {"index": 129040},
                {"index": 128117},
                {"index": 128885},
                {"index": 128893},
                {"index": 62867},
                {"index": 62866},
                {"index": 62865},
                {"index": 128839},
                {"index": 128722},
                {"index": 128721},
                {"index": 62837},
                {"index": 128842},
                {"index": 128720},
                {"index": 128841},
                {"index": 128818},
                {"index": 128817},
                {"index": 128816},
                {"index": 128726},
                {"index": 128725},
                {"index": 128724},
                {"index": 128723},
                {"index": 128275},
                {"index": 128483},
                {"index": 128792},
                {"index": 128782},
                {"index": 128781},
                {"index": 128274},
                {"index": 128527},
                {"index": 128360},
                {"index": 128892},
                {"index": 128200},
                {"index": 62615},
                {"index": 128177},
                {"index": 128178},
                {"index": 128175},
                {"index": 127995},
                {"index": 127647},
                {"index": 127645},
                {"index": 128207},
                {"index": 127784},
                {"index": 62916},
                {"index": 128483},
                {"index": 128482},
                {"index": 62805},
                {"index": 128981},
                {"index": 128980},
                {"index": 128979},
                {"index": 128978},
                {"index": 127688},
                {"index": 129069},
                {"index": 128015},
                {"index": 128058},
                {"index": 128041},
                {"index": 62571},
                {"index": 127731},
                {"index": 127695},
                {"index": 129056},
                {"index": 128523},
                {"index": 128369},
                {"index": 128198},
                {"index": 127731},
                {"index": 62527},
                {"index": 128990},
                {"index": 129153},
                {"index": 129152},
                {"index": 129192},
                {"index": 129148},
                {"index": 129146},
                {"index": 129145},
                {"index": 129144},
                {"index": 129143},
                {"index": 129141},
                {"index": 129140},
                {"index": 129139},
                {"index": 129138},
                {"index": 129137},
                {"index": 129136},
                {"index": 62931},
                {"index": 129023},
                {"index": 128056},
                {"index": 128116},
                {"index": 129031},
                {"index": 129030},
                {"index": 129029},
                {"index": 129028},
                {"index": 129027},
                {"index": 129026},
                {"index": 62934},
                {"index": 128200},
                {"index": 62571},
                {"index": 127955},
                {"index": 128483},
                {"index": 62417},
                {"index": 128888},
                {"index": 128887},
                {"index": 128886},
                {"index": 128720},
                {"index": 128816},
                {"index": 128207},
                {"index": 128884},
                {"index": 127930},
                {"index": 127694},
                {"index": 129302},
                {"index": 129314},
                {"index": 129347},
                {"index": 63041},
                {"index": 129289},
                {"index": 129203},
                {"index": 129126},
                {"index": 128500},
                {"index": 128499},
                {"index": 128498},
                {"index": 128497},
                {"index": 128484},
                {"index": 62733},
                {"index": 127815},
                {"index": 127971},
                {"index": 127960},
                {"index": 62400},
                {"index": 128480},
                {"index": 127680},
                {"index": 127693},
                {"index": 127694},
                {"index": 63064},
                {"index": 129302},
                {"index": 63041},
                {"index": 129296},
                {"index": 129294},
                {"index": 129289},
                {"index": 127691},
                {"index": 128500},
                {"index": 128479},
                {"index": 128478},
                {"index": 128472},
                {"index": 128498},
                {"index": 128477},
                {"index": 128497},
                {"index": 128851},
                {"index": 128484},
                {"index": 129015},
                {"index": 128483},
                {"index": 128482},
                {"index": 62734},
                {"index": 128360},
                {"index": 128503},
                {"index": 127815},
                {"index": 128176},
                {"index": 127977},
                {"index": 127971},
                {"index": 127960},
                {"index": 62400},
                {"index": 128480},
                {"index": 127601},
                {"index": 128814},
                {"index": 129303},
                {"index": 127688},
                {"index": 127856},
                {"index": 128393},
                {"index": 62461},
                {"index": 129289},
                {"index": 129264},
                {"index": 129259},
                {"index": 128835},
                {"index": 128827},
                {"index": 128451},
                {"index": 129187},
                {"index": 62991},
                {"index": 62976},
                {"index": 129042},
                {"index": 129070},
                {"index": 128525},
                {"index": 129054},
                {"index": 129039},
                {"index": 129035},
                {"index": 128418},
                {"index": 128878},
                {"index": 128049},
                {"index": 128832},
                {"index": 128830},
                {"index": 128829},
                {"index": 128828},
                {"index": 128767},
                {"index": 128766},
                {"index": 128023},
                {"index": 62768},
                {"index": 128453},
                {"index": 128988},
                {"index": 62717},
                {"index": 128291},
                {"index": 128987},
                {"index": 128310},
                {"index": 128285},
                {"index": 128284},
                {"index": 62702},
                {"index": 128383},
                {"index": 128350},
                {"index": 128186},
                {"index": 128185},
                {"index": 128184},
                {"index": 128183},
                {"index": 127782},
                {"index": 128352},
                {"index": 62676},
                {"index": 62656},
                {"index": 128254},
                {"index": 128253},
                {"index": 62644},
                {"index": 62621},
                {"index": 128153},
                {"index": 128075},
                {"index": 128078},
                {"index": 127795},
                {"index": 127971},
                {"index": 127946},
                {"index": 62516},
                {"index": 127870},
                {"index": 127725},
                {"index": 62400},
                {"index": 128748},
                {"index": 62384},
                {"index": 127681},
                {"index": 127680},
                {"index": 62378},
                {"index": 128801},
                {"index": 129043},
                {"index": 128378},
                {"index": 127731},
                {"index": 62527},
                {"index": 62404},
                {"index": 129215},
                {"index": 129167},
                {"index": 129024},
                {"index": 129355},
                {"index": 127797},
                {"index": 128852},
                {"index": 129059},
                {"index": 128227},
                {"index": 128852},
                {"index": 128684},
                {"index": 128433},
                {"index": 62582},
                {"index": 129084},
                {"index": 127886},
                {"index": 129084},
                {"index": 128686},
                {"index": 62705},
                {"index": 62583},
                {"index": 128438},
                {"index": 128915},
                {"index": 128646},
                {"index": 62705},
                {"index": 128037},
                {"index": 128438},
                {"index": 128907},
                {"index": 128034},
                {"index": 128059},
                {"index": 62570},
                {"index": 128059},
                {"index": 62570},
                {"index": 128488},
                {"index": 128532},
                {"index": 128487},
                {"index": 128168},
                {"index": 128175},
                {"index": 129069},
                {"index": 62702},
                {"index": 128299},
                {"index": 62659},
                {"index": 127680},
                {"index": 128482},
                {"index": 62702},
                {"index": 128299},
                {"index": 129173},
                {"index": 129172},
                {"index": 129171},
                {"index": 129170},
                {"index": 129169},
                {"index": 129168},
                {"index": 62759},
                {"index": 128116},
                {"index": 128393},
                {"index": 128046},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128933},
                {"index": 128932},
                {"index": 128931},
                {"index": 128930},
                {"index": 129034},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128933},
                {"index": 128932},
                {"index": 128931},
                {"index": 128930},
                {"index": 129034},
                {"index": 129033},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 128154},
                {"index": 127687},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128933},
                {"index": 128932},
                {"index": 128931},
                {"index": 128930},
                {"index": 129034},
                {"index": 129033},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127854},
                {"index": 127792},
                {"index": 128929},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128933},
                {"index": 128932},
                {"index": 128931},
                {"index": 128930},
                {"index": 129034},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128931},
                {"index": 129034},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128931},
                {"index": 129034},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 129194},
                {"index": 129193},
                {"index": 129353},
                {"index": 128315},
                {"index": 128937},
                {"index": 128934},
                {"index": 128931},
                {"index": 129034},
                {"index": 128163},
                {"index": 127899},
                {"index": 128936},
                {"index": 127792},
                {"index": 128929},
                {"index": 62697},
                {"index": 129266},
                {"index": 127921},
                {"index": 62467},
                {"index": 128117},
                {"index": 128804},
                {"index": 128604},
                {"index": 128715},
                {"index": 62768},
                {"index": 128602},
                {"index": 128599},
                {"index": 128209},
                {"index": 128470},
                {"index": 127858},
                {"index": 127688},
                {"index": 128224},
                {"index": 127958},
                {"index": 128170},
                {"index": 128176},
                {"index": 128003},
                {"index": 128071},
                {"index": 128008},
                {"index": 127810},
                {"index": 127649},
                {"index": 129331},
                {"index": 129321},
                {"index": 129311},
                {"index": 129248},
                {"index": 129247},
                {"index": 129070},
            ],
        },
        {
            "question": "What genes play role in ovarian cancer?",
            "expected_answer": "The genes that play a role in ovarian cancer include fam107a, mre11, dph1, slc22a10, gpr150, kansl1, arl11, muc16, chmp4c, lrrc46, macir, brip1, c19orf12, plekhf1, wdr77, trmt11, prtfdc1, lrrc59, rnf43, haus6, bnc2, cdk12, desi2, babam1, tiparp, spdef, sulf1, jmjd6, rras2, pop4, srsf10, camkk2, atg7, yap1, dlc1, tubb3, msln, hdac6, atg5, dyrk1b, prc1, selenbp1, tnfsf10, uri1, skap1, tp63, itga8, bap1, wnt7a, tyms, tp53bp1, tp53, tlr4, tert, zeb1, stk11, stat3, sparc, sod2, sod1, smarca4, slc5a5, slc2a1, skp2, rbl2, rad51d, rad51c, nectin2, pten, klk10, map2k1, mapk3, mapk1, ppp1cc, pms2, pms1, pik3r1, pik3ca, prkn, opcml, nme2, myc, msh2, mlh1, mki67, met, epcam, kras, il11ra, cxcl8, il6st, il6, hoxd11, hoxd9, hoxd1, hoxb9, msh6, grik2, nr5a1, mtor, folr1, fgf1, fasn, mecom, esr1, ercc6, ercc4, ereg, erbb2, egfr, ednra, dok1, gadd45a, cyp1b1, ctnnb1, cdkn1b, cdh1, ccnh, ccne1, ccnd2, cav1, brca2, braf, brca1, bcl9, atr, atp7b, atf3, areg, aqp3, birc5, xiap, anxa3, alox12b, alox5, akt2 and akt1.",
            "expected_cypher": 'MATCH (d:disease {name: "ovarian cancer"})-[:associated_with]->(g:gene_or_protein) RETURN g.name',
            "subgraph_query": 'MATCH (d:disease {name: "ovarian cancer"})-[a:associated_with]->(g:gene_or_protein) RETURN d, a, g',
            "nodes": [
                {"index": 444},
                {"index": 2249},
                {"index": 5580},
                {"index": 34016},
                {"index": 34884},
                {"index": 21969},
                {"index": 34883},
                {"index": 22063},
                {"index": 5791},
                {"index": 12158},
                {"index": 13905},
                {"index": 6809},
                {"index": 34882},
                {"index": 6370},
                {"index": 5512},
                {"index": 12947},
                {"index": 6827},
                {"index": 7769},
                {"index": 12936},
                {"index": 7311},
                {"index": 13516},
                {"index": 11472},
                {"index": 10611},
                {"index": 1932},
                {"index": 34033},
                {"index": 2147},
                {"index": 12680},
                {"index": 3868},
                {"index": 4927},
                {"index": 372},
                {"index": 2601},
                {"index": 12491},
                {"index": 2861},
                {"index": 2840},
                {"index": 7689},
                {"index": 1205},
                {"index": 4137},
                {"index": 4769},
                {"index": 1322},
                {"index": 3304},
                {"index": 3609},
                {"index": 1505},
                {"index": 6996},
                {"index": 3152},
                {"index": 3788},
                {"index": 6373},
                {"index": 6757},
                {"index": 7817},
                {"index": 1709},
                {"index": 3509},
                {"index": 1796},
                {"index": 1785},
                {"index": 3259},
                {"index": 5129},
                {"index": 449},
                {"index": 5998},
                {"index": 729},
                {"index": 9714},
                {"index": 4959},
                {"index": 361},
                {"index": 492},
                {"index": 7713},
                {"index": 8385},
                {"index": 86},
                {"index": 5265},
                {"index": 1367},
                {"index": 5869},
                {"index": 4278},
                {"index": 177},
                {"index": 6439},
                {"index": 1831},
                {"index": 866},
                {"index": 418},
                {"index": 1871},
                {"index": 3556},
                {"index": 10031},
                {"index": 43},
                {"index": 1641},
                {"index": 1340},
                {"index": 5958},
                {"index": 3559},
                {"index": 25},
                {"index": 426},
                {"index": 2423},
                {"index": 1782},
                {"index": 67},
                {"index": 12733},
                {"index": 2621},
                {"index": 9521},
                {"index": 2978},
                {"index": 1533},
                {"index": 1567},
                {"index": 2625},
                {"index": 7162},
                {"index": 7514},
                {"index": 4136},
                {"index": 3246},
                {"index": 8936},
                {"index": 668},
                {"index": 1558},
                {"index": 10129},
                {"index": 2036},
                {"index": 4845},
                {"index": 420},
                {"index": 375},
                {"index": 695},
                {"index": 4752},
                {"index": 6385},
                {"index": 822},
                {"index": 125},
                {"index": 9207},
                {"index": 4195},
                {"index": 6779},
                {"index": 12769},
                {"index": 769},
                {"index": 2104},
                {"index": 2543},
                {"index": 2295},
                {"index": 406},
                {"index": 4435},
                {"index": 489},
                {"index": 80},
                {"index": 1980},
                {"index": 554},
                {"index": 4842},
                {"index": 395},
                {"index": 8946},
                {"index": 882},
                {"index": 10285},
                {"index": 832},
                {"index": 4084},
                {"index": 699},
                {"index": 6095},
                {"index": 7694},
                {"index": 3631},
                {"index": 3380},
                {"index": 1057},
            ],
        },
        {
            "question": "Which medications are currently available for treating Parkinson's disease?",
            "expected_answer": "The medications currently available for treating Parkinson's disease include benzatropine, apomorphine, levodopa, istradefylline, entacapone, amantadine, droxidopa, safinamide, biperiden, selegiline, orphenadrine, rotigotine, ropinirole, rasagiline, pergolide, bromocriptine, tolcapone, opicapone, trihexyphenidyl, procyclidine, cycrimine, pramipexole, pimavanserin, piribedil, carbidopa, metixene, and melevodopa. These medications help manage the symptoms and improve the quality of life for individuals with Parkinson's disease.",
            "expected_cypher": 'MATCH (d:disease {name: "parkinson disease"})-[:indication]->(drug:drug) RETURN drug.name',
            "nodes": [
                {"index": 14139},
                {"index": 14220},
                {"index": 14321},
                {"index": 14452},
                {"index": 14806},
                {"index": 14830},
                {"index": 14831},
                {"index": 14904},
                {"index": 15124},
                {"index": 15138},
                {"index": 15147},
                {"index": 15193},
                {"index": 15291},
                {"index": 15333},
                {"index": 15490},
                {"index": 15493},
                {"index": 15772},
                {"index": 15812},
                {"index": 16054},
                {"index": 16056},
                {"index": 16315},
                {"index": 16842},
                {"index": 17270},
                {"index": 17414},
                {"index": 19207},
                {"index": 20153},
                {"index": 20591},
            ],
        },
        {
            "question": "What are the likely causes of chronic kidney disease?",
            "expected_answer": "The likely causes of chronic kidney disease include exposure to atrazine, cadmium, lead, mercury, paraquat, perfluorooctanoic acid, alachlor, aldicarb, chlordan, coumaphos, imazethapyr, metalaxyl, metolachlor, parathion, pendimethalin, permethrin, petroleum and phorate.",
            "expected_cypher": 'MATCH (disease:disease {name: "chronic kidney disease"})-[:linked_to]->(exposure:exposure) RETURN DISTINCT exposure.name',
            "subgraph_query": 'MATCH (disease:disease {name: "chronic kidney disease"})-[l:linked_to]->(exposure:exposure) RETURN disease, l, exposure',
            "nodes": [
                {"index": 61711},
                {"index": 61717},
                {"index": 61757},
                {"index": 61760},
                {"index": 61784},
                {"index": 61794},
                {"index": 61876},
                {"index": 61877},
                {"index": 61937},
                {"index": 61961},
                {"index": 62046},
                {"index": 62063},
                {"index": 62073},
                {"index": 62097},
                {"index": 62098},
                {"index": 62104},
                {"index": 62106},
                {"index": 62108},
            ],
        },
        {
            "question": "What are symptoms of rheumatoid arthritis?",
            "expected_answer": "Symptoms of rheumatoid arthritis include vasculitis, a positive rheumatoid factor, weight loss, fever, interphalangeal joint erosions, positivity for anti-citrullinated protein antibodies, arthralgia, swan neck-like deformities of the fingers, digital flexor tenosynovitis, elevated C-reactive protein levels, fatigue, joint stiffness, polyarticular arthritis, elevated erythrocyte sedimentation rate, and joint swelling. These symptoms collectively contribute to the clinical diagnosis and management of the condition.",
            "expected_cypher": 'MATCH (d:disease {name: "rheumatoid arthritis"})-[:phenotype_present]->(p:effect_or_phenotype) RETURN p.name',
            "nodes": [
                {"index": 23956},
                {"index": 85299},
                {"index": 23736},
                {"index": 22952},
                {"index": 86562},
                {"index": 92247},
                {"index": 23901},
                {"index": 86513},
                {"index": 89234},
                {"index": 88592},
                {"index": 25620},
                {"index": 24045},
                {"index": 23937},
                {"index": 86388},
                {"index": 85550},
                {"index": 84874},
            ],
        },
        {
            "question": "Which biomarkers are associated with a diagnosis of Alport syndrome?",
            "expected_answer": "The biomarkers associated with a diagnosis of Alport syndrome include the genes nphs2, col4a5, col4a4, myh9 and col4a3. These genes are crucial for understanding the genetic basis of the condition.",
            "expected_cypher": 'MATCH (d:disease {name: "alport syndrome"})-[:associated_with]->(g:gene_or_protein) RETURN DISTINCT g.name',
            "subgraph_query": 'MATCH (d:disease {name: "alport syndrome"})-[a:associated_with]->(gp:gene_or_protein) RETURN d, a, gp',
            "nodes": [{"index": 11428}, {"index": 909}, {"index": 11545}, {"index": 209}, {"index": 8571}],
        },
        {
            "question": "How many biomarkers are associated with a diagnosis of Alport syndrome?",
            "expected_answer": "There are 5 biomarkers associated with the diagnosis of Alport syndrome.",
            "expected_cypher": 'MATCH (d:disease {name: "alport syndrome"})-[:associated_with]->(gp:gene_or_protein) RETURN COUNT(gp)',
            "subgraph_query": 'MATCH (d:disease {name: "alport syndrome"})-[a:associated_with]->(gp:gene_or_protein) RETURN d, a, gp',
            "nodes": [{"index": 11428}, {"index": 909}, {"index": 11545}, {"index": 209}, {"index": 8571}],
        },
    ]

    eval_sample_addition = EvalSampleAddition(
        graph=build_neo4j_graph(),
        subgraph_extractor=LLMSubGraphExtractor(
            model=load_chat_model(),
        ),
        path_to_json=Path("src/fact_finder/evaluator/evaluation_samples.json"),
    )

    for sample in tqdm.tqdm(manual_samples):

        eval_sample_addition.add_to_evaluation_sample_json(
            question=sample["question"],
            expected_cypher=sample["expected_cypher"],
            source="manual",
            expected_answer=sample["expected_answer"],
            is_answerable=True,
            nodes=sample["nodes"],
            subgraph_query=sample.get("subgraph_query"),
        )
