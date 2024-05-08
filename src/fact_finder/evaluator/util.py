import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import tqdm
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from fact_finder.evaluator.evaluation_sample import EvaluationSample
from fact_finder.evaluator.evaluation_samples import manual_samples
from fact_finder.tools.sub_graph_extractor import LLMSubGraphExtractor
from fact_finder.utils import build_neo4j_graph, load_chat_model

import pickle


def save_pickle(object: Any, path: str = "filename.pickle"):
    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str = "filename.pickle"):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
