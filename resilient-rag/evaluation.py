# src/evaluations.py
import json
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.rag import RAGEngine
from src.config import Config

config = Config()

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY)
)


class RAGEvaluator:
    def __init__(self):
        self.rag_engine = RAGEngine()

    def load_test_data(self, filepath: str = "data/test_set.json"):
        """Load test data from a JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def run_evaluation(self):
        print("Starting RAG Evaluation")
        test_data = self.load_test_data()

        questions = []
        references = []  # <-- RAGAS expects 'reference' (string), not 'ground_truth' (list)
        answers = []
        contexts = []

        print(f"Processing {len(test_data)} test cases")
        for case in test_data:
            q = case["question"]
            # your JSON uses 'ground_truth'; map it to a single-string 'reference'
            ref = case["ground_truth"]

            result = self.rag_engine.query(q)

            # RAGAS columns:
            # - question: str
            # - answer: str
            # - contexts: List[str]
            # - reference: str   <-- single string, not a list
            questions.append(q)
            references.append(ref)
            answers.append(result["answer"])
            contexts.append(result["contexts"])  # already List[str]

        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "reference": references,
        }

        dataset = Dataset.from_dict(data_dict)

        print("Evaluating with RAGAS...")
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm
        )
        return results


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation()

    print("\nEvaluation Report")
    print(results)

    df = results.to_pandas()
    df.to_csv("evaluation_report.csv", index=False)
    print("Detailed report saved to 'evaluation_report.csv'")
