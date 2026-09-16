import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from langchain_groq import ChatGroq

from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "golden/retriever_deepeval_goldens.json"
JUDGE_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
THRESHOLD = 0.7


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return ChatGroq(model=self.model_name, temperature=0, max_retries=5)

    def generate(self, prompt, schema=None, **kwargs):
        model = self.model.with_structured_output(schema) if schema else self.model
        response = model.invoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    async def a_generate(self, prompt, schema=None, **kwargs):
        model = self.model.with_structured_output(schema) if schema else self.model
        response = await model.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    def get_model_name(self):
        return self.model_name

    def supports_structured_outputs(self):
        return True


def run():
    with open(GOLDEN_PATH, encoding="utf-8") as file:
        goldens = json.load(file)

    retriever = build_retriever()
    test_cases = []
    for golden in goldens:
        retrieved = retriever.invoke(golden["query"])
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                retrieval_context=[doc.page_content for doc in retrieved],
                actual_output="(generator not evaluated in this run)",
            )
        )

    judge = GroqDeepEvalLLM(JUDGE_MODEL)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        hyperparameters={
            "retriever": "chroma",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )


if __name__ == "__main__":
    result = run()
    print(f"retriever evaluation completed: {len(result.test_results)} cases")