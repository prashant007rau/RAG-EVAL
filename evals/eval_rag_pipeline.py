# eval_rag_pipeline.py
import os
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from langchain_groq import ChatGroq

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "golden" / "faithfulness_dataset.json"
JUDGE_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
THRESHOLD = 0.7
MAX_CASES = int(os.getenv("EVAL_MAX_CASES", "3"))


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return ChatGroq(model=self.model_name, temperature=0, max_retries=5)

    def generate(self, prompt, schema=None, **kwargs):
        response = self.model.invoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    async def a_generate(self, prompt, schema=None, **kwargs):
        response = await self.model.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    def get_model_name(self):
        return self.model_name

    def supports_structured_outputs(self):
        return False


def run(rag):
    goldens = load_goldens(GOLDEN_PATH)
    if MAX_CASES and MAX_CASES > 0 and MAX_CASES < len(goldens):
        goldens = goldens[:MAX_CASES]

    judge = GroqDeepEvalLLM(JUDGE_MODEL)
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"],
            )
        )

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())