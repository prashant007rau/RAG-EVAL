"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from langchain_groq import ChatGroq

from src.generator import generate   # your generator: generate(query, context) -> answer
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "golden" / "faithfulness_dataset.json"
JUDGE_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
THRESHOLD = 0.7


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return ChatGroq(model=self.model_name, temperature=0, max_retries=5)

    def generate(self, prompt, schema=None, **kwargs):
        # Groq does not reliably accept the schema/tool-call format that DeepEval
        # emits for verdict extraction. Use plain model.invoke instead and let the
        # metric parser handle the JSON response text.
        response = self.model.invoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    async def a_generate(self, prompt, schema=None, **kwargs):
        response = await self.model.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else response

    def get_model_name(self):
        return self.model_name

    def supports_structured_outputs(self):
        return False


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    judge = GroqDeepEvalLLM(JUDGE_MODEL)
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())