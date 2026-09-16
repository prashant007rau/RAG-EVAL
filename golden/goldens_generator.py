import os, re, glob, json, random
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Synthesizer
from langchain_groq import ChatGroq
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()


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


# --- reuse your own VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in glob.glob("data/*.vtt"):
        with open(path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
context_limit = int(os.getenv("GROQ_CONTEXTS", "3"))
sample = random.sample(chunks, min(context_limit, len(chunks)))
contexts = [[c] for c in sample]                          # each context = one chunk

groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
max_concurrent = int(os.getenv("GROQ_MAX_CONCURRENT", "1"))
synthesizer = Synthesizer(
    model=GroqDeepEvalLLM(groq_model),
    max_concurrent=max_concurrent,
    evolution_config=EvolutionConfig(num_evolutions=0),
)
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,       # <-- THIS gives you the ideal_answer
    max_goldens_per_context=1,          # 1 question per chunk -> ~12 goldens
)


# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

with open("golden/retriever_deepeval_goldens.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> golden/retriever_deepeval_goldens.json")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")