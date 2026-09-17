import importlib


def test_eval_harness_exists_and_loads_golden_dataset():
    module = importlib.import_module("evals.harness")
    goldens = module.load_goldens("golden/faithfulness_dataset.json")

    assert isinstance(goldens, list)
    assert goldens
    assert "query" in goldens[0]
    assert "ideal_context" in goldens[0]
