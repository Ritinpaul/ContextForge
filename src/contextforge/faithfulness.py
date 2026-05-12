from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float | None
    error: str | None = None


def compute_faithfulness_ragas(*, question: str, answer: str, contexts: list[str]) -> FaithfulnessResult:
    """Compute faithfulness with RAGAS if available.

    Returns score in [0, 1] when successful.

    If RAGAS isn't installed or evaluation fails, returns score=None and an error.
    """

    try:
        from datasets import Dataset
    except Exception as e:  # pragma: no cover
        return FaithfulnessResult(score=None, error=f"datasets import failed: {e}")

    try:
        from ragas import evaluate
    except Exception as e:  # pragma: no cover
        return FaithfulnessResult(score=None, error=f"ragas import failed: {e}")

    # Metric import has changed across ragas versions.
    metric = None
    try:  # older versions
        from ragas.metrics import faithfulness as faithfulness_metric

        metric = faithfulness_metric
    except Exception:
        try:  # newer versions
            from ragas.metrics import Faithfulness

            metric = Faithfulness()
        except Exception as e:
            return FaithfulnessResult(score=None, error=f"ragas faithfulness metric import failed: {e}")

    # RAGAS expects list-of-contexts per row
    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )

    # RAGAS typically needs an LLM to judge faithfulness.
    # We provide a local HuggingFace pipeline-based LLM to keep Windows runnable.
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline

        judge_pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
        )
        llm = HuggingFacePipeline(pipeline=judge_pipe)
    except Exception as e:  # pragma: no cover
        return FaithfulnessResult(score=None, error=f"failed to init local judge LLM: {e}")

    try:
        result = evaluate(ds, metrics=[metric], llm=llm)
    except TypeError:
        # some versions use `run_config` etc; try minimal call
        try:
            result = evaluate(ds, metrics=[metric], llm=llm, embeddings=None)
        except Exception as e:
            return FaithfulnessResult(score=None, error=f"ragas evaluate failed: {e}")
    except Exception as e:
        return FaithfulnessResult(score=None, error=f"ragas evaluate failed: {e}")

    try:
        # result is a ragas.dataset.Schema or pandas-like
        # Most versions return a dict-like object with metric name.
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            val = float(df.iloc[0]["faithfulness"])
            return FaithfulnessResult(score=val)
        if isinstance(result, dict) and "faithfulness" in result:
            return FaithfulnessResult(score=float(result["faithfulness"]))
    except Exception:
        pass

    # Fall back: try attribute / indexing
    try:
        val = float(result["faithfulness"][0])
        return FaithfulnessResult(score=val)
    except Exception as e:
        return FaithfulnessResult(score=None, error=f"could not parse ragas result: {e}")
