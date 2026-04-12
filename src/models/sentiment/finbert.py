"""FinBERT-based financial sentiment analyser.

Lazily loads the ``ProsusAI/finbert`` model on first use so that import time
stays fast and the heavy PyTorch + transformers dependencies are only required
at runtime.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class FinBERTAnalyzer:
    """Financial sentiment analysis via FinBERT.

    The model and tokenizer are loaded lazily on the first call to
    :meth:`analyze` or :meth:`analyze_batch`.  If the ``transformers``
    library is not installed a graceful fallback returns neutral sentiment.
    """

    _LABEL_MAP: dict[str, str] = {
        "LABEL_0": "positive",
        "LABEL_1": "negative",
        "LABEL_2": "neutral",
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name
        self._pipeline: Any = None
        self._available: bool | None = None  # None = not yet checked

        logger.info("finbert_analyzer_init", model_name=model_name)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Attempt to load the transformers pipeline.

        Returns ``True`` on success, ``False`` if transformers is missing.
        """
        if self._available is not None:
            return self._available

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            self._available = True
            logger.info("finbert_model_loaded", model=self.model_name)
        except ImportError:
            self._available = False
            logger.warning(
                "transformers_not_installed",
                msg="FinBERT unavailable — returning neutral fallback",
            )
        except Exception:
            self._available = False
            logger.exception("finbert_load_failed")

        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """Analyse a list of texts and return sentiment results.

        Parameters
        ----------
        texts:
            Raw text strings (headlines, articles, tweets, etc.).

        Returns
        -------
        list[dict]
            Each dict contains ``sentiment`` (str) and ``score`` (float).
        """
        if not texts:
            return []

        if not self._load():
            return self._neutral_fallback(texts)

        try:
            raw_results = self._pipeline(texts)  # type: ignore[misc]
        except Exception:
            logger.exception("finbert_inference_failed")
            return self._neutral_fallback(texts)

        results: list[dict[str, Any]] = []
        for raw in raw_results:
            label = self._LABEL_MAP.get(raw["label"], "neutral")
            results.append({
                "sentiment": label,
                "score": float(raw["score"]),
            })

        logger.debug("finbert_analyze_complete", n_texts=len(texts))
        return results

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Analyse texts in batches for efficiency.

        Parameters
        ----------
        texts:
            Raw text strings.
        batch_size:
            Number of texts per inference batch.

        Returns
        -------
        list[dict]
            Same format as :meth:`analyze`.
        """
        if not texts:
            return []

        if not self._load():
            return self._neutral_fallback(texts)

        all_results: list[dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_results.extend(self.analyze(batch))

        logger.info(
            "finbert_batch_complete",
            total=len(texts),
            batch_size=batch_size,
        )
        return all_results

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_fallback(texts: list[str]) -> list[dict[str, Any]]:
        """Return neutral sentiment for every input text."""
        return [{"sentiment": "neutral", "score": 0.0} for _ in texts]
