import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryTracker:
    """Tracks RAG query metrics using MLflow."""

    def __init__(self, experiment_name: str = "documind-rag-queries", tracking_uri: str = "mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow = None
        self._enabled = False
        self._setup()

    def _setup(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._enabled = True
            logger.info(f"MLflow tracking enabled → experiment: '{self.experiment_name}'")
        except ImportError:
            logger.warning("MLflow not installed. Query tracking disabled.")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Tracking disabled.")

    def log_query(
        self,
        query: str,
        answer: str,
        latency: float,
        chunks_retrieved: int,
        input_tokens: int,
        output_tokens: int,
        model: str,
        sources: list[str],
    ) -> None:
        if not self._enabled:
            return

        with self._mlflow.start_run(run_name=f"query_{datetime.now().strftime('%H%M%S')}"):
            self._mlflow.log_params({
                "model": model,
                "query_length": len(query),
                "sources_count": len(set(sources)),
            })
            self._mlflow.log_metrics({
                "latency_seconds": latency,
                "chunks_retrieved": chunks_retrieved,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "answer_length": len(answer),
            })
            self._mlflow.log_text(query, "query.txt")
            self._mlflow.log_text(answer, "answer.txt")
            self._mlflow.log_text(", ".join(sources), "sources.txt")

    def log_ingestion(self, doc_name: str, chunk_count: int, embed_time: float) -> None:
        if not self._enabled:
            return

        with self._mlflow.start_run(run_name=f"ingest_{doc_name[:20]}"):
            self._mlflow.log_params({"document": doc_name})
            self._mlflow.log_metrics({
                "chunks_created": chunk_count,
                "embedding_time_seconds": embed_time,
            })
