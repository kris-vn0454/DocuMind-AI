import logging
import time
from dataclasses import dataclass, field

from src.generation.llm_client import GrokClient, LLMResponse
from src.retrieval.retriever import RetrievedContext, SemanticRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are DocuMind AI, an enterprise document intelligence assistant.
Answer questions accurately and concisely based ONLY on the provided document context.
Always cite the source document name when referencing information.
Structure your answers with clear formatting when appropriate.
If the answer is not present in the provided context, respond:
"The provided documents do not contain sufficient information to answer this question."
Never fabricate facts or draw on knowledge outside the provided context."""


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: list[RetrievedContext]
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    context_chunks_used: int = 0

    @property
    def unique_sources(self) -> list[str]:
        return list(dict.fromkeys(ctx.source for ctx in self.sources))


class RAGPipeline:
    def __init__(self, retriever: SemanticRetriever, llm: GrokClient, system_prompt: str | None = None):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or _SYSTEM_PROMPT

    def _build_prompt(self, query: str, contexts: list[RetrievedContext]) -> str:
        context_block = self.retriever.format_context_for_prompt(contexts)
        return (
            f"### Relevant Document Context:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:"
        )

    def query(self, question: str) -> RAGResult:
        t0 = time.perf_counter()

        contexts = self.retriever.retrieve(question)
        prompt = self._build_prompt(question, contexts)
        llm_response = self.llm.generate(prompt, system_prompt=self.system_prompt)

        latency = time.perf_counter() - t0
        logger.info(
            f"RAG query completed in {latency:.2f}s | "
            f"chunks={len(contexts)} | tokens={llm_response.total_tokens}"
        )

        return RAGResult(
            query=question,
            answer=llm_response.content,
            sources=contexts,
            model=llm_response.model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            latency_seconds=latency,
            context_chunks_used=len(contexts),
        )

    def summarize_document(self, document_name: str, doc_content: str) -> str:
        prompt = (
            f"Please provide a comprehensive summary of the following document: '{document_name}'\n\n"
            f"Document Content:\n{doc_content[:8000]}\n\n"
            f"Provide:\n"
            f"1. Executive Summary (2-3 sentences)\n"
            f"2. Key Topics Covered (bullet points)\n"
            f"3. Important Findings or Conclusions\n"
            f"4. Suggested Use Cases for this document"
        )
        response = self.llm.generate(prompt, system_prompt=self.system_prompt)
        return response.content

    def extract_key_insights(self, contexts: list[RetrievedContext]) -> str:
        context_text = self.retriever.format_context_for_prompt(contexts)
        prompt = (
            f"Based on the following document excerpts, extract and list the most important "
            f"key insights, facts, and actionable information:\n\n{context_text}\n\n"
            f"Format as a structured list with categories."
        )
        response = self.llm.generate(prompt, system_prompt=self.system_prompt)
        return response.content
