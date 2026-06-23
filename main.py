"""
DocuMind AI — Main Entry Point
Generates sample documents and demonstrates the full RAG pipeline.

Usage:
    python main.py                    # Full demo (requires ANTHROPIC_API_KEY)
    python main.py --generate-samples # Generate sample docs only
    python main.py --ingest-only      # Ingest sample docs without querying
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_sample_documents():
    """Create sample .txt documents in data/sample_docs/"""
    sample_dir = Path("data/sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "01_retrieval_augmented_generation.txt": """
# Retrieval-Augmented Generation (RAG): A Comprehensive Overview

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language model (LLM) outputs
by incorporating relevant information retrieved from external knowledge bases. Introduced by Meta AI researchers
in 2020, RAG addresses the key limitation of LLMs: static knowledge frozen at training time.

## Core Components of a RAG System
1. Document Ingestion Pipeline: Raw documents are loaded, cleaned, and chunked into smaller segments.
2. Embedding Model: Each chunk is converted into a dense vector representation capturing semantic meaning.
3. Vector Database: Embeddings are indexed for fast similarity search (e.g., FAISS, Pinecone, Chroma).
4. Retriever: Given a user query, the retriever finds the most semantically similar chunks.
5. Generator: An LLM synthesizes a final answer using the retrieved context plus the original query.

## Why RAG Outperforms Pure LLMs
- Reduces hallucinations by grounding answers in real documents
- Enables access to proprietary or up-to-date information
- Provides source citations, making answers auditable
- Cheaper than fine-tuning for domain adaptation
- Supports dynamic knowledge updates without retraining

## Advanced RAG Techniques
Naive RAG simply retrieves and generates. Advanced techniques include:
- Hybrid Search: Combining dense (semantic) and sparse (BM25) retrieval
- Reranking: Using cross-encoder models to rerank retrieved results
- HyDE (Hypothetical Document Embeddings): Generating a hypothetical answer for better retrieval
- Self-RAG: The model decides when and what to retrieve
- RAPTOR: Recursive clustering and summarization for hierarchical retrieval

## Evaluation Metrics for RAG Systems
- Faithfulness: Is the answer consistent with the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: How relevant are the retrieved chunks?
- Context Recall: Are all relevant chunks retrieved?
Tools like RAGAS, TruLens, and DeepEval automate RAG evaluation.

## Production Considerations
When deploying RAG systems at scale:
- Chunk size matters: 256-512 tokens is the typical sweet spot
- Overlap between chunks prevents information loss at boundaries
- Metadata filtering can pre-narrow the search space
- Caching frequent queries reduces latency and cost
- Monitoring query patterns helps identify knowledge gaps

## Industry Applications
RAG is widely used in enterprise document Q&A systems, legal research platforms, medical knowledge bases,
customer support automation, and internal knowledge management. Companies like Anthropic, OpenAI, and
Cohere all provide RAG-optimized APIs and toolkits.
""",

        "02_machine_learning_fundamentals.txt": """
# Machine Learning Fundamentals: A Practitioner's Guide

## Types of Machine Learning
Machine learning encompasses three primary paradigms:

### 1. Supervised Learning
The algorithm learns a mapping from inputs to outputs using labeled training data.
- Classification: Predict discrete categories (spam detection, image recognition, churn prediction)
- Regression: Predict continuous values (house prices, stock returns, demand forecasting)
Key algorithms: Linear/Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, Neural Networks

### 2. Unsupervised Learning
The algorithm discovers patterns in unlabeled data.
- Clustering: Group similar data points (K-Means, DBSCAN, Hierarchical)
- Dimensionality Reduction: Compress data while preserving structure (PCA, t-SNE, UMAP)
- Anomaly Detection: Identify unusual patterns (Isolation Forest, Autoencoders)

### 3. Reinforcement Learning
An agent learns by interacting with an environment and receiving rewards.
Applications: Game playing (AlphaGo), robotics, recommendation systems, trading strategies.

## The Machine Learning Pipeline
1. Problem Framing: Define business objective as ML task
2. Data Collection: Gather relevant, sufficient, representative data
3. Exploratory Data Analysis: Understand distributions, correlations, anomalies
4. Feature Engineering: Transform raw data into informative features
5. Model Selection: Choose appropriate algorithm(s)
6. Training & Validation: Fit model with cross-validation
7. Hyperparameter Tuning: Optimize model configuration
8. Evaluation: Measure performance on held-out test set
9. Deployment: Serve model predictions via API
10. Monitoring: Track data drift and model degradation

## Model Evaluation Metrics

### Classification Metrics
- Accuracy: Overall correct predictions (misleading with imbalanced classes)
- Precision: Of all positive predictions, how many are correct?
- Recall (Sensitivity): Of all actual positives, how many did we catch?
- F1-Score: Harmonic mean of Precision and Recall
- AUC-ROC: Area under the Receiver Operating Characteristic curve (threshold-independent)
- AUC-PR: More informative than AUC-ROC for imbalanced datasets

### Regression Metrics
- MAE: Mean Absolute Error (interpretable, robust to outliers)
- RMSE: Root Mean Squared Error (penalizes large errors)
- MAPE: Mean Absolute Percentage Error (scale-independent)
- R²: Coefficient of determination (proportion of variance explained)

## Regularization Techniques
Regularization prevents overfitting by constraining model complexity:
- L1 (Lasso): Drives some coefficients to exactly zero, enabling feature selection
- L2 (Ridge): Shrinks all coefficients, handles multicollinearity well
- ElasticNet: Combines L1 and L2 penalties
- Dropout: Randomly zeros neurons during neural network training
- Early Stopping: Halt training when validation loss stops improving

## Ensemble Methods
Combining multiple models typically outperforms any single model:
- Bagging (Bootstrap Aggregating): Train models on bootstrapped subsets (Random Forest)
- Boosting: Sequentially train models to correct predecessor errors (XGBoost, LightGBM, AdaBoost)
- Stacking: Train a meta-model on predictions from base models
- Voting: Combine predictions by majority vote or averaging

## Handling Imbalanced Data
Class imbalance is common in fraud detection, medical diagnosis, and churn prediction:
- Oversampling: SMOTE generates synthetic minority class samples
- Undersampling: Randomly remove majority class samples
- Class weights: Penalize misclassification of minority class more heavily
- Threshold tuning: Adjust decision boundary based on business cost-benefit analysis
""",

        "03_deep_learning_and_transformers.txt": """
# Deep Learning and Transformer Architecture

## Neural Network Fundamentals
A neural network consists of layers of interconnected neurons (nodes).
Each connection has a learnable weight; each neuron applies an activation function.
Training uses backpropagation with gradient descent to minimize a loss function.

### Key Activation Functions
- ReLU (Rectified Linear Unit): f(x) = max(0, x) — default choice for hidden layers
- Sigmoid: Maps to (0,1), used for binary output layers
- Softmax: Maps to probability distribution, used for multi-class output
- GELU: Smooth approximation of ReLU, used in modern transformers (BERT, GPT)
- SiLU/Swish: Self-gated activation, used in LLaMA and other modern models

## Convolutional Neural Networks (CNNs)
Designed for grid-structured data (images, time series).
- Convolutional layers detect local patterns (edges, textures, shapes)
- Pooling layers reduce spatial dimensions while retaining features
- Fully connected layers perform final classification/regression
Landmark architectures: AlexNet → VGG → ResNet → EfficientNet → Vision Transformer (ViT)

## Recurrent Neural Networks (RNNs) and LSTMs
Process sequential data by maintaining a hidden state across timesteps.
- Vanilla RNN: Suffers from vanishing gradient problem with long sequences
- LSTM (Long Short-Term Memory): Gates control information flow, solving vanishing gradients
- GRU (Gated Recurrent Unit): Simpler than LSTM, comparable performance
Applications: Time series, NLP (pre-transformer era), speech recognition

## The Transformer Architecture
Published in "Attention Is All You Need" (Vaswani et al., 2017), transformers revolutionized AI.

### Self-Attention Mechanism
Allows each token to attend to all other tokens in the sequence.
Computes Query (Q), Key (K), Value (V) matrices and calculates attention scores:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
Multi-head attention runs attention in parallel across multiple "heads" to capture different patterns.

### Transformer Components
1. Input Embeddings + Positional Encoding: Represent tokens and their positions
2. Multi-Head Self-Attention: Capture relationships between all tokens
3. Feed-Forward Network: Apply non-linear transformation to each position
4. Layer Normalization: Stabilize training
5. Residual Connections: Enable deeper networks without vanishing gradients

## Large Language Models (LLMs)
LLMs are transformers pre-trained on massive text corpora:
- GPT family (OpenAI): Decoder-only, autoregressive text generation
- BERT family (Google): Encoder-only, masked language modeling
- T5/PaLM/Gemini: Encoder-decoder or decoder-only at scale
- Claude (Anthropic): Constitutional AI training for safety and helpfulness
- LLaMA (Meta): Open-weights models enabling research and customization

### LLM Training Stages
1. Pre-training: Self-supervised learning on web-scale text (hundreds of billions of tokens)
2. Instruction Fine-tuning (SFT): Learn to follow instructions from human-curated examples
3. RLHF (Reinforcement Learning from Human Feedback): Align outputs with human preferences
4. Constitutional AI: Use AI feedback to ensure safety and harmlessness

## Fine-tuning vs RAG vs Prompting
- Zero/Few-shot Prompting: Use in-context examples; no training; fastest to deploy
- RAG: Retrieve relevant context dynamically; no weight updates; ideal for knowledge-intensive tasks
- Fine-tuning: Update model weights on domain-specific data; best for style/format adaptation
- LoRA/QLoRA: Parameter-efficient fine-tuning; update only low-rank adapter matrices

## Evaluation of Language Models
- Perplexity: Measures how well the model predicts a test set (lower = better)
- BLEU/ROUGE: N-gram overlap metrics for text generation
- BERTScore: Semantic similarity using contextual embeddings
- Human evaluation: Preferred for open-ended generation tasks
- Benchmark suites: MMLU, HellaSwag, TruthfulQA, HumanEval (coding)
""",

        "04_mlops_and_model_deployment.txt": """
# MLOps: Taking Machine Learning Models to Production

## What is MLOps?
MLOps (Machine Learning Operations) applies DevOps principles to machine learning systems.
It encompasses practices, tools, and processes for automating and monitoring ML in production.
The goal: reduce time from model training to production value while ensuring reliability.

## The ML Lifecycle
1. Data Engineering: Collection, validation, versioning, feature engineering
2. Model Development: Experimentation, training, evaluation, selection
3. Model Deployment: Packaging, serving, A/B testing, shadow mode
4. Production Monitoring: Data drift, model drift, performance degradation, alerts
5. Feedback Loops: Capture production data to retrain and improve models

## Key MLOps Components

### Experiment Tracking
Track all experiments to reproduce results and compare approaches.
- MLflow: Open-source platform for tracking runs, parameters, metrics, and artifacts
- Weights & Biases: Collaborative experiment tracking with rich visualizations
- Neptune.ai: Metadata store for ML experiments

### Data Versioning
- DVC (Data Version Control): Git for data and models; track large files in cloud storage
- Delta Lake: ACID transactions for data lakes
- Feature stores (Feast, Tecton): Centralize feature computation and serving

### Model Registry
A model registry provides versioning, lineage, and lifecycle management:
- MLflow Model Registry: Stage models (Staging → Production → Archived)
- Hugging Face Hub: For NLP and vision models
- Amazon SageMaker Model Registry: AWS-native registry with approval workflows

### Model Serving
- REST API (FastAPI + Uvicorn): Simple, flexible deployment
- Triton Inference Server (NVIDIA): High-performance multi-model serving
- TorchServe: PyTorch-native serving
- TensorFlow Serving: TensorFlow-native serving
- BentoML: Framework-agnostic model packaging and serving

### Containerization and Orchestration
- Docker: Package model with all dependencies for consistent environments
- Kubernetes: Orchestrate containerized model services for scalability
- Helm Charts: Kubernetes package manager for deployment configuration

## Model Monitoring: The Most Underrated MLOps Practice
Models degrade in production because the world changes. Monitor:
- Data Drift: Input feature distribution shifts from training distribution
  - Methods: KL divergence, Population Stability Index (PSI), Kolmogorov-Smirnov test
- Concept Drift: Relationship between features and target changes
  - Methods: Monitor prediction distribution, track accuracy on labeled samples
- Operational Metrics: Latency, throughput, error rates, memory usage

Tools: Evidently AI, Arize AI, WhyLabs, Seldon Alibi Detect

## CI/CD for Machine Learning

### Continuous Integration
- Automated testing: Unit tests for data processing, model loading, API endpoints
- Data validation: Great Expectations, Pandera ensure data quality
- Model validation: Automated checks against business metric thresholds before promotion

### Continuous Delivery
- Automated retraining pipelines triggered by drift detection or schedule
- Blue-green deployments: Run old and new models in parallel
- Canary releases: Route small % of traffic to new model
- Shadow mode: Run new model without serving its predictions, compare offline

## Feature Engineering at Scale
- Feature Stores provide a single source of truth for features
- Offline store: Historical features for training (data warehouse/lake)
- Online store: Low-latency feature retrieval for real-time serving (Redis, DynamoDB)
- Feature sharing across teams prevents redundant computation

## Recommended MLOps Stack (2024-2026)
- Orchestration: Apache Airflow, Prefect, or Dagster
- Training: Ray Train, Spark ML, or cloud-managed training (SageMaker, Vertex AI)
- Tracking: MLflow or W&B
- Serving: FastAPI + Docker + Kubernetes or cloud-managed endpoints
- Monitoring: Evidently AI + Grafana
- Data versioning: DVC
- Feature store: Feast (open-source) or Tecton (enterprise)

## Common MLOps Anti-Patterns
- Training-serving skew: Features computed differently at training vs inference time
- No monitoring: Models deployed without any performance tracking
- Manual deployment: Slow, error-prone, non-reproducible model releases
- Single environment: No separation between dev, staging, and production
- No rollback plan: Unable to quickly revert a bad model deployment
""",

        "05_data_science_career_guide.txt": """
# Data Science Career Guide: Skills, Roles, and Interview Preparation

## The Modern Data Science Landscape
Data science has evolved significantly. In 2025-2026, the field branches into:
- ML Engineer: Production focus, software engineering + ML
- Research Scientist: Algorithms, publications, novel architectures
- Data Scientist: Business insights, experimentation, modeling
- AI Engineer: LLM applications, RAG systems, prompt engineering
- MLOps Engineer: Pipelines, infrastructure, monitoring

## Core Technical Skills

### Programming & Tools
- Python: Pandas, NumPy, Scikit-learn, PyTorch/TensorFlow, FastAPI
- SQL: Window functions, CTEs, query optimization, analytical queries
- Spark: Distributed data processing for large-scale datasets
- Cloud: AWS (SageMaker, S3, Lambda), GCP (Vertex AI, BigQuery), Azure ML
- Version Control: Git, GitHub/GitLab, branching strategies

### Statistics & Mathematics
- Probability Theory: Bayes theorem, distributions, joint/marginal/conditional probability
- Statistical Inference: Hypothesis testing, p-values, confidence intervals, power analysis
- Linear Algebra: Matrix operations, eigendecomposition, SVD (PCA foundation)
- Calculus: Gradient descent, partial derivatives (backpropagation foundation)
- Experimental Design: A/B testing, multi-armed bandits, causal inference

### Machine Learning
- Supervised: Regression, classification, ensemble methods
- Unsupervised: Clustering, anomaly detection, dimensionality reduction
- Deep Learning: CNNs, RNNs, Transformers, transfer learning
- NLP: Text preprocessing, embeddings, LLMs, RAG
- Time Series: ARIMA, Prophet, LSTM, N-BEATS

## High-Demand Skills in 2025-2026
1. LLM Engineering: Prompt engineering, RAG, fine-tuning, evaluation
2. MLOps: CI/CD for ML, model monitoring, feature stores
3. Causal Inference: Beyond correlation, A/B testing at scale
4. Responsible AI: Fairness, bias detection, explainability (SHAP, LIME)
5. Real-time ML: Streaming data, online learning, low-latency serving

## Portfolio Project Recommendations
Strong portfolios demonstrate business impact and technical depth:
- RAG System: Build a document Q&A platform using LangChain/custom RAG
- MLOps Pipeline: End-to-end pipeline with tracking, serving, and monitoring
- Time Series Forecasting: Multi-model forecasting with uncertainty quantification
- NLP Classification: Fine-tuned transformer for domain-specific text classification
- Recommendation System: Collaborative filtering with real or synthetic data

## Interview Preparation

### ML Concepts Questions
- Explain the bias-variance tradeoff
- When would you use L1 vs L2 regularization?
- How does gradient boosting work?
- What is the curse of dimensionality?
- Explain the attention mechanism in transformers

### Coding Interviews
- Implement gradient descent from scratch
- Write a custom cross-validation function
- Implement K-Means clustering
- Build a simple neural network with NumPy
- SQL window functions and analytical queries

### Case Studies
- Design an ML system for fraud detection at Stripe
- How would you build a recommendation system for Netflix?
- Diagnose why a deployed model's performance degraded last week

## Salary Benchmarks (US Market, 2025-2026)
- Junior Data Scientist (0-2 years): $90K-$130K
- Mid-level Data Scientist (3-5 years): $130K-$180K
- Senior Data Scientist (6+ years): $180K-$250K
- Staff/Principal Data Scientist: $250K-$400K+
- ML Engineer (Mid): $140K-$200K
- Research Scientist (Top Labs): $200K-$500K+

## Key Certifications (2025)
- AWS Certified Machine Learning - Specialty
- Google Professional Machine Learning Engineer
- TensorFlow Developer Certificate
- Databricks Certified ML Professional
"""
    }

    for filename, content in samples.items():
        (sample_dir / filename).write_text(content.strip(), encoding="utf-8")

    logger.info(f"Generated {len(samples)} sample documents in '{sample_dir}'")
    return list(samples.keys())


def run_demo(pipeline, sample_questions: list[str]):
    """Run a demo query session."""
    print("\n" + "=" * 70)
    print("  DocuMind AI — Live Demo")
    print("=" * 70)

    for i, question in enumerate(sample_questions, 1):
        print(f"\n[Query {i}] {question}")
        print("-" * 60)
        try:
            result = pipeline.query(question)
            print(f"Answer:\n{result.answer}")
            print(f"\nSources: {', '.join(result.unique_sources)}")
            print(f"Latency: {result.latency_seconds:.2f}s | Tokens: {result.input_tokens + result.output_tokens}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="DocuMind AI — Document Intelligence Platform")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample documents only")
    parser.add_argument("--ingest-only", action="store_true", help="Ingest sample docs without running queries")
    parser.add_argument("--no-demo", action="store_true", help="Skip the demo queries")
    args = parser.parse_args()

    print("\n🧠 DocuMind AI — Enterprise Document Intelligence Platform")
    print("=" * 70)

    # 1. Generate sample documents
    print("\n[1/4] Generating sample documents...")
    filenames = generate_sample_documents()
    for f in filenames:
        print(f"  ✅ {f}")

    if args.generate_samples:
        print("\nSample documents generated. Run 'streamlit run app/streamlit_app.py' to explore.")
        return

    # 2. Load configuration
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # 3. Initialize pipeline
    print("\n[2/4] Initializing AI pipeline...")
    from src.embeddings.embedder import Embedder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.generation.llm_client import GrokClient
    from src.generation.rag_pipeline import RAGPipeline
    from src.ingestion.document_loader import load_documents_from_directory
    from src.ingestion.text_chunker import RecursiveTextChunker
    from src.retrieval.retriever import SemanticRetriever

    embedder = Embedder(
        model_name=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"]["batch_size"],
    )
    vector_store = FAISSVectorStore(
        dimension=cfg["embeddings"]["dimension"],
        store_path=cfg["vector_store"]["path"],
    )
    vector_store.load()
    print("  ✅ Embedder and vector store ready")

    # 4. Ingest documents
    print("\n[3/4] Ingesting sample documents...")
    chunker = RecursiveTextChunker(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        min_chunk_size=cfg["chunking"]["min_chunk_size"],
    )

    docs = load_documents_from_directory("data/sample_docs")
    total_chunks = 0
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        embeddings = embedder.embed([c.content for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
        vector_store.register_document(doc.id, {
            "document_id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunk_count": len(chunks),
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "content": doc.content,
        })
        total_chunks += len(chunks)
        print(f"  ✅ {doc.filename} → {len(chunks)} chunks")

    vector_store.save()
    print(f"\n  Total: {len(docs)} documents, {total_chunks} chunks indexed")

    if args.ingest_only:
        print("\nIngestion complete. Run 'streamlit run app/streamlit_app.py' to explore.")
        return

    # 5. Demo queries
    if not args.no_demo:
        llm = GrokClient(
            model=cfg["llm"]["model"],
            max_tokens=cfg["llm"]["max_tokens"],
        )
        if not llm.is_available():
            print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping demo queries.")
            print("   Set it in your .env file and re-run for the full demo.")
        else:
            retriever = SemanticRetriever(embedder=embedder, vector_store=vector_store)
            pipeline = RAGPipeline(retriever=retriever, llm=llm)

            print("\n[4/4] Running demo queries...")
            demo_questions = [
                "What is RAG and what are its main components?",
                "How do I handle class imbalance in machine learning?",
                "What are the key MLOps practices for production ML?",
            ]
            run_demo(pipeline, demo_questions)

    print("\n" + "=" * 70)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Start the Streamlit UI:  streamlit run app/streamlit_app.py")
    print("  2. Start the API server:    uvicorn src.api.main:app --reload --port 8000")
    print("  3. API docs available at:  http://localhost:8000/docs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
