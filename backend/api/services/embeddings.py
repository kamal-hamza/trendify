"""
Vector embedding service for semantic entity resolution.
Uses FREE local alternatives:
- sentence-transformers for embeddings (no API key needed)
- Ollama for LLM resolution (runs locally, completely free)
"""

import json
import os
from typing import List, Optional

import numpy as np
import requests


class EmbeddingService:
    """
    Generate vector embeddings for semantic similarity matching.
    Uses sentence-transformers (local, free, no API key needed).
    """

    def __init__(self):
        """Initialize the embedding service with sentence-transformers."""
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self._init_sentence_transformers()

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers model (local)"""
        try:
            from sentence_transformers import SentenceTransformer

            # Use a fast, lightweight model (384 dimensions)
            model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(model_name)
            self.dimension = 384
            print(f"Loaded sentence-transformers model: {model_name}")

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the vector embedding
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient).

        Args:
            texts: List of input texts

        Returns:
            List of vector embeddings
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t if t and t.strip() else " " for t in texts]

        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1 (typically 0 to 1 for embeddings)
        """
        if not vec1 or not vec2:
            return 0.0

        if len(vec1) != len(vec2):
            raise ValueError(
                f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}"
            )

        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)

    @staticmethod
    def find_similar(
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        threshold: float = 0.8,
    ) -> List[tuple]:
        """
        Find candidates similar to query above a threshold.

        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate vectors
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        similarities = []

        for idx, candidate in enumerate(candidate_embeddings):
            similarity = EmbeddingService.cosine_similarity(
                query_embedding, candidate
            )
            if similarity >= threshold:
                similarities.append((idx, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


class LLMResolver:
    """
    LLM-based entity resolution using OpenRouter's free models.
    No installation required - uses cloud API with free tier.
    Get API key: https://openrouter.ai/keys
    """

    def __init__(self, api_key: str = None):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "openrouter/auto"  # Automatically routes to free models
        self.timeout = 30

    def is_available(self) -> bool:
        """Check if OpenRouter API key is configured"""
        return bool(self.api_key)

    def resolve_entities(self, keywords: List[str]) -> dict:
        """
        Use OpenRouter to identify which keywords are synonyms and group them.

        Args:
            keywords: List of keyword strings to resolve

        Returns:
            Dictionary mapping canonical names to lists of synonyms
            Example: {
                "Claude": ["Claude 4.6", "Sonnet 4.6", "Opus 4.6", "Anthropic 4.6"],
                "OpenClaw": ["OpenClaw", "Open-Claw", "Clawd"]
            }
        """
        if not keywords:
            return {}

        # Check if API key is configured
        if not self.is_available():
            print(
                "Warning: OPENROUTER_API_KEY not set. "
                "Get a free API key at https://openrouter.ai/keys"
            )
            return {}

        prompt = self._build_resolution_prompt(keywords)

        try:
            response = self._call_openrouter(prompt)

            # Parse the JSON response
            # Try to extract JSON from response (LLM might add extra text)
            result = self._extract_json(response)
            return result

        except Exception as e:
            print(f"LLM resolution error: {e}")
            return {}

    def _build_resolution_prompt(self, keywords: List[str]) -> str:
        """Build the prompt for entity resolution"""
        keywords_text = "\n".join([f"- {kw}" for kw in keywords])

        return f"""You are an entity resolution system for a tech trend tracking platform.

Given this list of keywords extracted from tech posts:

{keywords_text}

Your task:
1. Identify which keywords refer to the same technology/entity (synonyms, typos, variations)
2. Group them under a canonical name (the most common or official name)
3. Identify which are completely distinct entities

Return ONLY a JSON object in this format:
{{
  "canonical_name_1": ["synonym1", "synonym2", "variation1"],
  "canonical_name_2": ["synonym3", "typo1"],
  "standalone_name": ["standalone_name"]
}}

Rules:
- Use the most official/common name as the canonical key
- Include the canonical name in its own synonym list
- Different versions (e.g., "Claude 3.5" vs "Claude 4.6") should be grouped under the parent (e.g., "Claude")
- Be conservative: only group if you're confident they're the same thing

Return ONLY valid JSON, no explanation."""

    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API using free models"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,  # "openrouter/auto" routes to free models
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise entity resolution system. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response (might have extra text)"""
        # Try to find JSON in the response
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            # No JSON found, return empty dict
            return {}

        json_text = text[start_idx:end_idx]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try to clean up common issues
            json_text = json_text.replace("'", '"')  # Single to double quotes
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return {}


class HybridResolver:
    """
    Combines vector-based clustering with optional LLM resolution.
    Falls back gracefully if LLM is not available.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm_resolver = LLMResolver()

    def resolve_topics_to_entities(
        self, topics: List[str], similarity_threshold: float = 0.85
    ) -> dict:
        """
        Resolve topics using vector clustering first, then LLM for ambiguous cases.

        Args:
            topics: List of topic names
            similarity_threshold: Minimum cosine similarity to consider synonyms

        Returns:
            Dictionary mapping canonical names to lists of synonyms
        """
        if not topics:
            return {}

        # Step 1: Generate embeddings for all topics
        embeddings = self.embedding_service.generate_embeddings_batch(topics)

        # Step 2: Cluster by vector similarity
        clusters = self._cluster_by_similarity(
            topics, embeddings, threshold=similarity_threshold
        )

        # Step 3: If LLM is available, refine ambiguous clusters
        if self.llm_resolver.is_available() and len(clusters) > 1:
            # Get the first topic from each cluster for LLM resolution
            cluster_representatives = [cluster[0] for cluster in clusters]

            # Only use LLM if we have a reasonable number of clusters
            if 2 <= len(cluster_representatives) <= 50:
                try:
                    llm_result = self.llm_resolver.resolve_entities(cluster_representatives)

                    if llm_result:
                        # Merge vector clusters with LLM results
                        return self._merge_results(clusters, llm_result)
                except Exception as e:
                    print(f"LLM resolution failed: {e}, falling back to vector clustering")

        # Step 4: Return vector-based clusters if LLM unavailable
        result = {}
        for cluster in clusters:
            canonical = cluster[0]  # Use first as canonical
            result[canonical] = cluster

        return result

    def _cluster_by_similarity(
        self, topics: List[str], embeddings: List[List[float]], threshold: float
    ) -> List[List[str]]:
        """
        Cluster topics by vector similarity using agglomerative approach.

        Args:
            topics: List of topic names
            embeddings: Corresponding embeddings
            threshold: Similarity threshold

        Returns:
            List of clusters (each cluster is a list of topic names)
        """
        n = len(topics)
        clusters = [[i] for i in range(n)]  # Start with each topic in its own cluster

        # Calculate all pairwise similarities
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.embedding_service.cosine_similarity(
                    embeddings[i], embeddings[j]
                )
                similarities[i][j] = sim
                similarities[j][i] = sim

        # Agglomerative clustering
        while True:
            # Find the most similar pair of clusters
            max_sim = -1
            best_pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate average similarity between clusters
                    cluster_i = clusters[i]
                    cluster_j = clusters[j]

                    avg_sim = np.mean(
                        [similarities[ci][cj] for ci in cluster_i for cj in cluster_j]
                    )

                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        best_pair = (i, j)

            # Stop if no pair exceeds threshold
            if max_sim < threshold or best_pair is None:
                break

            # Merge the best pair
            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        # Convert indices to topic names
        result_clusters = []
        for cluster_indices in clusters:
            cluster_topics = [topics[idx] for idx in cluster_indices]
            result_clusters.append(cluster_topics)

        return result_clusters

    def _merge_results(
        self, vector_clusters: List[List[str]], llm_result: dict
    ) -> dict:
        """
        Merge vector-based clusters with LLM resolution results.

        Args:
            vector_clusters: Clusters from vector similarity
            llm_result: Entity groupings from LLM

        Returns:
            Combined result dictionary
        """
        # For simplicity, prefer LLM results but include vector clusters
        # that weren't resolved by LLM
        merged = dict(llm_result)

        # Track which topics were handled by LLM
        llm_topics = set()
        for synonyms in llm_result.values():
            llm_topics.update(synonyms)

        # Add vector clusters that weren't handled by LLM
        for cluster in vector_clusters:
            if not any(topic in llm_topics for topic in cluster):
                canonical = cluster[0]
                merged[canonical] = cluster

        return merged


# Singleton instances for reuse
_embedding_service: Optional[EmbeddingService] = None
_llm_resolver: Optional[LLMResolver] = None
_hybrid_resolver: Optional[HybridResolver] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service


def get_llm_resolver() -> LLMResolver:
    """Get or create LLM resolver singleton"""
    global _llm_resolver

    if _llm_resolver is None:
        _llm_resolver = LLMResolver()

    return _llm_resolver


def get_hybrid_resolver() -> HybridResolver:
    """Get or create hybrid resolver singleton"""
    global _hybrid_resolver

    if _hybrid_resolver is None:
        _hybrid_resolver = HybridResolver()

    return _hybrid_resolver