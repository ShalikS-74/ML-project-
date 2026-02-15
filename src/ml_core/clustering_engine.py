import logging
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    COVERAGE_WEIGHT,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    FREQUENCY_WEIGHT,
    WEIGHTED_WEIGHT,
)


class ClusteringEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.eps = DBSCAN_EPS
        self.min_samples = DBSCAN_MIN_SAMPLES

    def apply_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply DBSCAN clustering with cosine distance."""
        if embeddings.size == 0:
            return np.array([])
        if len(embeddings) == 1:
            return np.array([0], dtype=int)

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine",
        )
        cluster_labels = clustering.fit_predict(embeddings)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = int(np.sum(cluster_labels == -1))
        self.logger.info(f"DBSCAN found {n_clusters} clusters, {n_noise} outliers")
        return cluster_labels

    def generate_cluster_stats(self, cluster_labels: np.ndarray, questions: List[Dict]) -> List[Dict]:
        """Generate cluster statistics with multi-metric scoring."""
        if cluster_labels.size == 0:
            return []

        clusters: List[Dict] = []
        unique_labels = set(cluster_labels.tolist())
        if -1 in unique_labels:
            unique_labels.remove(-1)

        total_papers = max(len(set(q["paper"] for q in questions)), 1)
        total_questions = len(questions)

        for cluster_id in sorted(unique_labels):
            cluster_questions = [q for i, q in enumerate(questions) if cluster_labels[i] == cluster_id]
            if not cluster_questions:
                continue

            frequency = len(cluster_questions)
            papers_in_cluster = sorted(set(q["paper"] for q in cluster_questions))
            coverage_ratio = len(papers_in_cluster) / total_papers
            total_marks = sum((q.get("marks", 0) or 0) for q in cluster_questions)
            avg_marks = total_marks / frequency if frequency else 0
            topic_keywords = self._extract_topic_keywords(cluster_questions)

            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "questions": cluster_questions,
                    "frequency": frequency,
                    "coverage_ratio": coverage_ratio,
                    "papers_covered": papers_in_cluster,
                    "avg_marks": avg_marks,
                    "total_marks": total_marks,
                    "topic_keywords": topic_keywords,
                }
            )

        return self._calculate_trend_scores(clusters, total_questions)

    def _calculate_trend_scores(self, clusters: List[Dict], total_questions: int) -> List[Dict]:
        """TrendScore = 0.5*Coverage + 0.3*Frequency + 0.2*Weighted."""
        if not clusters:
            return []

        max_frequency = max(c["frequency"] for c in clusters) or 1
        max_weighted = max((c["frequency"] * c["avg_marks"] for c in clusters), default=1) or 1

        for cluster in clusters:
            freq_normalized = cluster["frequency"] / max_frequency
            weighted_normalized = (cluster["frequency"] * cluster["avg_marks"]) / max_weighted
            coverage_normalized = cluster["coverage_ratio"]

            trend_score = (
                COVERAGE_WEIGHT * coverage_normalized
                + FREQUENCY_WEIGHT * freq_normalized
                + WEIGHTED_WEIGHT * weighted_normalized
            )

            cluster["trend_score"] = trend_score
            cluster["frequency_rate"] = cluster["frequency"] / max(total_questions, 1)

        return sorted(clusters, key=lambda c: c["trend_score"], reverse=True)

    def _extract_topic_keywords(self, questions: List[Dict]) -> List[str]:
        """Extract representative keywords using TF-IDF."""
        texts = [q.get("text", "").strip() for q in questions if q.get("text")]
        if not texts:
            return []

        if len(texts) < 2:
            return [word for word in texts[0].split()[:3] if len(word) > 3]

        try:
            vectorizer = TfidfVectorizer(
                max_features=3,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-3:][::-1]
            return [feature_names[i] for i in top_indices]
        except Exception:
            return ["topic", "cluster", str(len(questions))]

    def tune_eps_parameter(self, embeddings: np.ndarray, questions: List[Dict]) -> float:
        """Tune DBSCAN eps parameter for practical cluster balance."""
        del questions
        if embeddings.size == 0:
            return self.eps

        eps_values = [0.15, 0.20, 0.25, 0.30, 0.35]
        best_eps = self.eps
        best_score = float("-inf")

        for eps in eps_values:
            self.eps = eps
            labels = self.apply_clustering(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            score = n_clusters - (n_noise * 0.1)
            if score > best_score:
                best_score = score
                best_eps = eps

        self.eps = best_eps
        self.logger.info(f"Tuned eps to {best_eps}")
        return best_eps

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Utility for diagnostics."""
        if embeddings.size == 0:
            return np.array([])
        return cosine_similarity(embeddings)
