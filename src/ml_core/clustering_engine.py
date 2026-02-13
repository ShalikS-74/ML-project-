import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CLUSTERING_METHOD, SIMILARITY_THRESHOLD


class ClusteringEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.clustering_method = CLUSTERING_METHOD

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        if embeddings.size == 0:
            return np.array([])
        return cosine_similarity(embeddings)

    def apply_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply clustering algorithm."""
        if embeddings.size == 0:
            return np.array([])
        if len(embeddings) == 1:
            return np.array([0], dtype=int)

        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)

        if self.clustering_method == "dbscan":
            model = DBSCAN(metric="precomputed", eps=(1.0 - self.similarity_threshold), min_samples=2)
            labels = model.fit_predict(distance_matrix)
        else:
            # Convert similarity threshold into distance threshold.
            distance_threshold = 1.0 - self.similarity_threshold
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = model.fit_predict(distance_matrix)

        self.logger.info(f"Found {len(set(labels))} clusters")
        return labels

    def tune_threshold(self, embeddings: np.ndarray, questions: List[Dict]) -> float:
        """Tune clustering threshold based on cohesion/coverage balance."""
        del questions  # Reserved for future supervised tuning.
        if embeddings.size == 0:
            return self.similarity_threshold

        thresholds = np.arange(0.75, 0.86, 0.02)
        original = self.similarity_threshold
        best_threshold = original
        best_score = float("-inf")

        for threshold in thresholds:
            self.similarity_threshold = float(threshold)
            labels = self.apply_clustering(embeddings)
            score = self.evaluate_clustering_quality(labels, embeddings)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        self.similarity_threshold = best_threshold
        self.logger.info(f"Tuned similarity threshold to {best_threshold:.2f}")
        return best_threshold

    def evaluate_clustering_quality(self, labels: np.ndarray, embeddings: np.ndarray) -> float:
        """Heuristic quality: reward cohesion and moderate cluster counts."""
        if labels.size == 0:
            return 0.0

        similarity = self.calculate_similarity_matrix(embeddings)
        unique_labels = sorted(set(labels.tolist()))
        total_points = len(labels)

        cohesion_sum = 0.0
        coverage = 0

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_size = len(indices)
            if cluster_size < 2:
                continue
            coverage += cluster_size
            cluster_sim = similarity[np.ix_(indices, indices)]
            upper = cluster_sim[np.triu_indices(cluster_size, k=1)]
            if upper.size > 0:
                cohesion_sum += float(np.mean(upper)) * cluster_size

        coverage_ratio = coverage / total_points
        avg_cohesion = cohesion_sum / max(coverage, 1)
        cluster_penalty = len(unique_labels) / max(total_points, 1)
        return (0.7 * avg_cohesion) + (0.3 * coverage_ratio) - (0.1 * cluster_penalty)

    def generate_cluster_stats(
        self,
        clusters: np.ndarray,
        questions: List[Dict],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Generate cluster statistics and metadata."""
        if clusters.size == 0:
            return []

        cluster_stats: List[Dict] = []
        for cluster_id in sorted(set(clusters.tolist())):
            indices = np.where(clusters == cluster_id)[0]
            cluster_questions = [questions[i] for i in indices]
            marks = [q.get("marks", 0) or 0 for q in cluster_questions]

            centroid = None
            similarity_score = 0.0
            if embeddings is not None and len(indices) > 0:
                cluster_embeddings = embeddings[indices]
                centroid_vec = np.mean(cluster_embeddings, axis=0)
                centroid = centroid_vec.tolist()
                if len(indices) > 1:
                    sim = cosine_similarity(cluster_embeddings)
                    upper = sim[np.triu_indices(len(indices), k=1)]
                    similarity_score = float(np.mean(upper)) if upper.size > 0 else 1.0
                else:
                    similarity_score = 1.0

            cluster_stats.append(
                {
                    "cluster_id": int(cluster_id),
                    "questions": cluster_questions,
                    "centroid": centroid,
                    "size": len(cluster_questions),
                    "avg_marks": float(np.mean(marks)) if marks else 0.0,
                    "similarity_score": similarity_score,
                    "topic_keywords": self.extract_topic_keywords(cluster_questions),
                }
            )

        return cluster_stats

    def extract_topic_keywords(self, questions: List[Dict]) -> List[str]:
        """Extract representative keywords using TF-IDF."""
        texts = [q.get("text", "").strip() for q in questions if q.get("text")]
        if not texts:
            return []

        try:
            vectorizer = TfidfVectorizer(max_features=5, stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                return []
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-5:][::-1]
            return [feature_names[i] for i in top_indices]
        except ValueError:
            return []

    def calculate_weighted_trends(self, clusters: List[Dict]) -> Dict:
        """Calculate frequency and marks-weighted trends."""
        total_questions = sum(len(cluster.get("questions", [])) for cluster in clusters)
        if total_questions == 0:
            return {}

        trends = {}
        for cluster in clusters:
            topic_name = (
                cluster.get("topic_keywords", [f"Topic_{cluster['cluster_id']}"])[0]
                if cluster.get("topic_keywords")
                else f"Topic_{cluster['cluster_id']}"
            )
            question_count = len(cluster.get("questions", []))
            frequency = question_count / total_questions
            total_marks = sum((q.get("marks", 0) or 0) for q in cluster.get("questions", []))
            avg_marks = total_marks / question_count if question_count else 0
            weighted_score = frequency * avg_marks
            trends[topic_name] = {
                "frequency": frequency,
                "avg_marks": avg_marks,
                "weighted_score": weighted_score,
                "question_count": question_count,
                "total_marks": total_marks,
            }

        return dict(sorted(trends.items(), key=lambda item: item[1]["weighted_score"], reverse=True))
