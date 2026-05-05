from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np

from src.models.paper import Paper
from src.models.schema import GraphNode, NodeLabel, GraphEdge, EdgeType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrendDetector:
    def __init__(self, min_cluster_size: int = 5):
        self.min_cluster_size = min_cluster_size

    def detect_clusters(self, nodes: List[GraphNode]) -> List[dict]:
        paper_nodes = [n for n in nodes if n.label == NodeLabel.PAPER]

        if len(paper_nodes) < self.min_cluster_size:
            logger.warning(f"节点数 {len(paper_nodes)} 不足，跳过趋势检测")
            return []

        adjacency = self._build_adjacency_matrix(paper_nodes)

        if adjacency is None or adjacency.sum() == 0:
            return self._keyword_based_clustering(paper_nodes)

        try:
            clusters = self._louvain_clustering(adjacency, paper_nodes)
        except Exception as e:
            logger.warning(f"Louvain 聚类失败: {e}，使用关键词聚类降级方案")
            clusters = self._keyword_based_clustering(paper_nodes)

        return self._enrich_clusters(clusters)

    def _build_adjacency_matrix(
        self, nodes: List[GraphNode]
    ) -> Optional[np.ndarray]:
        n = len(nodes)
        adj = np.zeros((n, n))
        id_to_idx = {node.id: i for i, node in enumerate(nodes)}

        has_edges = False
        for node in nodes:
            if node.embedding is not None:
                has_edges = True
                break

        if not has_edges:
            for i in range(n):
                for j in range(i + 1, n):
                    keywords_i = set(
                        nodes[i].properties.get("keywords", [])
                    )
                    keywords_j = set(
                        nodes[j].properties.get("keywords", [])
                    )
                    if keywords_i and keywords_j:
                        overlap = len(keywords_i & keywords_j)
                        total = len(keywords_i | keywords_j)
                        if total > 0:
                            adj[i, j] = adj[j, i] = overlap / total

        return adj if adj.sum() > 0 else None

    def _louvain_clustering(
        self,
        adjacency: np.ndarray,
        nodes: List[GraphNode],
    ) -> List[dict]:
        m = adjacency.sum()

        communities = {i: i for i in range(len(nodes))}
        improved = True

        while improved:
            improved = False
            for i in range(len(nodes)):
                current_comm = communities[i]
                best_comm = current_comm
                best_gain = 0

                neighbor_comms = set()
                for j in range(len(nodes)):
                    if adjacency[i, j] > 0:
                        neighbor_comms.add(communities[j])

                for new_comm in neighbor_comms:
                    if new_comm == current_comm:
                        continue

                    k_i = adjacency[i].sum()
                    k_i_in = sum(
                        adjacency[i, j]
                        for j in range(len(nodes))
                        if communities[j] == new_comm
                    )
                    sigma_tot = sum(
                        adjacency[k].sum()
                        for k in range(len(nodes))
                        if communities[k] == new_comm
                    )

                    gain = k_i_in / m - k_i * sigma_tot / (m * m) if m > 0 else 0
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = new_comm
                        improved = True

                if best_comm != current_comm:
                    communities[i] = best_comm

        clusters = defaultdict(list)
        for idx, comm in communities.items():
            clusters[comm].append(nodes[idx])

        return [
            {
                "nodes": nodes_list,
                "size": len(nodes_list),
                "label": self._generate_cluster_label(nodes_list),
            }
            for nodes_list in clusters.values()
            if len(nodes_list) >= self.min_cluster_size
        ]

    def _keyword_based_clustering(
        self, nodes: List[GraphNode]
    ) -> List[dict]:
        keyword_groups: Dict[str, List[GraphNode]] = defaultdict(list)

        for node in nodes:
            title = node.properties.get("title", "").lower()
            abstract = node.properties.get("abstract", "").lower()
            combined = title + " " + abstract

            matched = False
            import re

            category_keywords = {
                "ml_architectures": [
                    "transformer", "attention", "sequence model", "state space",
                    "mamba", "rnn", "lstm", "cnn",
                ],
                "nlp": [
                    "language model", "llm", "nlp", "text generation",
                    "machine translation", "summarization",
                ],
                "rl": [
                    "reinforcement learning", "policy gradient", "q-learning",
                    "actor-critic", "rlhf",
                ],
                "cv": [
                    "image", "vision", "segmentation", "object detection",
                    "diffusion model",
                ],
                "optimization": [
                    "optimization", "gradient descent", "adam", "sgd",
                    "convergence", "loss landscape",
                ],
                "graph": [
                    "graph neural", "gnn", "graph attention", "node embedding",
                    "knowledge graph",
                ],
                "safety": [
                    "safety", "alignment", "bias", "fairness", "ethics",
                    "interpretability", "explainability",
                ],
                "efficiency": [
                    "efficiency", "quantization", "pruning", "distillation",
                    "inference speed", "latency",
                ],
            }

            for category, keywords in category_keywords.items():
                if any(kw in combined for kw in keywords):
                    keyword_groups[category].append(node)
                    matched = True
                    break

            if not matched:
                keyword_groups["other"].append(node)

        return [
            {
                "nodes": nodes_list,
                "size": len(nodes_list),
                "label": self._generate_cluster_label(nodes_list),
            }
            for nodes_list in keyword_groups.values()
            if len(nodes_list) >= max(1, self.min_cluster_size // 2)
        ]

    def _generate_cluster_label(self, nodes: List[GraphNode]) -> str:
        if not nodes:
            return "Unknown"

        titles = [n.properties.get("title", "") for n in nodes]
        all_words = " ".join(titles).lower()

        import re
        words = re.findall(r'[a-z]+', all_words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "now", "also", "using", "via",
            "based", "new", "via", "towards",
        }

        word_freq = defaultdict(int)
        for w in words:
            if w not in stop_words and len(w) > 2:
                word_freq[w] += 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        return " + ".join(w for w, _ in top_words).title() if top_words else "Emerging Trend"

    def _enrich_clusters(self, clusters: List[dict]) -> List[dict]:
        for cluster in clusters:
            nodes = cluster["nodes"]
            cluster["top_papers"] = [
                {
                    "id": n.id,
                    "title": n.properties.get("title", "Unknown"),
                    "year": n.properties.get("year", "Unknown"),
                    "venue": n.properties.get("venue", ""),
                }
                for n in sorted(
                    nodes,
                    key=lambda n: n.properties.get("screening_score", 0),
                    reverse=True,
                )[:5]
            ]

            all_keywords = []
            for n in nodes:
                kw = n.properties.get("keywords", [])
                if isinstance(kw, list):
                    all_keywords.extend(kw)

            from collections import Counter
            kw_counter = Counter(all_keywords)
            cluster["keywords"] = [kw for kw, _ in kw_counter.most_common(5)]

            cluster["growth_pct"] = self._estimate_growth(nodes)

        return clusters

    def _estimate_growth(self, nodes: List[GraphNode]) -> float:
        import time
        from datetime import datetime

        now = datetime.now()
        recent_count = 0
        older_count = 0

        for node in nodes:
            created = node.created_at
            if created is None:
                older_count += 1
                continue
            days_ago = (now - created).days
            if days_ago <= 30:
                recent_count += 1
            elif days_ago <= 90:
                older_count += 1

        if older_count == 0:
            return 100.0 if recent_count > 0 else 0.0
        return round((recent_count / older_count) * 100, 1)

    def assign_papers_to_clusters(
        self,
        papers: List[Paper],
        clusters: List[dict],
    ) -> List[dict]:
        for paper in papers:
            best_cluster = None
            best_score = 0

            paper_text = (paper.title + " " + paper.abstract).lower()

            for cluster in clusters:
                keywords = cluster.get("keywords", [])
                score = sum(
                    1 for kw in keywords
                    if kw.lower() in paper_text
                )
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster and best_score > 0:
                best_cluster.setdefault("assigned_papers", []).append(
                    {
                        "id": paper.id,
                        "title": paper.title,
                    }
                )

        return clusters
