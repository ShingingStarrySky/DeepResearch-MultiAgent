from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional, Any

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.models.paper import Paper
from src.models.schema import (
    GraphNode,
    GraphEdge,
    NodeLabel,
    EdgeType,
    InterestProfile,
    DebateRecord,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraphManager:
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j = neo4j_client
        self._memory_nodes: Dict[str, GraphNode] = {}
        self._memory_edges: List[GraphEdge] = []
        self._use_memory = neo4j_client is None or neo4j_client.driver is None

    def add_paper_node(self, paper: Paper) -> GraphNode:
        node = GraphNode(
            id=paper.id,
            label=NodeLabel.PAPER,
            properties={
                "title": paper.title,
                "abstract": paper.abstract[:500],
                "year": paper.year,
                "venue": paper.venue,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "screening_score": paper.screening_score,
                "has_conflicts": paper.has_conflicts,
            },
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        if self._use_memory:
            self._memory_nodes[paper.id] = node
            return node

        query = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.year = $year,
            p.venue = $venue,
            p.doi = $doi,
            p.arxiv_id = $arxiv_id,
            p.screening_score = $screening_score,
            p.has_conflicts = $has_conflicts,
            p.last_updated = datetime()
        """
        self.neo4j.execute_write(query, node.properties)
        return node

    def add_claim_nodes(self, paper_id: str, claims: List):
        for claim in claims:
            node_id = f"claim_{paper_id}_{claim.id}"
            properties = {
                "content": claim.content,
                "verification": claim.verification.value,
                "evidence": claim.evidence or "",
                "confidence": claim.confidence,
            }

            if self._use_memory:
                self._memory_nodes[node_id] = GraphNode(
                    id=node_id,
                    label=NodeLabel.CLAIM,
                    properties=properties,
                    created_at=datetime.now(),
                )
                edge = GraphEdge(
                    source_id=paper_id,
                    target_id=node_id,
                    type=EdgeType.CONTAINS_CLAIM,
                    weight=claim.confidence,
                    created_at=datetime.now(),
                )
                self._memory_edges.append(edge)
            else:
                query = """
                MERGE (c:Claim {id: $id})
                SET c.content = $content,
                    c.verification = $verification,
                    c.evidence = $evidence,
                    c.confidence = $confidence
                WITH c
                MATCH (p:Paper {id: $paper_id})
                MERGE (p)-[:CONTAINS_CLAIM {weight: $confidence}]->(c)
                """
                self.neo4j.execute_write(
                    query, {**properties, "paper_id": paper_id}
                )

    def add_cross_reference_edge(
        self,
        source_paper_id: str,
        target_paper_id: str,
        relation_type: str,
        description: str,
        is_conflict: bool,
        weight: float = 1.0,
    ):
        if self._use_memory:
            edge_type = (
                EdgeType.CONTRADICTS if is_conflict else EdgeType.SIMILAR_TO
            )
            edge = GraphEdge(
                source_id=source_paper_id,
                target_id=target_paper_id,
                type=edge_type,
                properties={
                    "relation_type": relation_type,
                    "description": description,
                    "is_conflict": is_conflict,
                },
                weight=weight,
                created_at=datetime.now(),
            )
            self._memory_edges.append(edge)
        else:
            query = """
            MATCH (a:Paper {id: $source_id})
            MATCH (b:Paper {id: $target_id})
            MERGE (a)-[r:RELATED {type: $edge_type}]->(b)
            SET r.relation_type = $relation_type,
                r.description = $description,
                r.is_conflict = $is_conflict,
                r.weight = $weight
            """
            edge_type = (
                EdgeType.CONTRADICTS.value if is_conflict else EdgeType.SIMILAR_TO.value
            )
            self.neo4j.execute_write(
                query,
                {
                    "source_id": source_paper_id,
                    "target_id": target_paper_id,
                    "edge_type": edge_type,
                    "relation_type": relation_type,
                    "description": description,
                    "is_conflict": is_conflict,
                    "weight": weight,
                },
            )

    def add_debate_record(self, record: DebateRecord):
        node_id = record.id
        properties = {
            "topic": record.topic,
            "paper_a_id": record.paper_a_id,
            "paper_b_id": record.paper_b_id,
            "round_count": len(record.rounds),
            "has_verdict": record.human_verdict is not None,
            "verdict": record.human_verdict or "",
        }

        if self._use_memory:
            self._memory_nodes[node_id] = GraphNode(
                id=node_id,
                label=NodeLabel.DEBATE_TOPIC,
                properties=properties,
                created_at=record.created_at,
            )
            for paper_id in [record.paper_a_id, record.paper_b_id]:
                edge = GraphEdge(
                    source_id=paper_id,
                    target_id=node_id,
                    type=EdgeType.DEBATE_ABOUT,
                    weight=1.0,
                )
                self._memory_edges.append(edge)
        else:
            query = """
            MERGE (d:Debate {id: $id})
            SET d.topic = $topic,
                d.paper_a_id = $paper_a_id,
                d.paper_b_id = $paper_b_id,
                d.round_count = $round_count
            WITH d
            MATCH (a:Paper {id: $paper_a_id})
            MATCH (b:Paper {id: $paper_b_id})
            MERGE (a)-[:DEBATE_ABOUT]->(d)
            MERGE (b)-[:DEBATE_ABOUT]->(d)
            """
            self.neo4j.execute_write(query, properties)

    def get_all_paper_nodes(self) -> List[GraphNode]:
        if self._use_memory:
            return list(self._memory_nodes.values())

        results = self.neo4j.execute_read("MATCH (p:Paper) RETURN p")
        nodes = []
        for record in results:
            p = record["p"]
            node = GraphNode(
                id=p.get("id", ""),
                label=NodeLabel.PAPER,
                properties=dict(p),
                created_at=p.get("created_at"),
                last_updated=p.get("last_updated"),
            )
            nodes.append(node)
        return nodes

    def get_paper_count(self) -> int:
        if self._use_memory:
            return len(self._memory_nodes)
        result = self.neo4j.execute_read("MATCH (p:Paper) RETURN count(p) as cnt")
        return result[0]["cnt"] if result else 0

    def get_all_edges(self) -> List[GraphEdge]:
        if self._use_memory:
            return self._memory_edges

        results = self.neo4j.execute_read(
            "MATCH ()-[r]->() RETURN startNode(r).id as source, endNode(r).id as target, type(r) as type, r"
        )
        edges = []
        for record in results:
            edge = GraphEdge(
                source_id=record["source"],
                target_id=record["target"],
                type=EdgeType.SIMILAR_TO,
                properties=dict(record.get("r", {})),
                weight=record.get("r", {}).get("weight", 1.0),
            )
            edges.append(edge)
        return edges

    def get_stats(self) -> dict:
        paper_count = self.get_paper_count()
        edge_count = len(self.get_all_edges())

        claim_count = 0
        if self._use_memory:
            claim_count = sum(
                1 for n in self._memory_nodes.values()
                if n.label == NodeLabel.CLAIM
            )
        else:
            result = self.neo4j.execute_read(
                "MATCH (c:Claim) RETURN count(c) as cnt"
            )
            claim_count = result[0]["cnt"] if result else 0

        return {
            "paper_nodes": paper_count,
            "edges": edge_count,
            "claim_nodes": claim_count,
        }

    def sync_papers(
        self,
        papers: List[Paper],
    ) -> int:
        new_nodes = 0
        new_edges = 0
        for paper in papers:
            if paper.id not in self._memory_nodes:
                self.add_paper_node(paper)
                new_nodes += 1
            if paper.claims:
                self.add_claim_nodes(paper.id, paper.claims)
                new_nodes += len(paper.claims)
            for cr in paper.cross_references:
                self.add_cross_reference_edge(
                    paper.id,
                    cr.related_paper_id,
                    cr.relation_type,
                    cr.description,
                    cr.is_conflict,
                )
                new_edges += 1

        logger.info(
            f"知识图谱同步完成: {new_nodes} 新节点, {new_edges} 新边"
        )
        return new_nodes + new_edges
