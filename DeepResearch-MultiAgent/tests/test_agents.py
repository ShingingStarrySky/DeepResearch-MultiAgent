from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime

from src.models.paper import (
    Paper,
    PaperSection,
    SectionType,
    Claim,
    VerificationStatus,
    StructuredSummary,
    CrossReference,
)
from src.models.schema import (
    GraphNode,
    NodeLabel,
    GraphEdge,
    EdgeType,
    InterestProfile,
    InterestTopic,
    DebateRecord,
    DebateRound,
    DisputePanorama,
)
from src.llm.token_tracker import TokenTracker, TokenUsage


class TestPaperModel:
    def test_paper_creation(self):
        paper = Paper(
            id="test_001",
            title="Test Paper",
            authors=["Author A", "Author B"],
            abstract="This is a test abstract.",
        )
        assert paper.id == "test_001"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert not paper.processed
        assert not paper.is_selected

    def test_paper_screening(self):
        paper = Paper(
            id="test_002",
            title="ML Paper",
            abstract="Machine learning test.",
            screening_score=0.85,
            screening_reason="High relevance",
            is_selected=True,
        )
        assert paper.screening_score == 0.85
        assert paper.is_selected

    def test_paper_sections(self):
        paper = Paper(id="test_003", title="Test")
        section = PaperSection(
            type=SectionType.METHOD,
            title="Methodology",
            content_summary="We propose a new method.",
        )
        paper.sections = [section]
        assert len(paper.sections) == 1
        assert paper.sections[0].type == SectionType.METHOD

    def test_paper_claims(self):
        claim = Claim(
            id="claim_1",
            content="Our method achieves 95% accuracy.",
            section="Experiment",
            evidence="Table 1",
            verification=VerificationStatus.VERIFIED,
        )
        assert claim.verification == VerificationStatus.VERIFIED
        assert claim.evidence == "Table 1"

    def test_structured_summary(self):
        summary = StructuredSummary(
            core_contribution="A novel algorithm",
            method_limitations=["Limited dataset", "Single domain"],
            improvement_directions=["Multi-domain testing", "Larger scale"],
            overall_quality=0.8,
        )
        assert len(summary.method_limitations) == 2
        assert summary.overall_quality == 0.8

    def test_cross_reference(self):
        cr = CrossReference(
            related_paper_id="paper_b",
            related_paper_title="Another Paper",
            relation_type="CONFLICT",
            description="Opposite conclusions on same method",
            is_conflict=True,
        )
        assert cr.is_conflict
        assert cr.relation_type == "CONFLICT"


class TestSchemaModels:
    def test_graph_node(self):
        node = GraphNode(
            id="node_1",
            label=NodeLabel.PAPER,
            properties={"title": "Test", "year": 2024},
            created_at=datetime.now(),
        )
        assert node.label == NodeLabel.PAPER
        assert node.properties["year"] == 2024

    def test_graph_edge(self):
        edge = GraphEdge(
            source_id="paper_a",
            target_id="paper_b",
            type=EdgeType.CITES,
            weight=0.9,
        )
        assert edge.type == EdgeType.CITES
        assert edge.weight == 0.9

    def test_interest_profile(self):
        profile = InterestProfile(
            researcher_id="user_001",
            researcher_name="测试研究者",
            active_topics=[
                InterestTopic(
                    id="topic_1",
                    name="NLP",
                    keywords=["language model", "transformer"],
                    weight=0.8,
                ),
                InterestTopic(
                    id="topic_2",
                    name="CV",
                    keywords=["image", "vision"],
                    weight=0.6,
                ),
            ],
        )
        assert len(profile.active_topics) == 2
        weights = profile.get_keyword_weights()
        assert "language model" in weights

    def test_debate_record(self):
        record = DebateRecord(
            id="debate_001",
            topic="Test Topic",
            paper_a_id="paper_a",
            paper_b_id="paper_b",
            rounds=[
                DebateRound(
                    round_num=1,
                    proponent_argument="Argument A",
                    opponent_argument="Counter A",
                    token_cost=1000,
                ),
            ],
            panorama=DisputePanorama(
                core_disagreement="Test disagreement",
                a_experiment_rigor="High",
                b_experiment_rigor="Medium",
                suggested_resolution="More experiments needed",
            ),
        )
        assert len(record.rounds) == 1
        assert record.paper_a_id == "paper_a"
        assert record.panorama is not None
        assert record.panorama.a_experiment_rigor == "High"


class TestTokenTracker:
    def test_token_tracker_add(self):
        tracker = TokenTracker(session_id="test_session")
        tracker.add(100, 50, model="deepseek-chat", agent="TestAgent")
        assert tracker.total_tokens == 150
        assert tracker.total_calls == 1

    def test_token_tracker_by_model(self):
        tracker = TokenTracker()
        tracker.add(200, 100, model="deepseek-chat")
        tracker.add(100, 50, model="glm-4")
        assert tracker.total_tokens == 450
        assert tracker.total_calls == 2

    def test_token_tracker_by_agent(self):
        tracker = TokenTracker()
        tracker.add(100, 50, model="deepseek", agent="AgentA")
        tracker.add(200, 100, model="deepseek", agent="AgentB")
        assert tracker.get_by_agent("AgentA").total_tokens == 150
        assert tracker.get_by_agent("AgentB").total_tokens == 300

    def test_token_tracker_report(self):
        tracker = TokenTracker(session_id="test")
        tracker.add(500, 200, model="deepseek-chat", agent="Agent1")
        report = tracker.generate_report()
        assert report["session_id"] == "test"
        assert report["total_tokens"] == 700
        assert "deepseek-chat" in report["by_model"]
        assert "Agent1" in report["by_agent"]

    def test_token_tracker_reset(self):
        tracker = TokenTracker()
        tracker.add(100, 50)
        assert tracker.total_tokens == 150
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_calls == 0


class TestAgentBase:
    def test_base_agent_init(self):
        from src.agents.base_agent import BaseAgent

        class MockAgent(BaseAgent):
            def execute(self, *args, **kwargs):
                return {"status": "ok"}

        agent = MockAgent(name="MockAgent")
        assert agent.name == "MockAgent"
        status = agent.get_status()
        assert status["agent_name"] == "MockAgent"


class TestEmbeddingService:
    def test_embedding_service_init(self):
        from src.retrieval.embedding import EmbeddingService

        service = EmbeddingService()
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_fallback_encode(self):
        from src.retrieval.embedding import EmbeddingService

        service = EmbeddingService(model_name="nonexistent_model")
        service._model = None
        result = service._fallback_encode(["test text one", "test text two"])
        assert result is not None
        assert result.shape[0] == 2


class TestKnowledgeGraphManager:
    def test_memory_graph_manager(self):
        from src.knowledge_graph.graph_manager import KnowledgeGraphManager

        manager = KnowledgeGraphManager(neo4j_client=None)
        paper = Paper(
            id="test_paper", title="Test Paper", abstract="Test abstract."
        )
        node = manager.add_paper_node(paper)
        assert node.id == "test_paper"

        stats = manager.get_stats()
        assert stats["paper_nodes"] >= 1

    def test_memory_graph_stats(self):
        from src.knowledge_graph.graph_manager import KnowledgeGraphManager

        manager = KnowledgeGraphManager(neo4j_client=None)
        stats = manager.get_stats()
        assert "paper_nodes" in stats
        assert "edges" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
