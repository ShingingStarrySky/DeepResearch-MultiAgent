from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Dict

from src.agents.base_agent import BaseAgent
from src.llm.provider import ProviderType
from src.llm.token_tracker import TokenTracker
from src.models.paper import Paper
from src.models.schema import InterestProfile, GraphNode, NodeLabel
from src.knowledge_graph.trend_detector import TrendDetector


class MetaAgent(BaseAgent):
    SYSTEM_PROMPT = """你是一个元认知学术趋势分析Agent。基于知识图谱中的论文簇和趋势检测结果，
生成"本周领域前沿简报"。简报应包含:
1. 本周最重要的新兴研究趋势（1-3个）
2. 每个趋势的核心论文列表
3. 对研究者的建议
"""

    def __init__(
        self,
        token_tracker: Optional[TokenTracker] = None,
    ):
        super().__init__(
            name="MetaAgent",
            provider_type=ProviderType.DEEPSEEK,
            token_tracker=token_tracker,
        )
        self.trend_detector = TrendDetector()

    def execute(
        self,
        paper_nodes: List[GraphNode],
        new_papers: Optional[List[Paper]] = None,
    ) -> dict:
        self.log("每周知识图谱趋势分析开始...")

        clusters = self.trend_detector.detect_clusters(paper_nodes)
        significant = self._filter_significant_clusters(clusters)
        self.log(f"运行 GNN 社区检测... 发现 {len(significant)} 个新兴论文簇:")

        for i, cluster in enumerate(significant):
            self.log(
                f"  簇 {i + 1}: \"{cluster.get('label', 'Unknown')}\" "
                f"({cluster.get('size', 0)} 篇，中心度增长 +{cluster.get('growth_pct', 0)}%)"
            )

        if new_papers:
            cluster = self.trend_detector.assign_papers_to_clusters(
                new_papers, clusters
            )

        briefings = self._generate_briefings(significant)
        self.log('自动生成"本周领域前沿简报"并推送至研究者。')
        return briefings

    def weekly_scan_and_update(
        self,
        knowledge_graph_manager,
        interest_profiles: List[InterestProfile],
    ) -> dict:
        all_nodes = knowledge_graph_manager.get_all_paper_nodes()
        trend_report = self.execute(all_nodes)

        feedback = trend_report.get("feedback_signals", {})
        for profile in interest_profiles:
            self._update_interest_weights(profile, feedback)

        return trend_report

    def _filter_significant_clusters(self, clusters: List[dict]) -> List[dict]:
        significant = []
        for cluster in clusters:
            size = cluster.get("size", 0)
            growth = cluster.get("growth_pct", 0)
            if size >= 5 and growth >= self.settings.trend_analysis_interval_days * 10:
                significant.append(cluster)
        return sorted(significant, key=lambda c: c.get("growth_pct", 0), reverse=True)

    def _generate_briefings(self, clusters: List[dict]) -> dict:
        briefings = {}
        feedback_signals: Dict[str, dict] = {}

        for i, cluster in enumerate(clusters):
            label = cluster.get("label", f"Trend_{i}")
            papers = cluster.get("top_papers", [])

            papers_text = "\n".join(
                f"- {p.get('title', 'Unknown')} (ID: {p.get('id', 'N/A')})"
                for p in papers[:5]
            )

            messages = [
                self._build_system_message(self.SYSTEM_PROMPT),
                self._build_user_message(
                    f"趋势簇: {label}\n"
                    f"论文数量: {cluster.get('size', 0)}\n"
                    f"增长率: +{cluster.get('growth_pct', 0)}%\n"
                    f"核心论文:\n{papers_text}\n\n"
                    f"请生成一段100字以内的趋势简报，说明这个研究方向的意义和建议。"
                ),
            ]

            try:
                briefing = self._call_llm(messages, max_tokens=300)
            except Exception:
                briefing = f"新兴趋势: {label}，包含 {cluster.get('size', 0)} 篇相关论文，近期增长显著。"

            briefings[label] = {
                "briefing": briefing,
                "size": cluster.get("size", 0),
                "growth_pct": cluster.get("growth_pct", 0),
                "top_papers": papers[:5],
            }

            feedback_signals[label] = {
                "keywords": cluster.get("keywords", []),
                "weight_adjustment": min(cluster.get("growth_pct", 0) / 100, 0.5),
            }

        return {
            "briefings": briefings,
            "feedback_signals": feedback_signals,
            "generated_at": datetime.now().isoformat(),
        }

    def _update_interest_weights(
        self,
        profile: InterestProfile,
        feedback: Dict[str, dict],
    ):
        for topic in profile.active_topics:
            for trend_label, signal in feedback.items():
                if any(
                    kw.lower() in topic.name.lower() or kw.lower() in topic.description.lower()
                    for kw in signal.get("keywords", [])
                ):
                    adjustment = signal.get("weight_adjustment", 0.1)
                    topic.weight = min(1.0, topic.weight + adjustment)
                    self.log(
                        f"更新兴趣权重: {profile.researcher_id} "
                        f"对 '{topic.name}' 的权重 +{adjustment:.2f}"
                    )
        profile.last_updated = datetime.now()

    def process_feedback(
        self,
        researcher_id: str,
        briefing_label: str,
        is_important: bool,
        interest_profiles: List[InterestProfile],
    ) -> None:
        profile = next(
            (p for p in interest_profiles if p.researcher_id == researcher_id), None
        )
        if profile is None:
            return

        adjustment = 0.2 if is_important else -0.1
        for topic in profile.active_topics:
            if briefing_label.lower() in topic.name.lower():
                topic.weight = max(0.1, min(1.0, topic.weight + adjustment))
                self.log(
                    f"收到用户反馈: 研究者 {researcher_id} 标记'{briefing_label}'为"
                    f"{'重要' if is_important else '不相关'}，权重{'上调' if adjustment > 0 else '下调'} {abs(adjustment)}。"
                )
        profile.last_updated = datetime.now()
        self.log("逆向传播至检索Agent的兴趣权重图谱... 更新成功")
