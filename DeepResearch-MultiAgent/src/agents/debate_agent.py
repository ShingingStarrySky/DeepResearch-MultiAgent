from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.provider import ProviderType
from src.llm.token_tracker import TokenTracker
from src.models.paper import Paper, CrossReference
from src.models.schema import DebateRecord, DebateRound, DisputePanorama


class DebateAgent(BaseAgent):
    PROPONENT_SYSTEM = """你是一个学术辩论持方Agent。你的任务是为指定论文的观点进行辩护。
你需要基于论文中提供的证据链构建支持论证。请保持逻辑严密，引用论文中的具体实验数据或理论证明。
每次回应需要明确标注你引用的证据来源（如 Table 2, Figure 3 等）。"""

    OPPONENT_SYSTEM = """你是一个学术辩论反方Agent。你的任务是挑战持方论文的观点。
你需要基于对立论文及知识库中相关论据发起挑战。请指出持方论证中的逻辑漏洞、实验设置缺陷或理论假设不现实之处。
每次反驳需要引用具体论据。"""

    SUMMARIZER_SYSTEM = """你是一个学术辩论总结专家。请基于完整的辩论记录生成争议全景图。
全景图包含:
1. 核心分歧点
2. 持方论文实验严格性评价
3. 反方论文实验严格性评价
4. 持方论文理论假设评价
5. 反方论文理论假设评价
6. 建议解决方案
"""

    def __init__(
        self,
        token_tracker: Optional[TokenTracker] = None,
        max_rounds: int = 3,
    ):
        super().__init__(
            name="DebateAgent",
            provider_type=ProviderType.DEEPSEEK,
            token_tracker=token_tracker,
        )
        self.max_rounds = max_rounds or self.settings.debate_max_rounds

    def execute(
        self,
        paper_a: Paper,
        paper_b: Paper,
        topic: str = "",
    ) -> DebateRecord:
        topic = topic or self._generate_topic(paper_a, paper_b)
        self.log(f'启动对抗辩论: 主题 "{topic}"')

        record = DebateRecord(
            id=f"debate_{paper_a.id}_{paper_b.id}",
            topic=topic,
            paper_a_id=paper_a.id,
            paper_b_id=paper_b.id,
            rounds=[],
            created_at=datetime.now(),
        )

        proponent_context = self._build_paper_context(paper_a)
        opponent_context = self._build_paper_context(paper_b)

        proponent_last = f"(依据 {paper_a.id}) 主张: {paper_a.summary.core_contribution if paper_a.summary else paper_a.abstract[:300]}"
        opponent_last = ""

        for round_num in range(1, self.max_rounds + 1):
            dr = DebateRound(round_num=round_num, timestamp=datetime.now())

            proponent_arg = self._proponent_turn(
                topic, proponent_context, opponent_last, proponent_last, round_num
            )
            dr.proponent_argument = proponent_arg
            self.log(f"持方回应: {proponent_arg[:120]}...")

            opponent_arg = self._opponent_turn(
                topic, opponent_context, proponent_arg, round_num
            )
            dr.opponent_argument = opponent_arg
            self.log(f"反方反驳: {opponent_arg[:120]}...")

            record.rounds.append(dr)
            proponent_last = proponent_arg
            opponent_last = opponent_arg

        self.log(f"{self.max_rounds}轮辩论结束。生成争议全景图:")
        panorama = self._generate_panorama(record, paper_a, paper_b)
        record.panorama = panorama

        self.log(f"  - 论点分歧: {panorama.core_disagreement[:100]}")
        self.log(
            f"  - 实验严格性: {paper_a.id[:10]}({panorama.a_experiment_rigor}) | "
            f"{paper_b.id[:10]}({panorama.b_experiment_rigor})"
        )
        self.log(f"  - 建议: {panorama.suggested_resolution[:100]}")

        return record

    def _generate_topic(self, paper_a: Paper, paper_b: Paper) -> str:
        """自动从冲突论文中提取辩论主题"""
        conflicts_a = [cr for cr in paper_a.cross_references
                       if cr.is_conflict and cr.related_paper_id == paper_b.id]
        if conflicts_a:
            return conflicts_a[0].description[:100]

        return f"{paper_a.title[:50]} vs {paper_b.title[:50]} 的观点分歧"

    def _build_paper_context(self, paper: Paper) -> str:
        parts = [f"论文: {paper.title}"]
        if paper.summary:
            parts.append(f"核心贡献: {paper.summary.core_contribution}")
            if paper.summary.method_limitations:
                parts.append(f"方法局限: {'; '.join(paper.summary.method_limitations)}")
        if paper.claims:
            parts.append("关键声明:")
            for c in paper.claims:
                parts.append(
                    f"  - {c.content} (验证: {c.verification.value}, 证据: {c.evidence or 'N/A'})"
                )
        return "\n".join(parts)

    def _proponent_turn(
        self,
        topic: str,
        proponent_context: str,
        opponent_last: str,
        proponent_last: str,
        round_num: int,
    ) -> str:
        if round_num == 1:
            prompt = (
                f"辩论主题: {topic}\n\n"
                f"你的论文证据:\n{proponent_context}\n\n"
                f"请提出你的初始主张，引用论文中的具体证据。"
            )
        else:
            prompt = (
                f"辩论主题: {topic}\n\n"
                f"你上一轮的论证: {proponent_last[:500]}\n\n"
                f"反方上一轮的反驳: {opponent_last[:500]}\n\n"
                f"请回应反方的反驳，补充新的证据或澄清误解。"
            )

        messages = [
            self._build_system_message(self.PROPONENT_SYSTEM),
            self._build_user_message(prompt),
        ]

        try:
            return self._call_llm(messages, max_tokens=1500)
        except Exception as e:
            self.log(f"持方Agent调用失败: {e}", "WARNING")
            return proponent_last

    def _opponent_turn(
        self,
        topic: str,
        opponent_context: str,
        proponent_arg: str,
        round_num: int,
    ) -> str:
        prompt = (
            f"辩论主题: {topic}\n\n"
            f"你的论文证据:\n{opponent_context}\n\n"
            f"持方最新论证:\n{proponent_arg[:800]}\n\n"
            f"请反驳持方的论证，指出逻辑漏洞或实验缺陷。"
        )

        messages = [
            self._build_system_message(self.OPPONENT_SYSTEM),
            self._build_user_message(prompt),
        ]

        try:
            if round_num % 2 == 0:
                return self._call_llm(
                    messages,
                    model=self.settings.deepseek.pro_model,
                    max_tokens=1500,
                )
            else:
                return self._call_llm(messages, max_tokens=1500)
        except Exception as e:
            self.log(f"反方Agent调用失败: {e}", "WARNING")
            return "无法生成反驳，请检查API配置。"

    def _generate_panorama(
        self,
        record: DebateRecord,
        paper_a: Paper,
        paper_b: Paper,
    ) -> DisputePanorama:
        debate_text = "\n\n".join(
            f"第{r.round_num}轮:\n持方: {r.proponent_argument[:500]}\n反方: {r.opponent_argument[:500]}"
            for r in record.rounds
        )

        messages = [
            self._build_system_message(self.SUMMARIZER_SYSTEM),
            self._build_user_message(
                f"辩论主题: {record.topic}\n\n"
                f"持方论文 ({paper_a.id[:20]}...): {paper_a.title}\n"
                f"反方论文 ({paper_b.id[:20]}...): {paper_b.title}\n\n"
                f"辩论记录:\n{debate_text}\n\n"
                f"请生成争议全景图。输出格式:\n"
                f"核心分歧: <text>\n"
                f"持方实验严格性: <评价>\n"
                f"反方实验严格性: <评价>\n"
                f"持方理论假设: <评价>\n"
                f"反方理论假设: <评价>\n"
                f"建议方案: <text>"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=1500)
            return self._parse_panorama_result(result, paper_a, paper_b)
        except Exception as e:
            self.log(f"全景图生成失败: {e}", "WARNING")
            return DisputePanorama(
                core_disagreement=record.topic,
                a_experiment_rigor="未评估",
                b_experiment_rigor="未评估",
                a_theory_assumption="未评估",
                b_theory_assumption="未评估",
                suggested_resolution="需要人工判断",
                confidence=0.3,
            )

    def _parse_panorama_result(
        self, result: str, paper_a: Paper, paper_b: Paper
    ) -> DisputePanorama:
        panorama = DisputePanorama()
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("核心分歧"):
                panorama.core_disagreement = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("持方实验严格性"):
                panorama.a_experiment_rigor = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("反方实验严格性"):
                panorama.b_experiment_rigor = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("持方理论假设"):
                panorama.a_theory_assumption = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("反方理论假设"):
                panorama.b_theory_assumption = line.split(":", 1)[-1].strip() if ":" in line else line
            elif line.startswith("建议方案"):
                panorama.suggested_resolution = line.split(":", 1)[-1].strip() if ":" in line else line

        panorama.confidence = 0.7
        return panorama

    def process_conflicts(
        self,
        papers: List[Paper],
        knowledge_base: List[Paper],
    ) -> List[DebateRecord]:
        records: List[DebateRecord] = []

        for paper in papers:
            for cr in paper.cross_references:
                if not cr.is_conflict:
                    continue
                kb_paper = next(
                    (p for p in knowledge_base if p.id == cr.related_paper_id),
                    None,
                )
                if kb_paper is None:
                    continue

                topic = cr.description if cr.description else ""
                record = self.execute(paper, kb_paper, topic)
                records.append(record)

        self.log(
            f"争议全景图已推送至研究看板，等待人类裁决。共 {len(records)} 组辩论。"
        )
        return records
