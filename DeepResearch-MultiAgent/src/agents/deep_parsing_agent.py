from __future__ import annotations

import concurrent.futures
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.provider import ProviderType
from src.llm.token_tracker import TokenTracker
from src.models.paper import (
    Paper,
    PaperSection,
    SectionType,
    Claim,
    VerificationStatus,
    StructuredSummary,
    CrossReference,
)


class DeepParsingAgent(BaseAgent):
    SYSTEM_PROMPT_TEMPLATE = """你是一个学术论文深度解析专家。你将收到一篇论文的完整内容。
你需要严格按照五步链式任务进行解析，每一步的输出都依赖于前一步的结果。

当前步骤: {step_name}
上一步输出: {previous_output}

请仔细完成任务并输出结构化结果。
"""

    def __init__(
        self,
        token_tracker: Optional[TokenTracker] = None,
        max_workers: int = 3,
    ):
        super().__init__(
            name="DeepParsingAgent",
            provider_type=ProviderType.DEEPSEEK,
            token_tracker=token_tracker,
        )
        self.max_workers = max_workers

    def execute(
        self,
        papers: List[Paper],
        knowledge_base_papers: Optional[List[Paper]] = None,
    ) -> List[Paper]:
        knowledge_base_papers = knowledge_base_papers or []
        results: List[Paper] = []

        for paper in papers:
            self.log(f'开始处理候选论文: ID={paper.id} "{paper.title[:80]}..."')
            try:
                paper = self._parse_single_paper(paper, knowledge_base_papers)
                results.append(paper)
            except Exception as e:
                self.log(f"解析论文 {paper.id} 失败: {e}", "ERROR")
                results.append(paper)

        if len(papers) > 1:
            self._parallel_parse(papers[1:], knowledge_base_papers, results)

        conflict_count = sum(1 for p in results if p.has_conflicts)
        self.log(
            f"批量解析完成。本轮共识别 {conflict_count} 组观点冲突，均已触发辩论Agent。"
        )
        return results

    def _parse_single_paper(
        self,
        paper: Paper,
        knowledge_base: List[Paper],
    ) -> Paper:
        from datetime import datetime

        self.log("Step 1/5 结构树抽取…")
        paper.sections = self._step1_extract_structure(paper)

        self.log("Step 2/5 关键声明抽取…")
        paper.claims = self._step2_extract_claims(paper)

        self.log("Step 3/5 逻辑断层验证…")
        paper.claims = self._step3_verify_claims(paper)

        self.log("Step 4/5 结构化摘要生成…")
        paper.summary = self._step4_generate_summary(paper)

        self.log("Step 5/5 跨文献交叉引用分析…")
        cross_refs, has_conflicts = self._step5_cross_reference(paper, knowledge_base)
        paper.cross_references = cross_refs
        paper.has_conflicts = has_conflicts

        if has_conflicts:
            self.log(f"标记冲突: 论文 {paper.id} 与知识库存在观点冲突。触发辩论Agent。", "WARNING")

        paper.parsed_at = datetime.now()
        return paper

    def _step1_extract_structure(self, paper: Paper) -> List[PaperSection]:
        messages = [
            self._build_system_message(
                "你是一个论文结构分析专家。请从论文中识别IMRaD结构（动机-方法-实验-结论）"
                "以及各章节的标题和主要内容概括。"
                "输出格式: 每个章节一行，格式为 [类型]:[标题]:[内容摘要]。"
            ),
            self._build_user_message(
                f"论文标题: {paper.title}\n摘要: {paper.abstract}\n"
                f"论文全文 (如有): {paper.raw_data.get('full_text', paper.abstract)}\n\n"
                f"请分析论文结构。"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=2000)
            sections = self._parse_structure_result(result)
            return sections
        except Exception as e:
            self.log(f"结构抽取失败: {e}", "WARNING")
            return [
                PaperSection(
                    type=SectionType.UNKNOWN,
                    title="Full Paper",
                    content_summary=paper.abstract[:500],
                )
            ]

    def _parse_structure_result(self, result: str) -> List[PaperSection]:
        sections = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split(":", 2)
            if len(parts) < 3:
                sections.append(
                    PaperSection(
                        type=SectionType.UNKNOWN,
                        title="",
                        content_summary=line[:200],
                    )
                )
                continue

            type_str, title, summary = parts
            type_map = {
                "动机": SectionType.MOTIVATION,
                "方法": SectionType.METHOD,
                "实验": SectionType.EXPERIMENT,
                "结论": SectionType.CONCLUSION,
            }
            section_type = type_map.get(type_str.strip(), SectionType.UNKNOWN)
            sections.append(
                PaperSection(
                    type=section_type,
                    title=title.strip(),
                    content_summary=summary.strip(),
                )
            )
        return sections

    def _step2_extract_claims(self, paper: Paper) -> List[Claim]:
        sections_text = "\n".join(
            f"[{s.type.value}] {s.title}: {s.content_summary}" for s in paper.sections
        ) if paper.sections else paper.abstract

        messages = [
            self._build_system_message(
                "你是一个学术声明识别专家。请从论文中提取所有核心声明(Claim)。"
                "每个声明包括: 声明内容、所属章节、支撑证据位置(如Table 2, Figure 3等)。"
                "输出格式: 每行一个声明: [声明内容] | [章节] | [证据位置]。"
            ),
            self._build_user_message(
                f"论文: {paper.title}\n摘要: {paper.abstract}\n章节分析:\n{sections_text}\n\n"
                f"请提取关键声明（通常5-10条）。"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=2000)
            claims = self._parse_claims_result(result)
            self.log(f"共识别 {len(claims)} 条核心声明")
            return claims
        except Exception as e:
            self.log(f"声明抽取失败: {e}", "WARNING")
            return []

    def _parse_claims_result(self, result: str) -> List[Claim]:
        claims = []
        for i, line in enumerate(result.strip().split("\n")):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            content = parts[0].strip() if len(parts) > 0 else line
            section = parts[1].strip() if len(parts) > 1 else ""
            evidence = parts[2].strip() if len(parts) > 2 else None

            claims.append(
                Claim(
                    id=f"claim_{i}",
                    content=content,
                    section=section,
                    evidence=evidence,
                    verification=VerificationStatus.UNVERIFIED,
                )
            )
        return claims

    def _step3_verify_claims(self, paper: Paper) -> List[Claim]:
        if not paper.claims:
            return []

        claims_text = "\n".join(
            f"声明{i}: {c.content} [证据: {c.evidence or '未指定'}]"
            for i, c in enumerate(paper.claims)
        )

        messages = [
            self._build_system_message(
                "你是一个学术逻辑验证专家。请逐条检查每个声明与其实验数据/数学证明之间是否存在逻辑断层。"
                "验证每个声明: ✓(已验证)、⚠(部分验证/存疑)、✗(无法验证/矛盾)。"
                "输出格式: 每个声明一行: [状态] [声明ID] [验证备注]。"
            ),
            self._build_user_message(
                f"论文: {paper.title}\n声明列表:\n{claims_text}\n\n"
                f"论文内容摘要: {paper.abstract}\n\n"
                f"请逐条验证每个声明。"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=2000)

            lines = result.strip().split("\n")
            for i, line in enumerate(lines):
                if i >= len(paper.claims):
                    break

                line = line.strip()
                if line.startswith("✓"):
                    paper.claims[i].verification = VerificationStatus.VERIFIED
                elif line.startswith("⚠"):
                    paper.claims[i].verification = VerificationStatus.PARTIAL
                elif line.startswith("✗"):
                    paper.claims[i].verification = VerificationStatus.CONTRADICTED

                note_start = line.find("]") + 1 if "]" in line else 0
                paper.claims[i].verification_note = line[note_start:].strip()[:200]

                v_status = paper.claims[i].verification.value
                self.log(f"  {line[:3]} 声明{i}", "INFO")

        except Exception as e:
            self.log(f"声明验证失败: {e}", "WARNING")

        return paper.claims

    def _step4_generate_summary(self, paper: Paper) -> StructuredSummary:
        claims_text = "\n".join(
            f"- {c.content} (验证: {c.verification.value})" for c in paper.claims
        ) if paper.claims else "无声明"

        messages = [
            self._build_system_message(
                "你是一个学术论文总结专家。请基于论文的声明和验证结果，生成结构化摘要。"
                "摘要包含三部分: 核心贡献、方法局限、潜在改进方向。"
                "输出格式:\n核心贡献: <text>\n方法局限:\n- <limitation1>\n- <limitation2>\n改进方向:\n- <direction1>\n- <direction2>"
            ),
            self._build_user_message(
                f"论文: {paper.title}\n摘要: {paper.abstract}\n声明:\n{claims_text}\n\n"
                f"请生成结构化摘要。"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=1500)
            return self._parse_summary_result(result)
        except Exception as e:
            self.log(f"摘要生成失败: {e}", "WARNING")
            return StructuredSummary(
                core_contribution=paper.abstract[:300],
                method_limitations=["未识别"],
                improvement_directions=["未识别"],
                overall_quality=0.5,
            )

    def _parse_summary_result(self, result: str) -> StructuredSummary:
        contribution = ""
        limitations: List[str] = []
        improvements: List[str] = []
        current_section = ""

        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("核心贡献"):
                contribution = line.split(":", 1)[-1].strip() if ":" in line else ""
                current_section = "contribution"
            elif line.startswith("方法局限"):
                current_section = "limitations"
            elif line.startswith("改进方向"):
                current_section = "improvements"
            elif line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip()
                if current_section == "limitations":
                    limitations.append(item)
                elif current_section == "improvements":
                    improvements.append(item)
            elif current_section == "contribution":
                contribution += " " + line

        return StructuredSummary(
            core_contribution=contribution[:500] or result[:300],
            method_limitations=limitations[:5] or ["未详细说明"],
            improvement_directions=improvements[:5] or ["未详细说明"],
            overall_quality=0.7,
        )

    def _step5_cross_reference(
        self,
        paper: Paper,
        knowledge_base: List[Paper],
    ) -> tuple[List[CrossReference], bool]:
        cross_refs: List[CrossReference] = []
        has_conflicts = False

        if not knowledge_base:
            self.log("知识库为空，跳过交叉引用分析。")
            return cross_refs, has_conflicts

        kb_summary = "\n".join(
            f"- [{p.id}] {p.title[:100]} (贡献: {(p.summary.core_contribution if p.summary else p.abstract)[:100]})"
            for p in knowledge_base[:50]
        )

        messages = [
            self._build_system_message(
                "你是一个跨文献分析专家。请分析这篇论文与知识库中已有论文的关联关系。"
                "识别: 引用同一基础理论但得出相反结论、使用相同数据集但评估指标不同等深度关联。"
                "输出格式: 每行一个关联: [关系类型] [知识库论文ID] [描述]。"
                "对于观点冲突，关系类型使用 CONFLICT，并在前面添加 CONFLICT: 标记。"
            ),
            self._build_user_message(
                f"当前论文: [{paper.id}] {paper.title}\n摘要: {paper.abstract}\n"
                f"声明: {(paper.summary.core_contribution if paper.summary else paper.abstract)[:500]}\n\n"
                f"知识库论文:\n{kb_summary}\n\n"
                f"请分析关联关系，特别关注观点冲突。"
            ),
        ]

        try:
            result = self._call_llm(messages, max_tokens=2000)

            for line in result.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                is_conflict = line.startswith("CONFLICT:")
                if is_conflict:
                    line = line[9:].strip()
                    has_conflicts = True

                parts = line.split("]", 2)
                if len(parts) < 3:
                    continue

                rel_type = parts[0].replace("[", "").strip()
                kb_paper_id = parts[1].replace("[", "").strip()
                description = parts[2].strip()
                kb_paper = next(
                    (p for p in knowledge_base if p.id == kb_paper_id), None
                )

                cross_refs.append(
                    CrossReference(
                        related_paper_id=kb_paper_id,
                        related_paper_title=kb_paper.title if kb_paper else "Unknown",
                        relation_type=rel_type,
                        description=description,
                        is_conflict=is_conflict,
                    )
                )

            self.log(f"发现 {len(cross_refs)} 篇高度相关论文", "INFO")
        except Exception as e:
            self.log(f"交叉引用分析失败: {e}", "WARNING")

        return cross_refs, has_conflicts

    def _parallel_parse(
        self,
        papers: List[Paper],
        knowledge_base: List[Paper],
        results: List[Paper],
    ):
        self.log(f"继续处理剩余 {len(papers)} 篇论文... 并行启动 {self.max_workers} 个 worker。")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(self._parse_single_paper, p, knowledge_base): p
                for p in papers
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    paper = future.result()
                    results.append(paper)
                except Exception as e:
                    paper = futures[future]
                    self.log(f"并行解析 {paper.id} 失败: {e}", "ERROR")
