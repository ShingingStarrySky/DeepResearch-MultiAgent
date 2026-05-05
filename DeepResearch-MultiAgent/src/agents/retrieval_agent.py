from __future__ import annotations

import concurrent.futures
from typing import List, Dict, Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.llm.provider import ProviderType
from src.llm.token_tracker import TokenTracker
from src.models.paper import Paper
from src.models.schema import InterestProfile
from src.retrieval.arxiv_fetcher import ArxivFetcher
from src.retrieval.semantic_scholar import SemanticScholarClient
from src.retrieval.embedding import EmbeddingService


class RetrievalAgent(BaseAgent):
    SYSTEM_PROMPT = """你是一个学术论文初筛专家。你会收到一批论文的标题和摘要，以及研究者的兴趣权重图谱。
你需要对每篇论文进行评估，判断其与研究者课题的关联强度，并生成可解释的初筛推理链。

对于每篇入选的论文，必须输出:
1. 匹配权重分数 (0.0~1.0)
2. 入选理由 (引用具体关键词和课题)
3. 关联的研究者ID
"""

    def __init__(
        self,
        token_tracker: Optional[TokenTracker] = None,
    ):
        super().__init__(
            name="RetrievalAgent",
            provider_type=ProviderType.DEEPSEEK,
            token_tracker=token_tracker,
        )
        self.arxiv_fetcher = ArxivFetcher(max_results=200)
        self.semantic_scholar = SemanticScholarClient(
            api_key=self.settings.semantic_scholar.api_key,
        )
        self.embedding_service = EmbeddingService()

    def execute(
        self,
        interest_profiles: List[InterestProfile],
        categories: Optional[List[str]] = None,
        keywords: Optional[str] = None,
    ) -> List[Paper]:
        threshold = self.settings.screening_threshold
        max_papers = self.settings.max_papers_per_day
        self.log("开始从 arXiv, Semantic Scholar, DBLP 抓取论文...")

        arxiv_papers = self._fetch_from_arxiv(categories, keywords)
        ss_papers = self._fetch_from_semantic_scholar(keywords or "machine learning")
        all_papers = self._deduplicate(arxiv_papers + ss_papers)

        self.log(f"抓取完毕，共获得 {len(all_papers)} 篇新论文（含预印本 & 顶会）。")

        if not all_papers:
            return []

        self.log("运行初筛BERT模型… 基于权重图谱压缩候选集...")
        screened = self._semantic_screening(all_papers, interest_profiles, threshold)

        screened.sort(key=lambda p: p.screening_score, reverse=True)
        screened = screened[:max_papers]

        self.log(f"初筛完成，保留 {len(screened)} 篇高相关性论文（阈值 ≥ {threshold}）。")
        self.log("生成初筛推理链: 为每篇论文输出关联解释...")

        for paper in screened:
            reasoning = self._generate_screening_reasoning(paper, interest_profiles)
            paper.screening_reason = reasoning
            paper.is_selected = True
            self.log(f'示例: "{paper.title}" 匹配权重 {paper.screening_score:.2f}，'
                     f'理由: {reasoning[:100]}...')

        return screened

    def _fetch_from_arxiv(
        self,
        categories: Optional[List[str]] = None,
        keywords: Optional[str] = None,
    ) -> List[Paper]:
        try:
            return self.arxiv_fetcher.fetch_new_papers(
                categories=categories,
                keywords=keywords,
            )
        except Exception as e:
            self.log(f"arXiv 抓取失败: {e}", "WARNING")
            return []

    def _fetch_from_semantic_scholar(self, query: str) -> List[Paper]:
        try:
            return self.semantic_scholar.search_recent_papers(query=query, limit=100)
        except Exception as e:
            self.log(f"Semantic Scholar 抓取失败: {e}", "WARNING")
            return []

    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        seen: set = set()
        unique: List[Paper] = []
        for p in papers:
            dedup_key = p.doi or p.arxiv_id or p.title.lower()
            if dedup_key and dedup_key not in seen:
                seen.add(dedup_key)
                unique.append(p)
        return unique

    def _semantic_screening(
        self,
        papers: List[Paper],
        profiles: List[InterestProfile],
        threshold: float,
    ) -> List[Paper]:
        profile_texts = []
        for profile in profiles:
            topics_text = "; ".join(
                f"{t.name}: {t.description}" for t in profile.active_topics
            )
            profile_texts.append(topics_text)

        combined_profile = " ".join(profile_texts)
        profile_embedding = self.embedding_service.encode([combined_profile])

        if profile_embedding is None:
            for paper in papers:
                paper.screening_score = 0.5
            return papers[: self.settings.max_papers_per_day]

        paper_texts = [
            f"{p.title}. {p.abstract}" for p in papers
        ]
        paper_embeddings = self.embedding_service.encode(paper_texts)

        if paper_embeddings is None:
            for paper in papers:
                paper.screening_score = 0.5
            return papers[: self.settings.max_papers_per_day]

        scores = self.embedding_service.compute_similarity(
            profile_embedding[0], paper_embeddings
        )

        screened: List[Paper] = []
        for paper, score in zip(papers, scores):
            paper.screening_score = float(score)
            if score >= threshold:
                screened.append(paper)

        return screened

    def _generate_screening_reasoning(
        self,
        paper: Paper,
        profiles: List[InterestProfile],
    ) -> str:
        profile_summary = "\n".join(
            f"- 研究者 {p.researcher_id}: {[t.name for t in p.active_topics]}"
            for p in profiles
        )

        messages = [
            self._build_system_message(self.SYSTEM_PROMPT),
            self._build_user_message(
                f"论文标题: {paper.title}\n"
                f"摘要: {paper.abstract}\n"
                f"匹配分数: {paper.screening_score:.3f}\n\n"
                f"研究者兴趣图谱:\n{profile_summary}\n\n"
                f"请用1-2句话解释这篇论文为何入选，关联到哪个研究者和课题。"
            ),
        ]

        try:
            return self._call_llm(messages, max_tokens=200).strip()
        except Exception:
            keywords_str = ", ".join(paper.keywords[:3]) if paper.keywords else "N/A"
            return f"基于关键词 [{keywords_str}] 的语义匹配，关联度 {paper.screening_score:.2f}"
