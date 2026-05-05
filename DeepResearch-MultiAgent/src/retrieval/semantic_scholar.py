from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

import requests

from src.models.paper import Paper, PaperSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    PAPER_SEARCH_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "authors",
        "externalIds",
        "url",
        "venue",
        "publicationTypes",
        "fieldsOfStudy",
    ]

    def __init__(self, api_key: str = "", delay_seconds: float = 1.0):
        self.api_key = api_key
        self.delay_seconds = delay_seconds
        self._last_request: float = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.delay_seconds and self._last_request > 0:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request = time.time()

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def search_recent_papers(
        self,
        query: str = "machine learning",
        limit: int = 100,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        self._rate_limit()

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(self.PAPER_SEARCH_FIELDS),
        }

        resp = requests.get(
            f"{self.BASE_URL}/paper/search",
            params=params,
            headers=self._headers(),
            timeout=30,
        )

        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            resp = requests.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=self._headers(),
                timeout=30,
            )

        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        logger.info(f"SemanticScholar: 搜索 '{query}' 获取 {len(papers)} 篇论文")
        return papers

    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        self._rate_limit()

        params = {"fields": ",".join(self.PAPER_SEARCH_FIELDS)}
        resp = requests.get(
            f"{self.BASE_URL}/paper/{paper_id}",
            params=params,
            headers=self._headers(),
            timeout=30,
        )

        if resp.status_code == 404:
            logger.warning(f"Paper not found: {paper_id}")
            return None

        resp.raise_for_status()
        data = resp.json()
        return self._parse_paper(data)

    def get_citations(self, paper_id: str, limit: int = 50) -> List[Paper]:
        self._rate_limit()

        params = {
            "fields": ",".join(self.PAPER_SEARCH_FIELDS),
            "limit": min(limit, 100),
        }
        resp = requests.get(
            f"{self.BASE_URL}/paper/{paper_id}/citations",
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            paper = self._parse_paper(citing)
            if paper:
                papers.append(paper)
        return papers

    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        self._rate_limit()

        params = {
            "fields": ",".join(self.PAPER_SEARCH_FIELDS),
            "limit": min(limit, 100),
        }
        resp = requests.get(
            f"{self.BASE_URL}/paper/{paper_id}/references",
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            cited = item.get("citedPaper", {})
            paper = self._parse_paper(cited)
            if paper:
                papers.append(paper)
        return papers

    def _parse_paper(self, item: dict) -> Optional[Paper]:
        try:
            authors = [a.get("name", "") for a in item.get("authors", [])]
            external_ids = item.get("externalIds", {}) or {}

            paper = Paper(
                id=item.get("paperId", ""),
                source=PaperSource.SEMANTIC_SCHOLAR,
                title=item.get("title", ""),
                authors=authors,
                abstract=item.get("abstract", ""),
                doi=external_ids.get("DOI"),
                arxiv_id=external_ids.get("ArXiv"),
                year=item.get("year"),
                venue=item.get("venue", ""),
                url=item.get("url", ""),
                fields_of_study=item.get("fieldsOfStudy", []),
                fetched_at=datetime.now(),
                raw_data={"ss_raw": item},
            )
            return paper
        except Exception as e:
            logger.error(f"解析 Semantic Scholar paper 失败: {e}")
            return None
