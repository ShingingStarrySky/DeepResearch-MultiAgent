from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional

import feedparser

from src.models.paper import Paper, PaperSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivFetcher:
    BASE_API_URL = "http://export.arxiv.org/api/query"

    CATEGORIES = {
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language",
        "cs.LG": "Machine Learning",
        "cs.CV": "Computer Vision",
        "cs.NE": "Neural and Evolutionary Computing",
        "stat.ML": "Machine Learning (Statistics)",
        "math.OC": "Optimization and Control",
    }

    def __init__(self, max_results: int = 200, delay_seconds: float = 3.0):
        self.max_results = max_results
        self.delay_seconds = delay_seconds
        self._last_request: float = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.delay_seconds and self._last_request > 0:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request = time.time()

    def fetch_new_papers(
        self,
        categories: Optional[List[str]] = None,
        keywords: Optional[str] = None,
    ) -> List[Paper]:
        if categories is None:
            categories = list(self.CATEGORIES.keys())

        search_query_parts = []

        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        search_query_parts.append(f"({cat_query})")

        if keywords:
            search_query_parts.append(f"({keywords})")

        search_query = " AND ".join(search_query_parts)
        papers = self._search(search_query)
        logger.info(f"ArxivFetcher: 从 {len(categories)} 个分类获取 {len(papers)} 篇论文")
        return papers

    def _search(self, search_query: str) -> List[Paper]:
        self._rate_limit()

        params = {
            "search_query": search_query,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        feed = feedparser.parse(self.BASE_API_URL, params=params)

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Arxiv API 解析警告: {feed.bozo_exception}")

        papers = []
        for entry in feed.entries:
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        time.sleep(self.delay_seconds)
        return papers

    def _parse_entry(self, entry) -> Optional[Paper]:
        try:
            arxiv_id = entry.id.split("/abs/")[-1]
            authors = [author.name for author in entry.authors] if hasattr(entry, "authors") else []

            paper = Paper(
                id=f"arxiv_{arxiv_id}",
                source=PaperSource.ARXIV,
                title=entry.title.strip().replace("\n", " "),
                authors=authors,
                abstract=entry.summary.strip().replace("\n", " "),
                arxiv_id=arxiv_id,
                url=entry.id,
                published=(
                    datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                    if hasattr(entry, "published")
                    else datetime.now()
                ),
                fetched_at=datetime.now(),
                raw_data={"arxiv_raw": entry},
            )

            if hasattr(entry, "arxiv_doi"):
                paper.doi = entry.arxiv_doi

            return paper
        except Exception as e:
            logger.error(f"解析 arXiv entry 失败: {e}")
            return None

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        self._rate_limit()
        params = {"id_list": arxiv_id}
        feed = feedparser.parse(self.BASE_API_URL, params=params)
        entries = feed.entries
        if entries:
            return self._parse_entry(entries[0])
        return None
