from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Any

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    DBLP = "dblp"
    CUSTOM = "custom"


class SectionType(str, Enum):
    MOTIVATION = "motivation"
    METHOD = "method"
    EXPERIMENT = "experiment"
    CONCLUSION = "conclusion"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


class Claim(BaseModel):
    id: str = Field(default="", description="声明唯一ID")
    content: str = Field(default="", description="声明文本内容")
    section: str = Field(default="", description="所属章节")
    evidence: Optional[str] = Field(default=None, description="支撑证据(表格/图表/证明)")
    verification: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED, description="逻辑验证状态"
    )
    verification_note: Optional[str] = Field(default=None, description="验证备注")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")


class PaperSection(BaseModel):
    type: SectionType = Field(default=SectionType.UNKNOWN, description="章节类型")
    title: str = Field(default="", description="章节标题")
    content_summary: str = Field(default="", description="章节内容摘要")
    key_claims: List[Claim] = Field(default_factory=list, description="关键声明列表")


class CrossReference(BaseModel):
    related_paper_id: str = Field(default="", description="关联论文ID")
    related_paper_title: str = Field(default="", description="关联论文标题")
    relation_type: str = Field(default="", description="关联类型")
    description: str = Field(default="", description="关联描述")
    is_conflict: bool = Field(default=False, description="是否为观点冲突")


class StructuredSummary(BaseModel):
    core_contribution: str = Field(default="", description="核心贡献")
    method_limitations: List[str] = Field(default_factory=list, description="方法局限")
    improvement_directions: List[str] = Field(default_factory=list, description="潜在改进方向")
    overall_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="综合质量评分")


class Paper(BaseModel):
    id: str = Field(default="", description="论文唯一ID")
    source: PaperSource = Field(default=PaperSource.ARXIV, description="来源")
    title: str = Field(default="", description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(default="", description="摘要")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    year: Optional[int] = Field(default=None, description="出版年份")
    venue: Optional[str] = Field(default=None, description="发表会议/期刊")
    url: Optional[str] = Field(default=None, description="论文URL")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    fields_of_study: Optional[List[str]] = Field(default=None, description="研究领域")
    published: Optional[datetime] = Field(default=None, description="发表日期")

    screening_score: float = Field(default=0.0, description="初筛匹配分数")
    screening_reason: str = Field(default="", description="初筛入选理由")
    is_selected: bool = Field(default=False, description="是否入选候选")

    sections: List[PaperSection] = Field(default_factory=list, description="结构树")
    claims: List[Claim] = Field(default_factory=list, description="关键声明")
    summary: Optional[StructuredSummary] = Field(default=None, description="结构化摘要")
    cross_references: List[CrossReference] = Field(
        default_factory=list, description="交叉引用"
    )
    has_conflicts: bool = Field(default=False, description="是否有观点冲突")

    fetched_at: Optional[datetime] = Field(default=None, description="获取时间")
    parsed_at: Optional[datetime] = Field(default=None, description="解析完成时间")
    raw_data: dict = Field(default_factory=dict, description="原始API数据")

    @property
    def processed(self) -> bool:
        return self.parsed_at is not None

    @property
    def conflict_count(self) -> int:
        return sum(1 for cr in self.cross_references if cr.is_conflict)
