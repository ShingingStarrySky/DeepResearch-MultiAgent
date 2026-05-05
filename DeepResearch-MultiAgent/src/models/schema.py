from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Any

from pydantic import BaseModel, Field


class NodeLabel(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    CLAIM = "Claim"
    METHOD = "Method"
    DATASET = "Dataset"
    TREND = "Trend"
    DEBATE_TOPIC = "DebateTopic"
    RESEARCHER = "Researcher"


class EdgeType(str, Enum):
    CITES = "CITES"
    AUTHORS = "AUTHORS"
    CONTAINS_CLAIM = "CONTAINS_CLAIM"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    USES_DATASET = "USES_DATASET"
    USES_METHOD = "USES_METHOD"
    SIMILAR_TO = "SIMILAR_TO"
    BELONGS_TO_TREND = "BELONGS_TO_TREND"
    DEBATE_ABOUT = "DEBATE_ABOUT"


class GraphNode(BaseModel):
    id: str = Field(default="", description="节点唯一ID")
    label: NodeLabel = Field(default=NodeLabel.PAPER, description="节点标签")
    properties: dict = Field(default_factory=dict, description="节点属性")
    embedding: Optional[List[float]] = Field(default=None, description="节点向量嵌入")
    created_at: Optional[datetime] = Field(default=None, description="创建时间")
    last_updated: Optional[datetime] = Field(default=None, description="最后更新时间")


class GraphEdge(BaseModel):
    source_id: str = Field(default="", description="源节点ID")
    target_id: str = Field(default="", description="目标节点ID")
    type: EdgeType = Field(default=EdgeType.CITES, description="边类型")
    properties: dict = Field(default_factory=dict, description="边属性")
    weight: float = Field(default=1.0, description="边权重")
    created_at: Optional[datetime] = Field(default=None, description="创建时间")


class InterestTopic(BaseModel):
    id: str = Field(default="", description="课题ID")
    name: str = Field(default="", description="课题名称")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    weight: float = Field(default=1.0, description="兴趣权重 (0.0~1.0)")
    description: str = Field(default="", description="课题描述")


class InterestProfile(BaseModel):
    researcher_id: str = Field(default="", description="研究者ID")
    researcher_name: str = Field(default="", description="研究者名称")
    active_topics: List[InterestTopic] = Field(default_factory=list, description="活跃课题列表")
    total_weight: float = Field(default=1.0, description="总权重")
    last_updated: Optional[datetime] = Field(default=None, description="上次更新时间")

    def get_keyword_weights(self) -> dict:
        weights: dict = {}
        for topic in self.active_topics:
            for kw in topic.keywords:
                weights[kw] = weights.get(kw, 0) + topic.weight * self.total_weight
        return weights


class DebateRecord(BaseModel):
    id: str = Field(default="", description="辩论记录ID")
    topic: str = Field(default="", description="辩论主题")
    paper_a_id: str = Field(default="", description="持方论文ID")
    paper_b_id: str = Field(default="", description="反方论文ID")
    rounds: List[DebateRound] = Field(default_factory=list, description="辩论轮次")
    panorama: Optional[DisputePanorama] = Field(default=None, description="争议全景图")
    human_verdict: Optional[str] = Field(default=None, description="人类裁决结果")
    created_at: Optional[datetime] = Field(default=None, description="创建时间")


class DebateRound(BaseModel):
    round_num: int = Field(default=1, description="轮次编号")
    proponent_argument: str = Field(default="", description="持方论证")
    opponent_argument: str = Field(default="", description="反方反驳")
    timestamp: Optional[datetime] = Field(default=None, description="时间戳")
    token_cost: int = Field(default=0, description="Token消耗")


class DisputePanorama(BaseModel):
    core_disagreement: str = Field(default="", description="核心分歧点")
    a_experiment_rigor: str = Field(default="", description="持方论文实验严格性评价")
    b_experiment_rigor: str = Field(default="", description="反方论文实验严格性评价")
    a_theory_assumption: str = Field(default="", description="持方理论假设评价")
    b_theory_assumption: str = Field(default="", description="反方理论假设评价")
    suggested_resolution: str = Field(default="", description="建议解决方案")
    confidence: float = Field(default=0.0, description="全景图置信度")
