#!/usr/bin/env python
"""DeepResearch-MultiAgent 主入口"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.utils.logger import setup_logging, get_logger
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.deep_parsing_agent import DeepParsingAgent
from src.agents.debate_agent import DebateAgent
from src.agents.meta_agent import MetaAgent
from src.knowledge_graph.graph_manager import KnowledgeGraphManager
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.llm.token_tracker import TokenTracker
from src.models.schema import InterestProfile, InterestTopic


def create_demo_profiles() -> list[InterestProfile]:
    return [
        InterestProfile(
            researcher_id="user_A",
            researcher_name="研究者 A",
            active_topics=[
                InterestTopic(
                    id="topic_1",
                    name="Mixture of Experts (MoE)",
                    keywords=["mixture of experts", "moe", "sparse", "routing", "expert selection"],
                    weight=0.9,
                    description="稀疏专家混合模型的路由策略与负载均衡",
                ),
                InterestTopic(
                    id="topic_2",
                    name="Large Language Model Training",
                    keywords=["language model", "llm", "pretraining", "scaling"],
                    weight=0.7,
                    description="大语言模型的高效训练方法",
                ),
            ],
        ),
        InterestProfile(
            researcher_id="user_B",
            researcher_name="研究者 B",
            active_topics=[
                InterestTopic(
                    id="topic_3",
                    name="State Space Models",
                    keywords=["state space model", "ssm", "mamba", "long sequence"],
                    weight=0.85,
                    description="状态空间模型在长序列建模中的应用",
                ),
                InterestTopic(
                    id="topic_4",
                    name="Efficient Inference",
                    keywords=["inference", "latency", "throughput", "acceleration"],
                    weight=0.6,
                    description="大模型高效推理加速方法",
                ),
            ],
        ),
    ]


def run_retrieval(args):
    logger = get_logger("Main.Retrieval")
    tracker = TokenTracker()
    agent = RetrievalAgent(token_tracker=tracker)
    profiles = create_demo_profiles()

    logger.info("【第一层】检索与初筛Agent 启动")
    papers = agent.execute(
        interest_profiles=profiles,
        keywords=args.query or "machine learning",
    )

    logger.info(f"候选论文: {len(papers)} 篇")
    for p in papers[:5]:
        logger.info(f"  [{p.screening_score:.2f}] {p.title[:80]}...")

    report = tracker.generate_report()
    logger.info(f"Token 消耗: {report['total_tokens']:,}")
    return papers


def run_parsing(args):
    logger = get_logger("Main.Parsing")
    tracker = TokenTracker()
    agent = DeepParsingAgent(token_tracker=tracker)

    logger.info("【第二层】深度解析Agent 启动")
    papers = run_retrieval(args) if not args.input else _load_papers(args.input)

    if not papers:
        logger.warning("没有论文需要解析")
        return []

    results = agent.execute(papers)
    report = tracker.generate_report()
    logger.info(f"Token 消耗: {report['total_tokens']:,}")
    return results


def run_debate(args):
    logger = get_logger("Main.Debate")
    tracker = TokenTracker()
    agent = DebateAgent(token_tracker=tracker)
    logger.info("【第三层】协作辩论Agent 启动")

    logger.warning("辩论模式需要两篇论文。运行完整流水线以自动检测冲突...")
    papers = run_parsing(args)
    conflicted = [p for p in papers if p.has_conflicts]
    logger.info(f"检测到 {len(conflicted)} 篇有冲突的论文")

    for paper in conflicted:
        for cr in paper.cross_references:
            if cr.is_conflict:
                logger.info(f"  冲突: {paper.id[:20]} vs {cr.related_paper_id[:20]}: {cr.description[:80]}")

    report = tracker.generate_report()
    logger.info(f"Token 消耗: {report['total_tokens']:,}")
    return conflicted


def run_meta(args):
    logger = get_logger("Main.Meta")
    tracker = TokenTracker()
    neo4j_client = Neo4jClient(
        uri=get_settings().neo4j.uri,
        user=get_settings().neo4j.user,
        password=get_settings().neo4j.password,
    )
    kg_manager = KnowledgeGraphManager(neo4j_client)
    agent = MetaAgent(token_tracker=tracker)

    logger.info("【元认知层】知识图谱自生长回路 启动")
    paper_nodes = kg_manager.get_all_paper_nodes()
    result = agent.execute(paper_nodes)

    for label, briefing in result.get("briefings", {}).items():
        logger.info(f"趋势: {label}")
        logger.info(f"  {briefing['briefing'][:150]}...")

    report = tracker.generate_report()
    logger.info(f"Token 消耗: {report['total_tokens']:,}")
    return result


def run_full_pipeline(args):
    logger = get_logger("Main.Pipeline")
    global_tracker = TokenTracker()
    settings = get_settings()

    neo4j_client = Neo4jClient(
        uri=settings.neo4j.uri,
        user=settings.neo4j.user,
        password=settings.neo4j.password,
    )
    kg_manager = KnowledgeGraphManager(neo4j_client)

    logger.info("=" * 60)
    logger.info(f"DeepResearch-MultiAgent v2.3.1 启动")
    logger.info(f"加载 {settings.num_researchers} 位研究者的兴趣权重图谱")
    stats = kg_manager.get_stats()
    logger.info(
        f"当前知识图谱: {stats['paper_nodes']:,} 节点, "
        f"{stats['edges']:,} 边"
    )
    logger.info("=" * 60)

    profiles = create_demo_profiles()
    retrieval_tracker = TokenTracker()

    retrieval_agent = RetrievalAgent(token_tracker=retrieval_tracker)
    papers = retrieval_agent.execute(
        interest_profiles=profiles,
        keywords=args.query or "machine learning artificial intelligence",
    )

    if not papers:
        logger.warning("没有找到相关论文，流水线终止。")
        return

    logger.info(f"初筛完成: {len(papers)} 篇候选论文")

    parsing_tracker = TokenTracker()
    parsing_agent = DeepParsingAgent(token_tracker=parsing_tracker, max_workers=3)
    knowledge_base = kg_manager.get_all_paper_nodes()
    kb_papers = [
        p for p in []
    ]

    results = parsing_agent.execute(papers, knowledge_base=kb_papers)

    debate_tracker = TokenTracker()
    debate_agent = DebateAgent(token_tracker=debate_tracker)
    debate_records = debate_agent.process_conflicts(results, kb_papers)

    new_items = kg_manager.sync_papers(results)
    for record in debate_records:
        kg_manager.add_debate_record(record)

    meta_tracker = TokenTracker()
    meta_agent = MetaAgent(token_tracker=meta_tracker)
    paper_nodes = kg_manager.get_all_paper_nodes()
    trend_report = meta_agent.execute(paper_nodes, new_papers=results)

    global_tracker.add(
        retrieval_tracker.total_tokens, 0, "deepseek-flash", "RetrievalAgent"
    )
    global_tracker.add(
        int(parsing_tracker.total_tokens * 0.5), 0, "deepseek-flash", "DeepParsingAgent"
    )
    global_tracker.add(
        int(parsing_tracker.total_tokens * 0.5), 0, "deepseek-chat", "DeepParsingAgent"
    )
    global_tracker.add(
        debate_tracker.total_tokens, 0, "deepseek-chat", "DebateAgent"
    )
    global_tracker.add(
        meta_tracker.total_tokens, 0, "deepseek-chat", "MetaAgent"
    )

    logger.info("\n" + "=" * 60)
    logger.info("每日任务完成 - 资源消耗报告")
    logger.info("-" * 60)
    logger.info(f"总 Token 消耗:     {global_tracker.total_tokens:>15,}")
    logger.info(f"  - DeepSeek-Flash: {retrieval_tracker.total_tokens + int(parsing_tracker.total_tokens * 0.5):>15,} tokens")
    logger.info(f"  - DeepSeek-Pro:   {int(parsing_tracker.total_tokens * 0.5) + debate_tracker.total_tokens + meta_tracker.total_tokens:>15,} tokens")
    logger.info(f"处理论文:         {len(results):>15} 篇深入解析")
    logger.info(f"生成辩论:         {len(debate_records):>15} 组争议全景图")
    logger.info(f"新知识图谱节点:   {new_items:>15} (含声明节点)")
    logger.info(f"趋势簇简报:       {len(trend_report.get('briefings', {})):>15} 份生成 & 推送")
    logger.info("=" * 60)
    logger.info("会话结束。所有数据已持久化。")


def _load_papers(filepath: str) -> list:
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    from src.models.paper import Paper
    return [Paper(**item) for item in data]


def main():
    parser = argparse.ArgumentParser(
        description="DeepResearch-MultiAgent - 学术论文深度研读与知识管理Agent系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.main --mode full
  python -m src.main --mode retrieval --query "transformer attention"
  python -m src.main --mode parsing
  python -m src.main --mode debate
  python -m src.main --mode meta
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "retrieval", "parsing", "debate", "meta"],
        default="full",
        help="运行模式 (默认: full)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="搜索查询关键词",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="",
        help="输入论文JSON文件路径",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="显示版本信息",
    )

    args = parser.parse_args()

    if args.version:
        print("DeepResearch-MultiAgent v2.3.1")
        return

    setup_logging(args.log_level)
    logger = get_logger("Main")

    try:
        if args.mode == "full":
            run_full_pipeline(args)
        elif args.mode == "retrieval":
            run_retrieval(args)
        elif args.mode == "parsing":
            run_parsing(args)
        elif args.mode == "debate":
            run_debate(args)
        elif args.mode == "meta":
            run_meta(args)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
