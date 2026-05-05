[README.md](https://github.com/user-attachments/files/27409538/README.md)
# DeepResearch-MultiAgent

![Version](https://img.shields.io/badge/version-2.3.1-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**DeepResearch-MultiAgent** 是一个面向学术论文的深度研读与知识管理多Agent系统。它通过三层长链推理与多Agent协作架构，帮助研究人员从每日数百篇新论文中快速筛选、深度理解并进行跨文献关联推理。

## 核心痛点

研究人员面对每日数百篇新论文时，快速筛选、深度理解与跨文献关联推理效率低下。一篇核心论文的深度消化需要4-6小时，且难以快速发现多篇论文间隐含的论点冲突或技术演进脉络。

## 系统架构

### 三层多Agent协作架构

```
┌─────────────────────────────────────────────────────┐
│                   DeepResearch-MultiAgent            │
│                                                     │
│  第一层: 检索与初筛Agent (RetrievalAgent)              │
│  ├─ Semantic Scholar API + 自定义爬虫                 │
│  ├─ BERT语义向量化 + 兴趣权重图谱                      │
│  └─ 500篇 → 30篇 候选论文                            │
│                                                     │
│  第二层: 深度解析Agent (DeepParsingAgent)              │
│  ├─ Step 1: 结构树抽取 (动机-方法-实验-结论)           │
│  ├─ Step 2: 关键声明抽取                              │
│  ├─ Step 3: 逻辑断层验证                              │
│  ├─ Step 4: 结构化摘要生成                            │
│  └─ Step 5: 跨文献交叉引用分析                         │
│                                                     │
│  第三层: 协作辩论Agent (DebateAgent)                   │
│  ├─ 持方Agent (Proponent)                            │
│  ├─ 反方Agent (Opponent)                             │
│  └─ 三轮"主张-反驳-总结"对抗推理                       │
│                                                     │
│  知识图谱自生长回路 + 元认知Agent (MetaAgent)           │
│  └─ GNN趋势检测 → 简报生成 → 反馈逆向传播              │
└─────────────────────────────────────────────────────┘
```

### 知识图谱自生长回路

每周扫描知识库中累积的结构化摘要与争议全景图，运用图神经网络算法发现新兴研究趋势，自动生成"本周领域前沿简报"。用户反馈信号逆向传播，调整检索Agent的兴趣权重与解析Agent的关联判断阈值。

## 项目结构

```
DeepResearch-MultiAgent/
├── config/                     # 配置文件
│   ├── __init__.py
│   ├── settings.py             # 全局配置管理
│   └── default_config.yaml     # 默认配置
├── src/                        # 源代码
│   ├── __init__.py
│   ├── main.py                 # 主入口
│   ├── agents/                 # Agent模块
│   │   ├── base_agent.py       # Agent基类
│   │   ├── retrieval_agent.py  # 检索与初筛Agent
│   │   ├── deep_parsing_agent.py  # 深度解析Agent
│   │   ├── debate_agent.py     # 协作辩论Agent
│   │   └── meta_agent.py       # 元认知Agent
│   ├── llm/                    # LLM调用层
│   │   ├── provider.py         # 多Provider抽象
│   │   ├── deepseek_client.py  # DeepSeek API客户端
│   │   └── token_tracker.py    # Token消耗追踪器
│   ├── retrieval/              # 论文检索层
│   │   ├── arxiv_fetcher.py    # arXiv抓取器
│   │   ├── semantic_scholar.py # Semantic Scholar API
│   │   └── embedding.py        # 语义向量化模块
│   ├── knowledge_graph/        # 知识图谱
│   │   ├── graph_manager.py    # 图谱管理器
│   │   ├── neo4j_client.py     # Neo4j客户端
│   │   └── trend_detector.py   # GNN趋势检测
│   ├── models/                 # 数据模型
│   │   ├── paper.py            # 论文模型
│   │   └── schema.py           # 图数据库Schema
│   └── utils/                  # 工具函数
│       ├── logger.py           # 日志系统
│       └── helpers.py          # 辅助函数
├── scripts/                    # 脚本
│   └── run_daily_task.py       # 每日任务执行脚本
├── tests/                      # 测试
│   ├── __init__.py
│   └── test_agents.py          # Agent单元测试
├── .env.example                # 环境变量模板
├── requirements.txt            # Python依赖
├── pyproject.toml              # 项目元数据
└── README.md                   # 项目说明
```

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 14+ (数据持久化)
- Neo4j 5.x (知识图谱存储)

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/DeepResearch-MultiAgent.git
cd DeepResearch-MultiAgent

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入API密钥等信息
```

### 运行

```bash
# 执行每日例行任务（完整流水线）
python scripts/run_daily_task.py

# 仅运行检索与初筛
python -m src.main --mode retrieval

# 仅运行深度解析
python -m src.main --mode parsing

# 运行协作辩论
python -m src.main --mode debate --topic "Your Topic"

# 运行元认知趋势分析
python -m src.main --mode meta
```

## 支持的LLM Provider

| Provider | Model | 用途 |
|----------|-------|------|
| DeepSeek | deepseek-chat (V4-pro) | 高精度推理、辩论Agent |
| DeepSeek | deepseek-flash | 批量论文解析 |
| GLM-5.1 | glm-5.1 | 摘要生成、初筛推理 |
| OpenAI (可选) | gpt-4o | 备用推理 |

## 核心特性

- **三层长链推理**: 检索→解析→辩论，层层递进
- **可解释推理链**: 每个Agent输出完整的推理过程
- **观点冲突检测**: 自动发现论文间的论点矛盾
- **对抗性辩论**: 持方/反方多轮推理
- **知识图谱自生长**: 基于GNN的趋势发现与反馈学习
- **Token消耗精确追踪**: 每个Agent/每篇论文的Token用量可追溯
- **模块化插件架构**: 易于扩展新的检索源、LLM Provider和分析Agent

## API Token申请说明

本项目需要以下API Token：

1. **Semantic Scholar API** (免费): https://www.semanticscholar.org/product/api
2. **DeepSeek API**: https://platform.deepseek.com/api_keys
3. **GLM-5.1 API** (可选): https://open.bigmodel.cn/
4. **OpenAI API** (可选): https://platform.openai.com/api-keys

所有Token配置在 `.env` 文件中管理，参考 `.env.example`。

## 开发计划

- [ ] 支持更多论文源 (PubMed, IEEE Xplore)
- [ ] Web UI 研究看板
- [ ] 多语言论文支持 (中文/日文)
- [ ] RAG增强的论文问答
- [ ] 协作白标功能 (多人实时标注)

## License

MIT License

## 贡献者

欢迎提交 Issue 和 Pull Request！
