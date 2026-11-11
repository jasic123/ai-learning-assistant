# 混合检索、RRF 与精排

## 当前链路

```text
问题 ──→ Embedding → FAISS ──┐
     └─→ jieba → BM25+ ─────┤
                           ↓
                      等权 RRF 融合
                           ↓
                  可选远程 Rerank 精排
                           ↓
                     top_k 来源片段
                           ↓
                     LLM 带来源回答
```

默认 `hybrid`，两路各召回 30 条，按 Chunk.id 去重融合后保留 30 条，最后返回 4 条。
开启 `--rerank` 时先对这 30 条候选精排，再取最终 4 条；不会先截为 4 条再精排。
`--candidate-k` 必须不小于 `--top-k`，上限 200；最终 `--top-k` 范围为 1～20。

## 模块职责

| 文件 | 职责 |
|---|---|
| app/rag/retrieval/bm25.py | 文档与问题统一分词、BM25+ 关键词召回 |
| app/rag/retrieval/fusion.py | 两路等权 RRF、按 Chunk.id 去重 |
| app/rag/retrieval/reranker.py | 对融合候选调用精排客户端，保留来源及原检索分数 |
| app/rag/retrieval/retrievers.py | 固定版本加载、召回、融合、精排编排 |
| app/llm/client.py | 官方 OpenAI SDK 调用、扩展 Rerank 协议及响应校验 |

## BM25 与版本一致性

使用 jieba 搜索分词，中英文统一 NFKC 字符规范化、大小写和标点过滤。
采用 rank-bm25 的 BM25+（k1=1.5，b=0.75，delta=1.0），适合中文资料与少量文档的初期验证。
BM25+ 对未匹配文档也有基线分，因此实现额外排除与问题没有共同词项的片段。
仅中文标点的问题返回空关键词结果，不会把所有无关片段作为命中。

从已发布版本的 chunks.json 在内存延迟构建关键词索引，与同版本 FAISS 行号、来源一致。
同一 KnowledgeIndex 实例复用该索引；CURRENT 更新后重新创建实例才能加载新版本。
不新增正文数据库，不重复读取 PDF，不要求重建已有向量。
当前每次关键词查询对全部片段评分，适合单机初期规模；大规模时需要评估构建时间、内存和查询延迟。

## RRF 公式与分数

每条候选分数为 `0.5/(60+向量排名) + 0.5/(60+关键词排名)`，排名从 1 开始，
未进入某一路候选的项在该路贡献为 0。`--rrf-k` 可调整平滑常数，默认 60。
这里 0.5 : 0.5 是对两路排名贡献等权，不是把 BM25 原始分与余弦原始分直接相加。
同一通道重复 ID 只计一次，跨通道 ID 合并并累加贡献；同分保持首次召回顺序。

SearchResult.score_type 区分 cosine、bm25、rrf、rerank。
Rerank 结果用 score 保存精排分，用 retrieval_score 保留精排前分数；Chunk 正文和来源不被修改。
不同类型的分数不能直接比较，也不能视为答案正确概率。

## 使用命令

在项目根目录，使用 personal_assistant 环境运行：

```powershell
# 默认混合召回 + RRF，不启用精排
python -m app.rag search --kb 教育资料 --question "Python 列表如何追加元素？"

# 关键词检索完全本地运行，不需要 Embedding/LLM 密钥
python -m app.rag search --kb 教育资料 --question "append" --mode bm25

# 纯向量检索，便于对比基线
python -m app.rag search --kb 教育资料 --question "如何修改列表？" --mode vector

# 完整混合召回 + RRF + 精排 + 问答
python -m app.rag ask --kb 教育资料 --question "Python 列表如何追加元素？" --mode hybrid --candidate-k 30 --top-k 4 --rerank
```

`search --mode bm25` 仍读取知识库版本文件；需要之前已入库的知识库。
`ask` 额外需要 LLM 配置。`--rerank` 可叠加在任一召回模式上。

## 精排服务配置

在项目已有 .env 中补充下面三项，真实值只在本机填写；未使用 --rerank 时不用填：

```dotenv
RERANK_API_BASE=服务商提供的API基址
RERANK_API_KEY=精排服务密钥
RERANK_MODEL=服务商支持的精排模型名称
```

基址通常含 /v1，不含 /rerank。精排客户端不自动复用 Embedding 或聊天密钥。
通过官方 SDK 的自定义 post 方法接入支持下面协议的远程 /rerank 服务，普通聊天/Embedding 服务不一定支持该接口：

```json
{"model":"模型名","query":"问题","documents":["候选甲","候选乙"],"top_n":2,"return_documents":false}
```

期望响应包含 `results` 列表，每项有整数 index 和有限数值 relevance_score。
允许按相关性乱序返回，客户端会恢复为输入顺序；缺项、重复编号、非法分数全部报错。
启用精排后接口不可用会明确失败，不悄悄返回未精排结果。
候选正文会发往配置的精排服务；模型长度限制、费用和超时按服务商规则处理。
本次未安装或下载本地交叉编码器模型。

## 验证与效果比较

```powershell
python -m unittest discover -s tests -p test_hybrid_retrieval.py -v
python -m unittest discover -s tests -v

# 真实手册验收：首次重建选定正文范围，后续各模式复用同一索引
python -m scripts.validate_rag --mode vector --rebuild
python -m scripts.validate_rag --mode bm25
python -m scripts.validate_rag --mode hybrid
python -m scripts.validate_rag --mode hybrid --rerank
```

报告分别保存到 artifacts/rag-validation/vector.json、bm25.json、hybrid.json、hybrid-rerank.json。
脚本只验证指定问题的目标页召回和引用编号；事实正确性、引用支持关系、OCR 和代码仍需人工核验。
离线测试使用假向量、假精排分数与模拟 HTTP；不表示真实模型已经连通或检索质量已经提高。
调参应在相同索引和相同问题集上对比目标页/片段 Recall@k、排名和实际回答，不能只比较不同量纲的分数。
