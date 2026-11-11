# 第一阶段：终端 RAG 使用指南

## 数据类型与模块职责

- app/rag/domain/models.py：Document（清洗后一页）、Chunk、SearchResult。
- app/rag/ingestion/：PDF 读取、OCR、清洗、切片；pipeline.ingest_document 处理单文件。
- app/rag/indexing/：批量 Embedding、FAISS/JSON 存取及索引发布；pipeline.build_index 串联入库。
- app/rag/retrieval/：FAISS + BM25+、RRF、可选精排，返回带 score_type 的 SearchResult。
- app/llm/client.py：Agent 与 RAG 共用的官方 OpenAI SDK 模型客户端。
- app/rag/service.py：对外检索服务；domain/contracts.py 定义请求和响应。
- app/rag/services/：终端独立问答辅助。
- app/tools/knowledge.py：Agent 知识查询工具，返回资料与来源。
- app/rag/db/：SQLite 知识库、文件和构建记录；共享路径校验在 app/config.py。
- scripts/：手动预览、单份 PDF 入库验收、真实问答验收。

详见 [项目架构说明](ARCHITECTURE.md)。RAG 命令入口为 python -m app.rag，数据库命令入口为 python -m app.rag.db。

终端命令不变。Python 调用中，document_chunks() 的第一项返回值已改为 list[Chunk]，使用 chunk.text、chunk.source、chunk.page；需要字典时调用 chunk.to_dict()。检索结果使用 hit.chunk.text 和 hit.score。旧 JSON 索引字段仍可读取，本次重构不要求重建向量。

处理报告 processed_pages 统计已经加载并清洗的页面数，chunks 统计实际返回的片段数。为识别重复页眉，会先加载所选页，即使 max_chunks 提前限制片段数量。

## 已完成与边界

已实现：从项目 .env 读取模型配置；兼容接口聊天/Embedding 客户端；PDF 分页解析；字符切片；批量远程向量化；FAISS 余弦相似度检索；文件名、物理页码与来源编号；基于片段的问答；终端命令；离线回归测试。

当前支持 PDF、本地中英文 OCR、向量/关键词/混合检索，以及可选远程精排。SQLite 知识库、文件与索引构建元数据已实现，见 [元数据指南](SQLITE_METADATA.md)。Agent 知识工具循环已实现，见 [Agent 架构](AGENT_ARCHITECTURE.md)；前端和业务 HTTP 接口尚未实现。聊天模型按提示词决定资料是否足够，尚无经过标注集校准的自动拒答保证。检索分数不是可信度，来源引用也需要核验。

FAISS 和元数据采用版本目录 + CURRENT 指针保存。重建失败不会切换当前索引；旧版本暂时保留，占用磁盘，后续再加入清理。不要加载不可信来源的 FAISS 索引文件。原始资料不修改。

## 你的资料检查结果

资料位置：data/uploads/教育资料/，共 3 份 PDF。

- Python参考手册(第4版).pdf：538 页；第 10、30、100 页抽样只能提取到水印，不能将这些页当作有效正文。
- Python算法教程_中文版.pdf：334 页；第 10、30、100 页抽样没有提取到文字。
- 用Python进行自然语言处理.pdf：382 页；抽样有文字，但英文有较多断行，后续需要验证解析质量。

以上是抽样结果，不是全部页面的 OCR 结论。扩大到前 30 页时，参考手册有少量文字页产生 6 个切片，算法教程仍没有文字；不能据此断言参考手册整本都无文字。缺失正文如需参与问答，应先 OCR 或更换有正文文字层的版本。当前代码跳过字母、数字及中文等有效字符少于 40 个的页面，并在解析报告中记录空页和低文字量页面。页码均为 PDF 物理页码，可能与书内印刷页码不同。

## 1. 进入项目并配置模型

PowerShell：

~~~powershell
cd D:\Python\PythonProject1
~~~

项目已提供 .env；请在 PyCharm 打开它，在本机填写下面六项，不要把真实密钥粘贴到聊天或提交 Git：

~~~dotenv
LLM_API_BASE=你的聊天服务API基址
LLM_API_KEY=你的聊天服务密钥
LLM_MODEL=该服务实际支持的聊天模型名称
EMBEDDING_API_BASE=你的向量服务API基址
EMBEDDING_API_KEY=你的向量服务密钥
EMBEDDING_MODEL=该服务实际支持的向量模型名称
~~~

API_BASE 填提供商给出的版本基址（例如以 /v1 结束的基址），不包含 /chat/completions 或 /embeddings。即使同一提供商共用密钥，两项也要分别填写。系统环境变量优先于 .env；变更后重新运行命令。

当前通过官方 OpenAI SDK 调用兼容的 /chat/completions 和 /embeddings 接口。向量接口请求浮点编码，不假设模型名称或向量维度。更换 Embedding 服务、模型或切片方案后需要重新入库。默认 400 字符切片、60 字符重叠，不等于 token 数；仍须遵守实际模型输入限制。

## 2. 验证两个模型接口

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -X utf8 -m app.rag.cli smoke
~~~

该命令只发送一小段通用测试文本，不发送你的 PDF。应显示 Embedding 维度和聊天回复。缺配置时会打印缺少的字段，不会偷偷选用其他服务。

## 3. 本地检查文档（不调用 API）

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -X utf8 -m app.rag.cli inspect --kb 教育资料 --max-pages 30 --max-chunks 20
~~~

max-pages 是每份 PDF 的页数上限，max-chunks 是所有文件合计的切片上限。遇到总切片上限后会停止，因此部分文件可能尚未被处理。预览含原文，不要把敏感内容随意外发。

## 4. 小规模入库（调用 Embedding，会产生服务费用）

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -X utf8 -m app.rag.cli ingest --kb 教育资料 --max-pages 30 --max-chunks 20
~~~

这一步会将实际提取出的资料文本发送到 .env 指定的 Embedding 服务，并在 data/indexes/教育资料/ 保存索引。小规模验证成功后再决定是否全量入库。省略两个限制参数意味着处理全部 PDF 页面，可能有较多调用费用；扫描页默认先自动执行本地 OCR。

成功报告中的 documents 列出处理页数、空页、低文字量页和切片数。检索范围仅包含实际入库片段，不代表整本书已经可问答。

## 5. 先看检索片段

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -X utf8 -m app.rag.cli search --kb 教育资料 --question "自然语言处理可以做什么？" --top-k 4
~~~

默认检索会调用 Embedding 问题向量接口，同时执行本地 BM25+ 并进行 RRF 融合；显示片段、文件名、PDF 页码和标明类型的分数。--mode bm25 不调用 Embedding。请根据入库范围调整问题，确认片段有答案后再调用聊天模型。

## 6. 带来源问答

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -X utf8 -m app.rag.cli ask --kb 教育资料 --question "自然语言处理可以做什么？" --top-k 4
~~~

会将问题与召回片段发送给聊天服务；答案用 [S1] 等标记引用，终端显示编号对应的文件和页码。当前不能保证所有引用都正确，应人工核验。

完整链路通过 `--rerank` 启用精排，需另填 RERANK_API_BASE、RERANK_API_KEY、RERANK_MODEL；配置和模式对比见 [混合检索指南](HYBRID_RETRIEVAL.md)。

## 7. 开发验证与安装复现

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -m unittest discover -s tests -v
& "D:\miniconda\envs\personal_assistant\python.exe" -m pip check
~~~

测试使用临时生成的 PDF、假向量和 HTTP 模拟服务，不验证真实模型效果。真实模型效果需要另外运行手动验收，不能用假模型测试通过代替。

requirements.txt 是直接依赖列表；requirements.lock.txt 是本次安装环境的版本快照。环境为 Windows / Python 3.11。重建同类环境时使用：

~~~powershell
& "D:\miniconda\envs\personal_assistant\python.exe" -m pip install -r requirements.lock.txt
~~~

后端基础入口仍可用；当前 RAG 通过终端调用，尚未接入 HTTP 业务路由。


## PDF 清洗与切片更新（2026-10-25）

当前已接入 app/rag/ingestion/cleaners.py：统一换行、保守 Unicode 规范化、移除无效控制字符；识别本项目抽样观察到的 TopSage 边缘水印；移除与 PDF 物理页码一致的边缘页码；检测所选页面至少 3 页且 80% 以上出现的完全相同边缘短行。删除行及原因保存在入库报告的 page_diagnostics 中，供人工复核。只保留高频边缘文本是启发式，不保证每个删除项都正确。

普通提取出现大量一两个字符的碎行时尝试 pypdf layout 提取，要求其非空白文字量至少为原方式的 70% 且碎行比例下降。layout 不是 OCR，也不保证阅读顺序，返回 layout_fallback_check_reading_order 提示；应对照原 PDF 检查正文和代码顺序，不能只凭行数减少判定成功。

切片优先级：空行段落 → 中英文句末 → 换行 → 空格/分号 → 长块字符兜底。默认仍为 400 字符上限、目标重叠 60 字符；边界调整可能缩短实际重叠。保存代码缩进；offset 指清洗后该页文本位置，不是原始 PDF 字节位置。

当前局限：仍按页处理，跨页段落不会合并；超长代码块仍可能被硬切；没有标题层级、表格结构和 token 上限计数。OCR 不保证阅读顺序、代码缩进和符号准确。短且低文字量的有价值代码页仍可能被标记跳过，应检查报告。先用 10～20 个有来源的真实问题比较召回，再调整长度和重叠，不能把默认参数当作最优值。

## 扫描页标记与本地 OCR（2026-10-26）

load_pdf() 默认启用 OCR，原有调用不需要修改。先清洗文字层；有效字符不足 40 且包含图片时标记为疑似扫描页。未解决的英文碎字页面也可触发 OCR。只渲染候选页，在 CPU 上运行 RapidOCR 中英文模型，随后对 OCR 文字重新检测页眉页脚并清洗。OCR 达到最低文字量且没有英文碎行警告时才采用，否则保留原提取结果并记录待处理状态。没有图片的短文字页和真正空白页不会仅因文字少就调用 OCR。

~~~python
from pathlib import Path
from app.rag.ingestion.loaders import load_pdf

path = Path(r"D:\Python\PythonProject1\data\uploads\教育资料\Python参考手册(第4版).pdf")
documents, report = load_pdf(path, source=path.name, max_pages=3)
print("疑似扫描页：", report["suspected_scanned_pages"])
print("OCR 成功页：", report["ocr_pages"])
print("仍需 OCR 页：", report["needs_ocr_pages"])
print("OCR 失败页：", report["ocr_failed_pages"])
~~~

- suspected_scanned_pages：初始判定为疑似扫描的页面，成功后仍保留这一来源标记。
- ocr_candidate_pages：全部 OCR 候选，包括英文碎字页面。
- ocr_pages：成功采用 OCR 文字的页面，不保证识字、代码或事实内容完全正确。
- needs_ocr_pages：禁用、失败、依赖缺失或识别文字不足等尚未恢复有效文字的候选页。
- ocr_failed_pages：渲染/识别失败或依赖不可用的页。
- page_diagnostics：每页的 ocr_status、ocr_confidence、needs_ocr、has_images，以及原文字层字符量。置信度不是阅读顺序或答案可信度。

page_diagnostics 中 raw_chars 是最终清洗输入的字符数，OCR 成功时对应 OCR 文字；text_layer_chars 始终对应最初的文字层。low_text_pages 按最终清洗结果计算。empty_pages 表示最终没有可用提取文字，不代表视觉上一定是白纸。

仅检查文字层并记录候选页：load_pdf(..., ocr_enabled=False)，或 inspect/ingest 添加 --no-ocr。OCR 错误会记录到报告，单页失败不阻断其他页；依赖缺失时在当前 Conda 环境执行 pip install -r requirements.txt。OCR 模型随 rapidocr-onnxruntime 包安装，识别过程无须调用远程 API。

直接运行 tests/test_loader.py 可查看这本手册前 20 页。自动回归：python -m unittest discover -s tests -p test_ocr.py -v。初次识别需加载模型，大量页面会增加处理时间；当前尚未缓存 OCR 结果。旧索引不会自动补入扫描正文，检查识别效果后再重新 ingest。

新索引元数据保存 preprocessing_version 与实际切片参数。已有向量索引不会自动更新；验证清洗预览后需要主动重新 ingest。此次没有调用模型、没有修改原 PDF、没有重建旧索引。
