# 模型边界：此文件声明三张业务表；数据库连接及实际建表由 session.py 负责。
"""SQLite 业务元数据；正文和向量仍保存在索引版本目录。"""
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# 返回带 UTC 时区的 ISO 时间字符串，作为模型插入记录时的默认值生成函数。
def utc_now():
    # ISO 字符串保留时区，避免 SQLite 的 datetime 转换丢失 UTC 信息。
    return datetime.now(timezone.utc).isoformat()


# ORM 通用说明：Mapped[T] 声明映射属性及类型；T | None 通常表示数据库列允许 NULL。
# mapped_column 配置列的类型、主键、默认值和约束；JSON 属性更新时宜重新赋值以触发变更检测。
# SQLite 通常不强制 String(n) 的长度限制；严格长度校验需由业务或额外约束实现。
# 所有数据库模型共用的声明式基类；通过 Base.metadata 收集表结构，本身不对应业务表。
class Base(DeclarativeBase):
    pass


# 知识库表：一条记录代表一个知识库，保存简介及当前启用索引的概况。
class KnowledgeBase(Base):
    # 指定 SQLite 中的实际表名；定义类不会立即创建数据库或表。
    __tablename__ = "knowledgebases"

    # 知识库主键；文件表和构建记录表通过此编号关联知识库。
    id: Mapped[int] = mapped_column(primary_key=True)
    # 知识库名称；unique=True 禁止不同记录使用相同名称。
    name: Mapped[str] = mapped_column(String(255), unique=True)
    # 知识库简介，使用 Text 保存较长描述；默认是空字符串。
    kb_info: Mapped[str] = mapped_column(Text, default="")  # 供后续 Agent 判断知识库用途
    # 当前启用的索引版本编号，通常对应版本目录名；尚未建索引时允许为 None。此列未声明外键。
    current_generation: Mapped[str | None] = mapped_column(String(32))
    # 当前启用索引版本的切片总数，由业务代码维护，不由数据库自动统计。
    chunk_count: Mapped[int] = mapped_column(default=0)
    # 创建时间；default 传函数本身，在插入且未提供值时调用，不是在导入模块时计算。
    created_at: Mapped[str] = mapped_column(default=utc_now)
    # 最近更新时间；此处只有插入默认值，后续更新由业务代码显式设置。
    updated_at: Mapped[str] = mapped_column(default=utc_now)


# 文件表：一条记录代表知识库内一个文件的当前处理状态，不保存每个版本的文件历史快照。
class KnowledgeFile(Base):
    # 保存文件管理信息的数据库表名。
    __tablename__ = "knowledge_files"
    # 联合唯一约束：同一知识库内 source 不能重复，不同知识库可以保存相同文件路径。
    __table_args__ = (UniqueConstraint("knowledgebase_id", "source"),)

    # 文件记录的主键，与 PDF 内容中的页码无关。
    id: Mapped[int] = mapped_column(primary_key=True)
    # 所属知识库的外键；普通数据库索引用于加速按知识库查询文件，不是 FAISS 向量索引。
    knowledgebase_id: Mapped[int] = mapped_column(ForeignKey("knowledgebases.id"), index=True)
    # 保存相对路径便于迁移项目；相对路径的安全校验由业务代码负责。
    source: Mapped[str] = mapped_column(Text)  # uploads/知识库/ 内相对路径
    # 文件大小，单位为字节；文件不存在或大小未知时允许为 None。
    size_bytes: Mapped[int | None] = mapped_column()
    # pending 表示待处理，indexed 表示有切片入库，no_chunks 表示本次处理未生成切片。
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # indexed / no_chunks / pending；只描述当前版本，不代表整份文件都已入库。
    # 文件处理结果对应的索引版本；未参与当前构建时可以为 None。
    generation: Mapped[str | None] = mapped_column(String(32))
    # 原 PDF 总页数；尚未读取或未知时为 None，不等于本次处理页数。
    total_pages: Mapped[int | None] = mapped_column()
    # 报告记录的本次处理页数；受 max_pages 等参数限制，不保证每页都提取到有效文字。
    processed_pages: Mapped[int] = mapped_column(default=0)
    # 此文件在记录版本中贡献的切片数；与知识库总切片数区分。
    chunk_count: Mapped[int] = mapped_column(default=0)
    # JSON 保存清洗、OCR、异常页等报告；default=dict 为每条记录生成独立空字典。
    processing_report: Mapped[dict] = mapped_column(JSON, default=dict)
    # 首次登记此文件记录的时间，采用包含 UTC 时区的 ISO 字符串。
    created_at: Mapped[str] = mapped_column(default=utc_now)
    # 文件信息最近更新时间，由业务代码更新；default 不会自动跟踪后续修改。
    updated_at: Mapped[str] = mapped_column(default=utc_now)


# 构建记录表：一条记录代表一次索引构建，同一知识库可以保留多次构建记录。
class IndexBuild(Base):
    # 保存索引构建任务及配置的数据库表名。
    __tablename__ = "index_builds"

    # 32 位版本编号直接作为主键，并与磁盘索引版本目录对应，无需额外整数主键。
    generation: Mapped[str] = mapped_column(String(32), primary_key=True)
    # 本次构建所属知识库的外键；index=True 加速查询该知识库的构建历史。
    knowledgebase_id: Mapped[int] = mapped_column(ForeignKey("knowledgebases.id"), index=True)
    # 构建状态：building 为构建中，ready 为索引就绪，failed 为构建失败，由业务代码维护。
    status: Mapped[str] = mapped_column(String(32), default="building")
    # failed 仅记录异常类型，防止模型接口异常、配置内容进入数据库。
    # 失败时保存异常类名，例如 TimeoutError；成功或尚无错误时为 None。
    error_type: Mapped[str | None] = mapped_column(String(100))
    # 记录构建索引使用的 Embedding 模型，便于追溯并检查查询配置是否兼容。
    embedding_model: Mapped[str | None] = mapped_column(Text)
    # Embedding 服务地址，不保存 API Key；业务代码需确保地址本身不含密钥。
    embedding_base_url: Mapped[str | None] = mapped_column(Text)
    # 向量维度，例如 1024；构建尚未确定维度时允许为 None。
    dimension: Mapped[int | None] = mapped_column()
    # 这次构建写入索引的切片总数；保留历史值，不随知识库切换新版本而自动改变。
    chunk_count: Mapped[int] = mapped_column(default=0)
    # 清洗预处理逻辑的版本号，用于追溯不同版本索引的生成方式。
    preprocessing_version: Mapped[str | None] = mapped_column(String(100))
    # JSON 保存切片大小、重叠长度、处理页数、OCR 开关等构建参数。
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    # 开始构建的时间；插入记录且未传值时调用 utc_now 生成。
    started_at: Mapped[str] = mapped_column(default=utc_now)
    # 构建结束时间，成功或失败时由业务代码填写；进行中允许为 None。
    finished_at: Mapped[str | None] = mapped_column()
