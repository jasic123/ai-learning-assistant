# 会话入口：通过 with 管理数据库连接和事务，调用方无需重复编写提交、回滚和清理。
"""SQLite 连接与事务；导入模块时不建库、不创建文件。"""
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from app.config import DATA_DIR
from app.rag.db.models import Base


@contextmanager
def session_scope(data_dir=None, *, write=False):
    """调用时创建数据目录和表；data_dir 可注入临时目录用于隔离测试。"""
    # 测试可注入临时目录；正式运行默认使用项目 data 目录。
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    # 缺少上级目录时一并创建，目录已存在则复用。
    directory.mkdir(parents=True, exist_ok=True)
    # 引擎管理连接；timeout=30 是数据库锁等待时间，不是全部 SQL 的执行时限。
    engine = create_engine(URL.create("sqlite", database=str(directory / "assistant.db")),
                           connect_args={"timeout": 30})

    # 对引擎创建的每个新连接启用外键检查；下划线参数是此处未使用的连接记录。
    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _):
        cursor = connection.cursor()
        # SQLite 外键检查按连接开启，确保文件和构建记录不能引用不存在的知识库。
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        # 根据已注册模型创建缺失表；不会自动修改已有表结构或执行数据库迁移。
        Base.metadata.create_all(engine)
        # 退出此 with 时关闭会话；提交后保留已加载属性，不自动将对象刷新为最新数据库状态。
        with Session(engine, expire_on_commit=False) as session:
            try:
                if write:
                    # 写操作串行化，尤其不能让两个发布操作交叉替换 CURRENT。
                    # 提前取得写事务，其他写入需要等待；文件系统的 CURRENT 不受数据库回滚直接控制。
                    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                # 将会话交给调用方的 with 代码块；执行完或抛出异常后回到这里继续清理。
                yield session
                # 调用方正常结束后提交事务，使本次新增和修改正式生效。
                session.commit()
            except Exception:
                # 操作或提交失败时撤销本事务未提交的数据库修改，再把异常向外抛出。
                session.rollback()
                raise
    finally:
        # 每次调用都有独立引擎，结束时释放池中空闲连接，避免长期占用资源。
        engine.dispose()


# 仅进入并退出会话以触发目录、数据库和表初始化，不新增知识库或文件记录。
def init_db(data_dir=None):
    # 即使 with 内只有 pass，进入会话时也会执行目录创建及建表。
    with session_scope(data_dir):
        pass
