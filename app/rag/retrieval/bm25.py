"""关键词召回：从当前版本的 chunks.json 在内存中建立 BM25，不另存数据库。"""
from functools import lru_cache
import logging
import re
import unicodedata

from app.rag.domain.models import SearchResult


_QUERY_STOP_WORDS = {
    "在", "上", "中", "时", "如何", "怎样", "什么", "为什么", "哪个", "哪些", "是否",
    "分别", "作用", "用来", "进行", "使用", "通过", "一个", "以及", "和", "与", "的",
    "了", "是", "把", "对", "这", "那", "其", "而", "并", "或者", "吗", "呢",
    "可以", "能够", "怎么", "多少", "不", "还", "就", "会", "被", "所",
}

# Domain vocabulary bridges common learner wording and terms used in the books/code.
# Keep this list small and review it with the retrieval evaluation set.
_QUERY_TERM_ALIASES = {
    "defaultdict": ("default_factory", "setdefault", "keyerror"),
    "找零": ("denom", "coin", "change", "payed"),
    "贪心": ("greedy",),
    "词性": ("tag", "pos", "noun", "det"),
    "标签": ("tag", "pos"),
    "discourse": ("cs_semrep", "s_retrieve", "reading", "trace"),
    "歧义": ("ambiguous", "reading", "semrep"),
    "sql": ("load_parser", "sql0.fcfg", "sem"),
    "查询": ("query", "parser"),
}


@lru_cache(maxsize=1)
def _tokenizer():
    # 延迟加载分词器，普通模块导入和纯向量检索不初始化词典。
    import jieba
    jieba.setLogLevel(logging.WARNING)
    return jieba.Tokenizer()


def tokenize(text):
    """文档和问题使用同一规则：统一大小写和全角字符，搜索分词并过滤标点。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in _tokenizer().lcut_for_search(text, HMM=False)
            if re.search(r"[a-z0-9\u3400-\u9fff]", word)]


def query_terms(text, *, filter_query_stop_words=False, expand=True):
    terms = tokenize(text)
    if filter_query_stop_words:
        terms = [term for term in terms if term not in _QUERY_STOP_WORDS]
    if not expand:
        return terms
    expanded = list(terms)
    seen = set(expanded)
    for term in terms:
        for alias in _QUERY_TERM_ALIASES.get(term, ()):
            if alias not in seen:
                expanded.append(alias)
                seen.add(alias)
    return expanded


class BM25Retriever:
    """BM25+ 的 IDF 始终为正，避免少量文档时常见词出现零分或负分。"""

    def __init__(self, chunks):
        from rank_bm25 import BM25Plus
        self.chunks = chunks
        corpus = [tokenize(chunk.text) for chunk in chunks]
        # 全是标点时没有可索引词项，不能构造平均词数为零的 BM25 模型。
        self.model = BM25Plus(corpus, k1=1.5, b=0.75, delta=1.0) if any(corpus) else None

    def search(self, query, limit=30, *, filter_query_stop_words=False, expand_query=True):
        if not query.strip() or type(limit) is not int or limit < 1:
            raise ValueError("问题不能为空，关键词召回数量必须为正整数")
        terms = query_terms(query, filter_query_stop_words=filter_query_stop_words, expand=expand_query)
        if not terms or self.model is None:
            return []
        scores = self.model.get_scores(terms)
        # BM25+ 的 delta 会给未命中文档一个基线分，必须显式过滤没有共同词项的片段。
        matched = [i for i, frequencies in enumerate(self.model.doc_freqs)
                   if any(term in frequencies for term in terms)]
        ranked = sorted(matched, key=lambda i: (-float(scores[i]), i))[:limit]
        return [SearchResult(self.chunks[i], float(scores[i]), score_type="bm25") for i in ranked]
