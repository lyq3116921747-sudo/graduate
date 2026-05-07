from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import AgentConfig
from .utils import clean_text, normalize_text

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


def split_text_to_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            part = paragraph[start:end].strip()
            if part:
                chunks.append(part)
            if end >= len(paragraph):
                break
            start = max(end - overlap, start + 1)
        current = ""
    if current:
        chunks.append(current)
    return chunks


class PDFKnowledgeBase:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.cache_path = self.config.runtime_root / "rag" / "pdf_knowledge_cache.pkl"
        self.max_pages_per_pdf = 60
        self._chunks: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None

    def _iter_knowledge_files(self) -> List[Path]:
        files: List[Path] = []
        if not self.config.knowledge_root.exists():
            return files
        for pattern in ("*.pdf", "*.docx"):
            files.extend(sorted(self.config.knowledge_root.rglob(pattern)))
        deduped: List[Path] = []
        seen = set()
        for path in files:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(path)
        return deduped

    def _build_signature(self) -> List[Dict[str, Any]]:
        signature: List[Dict[str, Any]] = []
        for pdf_path in self._iter_knowledge_files():
            signature.append(
                {
                    "path": str(pdf_path.resolve()),
                    "mtime_ns": pdf_path.stat().st_mtime_ns,
                    "size": pdf_path.stat().st_size,
                }
            )
        return signature

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            payload = pickle.loads(self.cache_path.read_bytes())
        except Exception:
            return False
        if payload.get("signature") != self._build_signature():
            return False
        self._chunks = payload.get("chunks", [])
        self._metas = payload.get("metas", [])
        self._vectorizer = payload.get("vectorizer")
        self._matrix = payload.get("matrix")
        return bool(self._chunks and self._vectorizer is not None and self._matrix is not None)

    def _extract_pdf_chunks(self, pdf_path: Path) -> List[Dict[str, Any]]:
        if PdfReader is None:
            return []
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            return []

        extracted: List[Dict[str, Any]] = []
        page_count = min(len(reader.pages), self.max_pages_per_pdf)
        for page_idx in range(page_count):
            page = reader.pages[page_idx]
            try:
                page_text = normalize_text(page.extract_text() or "")
            except Exception:
                page_text = ""
            if not page_text:
                continue
            for chunk_idx, chunk in enumerate(split_text_to_chunks(page_text), start=1):
                extracted.append(
                    {
                        "text": chunk,
                        "meta": {
                            "path": str(pdf_path.resolve()),
                            "title": pdf_path.name,
                            "page": page_idx + 1,
                            "chunk_index": chunk_idx,
                        },
                    }
                )
        return extracted

    def _extract_docx_chunks(self, docx_path: Path) -> List[Dict[str, Any]]:
        if DocxDocument is None:
            return []
        try:
            document = DocxDocument(str(docx_path))
        except Exception:
            return []

        full_text = normalize_text("\n".join(paragraph.text for paragraph in document.paragraphs if clean_text(paragraph.text)))
        if not full_text:
            return []

        extracted: List[Dict[str, Any]] = []
        for chunk_idx, chunk in enumerate(split_text_to_chunks(full_text), start=1):
            extracted.append(
                {
                    "text": chunk,
                    "meta": {
                        "path": str(docx_path.resolve()),
                        "title": docx_path.name,
                        "page": 1,
                        "chunk_index": chunk_idx,
                    },
                }
            )
        return extracted

    def _extract_pages(self) -> None:
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []
        for knowledge_path in self._iter_knowledge_files():
            suffix = knowledge_path.suffix.lower()
            if suffix == ".pdf":
                extracted_items = self._extract_pdf_chunks(knowledge_path)
            elif suffix == ".docx":
                extracted_items = self._extract_docx_chunks(knowledge_path)
            else:
                extracted_items = []
            for item in extracted_items:
                chunks.append(item["text"])
                metas.append(item["meta"])

        self._chunks = chunks
        self._metas = metas
        if chunks:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                max_features=50000,
            )
            self._matrix = self._vectorizer.fit_transform(chunks)
        else:
            self._vectorizer = None
            self._matrix = None

        payload = {
            "signature": self._build_signature(),
            "chunks": self._chunks,
            "metas": self._metas,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(pickle.dumps(payload))

    def ensure_ready(self) -> None:
        if self._chunks and self._vectorizer is not None and self._matrix is not None:
            return
        if not self._load_cache():
            self._extract_pages()

    def _extract_query_terms(self, queries: Sequence[str]) -> List[str]:
        stop_terms = {
            "脓毒症",
            "感染",
            "诊断",
            "治疗",
            "指南",
            "定义",
            "器官功能障碍",
            "sofa",
            "sepsis",
            "guideline",
            "management",
        }
        terms: List[str] = []
        for query in queries:
            for token in re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,10}", query):
                normalized = token.lower()
                if normalized in stop_terms or normalized in terms:
                    continue
                terms.append(normalized)
        return terms

    def _title_bonus(self, title: str, query_terms: Sequence[str]) -> float:
        normalized_title = title.lower()
        bonus = 0.0
        for term in query_terms:
            if term and term in normalized_title:
                bonus += 0.03
        return min(bonus, 0.12)

    def search(self, queries: Sequence[str], top_k: int) -> Dict[str, Any]:
        self.ensure_ready()
        if not self._chunks or self._vectorizer is None or self._matrix is None:
            return {"items": [], "text": "", "sources": []}

        cleaned_queries = [clean_text(query) for query in queries if clean_text(query)]
        if not cleaned_queries:
            return {"items": [], "text": "", "sources": []}

        page_candidates: Dict[tuple[str, int], Dict[str, Any]] = {}
        for query in cleaned_queries:
            query_vector = self._vectorizer.transform([query])
            scores = self._matrix.dot(query_vector.T).toarray().ravel()
            page_scores: Dict[tuple[str, int], tuple[float, int]] = {}
            for index, raw_score in enumerate(scores):
                score = float(raw_score)
                if score <= 0:
                    continue
                meta = self._metas[index]
                page_key = (meta["path"], meta["page"])
                best = page_scores.get(page_key)
                if best is None or score > best[0]:
                    page_scores[page_key] = (score, index)

            for page_key, (score, index) in page_scores.items():
                meta = self._metas[index]
                candidate = page_candidates.setdefault(
                    page_key,
                    {
                        "path": meta["path"],
                        "title": meta["title"],
                        "page": meta["page"],
                        "best_index": index,
                        "score_sum": 0.0,
                        "score_max": 0.0,
                        "query_hits": 0,
                    },
                )
                candidate["score_sum"] += score
                candidate["score_max"] = max(candidate["score_max"], score)
                candidate["query_hits"] += 1
                if score >= candidate["score_max"]:
                    candidate["best_index"] = index

        if not page_candidates:
            return {"items": [], "text": "", "sources": []}

        query_terms = self._extract_query_terms(cleaned_queries)
        ranked_items = []
        for candidate in page_candidates.values():
            adjusted_score = (
                candidate["score_sum"]
                + candidate["score_max"] * 0.15
                + min(0.18, 0.06 * max(0, candidate["query_hits"] - 1))
                + self._title_bonus(candidate["title"], query_terms)
            )
            ranked_items.append((adjusted_score, candidate))
        ranked_items.sort(
            key=lambda item: (
                item[0],
                item[1]["query_hits"],
                item[1]["score_max"],
                -item[1]["page"],
            ),
            reverse=True,
        )

        items: List[Dict[str, Any]] = []
        selected_sources = set()
        doc_counts: Dict[str, int] = {}
        soft_doc_cap = 1 if max(1, top_k) <= 4 else 2
        hard_doc_cap = max(2, soft_doc_cap)

        for per_doc_cap in (soft_doc_cap, hard_doc_cap):
            for adjusted_score, candidate in ranked_items:
                if len(items) >= max(1, top_k):
                    break
                source_key = (candidate["path"], candidate["page"])
                if source_key in selected_sources:
                    continue
                if doc_counts.get(candidate["path"], 0) >= per_doc_cap:
                    continue
                selected_sources.add(source_key)
                doc_counts[candidate["path"]] = doc_counts.get(candidate["path"], 0) + 1
                best_index = int(candidate["best_index"])
                items.append(
                    {
                        "score": adjusted_score,
                        "title": candidate["title"],
                        "path": candidate["path"],
                        "page": candidate["page"],
                        "text": self._chunks[best_index],
                    }
                )
            if len(items) >= max(1, top_k):
                break

        text_lines: List[str] = []
        sources: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            tag = f"RAG{idx}"
            source = f"{item['path']}#page={item['page']}"
            snippet = clean_text(item["text"])[:280]
            text_lines.append(f"[{tag}] {item['title']} 第{item['page']}页：{snippet}")
            sources.append({"tag": tag, "source": source, "snippet": snippet})
        return {"items": items, "text": "\n".join(text_lines).strip(), "sources": sources}
