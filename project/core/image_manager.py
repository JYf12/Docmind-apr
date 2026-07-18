"""
Image Summary Generator & Store Manager

Index time: extract image references from markdown, call a lightweight VLM to
generate textual summaries, and produce LangChain Document objects suitable for
the existing Qdrant child-chunk collection.

Query time: ImageStoreManager provides JSON-backed lookup of image metadata
(image path, description, source) keyed by image_id, analogous to how
ParentStoreManager serves parent chunks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
from langchain_core.documents import Document
from utils import clear_directory_contents


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _image_id(source_stem: str, img_filename: str) -> str:
    """Deterministic image_id: {doc_stem}__{img_stem}.

    >>> _image_id("paper", "images/fig1.png")
    'paper__images_fig1'
    """
    clean = img_filename.replace("\\", "/").rsplit("/", 1)[-1]  # basename
    clean = Path(clean).stem
    return f"{source_stem}__{clean}"


def _encode_image_base64(image_path: str | Path) -> str:
    """Read an image file and return a data-URI suitable for LLM vision."""
    img_path = Path(image_path)
    ext = img_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")

    with open(img_path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


# ---------------------------------------------------------------------------
# ImageSummaryGenerator
# ---------------------------------------------------------------------------

class ImageSummaryGenerator:
    """Extract image references from a Markdown file, call a VLM to produce
    a textual summary for each image, and return Qdrant-compatible Document
    objects that can be indexed alongside text child chunks."""

    # Regex to find ![...](images/...) references inside Markdown
    _IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*images/[^)]+)\)")

    def __init__(self):
        self._vlm = None  # lazy init

    # ---- VLM lazy init ----------------------------------------------------

    def _get_vlm(self):
        if self._vlm is not None:
            return self._vlm

        provider = config.IMAGE_SUMMARY_PROVIDER or config.ACTIVE_LLM_CONFIG
        model = config.IMAGE_SUMMARY_MODEL

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self._vlm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_API_URL,
            )
        elif provider == "ollama":
            print("Using Ollama for image summaries")
            from langchain_openai import ChatOpenAI
            active_cfg = config.LLM_CONFIGS.get("ollama", {})
            self._vlm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key="ollama",
                base_url=active_cfg.get("url", "http://localhost:11434") + "/v1",
            )
        elif provider == "qwen":
            print("Using Qwen for image summaries")
            from langchain_openai import ChatOpenAI
            active_cfg = config.LLM_CONFIGS.get("qwen", {})
            self._vlm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=active_cfg.get("api_key", ""),
                base_url=active_cfg.get("base_url", ""),
            )
        # elif provider == "anthropic":
        #     from langchain_anthropic import ChatAnthropic
        #     self._vlm = ChatAnthropic(model=model, temperature=0)
        # elif provider == "google":
        #     from langchain_google_genai import ChatGoogleGenerativeAI
        #     self._vlm = ChatGoogleGenerativeAI(model=model, temperature=0)
        else:
            raise ValueError(f"Unsupported IMAGE_SUMMARY_PROVIDER: {provider}")
        print(f"✓ Initialized VLM for image summaries: {provider} / {model}")
        return self._vlm

    # ---- public API -------------------------------------------------------

    def extract_image_refs(self, md_path: str | Path) -> List[dict]:
        """Parse a markdown file and return every embedded image reference.

        Each entry:
          {"image_id": str, "image_path": str, "alt_text": str, "context_text": str}
        """
        md_path = Path(md_path)
        source_stem = md_path.stem
        md_dir = md_path.parent
        text = md_path.read_text(encoding="utf-8")

        refs: List[dict] = []
        seen_paths = set()
        for m in self._IMG_RE.finditer(text):
            alt_text = m.group(1) or ""             # e.g. "Figure 1: Example chart"
            img_rel = m.group(2)                     # e.g. "images/fig1.png"
            img_abs = md_dir / img_rel

            # skip duplicate image references (same file referenced multiple times)
            img_key = str(img_abs)
            if img_key in seen_paths:
                continue
            seen_paths.add(img_key)

            if not img_abs.exists():
                print(f"  ⚠ Image not found, skipping: {img_abs}")
                continue

            # extract surrounding text (±300 chars) as context for VLM  根据图片位置提取前后300个字符作为上下文
            start = max(0, m.start() - 300)
            end = min(len(text), m.end() + 300)
            context_text = text[start:end].strip()

            refs.append({
                "image_id": _image_id(source_stem, img_rel),
                "image_path": str(img_abs),
                "alt_text": alt_text,
                "context_text": context_text,
            })

        return refs

    def generate_summary(
        self, image_path: str | Path, context_text: str = "", alt_text: str = ""
    ) -> str:
        """Call a VLM to produce a detailed 1-3 sentence summary of an image.

        The context_text (surrounding markdown) helps the VLM understand the
        document context the image appears in.
        """
        vlm = self._get_vlm()
        image_uri = _encode_image_base64(image_path)

        prompt_parts = [
            "Describe this image in detail as a concise 1-3 sentence paragraph in Chinese. ",
            "Include: key objects, charts, diagrams, numbers, relationships, and notable visual elements. ",
            "Focus on information that would be relevant for document retrieval and question answering.\n\n",
        ]
        if alt_text:
            prompt_parts.append(f"The image alt-text hint is: '{alt_text}'.\n\n")
        if context_text:
            truncated = context_text[:500]
            prompt_parts.append(
                f"The image appears near the following document text (for context only): \n"
                f'"""{truncated}"""'
            )

        # Use langchain-openai ChatOpenAI format for the vision call
        from langchain_core.messages import HumanMessage

        content: list = [{"type": "text", "text": "".join(prompt_parts)}]
        content.append({
            "type": "image_url",
            "image_url": {"url": image_uri},
        })

        response = vlm.invoke([HumanMessage(content=content)])
        return str(response.content).strip()

    def create_summary_chunks(self, md_path: str | Path) -> List[Document]:
        """Full pipeline for one markdown file: extract image refs → generate
        summaries → return Qdrant-compatible Document objects.

        These Documents are stored in the SAME Qdrant collection as text child
        chunks — the image summary is a text chunk with is_image_summary=True
        in metadata.
        """
        refs = self.extract_image_refs(md_path)
        if not refs:
            return []

        source_stem = Path(md_path).stem
        docs: List[Document] = []

        for ref in refs:
            image_id = ref["image_id"]
            image_path = ref["image_path"]

            print(f"  🖼  Generating summary for {image_id} ...")
            try:
                summary = self.generate_summary(
                    image_path,
                    context_text=ref["context_text"],
                    alt_text=ref["alt_text"],
                )
                print(f"  📄 VLM summary for {image_id}: {summary[:100]}...")
            except Exception as e:
                print(f"  ⚠ VLM summary failed for {image_id}: {e}")
                summary = f"Image: {ref['alt_text'] or Path(image_path).name}"

            docs.append(Document(
                page_content=(
                    # f"[IMAGE SUMMARY — {image_id}]\n"
                    f"IMAGE SUMMARY Description: {summary}\n"
                    # f"Image ID: {image_id}"
                ),
                metadata={
                    "source": f"{source_stem}.pdf",
                    "image_id": image_id,
                    "image_path": image_path,
                    "is_image_summary": True,
                },
            ))

            # persist to ImageStoreManager as a side effect
            ImageStoreManager().save(
                image_id=image_id,
                image_path=image_path,
                description=summary,
                alt_text=ref["alt_text"],
                source_file=f"{source_stem}.pdf",
            )

        return docs


# ---------------------------------------------------------------------------
# ImageStoreManager
# ---------------------------------------------------------------------------

class ImageStoreManager:
    """JSON-backed store for image metadata, mirroring ParentStoreManager.

    Each image is stored as {image_id}.json under config.IMAGE_STORE_PATH.
    """

    def __init__(self, store_path: str | None = None):
        self._store_path = Path(store_path or config.IMAGE_STORE_PATH)
        self._store_path.mkdir(parents=True, exist_ok=True)

    # ---- CRUD -------------------------------------------------------------

    def save(
        self,
        image_id: str,
        image_path: str,
        description: str,
        alt_text: str = "",
        source_file: str = "",
    ) -> None:
        file_path = self._store_path / f"{image_id}.json"
        file_path.write_text(
            json.dumps({
                "image_id": image_id,
                "image_path": image_path,
                "description": description,
                "alt_text": alt_text,
                "source_file": source_file,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, image_id: str) -> dict:
        """Return a single image's metadata dict."""
        file_path = self._store_path / (
            image_id if image_id.lower().endswith(".json") else f"{image_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))

    def load_many(self, image_ids: List[str]) -> List[dict]:
        """Return metadata for a deduplicated set of image IDs, sorted."""
        unique = sorted(set(image_ids))
        results: List[dict] = []
        for iid in unique:
            try:
                results.append(self.load(iid))
            except FileNotFoundError:
                print(f"  ⚠ Image metadata not found: {iid}")
        return results

    def clear_store(self) -> None:
        self._store_path.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self._store_path)
