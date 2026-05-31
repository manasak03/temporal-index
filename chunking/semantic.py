"""Semantic chunking for catalog and waitlist CSVs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from paths import (
    DEFAULT_CHUNKS_JSONL,
    DEFAULT_MODELS_CSV,
    DEFAULT_WAITLIST_CSV,
)

DEFAULT_MODELS_PATH = DEFAULT_MODELS_CSV
DEFAULT_WAITLIST_PATH = DEFAULT_WAITLIST_CSV

COLLECTION_HINTS: list[tuple[str, str]] = [
    ("cosmograph daytona", "Cosmograph Daytona"),
    ("daytona", "Cosmograph Daytona"),
    ("submariner", "Submariner"),
    ("datejust", "Datejust"),
    ("gmt master", "GMT-Master II"),
    ("gmt-master", "GMT-Master II"),
    ("oyster perpetual", "Oyster Perpetual"),
    ("explorer ii", "Explorer II"),
    ("explorer", "Explorer"),
    ("sky-dweller", "Sky-Dweller"),
    ("yacht-master ii", "Yacht-Master II"),
    ("yacht-master", "Yacht-Master"),
    ("day-date", "Day-Date"),
    ("sea-dweller", "Sea-Dweller"),
    ("deepsea", "Deepsea Sea-Dweller"),
    ("milgauss", "Milgauss"),
    ("air king", "Air King"),
    ("cellini", "Cellini"),
    ("pearlmaster", "Pearlmaster"),
]

JUNK_MODEL_PATTERN = re.compile(r"^➢$|^nan$", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(r"\b1[0-9]{5}[A-Z0-9]*\b", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkRecord:
    """A retrieval-ready Markdown chunk with provenance metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


def _normalize(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1}


def _resolve_collection(category: str, model: str) -> str | None:
    haystack = f"{category} {model}".lower()
    for hint, collection in COLLECTION_HINTS:
        if hint in haystack:
            return collection
    return None


def _is_valid_waitlist_row(row: pd.Series) -> bool:
    model = _normalize(row.get("Model"))
    if not model or JUNK_MODEL_PATTERN.match(model):
        return False
    min_wait = _normalize(row.get("Min Wait Time"))
    max_wait = _normalize(row.get("Max Wait Time"))
    market = _normalize(row.get("Market Price VS Retail Price"))
    return bool(min_wait or max_wait or market)


def _reference_prefix(reference: str) -> str:
    digits = re.sub(r"\D", "", reference)
    return digits[:5] if len(digits) >= 5 else digits


def _subgroup_label(collection: str, group: pd.DataFrame) -> str:
    sizes = sorted({_normalize(size) for size in group["Size"].tolist() if _normalize(size)})
    prefixes = sorted(
        {_reference_prefix(_normalize(ref)) for ref in group["Reference"].tolist() if _normalize(ref)}
    )
    size_part = ", ".join(sizes) if sizes else "mixed sizes"
    prefix_part = ", ".join(prefixes[:4])
    if len(prefixes) > 4:
        prefix_part = f"{prefix_part}, …"
    return f"{collection} ({size_part} mm; ref families {prefix_part or 'n/a'})"


def _format_rrp(value: Any) -> str:
    text = _normalize(value)
    if not text:
        return "N/A"
    try:
        numeric = float(text)
        if numeric.is_integer():
            return f"${int(numeric):,}"
        return f"${numeric:,.2f}"
    except ValueError:
        return text


def _format_complication(value: Any) -> str:
    text = _normalize(value).strip(" ,")
    return text or "None"


def _catalog_row_markdown(row: pd.Series) -> str:
    reference = _normalize(row.get("Reference"))
    size = _normalize(row.get("Size"))
    description = _normalize(row.get("Description")) or "Standard"
    complication = _format_complication(row.get("Complication"))
    rrp = _format_rrp(row.get("RRP"))
    return (
        f"| {reference} | {size} | {description} | {complication} | {rrp} |"
    )


def _build_collection_table_markdown(collection: str, group: pd.DataFrame) -> str:
    subgroup = _subgroup_label(collection, group)
    references = sorted({_normalize(ref) for ref in group["Reference"].tolist() if _normalize(ref)})
    header = (
        f"# {subgroup}\n\n"
        f"Rolex **{collection}** catalog segment covering "
        f"{len(group)} reference variant(s).\n\n"
        f"**References:** {', '.join(references)}\n\n"
        "| Reference | Size (mm) | Dial / Description | Complications | RRP |\n"
        "|---|---:|---|---|---:|\n"
    )
    body = "\n".join(_catalog_row_markdown(row) for _, row in group.iterrows())
    return f"{header}{body}"


def _score_model_match(waitlist_model: str, row: pd.Series) -> float:
    wait_tokens = _tokenize(waitlist_model)
    if not wait_tokens:
        return 0.0

    catalog_text = " ".join(
        [
            _normalize(row.get("Reference")),
            _normalize(row.get("Description")),
            _normalize(row.get("Complication")),
            _normalize(row.get("Size")),
        ]
    )
    catalog_tokens = _tokenize(catalog_text)

    overlap = wait_tokens & catalog_tokens
    score = len(overlap)

    for ref in REFERENCE_PATTERN.findall(waitlist_model):
        if ref.upper() == _normalize(row.get("Reference")).upper():
            score += 5

    wait_lower = waitlist_model.lower()
    description_lower = _normalize(row.get("Description")).lower()
    if "steel" in wait_lower and "two-tone" not in wait_lower and "gold" not in wait_lower:
        if any(token in description_lower for token in ("standard", "black", "blue", "white")):
            score += 1
    if "gold" in wait_lower and "gold" in description_lower:
        score += 2
    if "diamond" in wait_lower and "diamond" in description_lower:
        score += 2

    return score


def _select_matching_models(
    waitlist_model: str,
    collection: str,
    models_df: pd.DataFrame,
    *,
    max_matches: int = 12,
    min_score: float = 1.0,
) -> pd.DataFrame:
    collection_df = models_df[models_df["Collection"] == collection].copy()
    if collection_df.empty:
        return collection_df

    scores = collection_df.apply(lambda row: _score_model_match(waitlist_model, row), axis=1)
    ranked = collection_df.assign(_match_score=scores)
    ranked = ranked[ranked["_match_score"] >= min_score].sort_values(
        by="_match_score", ascending=False
    )
    if ranked.empty:
        return collection_df.head(max_matches)
    return ranked.head(max_matches).drop(columns="_match_score")


def _build_unified_waitlist_chunk(
    row: pd.Series,
    models_df: pd.DataFrame,
) -> ChunkRecord | None:
    category = _normalize(row.get("Category"))
    model = _normalize(row.get("Model"))
    collection = _resolve_collection(category, model)
    if not collection:
        return None

    matched = _select_matching_models(model, collection, models_df)
    min_wait = _normalize(row.get("Min Wait Time"))
    max_wait = _normalize(row.get("Max Wait Time"))
    market_delta = _normalize(row.get("Market Price VS Retail Price")) or "Not reported"

    lines = [
        f"# {model}",
        "",
        "## Market & Availability",
        f"- **Category:** {category}",
        f"- **Collection:** {collection}",
        f"- **Minimum wait time:** {min_wait or 'Not specified'}",
        f"- **Maximum wait time:** {max_wait or 'Not specified'}",
        f"- **Market price vs retail (grey-market / resale vs RRP):** {market_delta}",
        "",
        "## Catalog Profile",
    ]

    if matched.empty:
        lines.append(
            f"No exact catalog rows matched; collection `{collection}` may still be relevant."
        )
    else:
        lines.extend(
            [
                "| Reference | Size (mm) | Dial / Description | Complications | RRP |",
                "|---|---:|---|---|---:|",
            ]
        )
        for _, catalog_row in matched.iterrows():
            lines.append(_catalog_row_markdown(catalog_row))
        references = sorted(
            {_normalize(ref) for ref in matched["Reference"].tolist() if _normalize(ref)}
        )
        lines.extend(["", f"**Matched references:** {', '.join(references)}"])

    metadata = {
        "chunk_type": "unified_waitlist",
        "category": category,
        "model": model,
        "collection": collection,
        "references": sorted(
            {_normalize(ref) for ref in matched.get("Reference", pd.Series(dtype=str)).tolist() if _normalize(ref)}
        ),
        "min_wait_time": min_wait,
        "max_wait_time": max_wait,
        "market_vs_retail": market_delta,
    }
    return ChunkRecord(text="\n".join(lines), metadata=metadata)


def _split_large_collection(
    collection: str,
    collection_df: pd.DataFrame,
    *,
    max_rows_per_chunk: int = 24,
) -> list[pd.DataFrame]:
    if len(collection_df) <= max_rows_per_chunk:
        return [collection_df]

    grouped: list[pd.DataFrame] = []
    for _, group in collection_df.groupby(
        collection_df["Reference"].map(lambda ref: _reference_prefix(_normalize(ref))),
        sort=True,
    ):
        if len(group) <= max_rows_per_chunk:
            grouped.append(group)
            continue

        for start in range(0, len(group), max_rows_per_chunk):
            grouped.append(group.iloc[start : start + max_rows_per_chunk])
    return grouped


def build_collection_catalog_chunks(
    models_path: Path | str = DEFAULT_MODELS_PATH,
    *,
    max_rows_per_chunk: int = 24,
) -> list[ChunkRecord]:
    models_df = pd.read_csv(models_path, sep=";")
    chunks: list[ChunkRecord] = []

    for collection, collection_df in models_df.groupby("Collection", sort=True):
        for subgroup in _split_large_collection(collection, collection_df, max_rows_per_chunk=max_rows_per_chunk):
            text = _build_collection_table_markdown(collection, subgroup)
            references = sorted(
                {_normalize(ref) for ref in subgroup["Reference"].tolist() if _normalize(ref)}
            )
            metadata = {
                "chunk_type": "collection_catalog",
                "collection": collection,
                "references": references,
                "row_count": len(subgroup),
            }
            chunks.append(ChunkRecord(text=text, metadata=metadata))

    return chunks


def build_unified_waitlist_chunks(
    waitlist_path: Path | str = DEFAULT_WAITLIST_PATH,
    models_path: Path | str = DEFAULT_MODELS_PATH,
) -> list[ChunkRecord]:
    waitlist_df = pd.read_csv(waitlist_path)
    models_df = pd.read_csv(models_path, sep=";")
    chunks: list[ChunkRecord] = []

    for _, row in waitlist_df.iterrows():
        if not _is_valid_waitlist_row(row):
            continue
        chunk = _build_unified_waitlist_chunk(row, models_df)
        if chunk is not None:
            chunks.append(chunk)

    return chunks


def build_all_chunks(
    models_path: Path | str = DEFAULT_MODELS_PATH,
    waitlist_path: Path | str = DEFAULT_WAITLIST_PATH,
    *,
    include_catalog_chunks: bool = True,
    max_rows_per_chunk: int = 24,
) -> list[ChunkRecord]:
    """Build unified waitlist+catalog chunks and optional collection catalog chunks."""
    chunks = build_unified_waitlist_chunks(waitlist_path=waitlist_path, models_path=models_path)
    if include_catalog_chunks:
        chunks.extend(
            build_collection_catalog_chunks(
                models_path=models_path,
                max_rows_per_chunk=max_rows_per_chunk,
            )
        )

    seen: set[str] = set()
    unique_chunks: list[ChunkRecord] = []
    for chunk in chunks:
        if chunk.text in seen:
            continue
        seen.add(chunk.text)
        unique_chunks.append(chunk)
    return unique_chunks


def save_chunks_jsonl(chunks: list[ChunkRecord], output_path: Path | str) -> None:
    import json

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps({"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata})
                + "\n"
            )


def main() -> None:
    chunks = build_all_chunks()
    save_chunks_jsonl(chunks, DEFAULT_CHUNKS_JSONL)
    print(f"Wrote {len(chunks)} semantic chunks to {DEFAULT_CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
