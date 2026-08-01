"""Coarse, keyword-based resource/producer tagging.

Deliberately simple: case-insensitive WORD-BOUNDARY matching of each resource's
name+aliases and each producer's aliases against an article's title+snippet.
No fuzzy matching, no embeddings — a later entity-resolution stage refines this.

Word boundaries (not raw substring) are used specifically so short tickers and
acronyms like "MU" or "TSM" don't match inside unrelated words (e.g. "aluminum").
"""
import json
import re
from typing import List, Optional, Tuple

Catalog = List[Tuple[str, List[str]]]  # [(id, [terms...]), ...]


def load_resource_catalog(conn) -> Catalog:
    """Build [(resource_id, [name + aliases...])] from the resource table."""
    catalog = []
    for resource_id, name, aliases_json in conn.execute(
        "SELECT resource_id, name, aliases FROM resource"
    ):
        terms = [name] + (json.loads(aliases_json) if aliases_json else [])
        catalog.append((resource_id, _dedupe(terms)))
    return catalog


def load_producer_catalog(alias_dict_path) -> Catalog:
    """Build [(producer_id, [aliases...])] from alias_dictionary.json."""
    data = json.loads(open(alias_dict_path, encoding="utf-8").read())
    return [(p["producer_id"], _dedupe(p.get("aliases", [])))
            for p in data.get("producers", [])]


def _dedupe(terms: List[str]) -> List[str]:
    seen, out = set(), []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _matches(term: str, text: str) -> bool:
    """Case-insensitive word-boundary match of `term` within `text`."""
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE) is not None


def tag_article(title: Optional[str], snippet: Optional[str],
                resource_catalog: Catalog,
                producer_catalog: Catalog) -> Tuple[List[str], Optional[str]]:
    """Tag one article.

    Returns (resource_ids, producer_id):
      - resource_ids: every resource whose name/aliases appear in the text.
        Empty list means the article is untagged noise (caller should discard).
      - producer_id: the first producer (catalog order) whose alias appears,
        or None. Nullable and secondary.
    """
    text = " ".join(part for part in (title, snippet) if part)
    if not text.strip():
        return [], None

    resource_ids = [rid for rid, terms in resource_catalog
                    if any(_matches(term, text) for term in terms)]

    producer_id = next(
        (pid for pid, terms in producer_catalog
         if any(_matches(term, text) for term in terms)),
        None,
    )
    return resource_ids, producer_id
