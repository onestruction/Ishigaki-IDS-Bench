#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


REQUIRED_KEYS = {
    "id",
    "messages",
    "language",
    "input_format",
    "turn_type",
    "domain",
    "ifc_versions",
}
TAXONOMY_KEYS = ["language", "input_format", "turn_type", "domain", "ifc_versions"]
IDS_NS = "http://standards.buildingsmart.org/IDS"
ALLOWED_FACETS = {"entity", "attribute", "classification", "material", "partOf", "property"}
FACET_TARGETS = {"facets", "spec", "datatype", "cardinality"}
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
FENCED_RE = re.compile(r"```(?:ids|xml)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


@dataclass
class Block:
    spec_idx: int | None
    section: str
    block_type: str
    content: dict[str, Any]

    def signature(self) -> tuple[str, str, str]:
        return (
            self.section,
            self.block_type,
            json.dumps(self.content, ensure_ascii=False, sort_keys=True),
        )


@dataclass
class Spec:
    idx: int
    name: str | None
    ifc_version: str | None
    description: Any | None
    instructions: Any | None
    facet_blocks: list[Block]
    core_facet_blocks: list[Block]


@dataclass
class ParsedIDS:
    metadata_blocks: list[Block]
    specs: list[Spec]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def load_dataset_rows(dataset_cfg: dict[str, Any], dataset_path_override: str | None) -> list[dict[str, Any]]:
    if dataset_path_override:
        return load_jsonl(Path(dataset_path_override))

    local_path = dataset_cfg.get("path")
    if local_path:
        return load_jsonl(Path(local_path))

    name = dataset_cfg.get("name")
    split = dataset_cfg.get("split", "test")
    if not name:
        raise ValueError("Set dataset.path or dataset.name in the config.")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install the 'datasets' package to load from Hugging Face.") from e

    data_files = dataset_cfg.get("data_files")
    if data_files:
        ds = load_dataset(name, data_files=data_files, split=split)
    else:
        ds = load_dataset(name, split=split)
    return [dict(row) for row in ds]


def normalize_ifc_versions(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, Sequence):
        raw = [str(v) for v in value]
    else:
        raw = []
    out: list[str] = []
    for item in raw:
        token = item.strip().upper().replace("-", "_")
        if not token:
            continue
        if "IFC2X3" in token:
            norm = "IFC2X3"
        elif "IFC4X3" in token:
            norm = "IFC4X3_ADD2"
        elif token.startswith("IFC4") or "IFC4" in token:
            norm = "IFC4"
        else:
            norm = token
        if norm not in out:
            out.append(norm)
    return out


def validate_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    missing = sorted(REQUIRED_KEYS - set(row))
    if missing:
        raise ValueError(f"Record {idx} is missing keys: {missing}")

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Record {idx} must contain at least two messages.")

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Record {idx} message {msg_idx} is not an object.")
        if msg.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Record {idx} message {msg_idx} has invalid role: {msg.get('role')}")
        if not isinstance(msg.get("content"), str) or not msg.get("content", "").strip():
            raise ValueError(f"Record {idx} message {msg_idx} has empty content.")

    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Record {idx} last message must be the gold assistant IDS.")

    row = dict(row)
    row["id"] = str(row["id"])
    row["ifc_versions"] = normalize_ifc_versions(row["ifc_versions"])
    if not row["ifc_versions"]:
        raise ValueError(f"Record {idx} has empty ifc_versions.")
    return row


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def strip_model_output(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    match = FENCED_RE.search(text)
    if match:
        text = match.group(1)
    return text.strip()


def openrouter_client(model_cfg: dict[str, Any]) -> tuple[OpenAI, str]:
    model_name = str(model_cfg.get("name") or model_cfg.get("model") or "").strip()
    if not model_name:
        raise ValueError("model.name must be set.")

    base_url = model_cfg.get("base_url")
    if base_url:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
        return OpenAI(api_key=api_key, base_url=str(base_url).rstrip("/")), model_name

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), model_name


def endpoint_matches_provider(endpoint: dict[str, Any], provider: str) -> bool:
    provider_norm = provider.strip().lower()
    tag = str(endpoint.get("tag") or "").lower()
    provider_name = str(endpoint.get("provider_name") or "").strip().lower().replace(" ", "-")
    if not provider_norm:
        return False
    return tag == provider_norm or tag.startswith(provider_norm + "/") or provider_name == provider_norm


def write_openrouter_metadata(config_path: Path, cfg: dict[str, Any], model_name: str, output_dir: Path) -> None:
    model_cfg = dict(cfg.get("model") or {})
    if model_cfg.get("base_url"):
        return

    provider_config = dict(model_cfg.get("provider") or {})
    provider_order = list(provider_config.get("order") or provider_config.get("only") or [])
    endpoint_snapshot: dict[str, Any] = {"selected_endpoints": [], "all_endpoints": [], "error": None}

    try:
        headers = {}
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"https://openrouter.ai/api/v1/models/{model_name}/endpoints"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        endpoints = data.get("endpoints") if isinstance(data, dict) else data
        endpoints = endpoints if isinstance(endpoints, list) else []
        selected: list[dict[str, Any]] = []
        for provider in provider_order:
            selected.extend([ep for ep in endpoints if endpoint_matches_provider(ep, str(provider))])
        endpoint_snapshot = {
            "selected_endpoints": selected,
            "all_endpoints": endpoints,
            "error": None,
        }
    except Exception as e:
        endpoint_snapshot["error"] = f"{type(e).__name__}: {e}"

    generation_keys = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "max_tokens",
        "enable_thinking",
        "reasoning_effort",
        "reasoning_exclude",
        "request_timeout_sec",
        "retries",
        "transforms",
    ]
    generation_parameters = {key: model_cfg.get(key) for key in generation_keys if key in model_cfg}

    write_json(
        output_dir / "openrouter_metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": f"openrouter:{model_name}",
            "model": model_name,
            "config_path": str(config_path),
            "generation_parameters": generation_parameters,
            "provider_config": provider_config,
            "endpoint_snapshot": endpoint_snapshot,
        },
    )


def build_request_kwargs(model_cfg: dict[str, Any], model_name: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"):
        if model_cfg.get(key) is not None:
            kwargs[key] = model_cfg[key]

    extra_body = dict(model_cfg.get("extra_body") or {})
    for key in ("transforms", "top_k", "min_p", "repetition_penalty"):
        value = model_cfg.get(key)
        if value is not None and value != {}:
            extra_body.setdefault(key, value)
    if model_cfg.get("enable_thinking") is True:
        extra_body.setdefault("enable_thinking", True)
    if model_cfg.get("base_url") and model_cfg.get("enable_thinking") is not None:
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", bool(model_cfg["enable_thinking"]))
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if model_cfg.get("reasoning_effort"):
        reasoning = dict(extra_body.get("reasoning") or {})
        reasoning.setdefault("effort", model_cfg["reasoning_effort"])
        if model_cfg.get("reasoning_exclude") is not None:
            reasoning.setdefault("exclude", bool(model_cfg["reasoning_exclude"]))
        extra_body["reasoning"] = reasoning
    if model_cfg.get("provider"):
        extra_body.setdefault("provider", model_cfg["provider"])
    if extra_body:
        kwargs["extra_body"] = extra_body

    if model_cfg.get("request_timeout_sec") is not None:
        kwargs["timeout"] = model_cfg["request_timeout_sec"]
    return kwargs


def generate_one(
    client: OpenAI,
    model_name: str,
    model_cfg: dict[str, Any],
    prompt: str,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    request_messages = [{"role": "system", "content": prompt}] + messages
    kwargs = build_request_kwargs(model_cfg, model_name, request_messages)
    last_error: Exception | None = None
    max_retries = int(model_cfg.get("retries", 2))
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            cleaned = strip_model_output(raw)
            if cleaned:
                return raw, cleaned
            last_error = RuntimeError("empty model output")
        except Exception as e:
            last_error = e
        if attempt < max_retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"generation failed: {last_error}") from last_error


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def normalize_text(text: str | None, upper: bool = False, lower: bool = False) -> str | None:
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    if upper:
        value = value.upper()
    if lower:
        value = value.lower()
    return value


def find_child(parent: ET.Element | None, local_name: str) -> ET.Element | None:
    if parent is None:
        return None
    for child in list(parent):
        if isinstance(child.tag, str) and strip_ns(child.tag) == local_name:
            return child
    return None


def canonicalize_xml(el: ET.Element) -> dict[str, Any]:
    children = [c for c in list(el) if isinstance(c.tag, str)]
    child_items = [canonicalize_xml(c) for c in children]
    child_items.sort(key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return {
        "tag": strip_ns(el.tag),
        "attrib": {k: v.strip() for k, v in sorted(el.attrib.items())},
        "text": normalize_text(el.text),
        "children": child_items,
    }


def parse_value_container(container: ET.Element | None, upper: bool = False) -> Any:
    if container is None:
        return None
    children = [c for c in list(container) if isinstance(c.tag, str)]
    if not children:
        return normalize_text(container.text, upper=upper)

    if all(strip_ns(c.tag) == "simpleValue" for c in children):
        values = [normalize_text(c.text, upper=upper) for c in children]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return values[0] if len(values) == 1 else sorted(values)

    if len(children) == 1:
        return canonicalize_xml(children[0])

    values = [canonicalize_xml(c) for c in children]
    values.sort(key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return values


def parse_facet_content(facet: ET.Element, targets: set[str]) -> dict[str, Any]:
    facet_type = strip_ns(facet.tag)
    if facet_type == "entity":
        return {
            "name": parse_value_container(find_child(facet, "name"), upper=True),
            "predefinedType": parse_value_container(find_child(facet, "predefinedType"), upper=True),
        }
    if facet_type == "attribute":
        return {
            "name": parse_value_container(find_child(facet, "name")),
            "value": parse_value_container(find_child(facet, "value")),
        }
    if facet_type == "classification":
        uri = normalize_text(facet.get("uri")) or parse_value_container(find_child(facet, "uri"))
        return {
            "system": parse_value_container(find_child(facet, "system")),
            "value": parse_value_container(find_child(facet, "value")),
            "uri": uri,
        }
    if facet_type == "material":
        uri = normalize_text(facet.get("uri")) or parse_value_container(find_child(facet, "uri"))
        return {"value": parse_value_container(find_child(facet, "value")), "uri": uri}
    if facet_type == "partOf":
        relation = normalize_text(facet.get("relation"), upper=True) or parse_value_container(
            find_child(facet, "relation"),
            upper=True,
        )
        entity_el = find_child(facet, "entity")
        entity = None
        if entity_el is not None:
            entity = {
                "name": parse_value_container(find_child(entity_el, "name"), upper=True),
                "predefinedType": parse_value_container(find_child(entity_el, "predefinedType"), upper=True),
            }
            entity = {k: v for k, v in entity.items() if v is not None} or None
        return {"entity": entity, "relation": relation}
    if facet_type == "property":
        content: dict[str, Any] = {
            "propertySet": parse_value_container(find_child(facet, "propertySet")),
            "baseName": parse_value_container(find_child(facet, "baseName")),
            "value": parse_value_container(find_child(facet, "value")),
        }
        if "datatype" in targets:
            content["dataType"] = normalize_text(facet.get("dataType"), upper=True)
        if "cardinality" in targets:
            content["cardinality"] = normalize_text(facet.get("cardinality"), lower=True)
        return content
    return {}


def parse_ids_file(path: Path) -> ParsedIDS:
    root = ET.parse(path).getroot()
    metadata_blocks: list[Block] = []
    specs = root.findall(f".//{{{IDS_NS}}}specification")
    if not specs:
        specs = [el for el in root.iter() if isinstance(el.tag, str) and strip_ns(el.tag) == "specification"]

    parsed_specs: list[Spec] = []
    for spec_idx, spec in enumerate(specs):
        facet_blocks: list[Block] = []
        core_blocks: list[Block] = []
        for section in ("applicability", "requirements"):
            section_el = find_child(spec, section)
            if section_el is None:
                continue
            for child in list(section_el):
                if not isinstance(child.tag, str):
                    continue
                facet_type = strip_ns(child.tag)
                if facet_type not in ALLOWED_FACETS:
                    continue
                core = {k: v for k, v in parse_facet_content(child, set()).items() if v is not None}
                full = {k: v for k, v in parse_facet_content(child, FACET_TARGETS).items() if v is not None}
                core_blocks.append(Block(spec_idx, section, facet_type, core))
                facet_blocks.append(Block(spec_idx, section, facet_type, full))

        parsed_specs.append(
            Spec(
                idx=spec_idx,
                name=normalize_text(spec.get("name")),
                ifc_version=normalize_text(spec.get("ifcVersion"), upper=True),
                description=parse_value_container(find_child(spec, "description")),
                instructions=parse_value_container(find_child(spec, "instructions")),
                facet_blocks=facet_blocks,
                core_facet_blocks=core_blocks,
            )
        )
    return ParsedIDS(metadata_blocks=metadata_blocks, specs=parsed_specs)


def blocks_for_spec(spec: Spec) -> list[Block]:
    blocks: list[Block] = []
    if spec.ifc_version is not None:
        blocks.append(Block(spec.idx, "spec", "ifcVersion", {"value": spec.ifc_version}))
    return blocks + spec.facet_blocks


def intersection_count(left: list[Block], right: list[Block]) -> int:
    a = Counter([x.signature() for x in left])
    b = Counter([x.signature() for x in right])
    return sum((a & b).values())


def match_specs(gold_specs: list[Spec], pred_specs: list[Spec]) -> list[int | None]:
    if not gold_specs:
        return []
    weights = [
        [intersection_count(g.core_facet_blocks, p.core_facet_blocks) for p in pred_specs]
        for g in gold_specs
    ]
    pred_count = len(pred_specs)

    if pred_count <= 18:
        @lru_cache(maxsize=None)
        def dp(i: int, used: int) -> tuple[int, tuple[int | None, ...]]:
            if i == len(gold_specs):
                return 0, ()
            best_score, best_assign = dp(i + 1, used)
            best_assign = (None,) + best_assign
            for j in range(pred_count):
                if (used >> j) & 1:
                    continue
                score, assign = dp(i + 1, used | (1 << j))
                score += weights[i][j]
                if score > best_score:
                    best_score = score
                    best_assign = (j,) + assign
            return best_score, best_assign

        return list(dp(0, 0)[1])

    assigned_gold: set[int] = set()
    assigned_pred: set[int] = set()
    output: list[int | None] = [None] * len(gold_specs)
    pairs = sorted(
        [(weights[i][j], i, j) for i in range(len(gold_specs)) for j in range(pred_count)],
        reverse=True,
    )
    for weight, i, j in pairs:
        if weight <= 0 or i in assigned_gold or j in assigned_pred:
            continue
        output[i] = j
        assigned_gold.add(i)
        assigned_pred.add(j)
    return output


def multiset_match(gold: list[Block], pred: list[Block]) -> tuple[int, int, int]:
    gold_counter = Counter([x.signature() for x in gold])
    pred_counter = Counter([x.signature() for x in pred])
    matched = sum((gold_counter & pred_counter).values())
    return matched, len(pred), len(gold)


def compute_metrics(matched: int, pred_total: int, gold_total: int) -> dict[str, float]:
    precision = 0.0 if pred_total == 0 else matched / pred_total
    recall = 1.0 if gold_total == 0 else matched / gold_total
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    score = 1.0 if gold_total == 0 else matched / gold_total
    return {"score": score, "precision": precision, "recall": recall, "f1": f1}


def score_ids_pair(pred_path: Path, gold_path: Path) -> dict[str, Any]:
    try:
        gold = parse_ids_file(gold_path)
        pred = parse_ids_file(pred_path)
    except Exception as e:
        return {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched": 0,
            "pred_block_count": 0,
            "gold_block_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }

    matched_total = 0
    pred_total = 0
    gold_total = 0
    mapping = match_specs(gold.specs, pred.specs)
    used_pred = {j for j in mapping if j is not None}

    for gold_idx, gold_spec in enumerate(gold.specs):
        pred_idx = mapping[gold_idx] if gold_idx < len(mapping) else None
        gold_blocks = blocks_for_spec(gold_spec)
        if pred_idx is None:
            gold_total += len(gold_blocks)
            continue
        pred_blocks = blocks_for_spec(pred.specs[pred_idx])
        matched, pred_count, gold_count = multiset_match(gold_blocks, pred_blocks)
        matched_total += matched
        pred_total += pred_count
        gold_total += gold_count

    for pred_spec in pred.specs:
        if pred_spec.idx not in used_pred:
            pred_total += len(blocks_for_spec(pred_spec))

    return {
        **compute_metrics(matched_total, pred_total, gold_total),
        "matched": matched_total,
        "pred_block_count": pred_total,
        "gold_block_count": gold_total,
        "error": None,
    }


def run_ids_audit(ids_path: Path, eval_cfg: dict[str, Any]) -> dict[str, Any] | None:
    if not eval_cfg.get("run_ids_audit", True):
        return None

    command = str(eval_cfg.get("ids_audit_command", "ids-tool"))
    if shutil.which(command) is None:
        return {
            "success": False,
            "return_code": None,
            "stdout": "",
            "stderr": f"{command} not found on PATH",
            "skipped": True,
        }

    args = list(eval_cfg.get("ids_audit_args", ["audit"]))
    timeout = int(eval_cfg.get("ids_audit_timeout_sec", 60))
    try:
        completed = subprocess.run(
            [command] + args + [str(ids_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": completed.returncode == 0,
            "return_code": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "skipped": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "return_code": -1,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "timeout",
            "skipped": False,
        }


def classify_audit(audit: dict[str, Any] | None) -> dict[str, Any]:
    if audit is None:
        return {"ids_ok": None, "structure_ok": None, "content_ok": None, "implementation_ok": None}
    if audit.get("skipped"):
        return {"ids_ok": None, "structure_ok": None, "content_ok": None, "implementation_ok": None}
    stdout = audit.get("stdout") or ""
    status = ""
    for line in stdout.splitlines():
        if "Completed with status:" in line:
            status = line.split("Completed with status:", 1)[1].strip().strip(".")
            break
    lowered = status.lower()
    if audit.get("success") or lowered == "ok" or lowered.endswith(" ok"):
        return {"ids_ok": 1, "structure_ok": 1, "content_ok": 1, "implementation_ok": 1}
    if "idsstructureerror" in lowered:
        return {"ids_ok": 0, "structure_ok": 0, "content_ok": None, "implementation_ok": 1}
    if "idscontenterror" in lowered:
        return {"ids_ok": 0, "structure_ok": 1, "content_ok": 0, "implementation_ok": 1}
    return {"ids_ok": 0, "structure_ok": None, "content_ok": None, "implementation_ok": 0}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit_rows = [r for r in rows if r.get("audit_metrics", {}).get("ids_ok") is not None]
    ids_ok = sum(int(r["audit_metrics"]["ids_ok"] == 1) for r in audit_rows) if audit_rows else None
    return {
        "count": len(rows),
        "generated": sum(1 for r in rows if r.get("generated")),
        "errors": sum(1 for r in rows if r.get("error")),
        "ids_ok": ids_ok,
        "audit_count": len(audit_rows),
        "facet_f1_mean": mean([float(r.get("facet", {}).get("f1", 0.0) or 0.0) for r in rows]),
        "facet_score_mean": mean([float(r.get("facet", {}).get("score", 0.0) or 0.0) for r in rows]),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": summarize_group(rows), "by": {}}
    for key in TAXONOMY_KEYS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(key)
            if isinstance(value, list):
                values = value
            else:
                values = [value]
            for item in values:
                groups[str(item)].append(row)
        summary["by"][key] = {name: summarize_group(group) for name, group in sorted(groups.items())}
    return summary


def evaluate_record(
    row: dict[str, Any],
    *,
    client: OpenAI,
    model_name: str,
    model_cfg: dict[str, Any],
    prompt: str,
    eval_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    sample_id = safe_id(row["id"])
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    inference_messages = messages[:-1]
    gold_text = messages[-1]["content"].strip()

    generated_dir = output_dir / "generated_ids"
    gold_dir = output_dir / "gold_ids"
    raw_dir = output_dir / "raw_outputs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    gold_path = gold_dir / f"{sample_id}.ids"
    pred_path = generated_dir / f"{sample_id}.ids"
    raw_path = raw_dir / f"{sample_id}.txt"
    gold_path.write_text(gold_text + "\n", encoding="utf-8")

    result: dict[str, Any] = {
        "id": row["id"],
        "language": row["language"],
        "input_format": row["input_format"],
        "turn_type": row["turn_type"],
        "domain": row["domain"],
        "ifc_versions": row["ifc_versions"],
        "gold_path": str(gold_path),
        "pred_path": str(pred_path),
        "raw_path": str(raw_path),
        "generated": False,
        "error": None,
    }

    try:
        raw, cleaned = generate_one(client, model_name, model_cfg, prompt, inference_messages)
        raw_path.write_text(raw, encoding="utf-8")
        pred_path.write_text(cleaned + "\n", encoding="utf-8")
        result["generated"] = True
        result["facet"] = score_ids_pair(pred_path, gold_path)
        audit = run_ids_audit(pred_path, eval_cfg)
        result["ids_audit"] = audit
        result["audit_metrics"] = classify_audit(audit)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["facet"] = {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched": 0,
            "pred_block_count": 0,
            "gold_block_count": 0,
            "error": result["error"],
        }
        result["ids_audit"] = None
        result["audit_metrics"] = classify_audit(None)
    return result


def run(config_path: Path, dataset_path_override: str | None, limit_override: int | None) -> None:
    load_dotenv()
    cfg = read_yaml(config_path)
    dataset_cfg = cfg.get("dataset", {})
    run_cfg = cfg.get("run", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("evaluation", {})

    rows = [validate_record(row, i) for i, row in enumerate(load_dataset_rows(dataset_cfg, dataset_path_override), 1)]
    limit = limit_override if limit_override is not None else run_cfg.get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No dataset rows to evaluate.")

    output_dir = Path(run_cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(cfg.get("prompt", {}).get("path", "prompts/ishigaki-ids-bench_0shot.txt"))
    prompt = prompt_path.read_text(encoding="utf-8")
    client, model_name = openrouter_client(model_cfg)
    write_openrouter_metadata(config_path, cfg, model_name, output_dir)
    workers = int(run_cfg.get("api_workers", 1))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                evaluate_record,
                row,
                client=client,
                model_name=model_name,
                model_cfg=model_cfg,
                prompt=prompt,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
            )
            for row in rows
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(future.result())

    results.sort(key=lambda x: x["id"])
    write_jsonl(output_dir / "predictions.jsonl", results)
    write_json(output_dir / "summary.json", build_summary(results))
    print(f"Wrote {output_dir / 'predictions.jsonl'}")
    print(f"Wrote {output_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ishigaki-IDS-Bench evaluation.")
    parser.add_argument("--config", default="config/eval-template.yaml")
    parser.add_argument("--dataset-path", default=None, help="Local JSONL override for pre-upload tests.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.config), args.dataset_path, args.limit)


if __name__ == "__main__":
    main()
