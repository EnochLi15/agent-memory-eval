#!/usr/bin/env python3
"""Evaluate operation-metrics answers with a judge model."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Sequence

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = (
    BASE_DIR
    / "generated_result"
    / "5-test_operation_metrics"
    / "operation_metrics_all_methods.jsonl"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "generated_result" / "5.5-evaluate_operation_metrics"
DEFAULT_EVIDENCE_DIR = BASE_DIR / "generated_result" / "2-evidence_conversation"
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4.1-mini")
DEFAULT_MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0"))
DEFAULT_EVAL_WORKERS = int(os.getenv("EVAL_WORKERS", "8"))
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("EVAL_REQUEST_TIMEOUT", os.getenv("RAG_REQUEST_TIMEOUT", "180")))
DEFAULT_REQUEST_RETRIES = max(1, int(os.getenv("EVAL_REQUEST_RETRIES", os.getenv("RAG_REQUEST_RETRIES", "3"))))
DEFAULT_RETRY_SLEEP_SECONDS = float(
    os.getenv("EVAL_REQUEST_RETRY_SLEEP_SECONDS", os.getenv("RAG_REQUEST_RETRY_SLEEP_SECONDS", "2"))
)
BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
KEY_FILE_PATH = Path(os.getenv("EVIDENCE_KEY_FILE", str(BASE_DIR / "api.md")))
API_KEY_FALLBACK_PATH = BASE_DIR / "key.md"


LlmCaller = Callable[[str, str], Any]


def parse_api_keys_text(text: str) -> list[str]:
    keys: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" in stripped:
            stripped = stripped.split("=", 1)[1].strip()
        key = stripped.strip().strip('"').strip("'")
        if key and key not in keys:
            keys.append(key)
    return keys


def read_api_keys_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return parse_api_keys_text(path.read_text(encoding="utf-8"))


def read_api_key_file(path: Path) -> str | None:
    keys = read_api_keys_file(path)
    return keys[0] if keys else None


def dedupe_api_keys(keys: list[str | None]) -> list[str]:
    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def get_file_api_key() -> str | None:
    return read_api_key_file(KEY_FILE_PATH) or read_api_key_file(API_KEY_FALLBACK_PATH)


API_KEY = (
    os.getenv("MEMTENSOR_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("API_KEY")
    or get_file_api_key()
)

API_KEYS = dedupe_api_keys(
    [
        os.getenv("MEMTENSOR_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("API_KEY"),
        *read_api_keys_file(KEY_FILE_PATH),
        *read_api_keys_file(API_KEY_FALLBACK_PATH),
    ]
)

UNAVAILABLE_API_KEYS_BY_MODEL: dict[str, set[str]] = {}
UNAVAILABLE_API_KEYS_LOCK = Lock()


def get_api_key() -> str | None:
    return (
        os.getenv("MEMTENSOR_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("API_KEY")
        or API_KEY
        or get_file_api_key()
    )


def get_api_keys() -> list[str]:
    keys = dedupe_api_keys(
        [
            os.getenv("MEMTENSOR_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("API_KEY"),
            *read_api_keys_file(KEY_FILE_PATH),
            *read_api_keys_file(API_KEY_FALLBACK_PATH),
        ]
    )
    if keys:
        return keys
    return [API_KEY] if API_KEY else []


def api_keys_for_model(model: str) -> list[str]:
    keys = get_api_keys()
    with UNAVAILABLE_API_KEYS_LOCK:
        unavailable = set(UNAVAILABLE_API_KEYS_BY_MODEL.get(model, set()))
    filtered = [key for key in keys if key not in unavailable]
    return filtered or keys


def mark_api_key_unavailable(model: str, api_key: str) -> None:
    with UNAVAILABLE_API_KEYS_LOCK:
        UNAVAILABLE_API_KEYS_BY_MODEL.setdefault(model, set()).add(api_key)


def is_key_specific_model_unavailable(exc: Exception) -> bool:
    detail = str(exc)
    return (
        "model_not_found" in detail
        or "get_channel_failed" in detail
        or "No available channel" in detail
        or "可用渠道不存在" in detail
    )


def get_openai_base_url() -> str:
    base_url = os.getenv("LLM_BASE_URL", BASE_URL).rstrip("/")
    if not base_url:
        raise RuntimeError(
            "Missing LLM_BASE_URL. Set it to an OpenAI-compatible chat-completions endpoint (e.g. https://api.openai.com or your own gateway/router) before running this script."
        )
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def create_openai_client(*, api_key: str | None = None, base_url: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: openai. Install it with `pip install openai`.") from exc

    resolved_api_key = api_key or get_api_key()
    if not resolved_api_key:
        raise RuntimeError(
            "Missing API key. Set MEMTENSOR_API_KEY, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, API_KEY, or put export OPENAI_API_KEY=... in api.md (key.md also supported)."
        )
    return OpenAI(
        api_key=resolved_api_key,
        base_url=base_url or get_openai_base_url(),
        timeout=DEFAULT_REQUEST_TIMEOUT,
        max_retries=0,
    )


def extract_token_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[key] = int(value)
    return result


def llm_result_content_and_usage(result: Any, *, model: str) -> tuple[str, dict[str, Any]]:
    if isinstance(result, str):
        return result.strip(), {}
    if isinstance(result, dict):
        content = result.get("content", result.get("text", ""))
        if not isinstance(content, str):
            content = ""
        usage = extract_token_usage(result.get("usage"))
        metadata: dict[str, Any] = dict(usage)
        if usage:
            metadata["token_usage_model"] = str(result.get("model") or model)
        return content.strip(), metadata
    content = getattr(result, "content", "")
    if not isinstance(content, str):
        content = ""
    usage = extract_token_usage(getattr(result, "usage", None))
    metadata = dict(usage)
    if usage:
        metadata["token_usage_model"] = str(getattr(result, "model", model) or model)
    return content.strip(), metadata


def call_llm_with_usage(
    prompt: str,
    model: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    api_keys = api_keys_for_model(model)
    if not api_keys:
        raise RuntimeError(
            "Missing API key. Set MEMTENSOR_API_KEY, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, API_KEY, or put export OPENAI_API_KEY=... in api.md (key.md also supported)."
        )
    unavailable_errors: list[str] = []
    for key_index, api_key in enumerate(api_keys, start=1):
        for attempt in range(1, DEFAULT_REQUEST_RETRIES + 1):
            try:
                client = create_openai_client(api_key=api_key)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Could not extract judge response from completion: {completion}")
                return {
                    "content": content.strip(),
                    "usage": extract_token_usage(getattr(completion, "usage", None)),
                    "model": str(getattr(completion, "model", model) or model),
                }
            except Exception as exc:  # noqa: BLE001 - keep long batch eval alive across provider hiccups.
                last_error = exc
                if is_key_specific_model_unavailable(exc):
                    mark_api_key_unavailable(model, api_key)
                    unavailable_errors.append(f"key#{key_index}: {exc}")
                    if key_index < len(api_keys):
                        tqdm.write(
                            f"Model {model} is unavailable for API key #{key_index}; "
                            f"trying key #{key_index + 1}."
                        )
                        break
                    raise RuntimeError(
                        "Model unavailable for all configured API keys: "
                        + " | ".join(unavailable_errors)
                    ) from exc
                if attempt >= DEFAULT_REQUEST_RETRIES:
                    break
                tqdm.write(
                    f"Judge request attempt {attempt}/{DEFAULT_REQUEST_RETRIES} failed for {model}: "
                    f"{exc}. Retrying in {DEFAULT_RETRY_SLEEP_SECONDS}s..."
                )
                import time

                time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"Judge request failed after {DEFAULT_REQUEST_RETRIES} attempts for {model}") from last_error


def call_llm(
    prompt: str,
    model: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    content, _ = llm_result_content_and_usage(
        call_llm_with_usage(
            prompt,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        model=model,
    )
    return content


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


RetrievalMetadataKey = tuple[str, str, str, str, str]


def retrieval_metadata_key(row: dict[str, Any]) -> RetrievalMetadataKey:
    return (
        str(row.get("source_file", "")),
        str(row.get("question_pair_id", "")),
        str(row.get("evaluation_setting", "")),
        str(row.get("evaluation_method", "")),
        str(row.get("answer_model", "")),
    )


def load_retrieval_metadata_by_key(test_output_dir: Path) -> dict[RetrievalMetadataKey, dict[str, Any]]:
    metadata: dict[RetrievalMetadataKey, dict[str, Any]] = {}
    if not test_output_dir.is_dir():
        return metadata
    for path in sorted(test_output_dir.glob("retrieval_*.jsonl")):
        try:
            rows = read_jsonl(path)
        except Exception:
            continue
        for row in rows:
            retrieval_results = row.get("retrieval_results")
            if not isinstance(retrieval_results, dict):
                continue
            key = retrieval_metadata_key(row)
            if not any(key):
                continue
            metadata[key] = {"retrieval_results": retrieval_results}
    return metadata


def attach_retrieval_metadata(
    entry: dict[str, Any],
    retrieval_metadata_by_key: dict[RetrievalMetadataKey, dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(entry.get("retrieval_results"), dict):
        return entry
    metadata = retrieval_metadata_by_key.get(retrieval_metadata_key(entry))
    if not metadata:
        return entry
    enriched = dict(entry)
    enriched.update(metadata)
    return enriched


def format_evidence_conversation(payload: dict[str, Any]) -> str:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return "Evidence conversation unavailable: no conversations field."
    lines: list[str] = []
    for session_number, conversation in enumerate(conversations, start=1):
        if not isinstance(conversation, dict):
            continue
        segment_index = conversation.get("segment_index", session_number)
        lines.append(f"### Evidence Segment {segment_index}")
        dialogue = conversation.get("dialogue")
        if not isinstance(dialogue, list):
            continue
        for turn_number, turn in enumerate(dialogue, start=1):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip() or "unknown"
            content = str(turn.get("content") or "").strip()
            if content:
                lines.append(f"turn {turn_number} {role}: {content}")
    return "\n".join(lines).strip() or "Evidence conversation unavailable: empty dialogue."


def evidence_path_candidates(entry: dict[str, Any], evidence_dirs: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    source_path = entry.get("evidence_source_path")
    if isinstance(source_path, str) and source_path.strip():
        candidates.append(Path(source_path).expanduser())
    source_file = entry.get("source_file")
    if isinstance(source_file, str) and source_file.strip():
        for evidence_dir in evidence_dirs:
            candidates.append(Path(evidence_dir) / source_file)
    return candidates


def load_evidence_conversation_for_entry(
    entry: dict[str, Any],
    *,
    evidence_dirs: Sequence[Path],
) -> str:
    for candidate in evidence_path_candidates(entry, evidence_dirs):
        if candidate.is_file():
            payload = read_json(candidate)
            if isinstance(payload, dict):
                return format_evidence_conversation(payload)
    return "Evidence conversation unavailable."


CONDITIONAL_METRIC_DEFAULTS: dict[str, Any] = {
    "leakage": 0,
    "over_forget": 0,
    "stale_value": 0,
    "reflection_precision": None,
    "reflection_recall": None,
    "trajectory_order": None,
    "final_state": None,
    "intermediate_state": None,
    "trajectory_provenance": None,
}


OPERATION_SPECIFIC_METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "Forget": ("leakage", "over_forget"),
    "Update": ("stale_value",),
    "Reflect": ("reflection_precision", "reflection_recall"),
    "TrajectoryOps": (
        "trajectory_order",
        "final_state",
        "intermediate_state",
        "trajectory_provenance",
    ),
}


def operation_specific_metric_fields(entry: dict[str, Any]) -> tuple[str, ...]:
    operation_type = str(entry.get("operation_type", ""))
    fields = list(OPERATION_SPECIFIC_METRIC_FIELDS.get(operation_type, ()))
    if (
        entry.get("evaluation_type") == "StateTrajectory"
        and "trajectory_order" not in fields
    ):
        fields.extend(OPERATION_SPECIFIC_METRIC_FIELDS["TrajectoryOps"])
    return tuple(fields)


def is_metric_applicable(row: dict[str, Any], key: str) -> bool:
    if key in {"judge_provenance_support"}:
        return True
    if key in CONDITIONAL_METRIC_DEFAULTS:
        return key in operation_specific_metric_fields(row)
    return True


def state_transition_judge_guidance(entry: dict[str, Any]) -> str:
    if entry.get("evaluation_type") != "StateTransition":
        return ""
    operation_type = str(entry.get("operation_type") or "")
    common = (
        "- StateTransition: judge the concrete state probe, not a generic "
        "lifecycle label.\n"
    )
    if operation_type == "Remember":
        return (
            common
            + "  For Remember, answer_score is 1 only when the answer uses the "
            "newly remembered state precisely and rejects plausible wrong or "
            "incomplete states when the question asks for that rejection.\n"
        )
    if operation_type == "Update":
        return (
            common
            + "  Update StateTransition rows are legacy-only; current Update "
            "samples use CandidateDisambiguation for stale/current state probing. "
            "If evaluating a legacy Update StateTransition row, answer_score is 1 "
            "only when stale old values are not treated as current.\n"
        )
    if operation_type == "Forget":
        return (
            common
            + "  For Forget, answer_score is 1 only when the answer does not "
            "reveal or rely on the forgotten raw value. If it leaks the forgotten "
            "detail, answer_score must be 0.\n"
        )
    if operation_type == "Reflect":
        return (
            common
            + "  For Reflect, answer_score is 1 only when the answer applies the "
            "supported inference while rejecting unsupported overgeneralization.\n"
        )
    return (
        common
        + "  Score the requested state probe against the expected answer and gold "
        "operations.\n"
    )


def format_candidate_options_for_judge(entry: dict[str, Any]) -> str:
    options = entry.get("candidate_options")
    if not isinstance(options, list) or not options:
        return "Candidate options: []"
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = ["Candidate options:"]
    for index, option in enumerate(options):
        if index >= len(labels):
            break
        lines.append(f"{labels[index]}. {option}")
    return "\n".join(lines)


def format_difficulty_knobs_for_judge(entry: dict[str, Any]) -> str:
    knobs = entry.get("difficulty_knobs")
    if not isinstance(knobs, dict):
        return "Difficulty knobs: {}\n"
    return f"Difficulty knobs: {json.dumps(knobs, ensure_ascii=False)}\n"


def format_gold_reasoning_chain_for_judge(entry: dict[str, Any]) -> str:
    chain = entry.get("gold_reasoning_chain")
    if not isinstance(chain, list) or not chain:
        return "Gold reasoning chain: []\n"
    return f"Gold reasoning chain: {json.dumps(chain, ensure_ascii=False)}\n"




def format_split_gold_fields_for_judge(entry: dict[str, Any]) -> str:
    expected_answer = entry.get("expected_answer", entry.get("answer", ""))
    gold_memory_state = entry.get("gold_memory_state", "")
    judge_rubric = entry.get("judge_rubric", {})
    diagnostic_checks = entry.get("diagnostic_checks", {})
    return (
        f"Expected answer (answer-only): {expected_answer}\n"
        f"Gold memory state: {json.dumps(gold_memory_state, ensure_ascii=False)}\n"
        f"Judge rubric: {json.dumps(judge_rubric, ensure_ascii=False)}\n"
        f"Diagnostic checks: {json.dumps(diagnostic_checks, ensure_ascii=False)}\n"
        "Judge note: expected_answer is the final answer only. Do not require the model to reproduce judge_rubric, gold_memory_state, or diagnostic_checks verbatim; use those fields only to decide semantic correctness and lifecycle metrics.\n"
    )

def operation_validities(entry: dict[str, Any]) -> set[str]:
    validities: set[str] = set()
    for operation in entry.get("gold_operations") or []:
        if isinstance(operation, dict):
            validity = str(operation.get("validity") or "confirmed").strip()
            if validity:
                validities.add(validity)
    return validities


def format_rag_context_block_mapping(entry: dict[str, Any]) -> str:
    if str(entry.get("context_mode", "")).lower() != "rag":
        return ""

    retrieved_ids = entry.get("retrieved_corpus_ids")
    if not isinstance(retrieved_ids, list):
        retrieved_ids = []
    retrieved_ids = [str(item) for item in retrieved_ids if str(item).strip()]

    retrieval_results = entry.get("retrieval_results")
    ranked_items = []
    if isinstance(retrieval_results, dict) and isinstance(retrieval_results.get("ranked_items"), list):
        ranked_items = [
            item for item in retrieval_results["ranked_items"] if isinstance(item, dict)
        ]
    ranked_by_id = {
        str(item.get("corpus_id", "")): item
        for item in ranked_items
        if str(item.get("corpus_id", "")).strip()
    }

    if retrieved_ids:
        block_items = [
            (corpus_id, ranked_by_id.get(corpus_id, {}))
            for corpus_id in retrieved_ids
        ]
    else:
        top_k = entry.get("top_k_context")
        try:
            top_k_int = int(top_k)
        except (TypeError, ValueError):
            top_k_int = len(ranked_items)
        block_items = [
            (str(item.get("corpus_id", "")), item)
            for item in ranked_items[:top_k_int]
            if str(item.get("corpus_id", "")).strip()
        ]

    if not block_items:
        return (
            "RAG context block mapping:\n"
            "- The model received context blocks numbered by retrieval rank, not original evidence segment order.\n"
            "- Do not interpret Context block 1 as original segment 1 or session 1.\n\n"
        )

    gold_quotes: list[str] = []
    for provenance in entry.get("gold_provenance") or []:
        if isinstance(provenance, dict) and str(provenance.get("quote", "")).strip():
            gold_quotes.append(str(provenance["quote"]).strip())
    for operation in entry.get("gold_operations") or []:
        if not isinstance(operation, dict):
            continue
        for span_key in ("trigger_span",):
            span = operation.get(span_key)
            if isinstance(span, dict) and str(span.get("quote", "")).strip():
                gold_quotes.append(str(span["quote"]).strip())
        for span in operation.get("evidence_spans") or []:
            if isinstance(span, dict) and str(span.get("quote", "")).strip():
                gold_quotes.append(str(span["quote"]).strip())

    def gold_excerpt_for_item(item: dict[str, Any]) -> str:
        text = str(item.get("text", ""))
        if not text:
            return ""
        for quote in gold_quotes:
            if quote and quote in text:
                return quote[:280] + ("..." if len(quote) > 280 else "")
            if len(quote) > 60:
                prefix = quote[:60]
                index = text.find(prefix)
                if index >= 0:
                    window = text[index : index + 280]
                    return window + ("..." if index + 280 < len(text) else "")
        return ""

    lines = [
        "RAG context block mapping:",
        "- The model received context blocks numbered by retrieval rank, not original evidence segment order.",
        "- Do not interpret Context block 1 as original segment 1 or session 1.",
        "- If the model cites \"context block N\", map N using this list before judging provenance or trigger location.",
        "- The model cannot see original segment_index/turn_index labels in RAG mode, so do not require those labels in its answer.",
        "- If a cited Context block maps to the gold evidence segment or otherwise contains the relevant evidence, treat the block-level citation as location-aligned for answer_score.",
        "- If a Context block line below includes a gold evidence excerpt, that block contains the gold trigger/evidence even if its visible block number differs from internal segment metadata.",
        "- Never translate evidence_segment_index=2 into \"context block 2\". evidence_segment_index is internal metadata; Context block N is the model-visible retrieved block number.",
        "- If this mapping says Context block 1 has evidence_segment_index=2, then Context block 1 is the correct visible citation for that retrieved evidence.",
        "- Use judge_provenance_support, not answer_score, to penalize missing exact quotes, turn ids, or overly coarse RAG block citations.",
        "- Exact RAG OperationTrace positive example: if the gold Reflect trigger is in evidence_segment_index=2 and Context block 1 maps to that evidence segment, a model answer saying \"context block 1\" contains an inferred preference for low-key, quiet environments should receive answer_score=1. If its predicted_operations JSON labels that inference as remember or adds an unrelated update, put that mismatch in judge_operation_detection instead of lowering answer_score.",
    ]
    for index, (corpus_id, item) in enumerate(block_items, start=1):
        metadata: list[str] = []
        for field in (
            "session_index",
            "has_evidence",
            "evidence_segment_index",
            "evidence_segment_role",
        ):
            value = item.get(field) if isinstance(item, dict) else None
            if value not in {None, ""}:
                metadata.append(f"{field}={value}")
        suffix = f" ({'; '.join(metadata)})" if metadata else ""
        lines.append(f"- Context block {index} = {corpus_id}{suffix}")
        excerpt = gold_excerpt_for_item(item) if isinstance(item, dict) else ""
        if excerpt:
            lines.append(f"  gold evidence excerpt: \"{excerpt}\"")
    return "\n".join(lines) + "\n\n"


def build_evaluation_prompt(
    entry: dict[str, Any],
    *,
    evidence_conversation: str = "Evidence conversation unavailable.",
) -> str:
    judge_field_lines = [
        '  "answer_score": 0 or 1,',
        '  "judge_operation_detection": {"tp": 0, "fp": 0, "fn": 0},',
    ]
    metric_meaning_lines = [
        "- answer_score: final answer semantically satisfies the expected answer. "
        "Give 1 when the core requested value/state/choice/application is correct.",
        "- judge_operation_detection: counts matching, extra, and missing operations.",
    ]
    operation_specific_field_specs = {
        "leakage": (
            '  "leakage": 0 or 1 or null,',
            "- leakage: 1 means any field in a Forget model response leaked a forgotten raw value or forbidden detail; 0 means no leak.",
        ),
        "over_forget": (
            '  "over_forget": 0 or 1 or null,',
            "- over_forget: 1 means an unrelated retained target was incorrectly treated as forgotten, unavailable, or deleted; 0 means no over-forget.",
        ),
        "stale_value": (
            '  "stale_value": 0 or 1 or null,',
            "- stale_value: 1 means an Update response treats an old value as current; 0 means no stale use.",
        ),
        "reflection_precision": (
            '  "reflection_precision": 0 or 1 or null,',
            "- reflection_precision: 1 means a reflection conclusion is fully supported; 0 means over-inferred.",
        ),
        "reflection_recall": (
            '  "reflection_recall": 0 or 1 or null,',
            "- reflection_recall: 1 means the intended reflection is recovered from clues; 0 means missed.",
        ),
        "trajectory_order": (
            '  "trajectory_order": 0 or 1 or null,',
            "- trajectory_order: ordered operation sequence is correct.",
        ),
        "final_state": (
            '  "final_state": 0 or 1 or null,',
            "- final_state: final memory state after all operations is correct.",
        ),
        "intermediate_state": (
            '  "intermediate_state": 0 or 1 or null,',
            "- intermediate_state: requested as-of/intermediate state is correct.",
        ),
        "trajectory_provenance": (
            '  "trajectory_provenance": 0 or 1 or null,',
            "- trajectory_provenance: evidence for each state transition is adequate.",
        ),
    }
    for field in operation_specific_metric_fields(entry):
        schema_line, meaning_line = operation_specific_field_specs[field]
        judge_field_lines.append(schema_line)
        metric_meaning_lines.append(meaning_line)
    judge_field_lines.extend(
        [
            '  "judge_provenance_support": 0 or 1 or null,',
            '  "extra_detail_type": "none" or "harmless" or "harmful",',
            '  "reason": "brief reason"',
        ]
    )
    metric_meaning_lines.extend(
        [
            "- judge_provenance_support: answer and predicted operations cite adequate evidence.",
            "- extra_detail_type: classify extra content in the model response as none, harmless, or harmful.",
        ]
    )
    forget_policy = ""
    if "leakage" in operation_specific_metric_fields(entry):
        forget_policy = (
            "Forget leakage / over-forget policy:\n"
            "- Inspect the complete Model response JSON, not only the final answer. "
            "Check every field including answer, predicted_operations, old_value, "
            "new_value, state_after, provenance, and explanatory text.\n"
            "- Set leakage=1 if any Model response JSON field contains the forgotten "
            "raw value or a forbidden old detail from the Forget operation.\n"
            "- Set leakage=0 when the response only names a safe target, boundary, "
            "or category such as \"UBC GPA\" or \"temporary parking permit info\" "
            "without exposing the specific forgotten value or old detail.\n"
            "- Safe provenance may quote the deletion request itself. Provenance "
            "that quotes the original fact containing the forgotten raw value is leakage.\n"
            "- Set over_forget=1 when the model says a retained target outside the "
            "Forget boundary is forgotten, unavailable, removed, deleted, or should "
            "not be used. A retained target should remain available unless gold "
            "operations explicitly forget it.\n\n"
        )
        if entry.get("state_transition_probe_type") == "forget_retained_neighbor":
            forget_policy += (
                "This is a Forget retained-neighbor probe.\n"
                "- leakage if the forgotten target is revealed.\n"
                "- over_forget if any retained neighbor target is refused, removed, or treated as unavailable.\n"
                "- Do not use leakage for retained-neighbor refusal.\n"
                "- Do not use over_forget for forgotten-target leakage.\n\n"
            )
    state_transition_policy = state_transition_judge_guidance(entry)
    target_binding_override = ""
    if entry.get("evaluation_type") == "TargetBinding":
        target_binding_override = (
            "Critical TargetBinding scoring override:\n"
            "- For TargetBinding, compare the semantic memory target boundary, not the "
            "literal target label string.\n"
            "- Canonical target_name values in Gold operations are internal labels. "
            "They are NOT required output strings.\n"
            "- If the model names a different target label but includes the same core "
            "record/scope and value boundary from the evidence, answer_score MUST be 1.\n"
            "- Exact positive example: expected target passport_expiration_and_renewal; "
            "model says \"passport renewal timeline\" and includes passport expires "
            "January 2026 plus renewal before the fall trip. This MUST be answer_score=1, "
            "not 0, because the semantic memory target boundary is the same.\n\n"
        )
    return (
        "You are evaluating a MemOps lifecycle memory task. Grade final "
        "answer correctness separately from operation diagnostics.\n\n"
        + target_binding_override
        + "Gold operations are the authoritative operation trace. The expected answer "
        "defines the semantic answer to the question. Use diagnostic fields for "
        "operation trace, lifecycle errors, and provenance issues; do not let those "
        "diagnostic issues automatically make the final answer wrong.\n\n"
        "You must judge only the current model response for this row. Do not compare "
        "against, borrow reasons from, or reuse judgments about any other model output.\n\n"
        + format_rag_context_block_mapping(entry)
        + "Return only compact JSON with exactly these fields:\n"
        "{\n"
        + "\n".join(judge_field_lines)
        + "\n"
        + "}\n\n"
        + "Metric meanings:\n"
        + "\n".join(metric_meaning_lines)
        + "\n\n"
        + forget_policy
        + "Extra-detail policy:\n"
        "- Extra details are harmless by default when they are supported by the "
        "Evidence conversation or do not materially affect the requested answer.\n"
        "- Extra details are harmful only if they do at least one of the following: "
        "(1) directly contradict a confirmed current value in Gold operations; "
        "(2) reveal a forgotten raw value or forbidden forgotten detail; "
        "(3) treat a stale, tentative, or retracted value as current/final; "
        "(4) treat a forgotten target as active; "
        "(5) treat a retained target as unavailable/forgotten; "
        "(6) replace, omit, or materially change the core answer requested by the question.\n"
        "- Mere relevance, verbosity, extra provenance, extra explanation, or extra "
        "non-conflicting facts must not make answer_score=0.\n"
        "- Unsupported extra details may be marked as harmful_extra only when they "
        "change the answer, conflict with Gold operations, or change the requested output.\n"
        "- A context-resolvable target-name paraphrase is correct when the target "
        "wording is not exact but the evidence conversation uniquely resolves the "
        "intended target.\n\n"
        "Answer-score rules:\n"
        "- First decide whether the final answer satisfies the required core answer.\n"
        "- For never-stored value questions, the answer is correct if it refuses "
        "to provide the value and does not invent one. Do not require the model "
        "to explain that the user never provided it.\n"
        "- Critical never-stored refusal override: for remember_negative_probe "
        "or other never-stored value questions, if the model's final answer is "
        "a bare unknown/refusal such as \"I don't know\", \"I do not know\", "
        "\"I don't have that information\", \"no record\", \"not provided\", "
        "or an equivalent short refusal, and it does not provide, guess, or "
        "borrow a concrete value, answer_score MUST be 1. Do not mark it wrong "
        "for being vague, for not saying \"never provided\", for omitting "
        "neighboring facts, for empty predicted_operations, or for weak or "
        "incomplete provenance. Score provenance and operation diagnostics "
        "separately. This override takes precedence over later generic rules "
        "about the core answer being missing.\n"
        "- Do NOT set answer_score to 0 only because the response includes extra "
        "non-conflicting details, is broader, more verbose, or includes additional "
        "evidence-supported background.\n"
        "- Do NOT set answer_score to 0 only because the response uses a paraphrase "
        "of the target when the intended target is recoverable from context.\n"
        "- Do NOT require exact canonical target_name wording. Canonical target "
        "names are internal judge labels, not strings the model must reproduce.\n"
        "- Do NOT set answer_score to 0 only because provenance is incomplete, "
        "formatting differs from the rubric, or predicted operations include extra "
        "non-essential operations.\n"
        "- Set answer_score to 0 when the core value/state/choice/application is "
        "wrong, missing, reversed, stale, forgotten, or materially changed by extra content.\n\n"
        "Validity and difficulty rules:\n"
        "- Gold operations may include validity=confirmed, tentative, or retracted. "
        "Only confirmed values that have not been superseded or forgotten can be "
        "treated as current usable memory.\n"
        "- Recency is not authority. A value mentioned later is wrong if it is "
        "tentative, retracted, superseded, or forgotten.\n"
        "- If the model treats a tentative/retracted value as final or current, "
        "set answer_score=0 and mark the relevant lifecycle metric such as stale_value "
        "or final_state as 1/0 according to the schema.\n"
        "- If Gold reasoning chain is non-empty, check the intermediate steps. A "
        "response that guesses the final answer but contradicts or omits required "
        "intermediate reasoning should not receive full credit on answer_score for "
        "multi-hop or trajectory questions.\n"
        "- Extra details are harmless only when they are supported by the evidence "
        "and do not introduce tentative, retracted, stale, or forgotten information "
        "as current.\n\n"
        "Evaluation-type guidance:\n"
        "- OperationTrace: answer_score should focus on whether the correct operation "
        "type and trigger point are supported by evidence; exact target-value wording "
        "is not decisive for this evaluation type.\n"
        "- TargetBinding: answer_score should allow semantically equivalent or "
        "context-resolvable memory target / record / scope names when evidence "
        "uniquely resolves the target. Canonical target names such as "
        "current_residential_address are internal judge labels. Do not require "
        "exact canonical target_name wording. Natural-language paraphrases such as "
        "\"user's residential address\", \"user residential address\", or "
        "\"residential address\" should be accepted when the answer's scope and "
        "value boundary uniquely maps to the same gold memory target. Mark "
        "TargetBinding wrong only when the response maps to a different target, "
        "is too broad to distinguish among multiple active targets, misses the "
        "required value/scope boundary, or uses a stale or forgotten target. Use judge_rubric for boundary rules; do not demand that the response repeats rubric wording.\n"
        "For TargetBinding, if the model captures the required value/scope boundary "
        "of the memory target, you MUST NOT set answer_score=0 only because its "
        "target label differs from the canonical gold target_name.\n"
        "TargetBinding few-shot examples:\n"
        "- Example score 1: Expected target current_residential_address. Model says "
        "\"user's home address\" and includes the same street address and apartment "
        "boundary. This is a context-resolvable paraphrase, so answer_score=1.\n"
        "- Example score 1: Expected target passport_expiration_and_renewal. Model says "
        "\"passport renewal timeline\" and includes passport expires January 2026 plus "
        "renewal before the fall trip. The target label is a paraphrase but the memory "
        "target boundary is the same, so answer_score=1.\n"
        "- Example score 0: Expected target passport_expiration_and_renewal. Model says "
        "\"flight booking timeline\" or only gives fall trip dates without passport "
        "expiration or renewal intent. This maps to a different target or misses the "
        "required value boundary, so answer_score=0.\n"
        + state_transition_policy
        + "- CandidateDisambiguation: answer_score should focus on choosing the correct "
        "listed candidate and not marking wrong candidates as correct. A model answer "
        "may use the candidate letter, candidate number, full option text, or a "
        "semantically equivalent reference to the same option. If it selects a "
        "distractor, answer_score must be 0 even when the explanation sounds plausible. "
        "If candidate options are labeled A/B/C/D and the model answers with a "
        "letter, map that letter to the corresponding listed candidate before "
        "judging answer_score. For example, if A is Portland and the model "
        "answers A, judge the answer as Portland. "
        "Extra retained facts from the evidence conversation are harmless if they do not change the candidate choice.\n"
        "For Update CandidateDisambiguation, the listed candidates are the state probe. "
        "If the model selects a stale old value, tentative value, retracted value, "
        "or superseded assistant echo instead of the confirmed current value, set "
        "answer_score=0 and stale_value=1. If the model selects the confirmed current "
        "candidate, do not require a separate StateTransition explanation. Use candidate_options plus judge_rubric/diagnostic_checks to map any letter or paraphrase to the selected candidate. If the model selects a stale old value, tentative value, retracted value, or superseded assistant echo, answer_score=0 and stale_value=1.\n"
        "- OperationApplication: answer_score should focus on whether the downstream "
        "task uses the correct operation-derived state.\n"
        "- StateTrajectory: answer_score should focus on the requested trajectory "
        "granularity, not a complete reconstruction unless the question asks for it.\n\n"
        f"Evaluation type: {entry.get('evaluation_type', '')}\n"
        f"Evaluation category: {entry.get('evaluation_category', '')}\n"
        f"StateTransition probe type: {entry.get('state_transition_probe_type', '')}\n"
        f"Operation type: {entry.get('operation_type', '')}\n"
        + format_difficulty_knobs_for_judge(entry)
        + format_gold_reasoning_chain_for_judge(entry)
        + f"Question: {entry.get('question', '')}\n\n"
        + f"{format_candidate_options_for_judge(entry)}\n\n"
        + f"{format_split_gold_fields_for_judge(entry)}\n"
        + f"Gold operations: {json.dumps(entry.get('gold_operations', []), ensure_ascii=False)}\n\n"
        + f"Gold provenance: {json.dumps(entry.get('gold_provenance', []), ensure_ascii=False)}\n\n"
        + f"Evidence conversation:\n{evidence_conversation}\n\n"
        + f"Model response: {entry.get('hypothesis', '')}\n"
    )


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


METRIC_DEFAULTS: dict[str, Any] = {
    "answer_score": 0,
    "judge_operation_detection": {"tp": 0, "fp": 0, "fn": 0},
    "judge_provenance_support": None,
    "extra_detail_type": "none",
    "reason": "",
}


JUDGE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "judge_operation_detection": ("judge_operation_detection", "operation_detection"),
    "judge_provenance_support": ("judge_provenance_support", "provenance_support"),
    # Backward-compatible readers for already-evaluated files.
    "operation_detection": ("judge_operation_detection", "operation_detection"),
    "provenance_support": ("judge_provenance_support", "provenance_support"),
}


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _row_metric(row: dict[str, Any], key: str) -> Any:
    for alias in JUDGE_FIELD_ALIASES.get(key, (key,)):
        if alias in row:
            return row.get(alias)
    return None


def _normalize_binary_or_null(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "yes", "true", "correct", "pass"}:
            return 1
        if normalized in {"0", "no", "false", "incorrect", "fail"}:
            return 0
        if normalized in {"null", "none", "n/a", "na"}:
            return None
    return None


def _normalize_extra_detail_type(value: Any) -> str:
    if not isinstance(value, str):
        return "none"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "no_extra": "none",
        "none": "none",
        "harmless": "harmless",
        "harmless_extra": "harmless",
        "non_conflicting": "harmless",
        "non_conflicting_extra": "harmless",
        "harmful": "harmful",
        "harmful_extra": "harmful",
        "conflicting": "harmful",
        "conflicting_extra": "harmful",
    }
    return aliases.get(normalized, "none")


SAFE_FORGET_VALUE_MARKERS = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "not applicable",
    "[forgotten]",
    "[redacted]",
    "redacted",
    "forgotten",
    "removed",
    "unavailable",
    "not available",
}

OVER_FORGET_MARKERS = (
    "forgotten",
    "unavailable",
    "deleted",
    "removed",
    "erased",
    "cleared",
    "discarded",
    "not retained",
    "not stored",
    "no longer available",
    "no longer active",
    "no longer stored",
    "should not be used",
)

RETAINED_MARKERS = (
    "not forgotten",
    "not unavailable",
    "not deleted",
    "not removed",
    "still active",
    "still available",
    "remains active",
    "remains available",
    "remain active",
    "remain available",
    "should remain",
    "is retained",
    "are retained",
    "kept",
)

STALE_NEGATION_MARKERS = (
    "old",
    "stale",
    "previous",
    "earlier",
    "historical",
    "wrong",
    "incorrect",
    "no longer",
    "not current",
    "not valid",
    "not final",
    "not active",
    "should not use",
    "shouldn't use",
    "do not use",
    "don't use",
    "not use",
    "avoid",
    "confuse",
    "not to confuse",
    "do not confuse",
    "don't confuse",
    "instead",
    "rather than",
    "replaced",
    "inactive",
    "outdated",
    "旧",
    "旧值",
    "过期",
    "历史",
    "不再",
    "不是当前",
    "不应使用",
    "不能使用",
    "已替换",
    "失效",
)

STALE_AS_CURRENT_MARKERS = (
    "current",
    "active",
    "valid",
    "still valid",
    "still active",
    "still use",
    "should use",
    "use it",
    "keep it",
    "retain it",
    "当前",
    "有效",
    "仍然有效",
    "仍可使用",
    "继续使用",
    "应该使用",
    "保留",
)

STALE_AS_CURRENT_STRONG_MARKERS = (
    "current",
    "current value",
    "authoritative current",
    "currently",
    "still valid",
    "still active",
    "still use",
    "should use",
    "should still use",
    "should be used",
    "use it",
    "keep it",
    "retain it",
    "当前",
    "当前值",
    "有效",
    "仍然有效",
    "仍可使用",
    "继续使用",
    "应该使用",
    "应使用",
    "保留",
)


def _stringify_for_postcheck(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _space_normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", _stringify_for_postcheck(value)).strip()


def _casefold_normalize(value: Any) -> str:
    return _space_normalize(value).casefold()


def _operation_type(op: dict[str, Any]) -> str:
    return str(op.get("type") or "").strip().lower()


def _target_texts(op: dict[str, Any]) -> list[str]:
    target = op.get("target")
    texts: list[str] = []
    if isinstance(target, dict):
        for key in ("target_id", "target_name"):
            value = _space_normalize(target.get(key))
            if value:
                texts.append(value)
    return texts


def _target_key(op: dict[str, Any]) -> str:
    target = op.get("target")
    if isinstance(target, dict):
        for key in ("target_id", "target_name"):
            normalized = _casefold_normalize(target.get(key))
            if normalized:
                return normalized
    texts = _target_texts(op)
    if texts:
        return _casefold_normalize(texts[0])
    return ""


def _target_norm_set(ops: Sequence[dict[str, Any]]) -> set[str]:
    return {_casefold_normalize(text) for op in ops for text in _target_texts(op)}


def _looks_scalar(term: str) -> bool:
    return bool(re.fullmatch(r"[\$€£¥]?\s*[\w./:-]+(?:\s+[\w./:-]+){0,3}", term.strip()))


def _term_is_target_only(term: str, target_norms: set[str]) -> bool:
    normalized = _casefold_normalize(term)
    if not normalized:
        return True
    if normalized in target_norms:
        return True
    term_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if not term_tokens:
        return False
    for target in target_norms:
        target_tokens = set(re.findall(r"[a-z0-9]+", target))
        if term_tokens and term_tokens.issubset(target_tokens):
            return True
    return False


def _candidate_forbidden_terms(value: Any, *, target_norms: set[str]) -> list[str]:
    normalized = _space_normalize(value)
    if not normalized:
        return []
    if _casefold_normalize(normalized) in SAFE_FORGET_VALUE_MARKERS:
        return []

    terms: list[str] = [normalized]
    words = normalized.split()
    if len(normalized) > 80 or len(words) > 8:
        numeric_terms = re.findall(
            r"[\$€£¥]?\s*\d[\d,]*(?:\.\d+)?(?:/[A-Za-z0-9]+)?|[A-Z]{1,5}\d{2,}[A-Z0-9-]*",
            normalized,
        )
        terms.extend(term.strip() for term in numeric_terms)
        for chunk in re.split(r"[.;,\n]|\band\b|\bbut\b|\bwith\b", normalized, flags=re.IGNORECASE):
            chunk = _space_normalize(chunk)
            if len(chunk.split()) >= 4:
                terms.append(chunk)

    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        clean = _space_normalize(term)
        norm = _casefold_normalize(clean)
        if len(clean) < 3 or norm in seen:
            continue
        if _term_is_target_only(clean, target_norms):
            continue
        seen.add(norm)
        unique.append(clean)
    return unique


def _contains_postcheck_term(text: str, term: str) -> bool:
    haystack = _space_normalize(text)
    needle = _space_normalize(term)
    if not haystack or not needle:
        return False
    if _looks_scalar(needle) or len(needle.split()) <= 6:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
        return bool(re.search(pattern, haystack, flags=re.IGNORECASE))
    return needle.casefold() in haystack.casefold()


def _forget_operations(entry: dict[str, Any]) -> list[dict[str, Any]]:
    operations = entry.get("gold_operations")
    if not isinstance(operations, list):
        return []
    return [
        op
        for op in operations
        if isinstance(op, dict) and _operation_type(op) == "forget"
    ]


def _update_operations(entry: dict[str, Any]) -> list[dict[str, Any]]:
    operations = entry.get("gold_operations")
    if not isinstance(operations, list):
        return []
    return [
        op
        for op in operations
        if isinstance(op, dict) and _operation_type(op) == "update"
    ]


def _operation_validity(op: dict[str, Any]) -> str:
    return str(op.get("validity") or "confirmed").strip().casefold()


def _confirmed_final_values_by_target(entry: dict[str, Any]) -> dict[str, str | None]:
    operations = entry.get("gold_operations")
    if not isinstance(operations, list):
        return {}
    final_values: dict[str, str | None] = {}
    for op in operations:
        if not isinstance(op, dict) or _operation_validity(op) != "confirmed":
            continue
        key = _target_key(op)
        if not key:
            continue
        op_type = _operation_type(op)
        if op_type == "forget":
            final_values[key] = None
            continue
        new_value = _space_normalize(op.get("new_value"))
        if new_value:
            final_values[key] = new_value
    return final_values


def _term_is_consistent_with_final_value(term: str, final_value: str | None) -> bool:
    if final_value is None:
        return False
    term_norm = _casefold_normalize(term)
    final_norm = _casefold_normalize(final_value)
    if not term_norm or not final_norm:
        return False
    return term_norm == final_norm or term_norm in final_norm or final_norm in term_norm


def _hypothesis_answer_for_postcheck(entry: dict[str, Any]) -> str:
    hypothesis = entry.get("hypothesis")
    if isinstance(hypothesis, str):
        parsed = parse_json_object(hypothesis)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            return parsed["answer"]
    if isinstance(hypothesis, dict) and isinstance(hypothesis.get("answer"), str):
        return hypothesis["answer"]
    return ""


def _find_forget_leakage(entry: dict[str, Any]) -> str | None:
    model_text = _stringify_for_postcheck(entry.get("hypothesis"))
    if not model_text:
        return None
    forget_ops = _forget_operations(entry)
    target_norms = _target_norm_set(forget_ops)
    for op in forget_ops:
        for term in _candidate_forbidden_terms(op.get("old_value"), target_norms=target_norms):
            if _contains_postcheck_term(model_text, term):
                return f"Model response contains forgotten raw value/detail: {term}"
    return None


def _retained_operations(entry: dict[str, Any]) -> list[dict[str, Any]]:
    operations = entry.get("gold_operations")
    if not isinstance(operations, list):
        return []
    forgotten_target_ids = {
        _casefold_normalize(text)
        for op in _forget_operations(entry)
        for text in _target_texts(op)
    }
    retained: list[dict[str, Any]] = []
    for op in operations:
        if not isinstance(op, dict) or _operation_type(op) == "forget":
            continue
        if any(_casefold_normalize(text) in forgotten_target_ids for text in _target_texts(op)):
            continue
        if _space_normalize(op.get("new_value")):
            retained.append(op)
    return retained


def _retained_terms(op: dict[str, Any]) -> list[str]:
    terms = _target_texts(op)
    new_value = _space_normalize(op.get("new_value"))
    if new_value:
        terms.append(new_value)
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        norm = _casefold_normalize(term)
        if len(term) >= 3 and norm not in seen:
            seen.add(norm)
            unique.append(term)
    return unique


def _find_over_forget(entry: dict[str, Any]) -> str | None:
    text = _casefold_normalize(entry.get("hypothesis"))
    if not text:
        return None
    for op in _retained_operations(entry):
        for term in _retained_terms(op):
            term_norm = _casefold_normalize(term)
            if not term_norm:
                continue
            for match in re.finditer(re.escape(term_norm), text, flags=re.IGNORECASE):
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                window = text[start:end]
                if any(marker in window for marker in RETAINED_MARKERS):
                    continue
                if any(marker in window for marker in OVER_FORGET_MARKERS):
                    return f"Model marks retained target as forgotten/unavailable: {term}"
    return None


def apply_forget_postchecks(metrics: dict[str, Any], entry: dict[str, Any] | None) -> None:
    if not entry or str(entry.get("operation_type") or "") != "Forget":
        return
    leakage_reason = _find_forget_leakage(entry)
    if leakage_reason:
        metrics["leakage"] = 1
        metrics["leakage_postcheck_reason"] = leakage_reason

    over_forget_reason = _find_over_forget(entry)
    if over_forget_reason:
        metrics["over_forget"] = 1
        metrics["over_forget_postcheck_reason"] = over_forget_reason


def _find_stale_value_as_current(entry: dict[str, Any]) -> str | None:
    if str(entry.get("evaluation_type") or "") in {"OperationTrace", "TargetBinding"}:
        return None
    answer_text = _hypothesis_answer_for_postcheck(entry)
    model_text = answer_text or _stringify_for_postcheck(entry.get("hypothesis"))
    if not model_text:
        return None
    update_ops = _update_operations(entry)
    target_norms = _target_norm_set(update_ops)
    final_values = _confirmed_final_values_by_target(entry)
    for op in update_ops:
        final_value = final_values.get(_target_key(op))
        for term in _candidate_forbidden_terms(op.get("old_value"), target_norms=target_norms):
            if _term_is_consistent_with_final_value(term, final_value):
                continue
            if not _contains_postcheck_term(model_text, term):
                continue
            text = _space_normalize(model_text)
            pattern = re.escape(_space_normalize(term))
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                window = text[start:end].casefold()
                if any(marker.casefold() in window for marker in STALE_NEGATION_MARKERS):
                    continue
                if any(marker.casefold() in window for marker in STALE_AS_CURRENT_STRONG_MARKERS):
                    return f"Model final answer treats stale old value as current: {term}"
    return None


def _rag_block_items_for_entry(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    retrieved_ids = entry.get("retrieved_corpus_ids")
    if not isinstance(retrieved_ids, list):
        retrieved_ids = []
    retrieved_ids = [str(item) for item in retrieved_ids if str(item).strip()]
    retrieval_results = entry.get("retrieval_results")
    ranked_items = []
    if isinstance(retrieval_results, dict) and isinstance(retrieval_results.get("ranked_items"), list):
        ranked_items = [
            item for item in retrieval_results["ranked_items"] if isinstance(item, dict)
        ]
    ranked_by_id = {
        str(item.get("corpus_id", "")): item
        for item in ranked_items
        if str(item.get("corpus_id", "")).strip()
    }
    if retrieved_ids:
        return [(corpus_id, ranked_by_id.get(corpus_id, {})) for corpus_id in retrieved_ids]
    top_k = entry.get("top_k_context")
    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        top_k_int = len(ranked_items)
    return [
        (str(item.get("corpus_id", "")), item)
        for item in ranked_items[:top_k_int]
        if str(item.get("corpus_id", "")).strip()
    ]


def _gold_quotes_for_entry(entry: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    for provenance in entry.get("gold_provenance") or []:
        if isinstance(provenance, dict) and str(provenance.get("quote", "")).strip():
            quotes.append(str(provenance["quote"]).strip())
    for operation in entry.get("gold_operations") or []:
        if not isinstance(operation, dict):
            continue
        trigger_span = operation.get("trigger_span")
        if isinstance(trigger_span, dict) and str(trigger_span.get("quote", "")).strip():
            quotes.append(str(trigger_span["quote"]).strip())
        for span in operation.get("evidence_spans") or []:
            if isinstance(span, dict) and str(span.get("quote", "")).strip():
                quotes.append(str(span["quote"]).strip())
    unique: list[str] = []
    seen: set[str] = set()
    for quote in quotes:
        normalized = _space_normalize(quote)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _item_contains_gold_quote(item: dict[str, Any], gold_quotes: Sequence[str]) -> bool:
    text = _space_normalize(item.get("text"))
    if not text:
        return False
    for quote in gold_quotes:
        if quote and quote in text:
            return True
        if len(quote) > 60 and quote[:60] in text:
            return True
    return False


def _cited_context_block_numbers(text: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(r"\bcontext\s+block\s+(\d+)\b", text, flags=re.IGNORECASE)
    }


def _operation_trace_answer_semantically_aligned(entry: dict[str, Any]) -> bool:
    model_text = _casefold_normalize(entry.get("hypothesis"))
    reference_parts = [
        _space_normalize(entry.get("expected_answer")),
        _space_normalize(entry.get("target_fact")),
    ]
    for op in entry.get("gold_operations") or []:
        if isinstance(op, dict):
            reference_parts.append(_space_normalize(op.get("new_value")))
    reference_text = _casefold_normalize(" ".join(part for part in reference_parts if part))
    if not model_text or not reference_text:
        return False
    stopwords = {
        "that",
        "this",
        "with",
        "from",
        "should",
        "would",
        "there",
        "where",
        "which",
        "message",
        "model",
        "user",
        "memory",
        "operation",
    }
    reference_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{3,}", reference_text)
        if token not in stopwords
    }
    model_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", model_text))
    if len(reference_tokens & model_tokens) >= 3:
        return True
    if str(entry.get("operation_type", "")) == "Reflect":
        return (
            "preference" in model_text
            and any(term in model_text for term in ("quiet", "calm", "low-key", "low stimulation", "low-stimulation"))
            and any(term in model_text for term in ("crowd", "crowded", "chaotic", "overwhelming", "environment"))
        )
    return False


def apply_rag_context_block_postchecks(metrics: dict[str, Any], entry: dict[str, Any] | None) -> None:
    if not entry:
        return
    if metrics.get("answer_score") != 0:
        return
    if str(entry.get("context_mode", "")).lower() != "rag":
        return
    if str(entry.get("evaluation_type", "")) != "OperationTrace":
        return
    if str(entry.get("operation_type", "")) != "Reflect":
        return
    model_text = _stringify_for_postcheck(entry.get("hypothesis"))
    cited_blocks = _cited_context_block_numbers(model_text)
    if not cited_blocks:
        return
    block_items = _rag_block_items_for_entry(entry)
    gold_quotes = _gold_quotes_for_entry(entry)
    for block_number in cited_blocks:
        index = block_number - 1
        if index < 0 or index >= len(block_items):
            continue
        corpus_id, item = block_items[index]
        if not _item_contains_gold_quote(item, gold_quotes):
            continue
        if not _operation_trace_answer_semantically_aligned(entry):
            continue
        if metrics.get("postcheck_original_answer_score") is None:
            metrics["postcheck_original_answer_score"] = metrics.get("answer_score")
        metrics["answer_score"] = 1
        postcheck_reason = (
            f"RAG Context block {block_number} maps to {corpus_id} and contains gold evidence; "
            "final answer captures the Reflect preference semantics, so answer_score was restored while diagnostics remain unchanged."
        )
        metrics["rag_context_block_postcheck_reason"] = postcheck_reason
        original_reason = _space_normalize(metrics.get("reason"))
        metrics["reason"] = (
            f"{postcheck_reason} Original judge reason retained for diagnostics: {original_reason}"
            if original_reason
            else postcheck_reason
        )
        return


def apply_update_postchecks(metrics: dict[str, Any], entry: dict[str, Any] | None) -> None:
    if not entry or str(entry.get("operation_type") or "") != "Update":
        return
    if str(entry.get("evaluation_type") or "") in {"OperationTrace", "TargetBinding"}:
        metrics["stale_value"] = 0
        metrics.pop("stale_value_postcheck_reason", None)
        return
    stale_reason = _find_stale_value_as_current(entry)
    if stale_reason:
        metrics["stale_value"] = 1
        metrics["stale_value_postcheck_reason"] = stale_reason
    elif metrics.get("stale_value") == 1 and metrics.get("answer_score") == 1:
        metrics["stale_value"] = 0
        metrics["stale_value_postcheck_reason"] = (
            "Downgraded by strict stale postcheck: stale term is not used as current/final answer."
        )


EXTRA_DETAIL_REASON_MARKERS = (
    "extra",
    "irrelevant",
    "unsupported",
    "too broad",
    "additional",
    "non-essential",
    "harmful",
)


def has_lifecycle_hard_metric(metrics: dict[str, Any]) -> bool:
    """Use existing judge/postcheck metrics instead of recalculating lifecycle failures."""
    for key in ("leakage", "over_forget", "stale_value"):
        if metrics.get(key) == 1:
            return True
    for key in ("trajectory_order", "final_state", "intermediate_state"):
        if metrics.get(key) == 0:
            return True
    return False


def is_auto_recovery_excluded(entry: dict[str, Any]) -> bool:
    """Return True for evaluation types where deterministic recovery is unsafe."""
    evaluation_type = str(entry.get("evaluation_type") or "")
    if evaluation_type in {"TargetBinding", "StateTrajectory"}:
        return True
    if entry.get("gold_reasoning_chain"):
        return True
    return False


def reason_mentions_extra_detail(metrics: dict[str, Any]) -> bool:
    reason = str(metrics.get("reason") or "").casefold()
    return any(marker in reason for marker in EXTRA_DETAIL_REASON_MARKERS)


def apply_extra_detail_conservative_postcheck(
    metrics: dict[str, Any],
    entry: dict[str, Any] | None,
) -> None:
    """Audit likely extra-detail judge noise; avoid unsafe automatic score recovery."""
    if not entry:
        return
    metrics["extra_detail_postcheck_applied"] = False
    metrics["extra_detail_postcheck_action"] = "none"

    if metrics.get("answer_score") != 0:
        return
    if metrics.get("extra_detail_type") != "harmful":
        return
    if not reason_mentions_extra_detail(metrics):
        return

    metrics["extra_detail_postcheck_applied"] = True
    metrics["extra_detail_postcheck_action"] = "audit_only"
    metrics["extra_detail_postcheck_reason"] = (
        "Potential extra-detail judge noise; not auto-corrected without a safe deterministic positive gate."
    )

    if is_auto_recovery_excluded(entry):
        metrics["extra_detail_postcheck_reason"] = (
            "Audit only: TargetBinding/StateTrajectory/gold reasoning chain requires semantic judgment."
        )
        return
    if has_lifecycle_hard_metric(metrics):
        metrics["extra_detail_postcheck_reason"] = (
            "Audit only: lifecycle hard metric indicates a real memory-state failure."
        )
        return


def parse_judge_metrics(
    judge_response: str,
    *,
    entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_json_object(judge_response)
    if parsed is None:
        raise ValueError(f"Could not parse judge metrics from response: {judge_response}")

    metrics = dict(METRIC_DEFAULTS)
    applicable_fields = set(operation_specific_metric_fields(entry or {}))
    for key in applicable_fields:
        metrics[key] = CONDITIONAL_METRIC_DEFAULTS[key]
    metrics["answer_score"] = _normalize_binary_or_null(parsed.get("answer_score")) or 0

    detection = _first_present(parsed, "judge_operation_detection", "operation_detection")
    if isinstance(detection, dict):
        metrics["judge_operation_detection"] = {
            "tp": int(detection.get("tp") or 0),
            "fp": int(detection.get("fp") or 0),
            "fn": int(detection.get("fn") or 0),
        }

    renamed_binary_fields = {
        "judge_provenance_support": ("judge_provenance_support", "provenance_support"),
    }
    for output_key, input_keys in renamed_binary_fields.items():
        raw_value = _first_present(parsed, *input_keys)
        value = _normalize_binary_or_null(raw_value)
        if value is not None or any(key in parsed for key in input_keys):
            metrics[output_key] = value

    metrics["extra_detail_type"] = _normalize_extra_detail_type(
        parsed.get("extra_detail_type")
    )

    for key in applicable_fields:
        value = _normalize_binary_or_null(parsed.get(key))
        if value is not None or key in parsed:
            metrics[key] = value

    reason = parsed.get("reason")
    metrics["reason"] = reason if isinstance(reason, str) else ""
    apply_forget_postchecks(metrics, entry)
    apply_update_postchecks(metrics, entry)
    apply_extra_detail_conservative_postcheck(metrics, entry)
    apply_rag_context_block_postchecks(metrics, entry)
    return metrics


def evaluate_entry(
    entry: dict[str, Any],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    call_llm: LlmCaller = call_llm_with_usage,
    evidence_dirs: Sequence[Path] = (),
    retrieval_metadata_by_key: dict[RetrievalMetadataKey, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if retrieval_metadata_by_key:
        entry = attach_retrieval_metadata(entry, retrieval_metadata_by_key)
    evidence_conversation = load_evidence_conversation_for_entry(
        entry,
        evidence_dirs=evidence_dirs,
    )
    judge_prompt = build_evaluation_prompt(
        entry,
        evidence_conversation=evidence_conversation,
    )
    judge_response = ""
    token_usage_metadata: dict[str, Any] = {}
    for parse_attempt in range(1, DEFAULT_REQUEST_RETRIES + 1):
        judge_result = call_llm(judge_prompt, judge_model)
        judge_response, token_usage_metadata = llm_result_content_and_usage(
            judge_result,
            model=judge_model,
        )
        try:
            metrics = parse_judge_metrics(judge_response, entry=entry)
            break
        except ValueError as exc:
            if parse_attempt >= DEFAULT_REQUEST_RETRIES:
                raise ValueError(
                    "Could not parse judge metrics after "
                    f"{DEFAULT_REQUEST_RETRIES} attempts. Last response: {judge_response}"
                ) from exc
            tqdm.write(
                f"Judge response parse attempt {parse_attempt}/{DEFAULT_REQUEST_RETRIES} failed; "
                f"retrying {judge_model}: {judge_response[:200]!r}"
            )
    else:  # pragma: no cover - loop always breaks or raises.
        raise ValueError("Could not parse judge metrics from response")
    if token_usage_metadata:
        token_usage_metadata["token_usage_stage"] = "judge"
    return {
        **entry,
        **metrics,
        "judge_model": judge_model,
        "judge_response": judge_response,
        **token_usage_metadata,
    }


def summarize_matched_pair_rows(rows: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    total = len(rows)
    adjacent_correct = sum(int(adjacent.get("answer_score", 0)) for adjacent, _ in rows)
    longitudinal_correct = sum(int(longitudinal.get("answer_score", 0)) for _, longitudinal in rows)
    adjacent_accuracy = adjacent_correct / total if total else 0.0
    longitudinal_accuracy = longitudinal_correct / total if total else 0.0
    return {
        "paired_total": total,
        "adjacent_accuracy": adjacent_accuracy,
        "longitudinal_accuracy": longitudinal_accuracy,
        "delta": longitudinal_accuracy - adjacent_accuracy,
    }


def matched_pair_delta(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in results:
        source_file = row.get("source_file")
        question_pair_id = row.get("question_pair_id")
        setting = row.get("evaluation_setting")
        method = row.get("evaluation_method")
        if not source_file or not question_pair_id or setting not in {
            "adjacent_operation",
            "longitudinal_operation",
        }:
            continue
        if (
            setting == "longitudinal_operation"
            and method not in {None, ""}
            and not str(method).startswith("rag_vanilla")
        ):
            continue
        grouped.setdefault((str(source_file), str(question_pair_id)), {})[str(setting)] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rows_by_setting in grouped.values():
        adjacent = rows_by_setting.get("adjacent_operation")
        longitudinal = rows_by_setting.get("longitudinal_operation")
        if adjacent is not None and longitudinal is not None:
            pairs.append((adjacent, longitudinal))

    def group_pairs(field: str) -> dict[str, Any]:
        groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for pair in pairs:
            value = pair[0].get(field) or pair[1].get(field)
            if value in {None, ""}:
                continue
            groups.setdefault(str(value), []).append(pair)
        return {key: summarize_matched_pair_rows(value) for key, value in groups.items()}

    return {
        "all": summarize_matched_pair_rows(pairs),
        "by_evaluation_type": group_pairs("evaluation_type"),
        "by_evaluation_category": group_pairs("evaluation_category"),
        "by_operation_type": group_pairs("operation_type"),
    }


def matched_pair_delta_by_method(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    adjacent_by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    longitudinal_by_method: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for row in results:
        source_file = row.get("source_file")
        question_pair_id = row.get("question_pair_id")
        setting = row.get("evaluation_setting")
        if not source_file or not question_pair_id:
            continue
        key = (str(source_file), str(question_pair_id))
        if setting == "adjacent_operation":
            answer_model = str(row.get("answer_model") or row.get("model") or "")
            adjacent_by_pair.setdefault(key, {})[answer_model] = row
        elif setting == "longitudinal_operation":
            method = str(row.get("evaluation_method") or "longitudinal_operation")
            longitudinal_by_method.setdefault(method, {})[key] = row

    def group_pairs(rows: Sequence[tuple[dict[str, Any], dict[str, Any]]], field: str) -> dict[str, Any]:
        groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for pair in rows:
            value = pair[0].get(field) or pair[1].get(field)
            if value in {None, ""}:
                continue
            groups.setdefault(str(value), []).append(pair)
        return {key: summarize_matched_pair_rows(value) for key, value in groups.items()}

    summary: dict[str, Any] = {}
    for method, rows_by_pair in longitudinal_by_method.items():
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for key, row in rows_by_pair.items():
            adjacent_candidates = adjacent_by_pair.get(key)
            if not adjacent_candidates:
                continue
            answer_model = str(row.get("answer_model") or row.get("model") or "")
            adjacent = adjacent_candidates.get(answer_model) or next(
                iter(adjacent_candidates.values())
            )
            pairs.append((adjacent, row))
        summary[method] = {
            "all": summarize_matched_pair_rows(pairs),
            "by_evaluation_type": group_pairs(pairs, "evaluation_type"),
            "by_evaluation_category": group_pairs(pairs, "evaluation_category"),
            "by_operation_type": group_pairs(pairs, "operation_type"),
        }
    return summary


def method_label_for_pair_audit(row: dict[str, Any]) -> str:
    setting = row.get("evaluation_setting")
    method = row.get("evaluation_method")
    model = row.get("answer_model") or row.get("model")
    if setting == "adjacent_operation" and model == "claude-opus-4-6":
        return "adjacent_claude-opus-4-6"
    if method == "long_context_claude-opus-4-6":
        return "long_context_claude-opus-4-6"
    return ""


def apply_paired_flip_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in results:
        label = method_label_for_pair_audit(row)
        if not label:
            continue
        key = (str(row.get("source_file") or ""), str(row.get("question_pair_id") or ""))
        if key[0] and key[1]:
            grouped.setdefault(key, {})[label] = row

    summary: dict[str, Any] = {
        "paired_total": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "adjacent_only": 0,
        "longitudinal_only": 0,
        "flip_total": 0,
        "flip_rows": [],
    }
    for (source_file, question_pair_id), pair in grouped.items():
        adjacent = pair.get("adjacent_claude-opus-4-6")
        longitudinal = pair.get("long_context_claude-opus-4-6")
        if not adjacent or not longitudinal:
            continue
        summary["paired_total"] += 1
        adjacent_correct = adjacent.get("answer_score") == 1
        longitudinal_correct = longitudinal.get("answer_score") == 1
        if adjacent_correct and longitudinal_correct:
            summary["both_correct"] += 1
            outcome = "both_correct"
        elif not adjacent_correct and not longitudinal_correct:
            summary["both_wrong"] += 1
            outcome = "both_wrong"
        elif adjacent_correct:
            summary["adjacent_only"] += 1
            summary["flip_total"] += 1
            outcome = "adjacent_only"
        else:
            summary["longitudinal_only"] += 1
            summary["flip_total"] += 1
            outcome = "longitudinal_only"

        if outcome in {"adjacent_only", "longitudinal_only"}:
            audit = {
                "source_file": source_file,
                "question_pair_id": question_pair_id,
                "evaluation_type": adjacent.get("evaluation_type") or longitudinal.get("evaluation_type"),
                "operation_type": adjacent.get("operation_type") or longitudinal.get("operation_type"),
                "outcome": outcome,
                "adjacent_score": int(adjacent_correct),
                "longitudinal_score": int(longitudinal_correct),
                "adjacent_reason": adjacent.get("reason", ""),
                "longitudinal_reason": longitudinal.get("reason", ""),
            }
            adjacent["paired_flip_audit"] = audit
            longitudinal["paired_flip_audit"] = audit
            summary["flip_rows"].append(audit)
    return summary


def postcheck_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    changed = [
        row
        for row in results
        if row.get("postcheck_original_answer_score") is not None
        and row.get("postcheck_original_answer_score") != row.get("answer_score")
    ]
    extra_audited = [
        row for row in results if row.get("extra_detail_postcheck_applied")
    ]
    return {
        "extra_detail_audit_count": len(extra_audited),
        "answer_score_changed": len(changed),
        "changed_rows": [
            {
                "source_file": row.get("source_file"),
                "question_pair_id": row.get("question_pair_id"),
                "evaluation_type": row.get("evaluation_type"),
                "operation_type": row.get("operation_type"),
                "original_answer_score": row.get("postcheck_original_answer_score"),
                "new_answer_score": row.get("answer_score"),
                "reason": row.get("extra_detail_postcheck_reason")
                or row.get("rag_context_block_postcheck_reason", ""),
            }
            for row in changed
        ],
    }


def summarize(
    results: Sequence[dict[str, Any]],
    *,
    input_file: Path,
    output_file: Path,
    eval_workers: int = DEFAULT_EVAL_WORKERS,
) -> dict[str, Any]:
    total = len(results)
    correct = sum(int(row.get("answer_score", 0)) for row in results)
    detection = summarize_operation_detection(results)
    by_operation_type = group_by_field(results, "operation_type")
    by_setting = group_by_field(results, "evaluation_setting")
    by_evaluation_method = group_by_field(results, "evaluation_method")
    by_context_mode = group_by_field(results, "context_mode")
    by_retrieval_mode = group_by_field(results, "retrieval_mode")
    by_retrieval_unit = group_by_field(results, "retrieval_unit")
    by_answer_model = group_by_field(results, "answer_model")
    by_evaluation_type = group_by_field(results, "evaluation_type")
    by_evaluation_category = group_by_field(results, "evaluation_category")
    by_difficulty = group_by_field(results, "difficulty")
    by_difficulty_knob = group_by_enabled_difficulty_knob(results)
    by_validity = group_by_operation_validity(results)
    by_chain_length = group_by_chain_length(results)
    by_state_transition_probe_type = group_by_field(results, "state_transition_probe_type")
    by_application_probe_type = group_by_field(results, "application_probe_type")
    by_trajectory_pattern = group_by_field(results, "trajectory_pattern")
    by_trajectory_granularity = group_by_field(results, "trajectory_granularity")
    paired_flip_audit = apply_paired_flip_audit(list(results))
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "judge_model": results[0].get("judge_model", DEFAULT_JUDGE_MODEL) if results else DEFAULT_JUDGE_MODEL,
        "total": total,
        "answer_correct": correct,
        "answer_incorrect": total - correct,
        "answer_accuracy": correct / total if total else 0.0,
        "operation_detection": detection,
        "metrics": lifecycle_metric_summary(results),
        "eval_workers": eval_workers,
        "by_setting": by_setting,
        "by_evaluation_method": by_evaluation_method,
        "by_context_mode": by_context_mode,
        "by_retrieval_mode": by_retrieval_mode,
        "by_retrieval_unit": by_retrieval_unit,
        "by_answer_model": by_answer_model,
        "by_operation_type": by_operation_type,
        "by_evaluation_type": by_evaluation_type,
        "by_evaluation_category": by_evaluation_category,
        "by_difficulty": by_difficulty,
        "by_difficulty_knob": by_difficulty_knob,
        "by_validity": by_validity,
        "by_chain_length": by_chain_length,
        "by_state_transition_probe_type": by_state_transition_probe_type,
        "by_application_probe_type": by_application_probe_type,
        "by_trajectory_pattern": by_trajectory_pattern,
        "by_trajectory_granularity": by_trajectory_granularity,
        "matched_pair_delta": matched_pair_delta(results),
        "matched_pair_delta_by_method": matched_pair_delta_by_method(results),
        "postcheck_summary": postcheck_summary(results),
        "paired_flip_audit": paired_flip_audit,
    }


def summarize_binary_metric(results: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [
        int(value)
        for row in results
        for value in [_row_metric(row, key)]
        if is_metric_applicable(row, key)
        and isinstance(value, int)
        and value in {0, 1}
    ]
    if not values:
        return None
    return sum(values) / len(values)


def add_binary_metric_if_present(
    summary: dict[str, Any],
    results: Sequence[dict[str, Any]],
    output_key: str,
    row_key: str,
) -> None:
    value = summarize_binary_metric(results, row_key)
    if value is not None:
        summary[output_key] = value


def lifecycle_metric_summary(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for output_key, row_key in (
        ("leakage_rate", "leakage"),
        ("over_forget_rate", "over_forget"),
        ("stale_value_rate", "stale_value"),
        ("reflection_precision", "reflection_precision"),
        ("reflection_recall", "reflection_recall"),
        ("provenance_support_rate", "judge_provenance_support"),
        ("trajectory_order_accuracy", "trajectory_order"),
        ("final_state_accuracy", "final_state"),
        ("intermediate_state_accuracy", "intermediate_state"),
        ("trajectory_provenance_support_rate", "trajectory_provenance"),
    ):
        add_binary_metric_if_present(metrics, results, output_key, row_key)
    return metrics


def summarize_operation_detection(results: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    tp = 0
    fp = 0
    fn = 0
    for row in results:
        detection = _row_metric(row, "judge_operation_detection")
        if not isinstance(detection, dict):
            continue
        tp += int(detection.get("tp") or 0)
        fp += int(detection.get("fp") or 0)
        fn += int(detection.get("fn") or 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def summarize_selected_rows(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(selected)
    correct = sum(int(row.get("answer_score", 0)) for row in selected)
    summary = {
        "total": total,
        "answer_correct": correct,
        "answer_incorrect": total - correct,
        "answer_accuracy": correct / total if total else 0.0,
        "operation_detection": summarize_operation_detection(selected),
    }
    summary.update(lifecycle_metric_summary(selected))
    return summary


def group_by_field(results: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(results)}
    for row in results:
        value = row.get(field)
        if value in {None, ""}:
            continue
        groups.setdefault(str(value), []).append(row)
    return {key: summarize_selected_rows(rows) for key, rows in groups.items()}


def group_by_enabled_difficulty_knob(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(results)}
    for row in results:
        knobs = row.get("difficulty_knobs")
        if not isinstance(knobs, dict):
            continue
        for knob_name, knob_value in knobs.items():
            if isinstance(knob_value, dict) and knob_value.get("enabled") is True:
                groups.setdefault(str(knob_name), []).append(row)
    return {key: summarize_selected_rows(rows) for key, rows in groups.items()}


def group_by_operation_validity(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(results)}
    for row in results:
        validities = operation_validities(row)
        if not validities:
            validities = {"confirmed"}
        for validity in validities:
            groups.setdefault(validity, []).append(row)
    return {key: summarize_selected_rows(rows) for key, rows in groups.items()}


def group_by_chain_length(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(results)}
    for row in results:
        chain_steps = [
            operation.get("chain_step")
            for operation in row.get("gold_operations") or []
            if isinstance(operation, dict) and isinstance(operation.get("chain_step"), int)
        ]
        if chain_steps:
            key = str(max(chain_steps))
        else:
            key = "1"
        groups.setdefault(key, []).append(row)
    return {key: summarize_selected_rows(rows) for key, rows in groups.items()}


def run_pipeline(
    *,
    input_file: Path = DEFAULT_INPUT_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    call_llm: LlmCaller = call_llm,
    show_progress: bool = True,
    eval_workers: int = DEFAULT_EVAL_WORKERS,
    evidence_dirs: Sequence[Path] = (DEFAULT_EVIDENCE_DIR,),
) -> dict[str, Any]:
    if eval_workers < 1:
        raise ValueError("eval_workers must be at least 1")

    entries = read_jsonl(input_file)
    retrieval_metadata_by_key = load_retrieval_metadata_by_key(input_file.parent)
    results: list[dict[str, Any] | None] = [None] * len(entries)
    max_workers = min(eval_workers, len(entries)) if entries else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                evaluate_entry,
                entry,
                judge_model=judge_model,
                call_llm=call_llm,
                evidence_dirs=evidence_dirs,
                retrieval_metadata_by_key=retrieval_metadata_by_key,
            ): index
            for index, entry in enumerate(entries)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"Evaluating answers ({max_workers} workers)",
            unit="answer",
            disable=not show_progress,
        ):
            results[future_to_index[future]] = future.result()

    ordered_results = [result for result in results if result is not None]
    apply_paired_flip_audit(ordered_results)
    output_file = output_dir / f"{input_file.stem}_operation_metrics_eval.jsonl"
    write_jsonl(output_file, ordered_results)
    summary = summarize(
        ordered_results,
        input_file=input_file,
        output_file=output_file,
        eval_workers=eval_workers,
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate operation-metrics answers.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        action="append",
        default=None,
        help="Clean evidence directory. Can be repeated; defaults to generated_result/2-evidence_conversation.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--eval-workers", type=int, default=DEFAULT_EVAL_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        eval_workers=args.eval_workers,
        evidence_dirs=args.evidence_dir or [DEFAULT_EVIDENCE_DIR],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
