"""Build a reproducible, score-blinded audit packet from already exposed answers.

This tool never runs the memory service, changes historical scores, or treats
agreement between judges as ground truth. Output directories must be new.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL = "memops-judge-audit-selection-v1"
QUESTION_TYPES = (
    "CandidateDisambiguation", "OperationApplication", "OperationTrace",
    "StateTransition", "TargetBinding", "StateTrajectory",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keyed(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"Duplicate input identity: {identity}")
        result[identity] = row
    return result


def prepare(primary, predictions, upstream, agreements_per_stratum=2):
    if agreements_per_stratum < 1:
        raise ValueError("At least one agreement per available type/verdict stratum is required")
    primary = keyed(primary, "qid")
    predictions = keyed(predictions, "qid")
    upstream = keyed(upstream, "question_id")
    if set(predictions) != set(upstream):
        raise ValueError("Saved answers and upstream results do not have identical identities")
    selected = {}
    agreements = defaultdict(list)
    comparisons = {}
    for qid, row in upstream.items():
        if row["hypothesis"] != predictions[qid]["answer"]:
            raise ValueError(f"Judges did not see the same saved answer: {qid}")
        old = primary.get(qid, {})
        if old.get("status") != "judged" or type(old.get("correct")) is not bool:
            raise ValueError(f"Missing primary verdict for saved answer: {qid}")
        if row.get("answer_score") not in (0, 1):
            raise ValueError(f"Nonbinary upstream result: {qid}")
        if row["evaluation_type"] not in QUESTION_TYPES:
            raise ValueError(f"Unknown question type: {qid}")
        other = bool(row["answer_score"])
        comparisons[qid] = {
            "qid": qid, "primary_correct": old["correct"],
            "upstream_correct": other, "primary_raw": old.get("raw"),
            "upstream_reason": row.get("reason"),
        }
        if old["correct"] != other:
            selected[qid] = "disagreement"
        else:
            agreements[(row["evaluation_type"], other)].append(qid)
    strata = []
    for kind in QUESTION_TYPES:
        for verdict in (False, True):
            available = agreements[(kind, verdict)]
            ranked = sorted(available, key=lambda q: digest((PROTOCOL + "\0" + q).encode()))
            picked = ranked[:agreements_per_stratum]
            selected.update((qid, "agreement") for qid in picked)
            strata.append({"evaluation_type": kind, "verdict": verdict,
                           "available": len(available), "selected": len(picked)})
    # Hash order prevents sorting by prior score or disagreement status.
    ids = sorted(selected, key=lambda q: digest(("blind\0" + q).encode()))
    blind, comparison, review = [], [], []
    for qid in ids:
        row = upstream[qid]
        blind.append({
            "qid": qid, "question": row["question"],
            "options": row.get("candidate_options"),
            "reference": row["expected_answer"], "rubric": row["judge_rubric"],
            "answer": row["hypothesis"], "evaluation_type": row["evaluation_type"],
        })
        comparison.append({**comparisons[qid], "selection": selected[qid]})
        review.append({"qid": qid, "reviewer": None, "reviewer_kind": None,
                       "correct": None, "criteria": [], "reason": None,
                       "status": "pending_independent_review"})
    summary = {
        "protocol": PROTOCOL, "compared_same_answers": len(upstream),
        "primary_planned": len(primary), "primary_without_saved_answer": len(set(primary) - set(upstream)),
        "selected": len(ids), "selection_counts": dict(Counter(selected.values())),
        "question_type_counts": dict(Counter(upstream[q]["evaluation_type"] for q in ids)),
        "agreements_per_type_and_verdict": agreements_per_stratum,
        "agreement_strata": strata, "review_completed": 0,
        "scope": "Already exposed regression answers; selection only, no calibration verdicts or new score. Agreement is not a reference label.",
    }
    return blind, comparison, review, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "predictions", "upstream", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--agreements-per-stratum", type=int, default=2)
    args = parser.parse_args()
    inputs = {k: getattr(args, k) for k in ("primary", "predictions", "upstream")}
    raw = {k: path.read_bytes() for k, path in inputs.items()}
    packets = prepare(**{k: [json.loads(line) for line in content.splitlines() if line.strip()]
                         for k, content in raw.items()}, agreements_per_stratum=args.agreements_per_stratum)
    args.output.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for name, rows in zip(("blind.jsonl", "prior-verdicts.jsonl", "reviews-pending.jsonl"), packets[:3]):
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (args.output / name).write_bytes(content)
        output_hashes[name] = digest(content)
    manifest = {**packets[3], "inputs": {k: {"path": str(inputs[k].resolve()), "sha256": digest(v)}
                                        for k, v in raw.items()}, "outputs_sha256": output_hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("protocol", "selected", "selection_counts", "review_completed")}))


if __name__ == "__main__":
    main()
