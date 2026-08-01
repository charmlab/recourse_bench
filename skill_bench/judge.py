from __future__ import annotations

import re
from pathlib import Path

import yaml

from skill_bench.schemas import FeatureResult, Judgment


def judge_worktree(
    worktree: Path,
    features_file: Path,
    baseline_root: Path | None = None,
) -> Judgment:
    with features_file.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    features_spec = spec.get("features", [])
    results: list[FeatureResult] = []

    for feat in features_spec:
        fid = feat["id"]
        title = feat["title"]
        check = feat.get("check", "glob")

        try:
            result = _evaluate(worktree, feat, check, baseline_root)
        except Exception as exc:
            result = FeatureResult(fid, title, "fail", f"judge error: {exc}")

        results.append(result)

    passed = sum(1 for r in results if r.verdict == "pass")
    failed = sum(1 for r in results if r.verdict == "fail")
    skipped = sum(1 for r in results if r.verdict == "skip")
    overall = "pass" if failed == 0 and passed > 0 else "fail"

    return Judgment(
        features=results,
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall=overall,
    )


def _matches(
    worktree: Path,
    pattern: str,
    *,
    fresh: bool,
    baseline_root: Path | None,
) -> list[Path]:
    matches = list(worktree.glob(pattern))
    if not fresh or baseline_root is None:
        return matches
    return [
        match
        for match in matches
        if not (baseline_root / match.relative_to(worktree)).exists()
    ]


def _evaluate(
    worktree: Path,
    feat: dict,
    check: str,
    baseline_root: Path | None,
) -> FeatureResult:
    fid = feat["id"]
    title = feat["title"]
    fresh = bool(feat.get("fresh", False))

    if check == "glob":
        pattern = feat["pattern"]
        min_count = feat.get("min_count", 1)
        matches = _matches(
            worktree,
            pattern,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if len(matches) >= min_count:
            return FeatureResult(fid, title, "pass", f"found {len(matches)} match(es) for {pattern!r}", str(matches[0].relative_to(worktree)))
        return FeatureResult(fid, title, "fail", f"expected >={min_count} match(es) for {pattern!r}, got {len(matches)}")

    if check == "glob_all":
        patterns = feat["patterns"]
        missing = []
        for pat in patterns:
            if not _matches(
                worktree,
                pat,
                fresh=fresh,
                baseline_root=baseline_root,
            ):
                missing.append(pat)
        if not missing:
            return FeatureResult(fid, title, "pass", "all required files found")
        return FeatureResult(fid, title, "fail", f"missing files matching: {missing}")

    if check == "file_contains":
        file_glob = feat["file_glob"]
        pattern = feat["pattern"]
        matches = _matches(
            worktree,
            file_glob,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if not matches:
            return FeatureResult(fid, title, "fail", f"no file found matching {file_glob!r}")
        target = matches[0]
        content = target.read_text(encoding="utf-8", errors="replace")
        if pattern in content:
            return FeatureResult(fid, title, "pass", f"found {pattern!r} in {target.name}", f"{target.relative_to(worktree)}")
        return FeatureResult(fid, title, "fail", f"{pattern!r} not found in {target.name}", f"{target.relative_to(worktree)}")

    if check == "file_contains_any":
        file_glob = feat["file_glob"]
        patterns = feat["patterns"]
        matches = _matches(
            worktree,
            file_glob,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if not matches:
            return FeatureResult(fid, title, "fail", f"no file found matching {file_glob!r}")
        target = matches[0]
        content = target.read_text(encoding="utf-8", errors="replace")
        found = [p for p in patterns if p in content]
        if found:
            return FeatureResult(fid, title, "pass", f"found {found[0]!r} in {target.name}", str(target.relative_to(worktree)))
        return FeatureResult(fid, title, "fail", f"none of {patterns} found in {target.name}")

    if check == "file_contains_regex":
        file_glob = feat["file_glob"]
        regex = feat["regex"]
        matches = _matches(
            worktree,
            file_glob,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if not matches:
            return FeatureResult(fid, title, "fail", f"no file found matching {file_glob!r}")
        target = matches[0]
        content = target.read_text(encoding="utf-8", errors="replace")
        if re.search(regex, content, re.MULTILINE):
            return FeatureResult(fid, title, "pass", f"regex matched in {target.name}", str(target.relative_to(worktree)))
        return FeatureResult(fid, title, "fail", f"regex {regex!r} did not match in {target.name}")

    if check == "report_pass_count":
        file_glob = feat["file_glob"]
        min_pass = feat.get("min_pass", 1)
        matches = _matches(
            worktree,
            file_glob,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if not matches:
            return FeatureResult(fid, title, "fail", f"no file found matching {file_glob!r}")
        content = matches[0].read_text(encoding="utf-8", errors="replace")
        pass_count = len(re.findall(r"\|\s*PASS\s*\|", content, re.IGNORECASE))
        if pass_count >= min_pass:
            return FeatureResult(fid, title, "pass", f"{pass_count} PASS rows found (expected >={min_pass})")
        return FeatureResult(fid, title, "fail", f"only {pass_count} PASS rows found, expected >={min_pass}")

    if check == "report_reference_values":
        file_glob = feat["file_glob"]
        allowed = set(feat["allowed"])
        matches = _matches(
            worktree,
            file_glob,
            fresh=fresh,
            baseline_root=baseline_root,
        )
        if not matches:
            return FeatureResult(fid, title, "fail", f"no file found matching {file_glob!r}")
        target = matches[0]
        rows: list[tuple[str, str]] = []
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("|"):
                continue
            columns = [column.strip() for column in line.strip().strip("|").split("|")]
            if len(columns) < 3 or columns[0] in {"Method", "---"}:
                continue
            rows.append((columns[0], columns[2]))
        if not rows:
            return FeatureResult(
                fid,
                title,
                "fail",
                "no method rows found in the report table",
                str(target.relative_to(worktree)),
            )
        invalid = [(method, value) for method, value in rows if value not in allowed]
        if invalid:
            details = ", ".join(f"{method}={value!r}" for method, value in invalid[:5])
            return FeatureResult(
                fid,
                title,
                "fail",
                f"invalid reference value(s): {details}",
                str(target.relative_to(worktree)),
            )
        return FeatureResult(
            fid,
            title,
            "pass",
            f"all {len(rows)} reference values use the allowed taxonomy",
            str(target.relative_to(worktree)),
        )

    return FeatureResult(fid, title, "skip", f"unknown check type: {check!r}")
