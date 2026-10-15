"""Data lineage: check docs/data_lineage.yaml against the code and the files, count rows, and render it.

1. Every table with a path must exist on disk; row counts are read live (parquet metadata, CSV lines, ...).
2. Every data path the pipeline code reads or writes (src/*.py, src/*.R, notebooks/explore.py) must belong to a
   table in the spec, so nothing the pipeline uses is left out.
3. Every operation's inputs and outputs must be tables in the spec.
Renders docs/data_lineage.md (overview + one Mermaid diagram per chapter, table cards with provenance, operations
with their math) and writes docs/data_lineage_counts.json for the notebook.

Usage: python src/data_lineage.py      (exit code 1 if any check fails)
"""
import fnmatch
import glob
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/data_lineage.yaml"
CODE = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "src").glob("*.R")) + [ROOT / "notebooks/explore.py"]
BASES = {"DER": "data/derived", "RAW": "data/raw", "REPO": "data/raw/repo/RiverShapeML/data",
         "RML": "data/raw/predict-riverML", "CACHE": "data/raw/channel_meas_by_site"}
EXTS = (".parquet", ".csv", ".txt", ".json", ".nc", ".pickle.dat")


def expand(pattern):
    """'a/{x,y}/b' -> ['a/x/b', 'a/y/b']."""
    m = re.search(r"\{([^{}]+)\}", pattern)
    return [p for alt in m.group(1).split(",") for p in expand(pattern[:m.start()] + alt + pattern[m.end():])] if m else [pattern]


def table_files(path):
    return sorted({f for p in expand(path) for f in glob.glob(str(ROOT / p))})


def count_rows(files):
    n = 0
    for f in files:
        if f.endswith(".parquet"):
            n += pq.ParquetFile(f).metadata.num_rows
        elif f.endswith(".csv") or f.endswith(".txt") and "BANKFULL" in f:
            with open(f, "rb") as fh:
                n += sum(1 for _ in fh) - 1
        elif f.endswith(".nc"):
            import xarray as xr
            with xr.open_dataset(f) as ds:
                n += int(ds.sizes.get("feature_id", 0))
        elif f.endswith(".json") and "channel-measurements" in f:
            n += len(json.load(open(f))["features"])
        else:
            n += 1
    return n


def code_paths():
    """Data paths referenced in the code, normalized to repo-relative globs."""
    refs = set()
    for f in CODE:
        src = f.read_text()
        for base, lit in re.findall(r"\b(DER|RAW|REPO|RML|CACHE)\s*/\s*f?\"([^\"]+)\"", src):
            refs.add((f.name, BASES[base] + "/" + re.sub(r"\{[^}]*\}", "*", lit)))
        for sub in re.findall(r"pages\(\s*\"([^\"]+)\"", src):
            refs.add((f.name, f"data/raw/{sub}/page_*.parquet"))
        for lit in re.findall(r"(?<![A-Z]\s/\s)(?<![A-Z]/\s)(?<![A-Z]\s/)(?<![A-Z]/)\"((?:\.\./)?data/[^\"]+)\"", src):
            refs.add((f.name, re.sub(r"\{[^}]*\}", "*", lit.removeprefix("../"))))
    return {(n, r) for n, r in refs if r.endswith(EXTS) or "/page_" in r}


def matches(ref, path):
    return any(fnmatch.fnmatch(ref, p) or fnmatch.fnmatch(p, ref) for p in expand(path))


def mermaid(spec, counts, chapter=None):
    T, O = spec["tables"], spec["operations"]
    nid = lambda s: re.sub(r"\W", "_", s)
    esc = lambda s: str(s).replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
    lines = ["flowchart TD"]
    if chapter is None:  # overview: chapters and their cross-chapter flows
        for c, title in spec["chapters"].items():
            n = sum(1 for t in T.values() if t["chapter"] == c)
            lines.append(f'  {c}["{c}. {esc(title)}<br/>{n} tables"]:::table')
        edges = {(T[i]["chapter"], o["chapter"]) for o in O.values() for i in o["inputs"] if T[i]["chapter"] != o["chapter"]}
        lines += [f"  {a} --> {b}" for a, b in sorted(edges)]
    else:
        ops = {k: o for k, o in O.items() if o["chapter"] == chapter}
        used = {t for o in ops.values() for t in o["inputs"] + o["outputs"]}
        shown = {k for k, t in T.items() if t["chapter"] == chapter} | used
        for k in sorted(shown):
            t = T[k]
            rows = f"{counts[k]:,} rows" if counts.get(k) is not None else "not stored"
            cls = "table" if t["chapter"] == chapter else "external"
            src = "" if t["chapter"] == chapter else f"<br/><i>from {t['chapter']}</i>"
            lines.append(f'  {nid(k)}["{esc(t["title"])}<br/>{rows}{src}"]:::{cls}')
        for k, o in ops.items():
            lines.append(f'  {nid("op_" + k)}(["{esc(o["title"])}"]):::op')
            lines += [f"  {nid(i)} --> {nid('op_' + k)}" for i in o["inputs"]]
            lines += [f"  {nid('op_' + k)} --> {nid(x)}" for x in o["outputs"]]
    lines += ["  classDef table fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A",
              "  classDef external fill:#ffffff,stroke:#888780,stroke-dasharray:4 3,color:#444441",
              "  classDef op fill:#E1F5EE,stroke:#0F6E56,color:#085041"]
    return "\n".join(lines)


def table_card(k, t, n):
    src = t.get("source") or {}
    prov = "\n".join(f"  - **{a}:** {v}" for a, v in src.items())
    cols = "\n".join(f"  - `{c}`: {v}" for c, v in (t.get("columns") or {}).items())
    rows = f"{n:,}" if n is not None else "not stored"
    return (f"#### {t['title']} (`{k}`)\n\n- **Path:** `{t['path']}`\n- **Rows:** {rows}\n- **Grain:** {t['grain']}\n"
            f"- **Key:** {', '.join(t.get('key') or []) or '-'}\n- **Source / provenance:**\n{prov}\n- **Columns:**\n{cols}\n")


def op_card(k, o, T):
    ins = ", ".join(f"`{i}`" for i in o["inputs"])
    outs = ", ".join(f"`{x}`" for x in o["outputs"])
    math = "\n".join(f"  - {m}" for m in o["math"])
    return f"#### {o['title']} (`{k}`)\n\n- **Script:** `{o['script']}`\n- **In:** {ins}\n- **Out:** {outs}\n- **Math:**\n{math}\n"


def main():
    spec = yaml.safe_load(SPEC.read_text())
    T, O = spec["tables"], spec["operations"]
    errors, counts = [], {}
    for k, t in T.items():
        if t.get("path") is None:
            counts[k] = None
            continue
        files = table_files(t["path"])
        if not files:
            errors.append(f"table {k}: no files at {t['path']}")
            counts[k] = None
        else:
            counts[k] = count_rows(files)
    for k, o in O.items():
        for x in o["inputs"] + o["outputs"]:
            if x not in T:
                errors.append(f"operation {k}: unknown table {x}")
    unmatched = sorted((n, r) for n, r in code_paths() if not any(t.get("path") and matches(r, t["path"]) for t in T.values()))
    errors += [f"{n} uses {r}, which is not in the lineage spec" for n, r in unmatched]

    (ROOT / "docs/data_lineage_counts.json").write_text(json.dumps(counts, indent=2))
    md = ["# Data lineage", "", "Every table in the final pipeline, where it came from, and the math connecting them.",
          "Generated by `src/data_lineage.py` from `docs/data_lineage.yaml`; row counts are live.", "",
          "## Overview", "", "```mermaid", mermaid(spec, counts), "```", ""]
    for c, title in spec["chapters"].items():
        md += [f"## {c}. {title}", "", "```mermaid", mermaid(spec, counts, c), "```", "", "### Tables", ""]
        md += [table_card(k, t, counts[k]) for k, t in T.items() if t["chapter"] == c]
        ops = [op_card(k, o, T) for k, o in O.items() if o["chapter"] == c]
        if ops:
            md += ["### Operations", ""] + ops
    (ROOT / "docs/data_lineage.md").write_text("\n".join(md))

    print(f"{len(T)} tables, {len(O)} operations; {sum(v is not None for v in counts.values())} stored tables counted")
    print("\n".join(f"  ERROR {e}" for e in errors) or "  all checks passed")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
