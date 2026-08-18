"""Render the condition-stratified BASE/SUPER AP json as a booktabs LaTeX table
(layout mirrors the per-condition analysis: Condition / N / Super / Base / Gap,
grouped by time of day, weather, scene type, crowd density).

    python -m analysis.make_condition_table
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
SRC = OUT / "bdd100k/eval/condition_stratified.json"
DST = Path(__file__).resolve().parent.parent / "outputs/figures/tab_condition_stratified.tex"

# pretty display names for crowd labels carry the median inline already
NICE = {"daytime": "Daytime", "night": "Night", "dawn/dusk": "Dawn/Dusk",
        "clear": "Clear", "overcast": "Overcast", "rainy": "Rainy", "snowy": "Snowy",
        "highway": "Highway", "city street": "City street", "residential": "Residential"}


def main():
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    by_head = {}
    for r in rows:
        by_head.setdefault(r["heading"], []).append(r)

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Condition-stratified AP@[.50:.95] on BDD100K val "
             f"({d['n']:,} images) for the frozen AnyDepth-YOLOv12s we adopt. "
             r"The SUPER$-$BASE "
             r"gap quantifies the per-condition benefit of the deep path. It is "
             r"input-dependent rather than uniform---sparse (crowd-low) scenes benefit "
             r"about $1.4\times$ more from the deep path than crowded ones---which is "
             r"what makes per-image routing worthwhile. "
             r"The crowd split uses the per-image object-count median "
             f"(${d['crowd_median']}$) as threshold.}}")
    L.append(r"\label{tab:condition_stratified}")
    L.append(r"\begin{tabular}{lrccc}")
    L.append(r"\toprule")
    L.append(r"Condition & $N$ & SUPER & BASE & Gap \\")
    L.append(r"\midrule")

    def fmt(r):
        return (f"{r['super'] * 100:.2f} & {r['base'] * 100:.2f} & "
                f"${r['gap'] * 100:+.2f}$ \\\\")

    # All row first
    allr = by_head.get("All", [])
    for r in allr:
        L.append(f"All & {r['n']:,} & {fmt(r)}")

    for head in ("Time of day", "Weather", "Scene type", "Crowd density"):
        if head not in by_head:
            continue
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{head}}}}} \\")
        for r in by_head[head]:
            lab = NICE.get(r["label"], r["label"])
            lab = lab.replace("<=", r"$\le$").replace("(> ", r"($>$ ")
            L.append(rf"\quad {lab} & {r['n']:,} & {fmt(r)}")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    DST.write_text("\n".join(L) + "\n")
    print(f"[*] -> {DST}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
