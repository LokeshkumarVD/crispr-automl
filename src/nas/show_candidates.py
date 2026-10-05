# src/nas/show_candidates.py
import json
import pandas as pd

from src.nas.nas_search import arch_to_str, LOG_PATH


def main():
    with open(LOG_PATH) as f:
        log = json.load(f)

    rows = []
    for h in log["history"]:
        rows.append({
            "candidate #": h["index"],
            "origin": h["origin"],
            "architecture": arch_to_str(h["architecture"]),
            "num layers": len(h["architecture"]),
            "params": h["params"],
            "fitness (val Spearman)": round(h["fitness"], 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("fitness (val Spearman)", ascending=False).reset_index(drop=True)

    print("All 12 candidates evaluated during the search, ranked by fitness:\n")
    print(df.to_string(index=False))

    print("\nReference baselines evaluated under the same mini-budget:")
    for name, score in log["reference"].items():
        print(f"  {name}: {round(score, 4)}")

    df.to_csv("reports/figures/nas_all_candidates.csv", index=False)
    print("\nSaved full table to reports/figures/nas_all_candidates.csv")


if __name__ == "__main__":
    main()