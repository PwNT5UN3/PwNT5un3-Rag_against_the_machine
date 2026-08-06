import json
import sys
from collections import defaultdict

def load_mappings(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for item in data:
        grouped[item["file_path"]].append(
            (item["first_character_index"], item["last_character_index"])
        )
    return grouped

def ranges_overlap(a, b, inclusive_end=True):
    a0, a1 = a
    b0, b1 = b
    if inclusive_end:
        return a0 <= b1 and b0 <= a1
    else:
        return a0 < b1 and b0 < a1

def main(file_a, file_b, out_path="result.json", inclusive_end=True):
    A = load_mappings(file_a)
    B = load_mappings(file_b)

    result = {
        "file_diffs": []
    }

    all_paths = sorted(set(A.keys()) | set(B.keys()))

    for p in all_paths:
        a_ranges = A.get(p, [])
        b_ranges = B.get(p, [])

        a_exact = set(a_ranges)
        b_exact = set(b_ranges)

        has_discrepancy = False
        a_only_discrepancy = []  # which A ranges participate (kept for debugging if you want)
        b_only_discrepancy = []  # which B ranges participate (kept for debugging if you want)

        # Detect discrepancies using your original logic:
        # A -> B: A ranges overlapping B but not exactly identical
        for ar in a_ranges:
            overlaps_with = []
            for br in b_ranges:
                if ranges_overlap(ar, br, inclusive_end=inclusive_end):
                    overlaps_with.append(list(br))
            if overlaps_with and ar not in b_exact:
                has_discrepancy = True
                a_only_discrepancy.append({
                    "a_range": list(ar),
                    "overlaps_b_ranges": overlaps_with
                })

        # B -> A: B ranges overlapping A but not exactly identical
        for br in b_ranges:
            if br in a_exact:
                continue
            overlaps_with = []
            for ar in a_ranges:
                if ranges_overlap(ar, br, inclusive_end=inclusive_end):
                    overlaps_with.append(list(ar))
            if overlaps_with:
                has_discrepancy = True
                b_only_discrepancy.append({
                    "b_range": list(br),
                    "overlaps_a_ranges": overlaps_with
                })

        if has_discrepancy:
            # Key change: include ALL ranges for the file (from both sides)
            file_report = {
                "file_path": p,
                "all_a_ranges": [list(r) for r in a_ranges],
                "all_b_ranges": [list(r) for r in b_ranges],

                # Optional: keep the original focused discrepancy info too
                "a_ranges_overlapping_b_not_identical": a_only_discrepancy,
                "b_ranges_overlapping_a_not_identical": b_only_discrepancy,
            }
            result["file_diffs"].append(file_report)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote diff report to {out_path}")

if __name__ == "__main__":
    # usage:
    # python overlap_nonidentical_to_json.py left.json right.json [inclusive]
    # inclusive: 1 for inclusive last_character_index (default), 0 for half-open [start,end)
    if len(sys.argv) < 3:
        print("Usage: python overlap_nonidentical_to_json.py left.json right.json [inclusive]")
        raise SystemExit(1)

    inclusive_end = True
    if len(sys.argv) >= 4:
        inclusive_end = (sys.argv[3] == "1")

    main(sys.argv[1], sys.argv[2], out_path="result.json", inclusive_end=inclusive_end)
