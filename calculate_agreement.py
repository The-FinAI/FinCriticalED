import json
import argparse
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
import math


def make_item_id(value):
    return (value.get("start"), value.get("end"))


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:20]


def fleiss_kappa(M):
    N = len(M)
    if N == 0:
        return float("nan")
    k = len(M[0])
    n_annotators = sum(M[0])

    P_i = []
    for row in M:
        if n_annotators <= 1:
            P_i.append(0.0)
            continue
        Pi = sum(r * (r - 1) for r in row) / (n_annotators * (n_annotators - 1))
        P_i.append(Pi)
    P_bar = sum(P_i) / N

    p_j = []
    for j in range(k):
        pj = sum(M[i][j] for i in range(N)) / (N * n_annotators)
        p_j.append(pj)
    P_e = sum(p * p for p in p_j)

    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    parser = argparse.ArgumentParser(description="Compute agreement from Label Studio export.")
    parser.add_argument("--file", required=True, help="Path to Label Studio JSON export.")
    parser.add_argument("--annotators", type=int, required=True,
                        help="Use tasks with exactly this number of annotators.")
    parser.add_argument("--pairwise", action="store_true",
                        help="When annotators >=3, also compute pairwise Cohen kappa inside those tasks.")
    args = parser.parse_args()

    data = load_annotations(args.file)
    desired = args.annotators

    annotator_span_counts = defaultdict(int)

    # === case 1: exactly 2 annotators -> simple pairwise ===
    if desired == 2:
        pair_labels_all = defaultdict(lambda: {"a": [], "b": []})
        global_label_set = set()

        for task in data:
            anns = task.get("annotations", [])
            users = list({ann.get("completed_by") for ann in anns})
            if len(users) != 2:
                continue

            a_id, b_id = users
            user2labels = {}
            all_items = set()

            for ann in anns:
                uid = ann.get("completed_by")
                labels_for_user = {}
                for res in ann.get("result", []):
                    if res.get("type") != "labels":
                        continue
                    value = res.get("value", {})
                    item_id = make_item_id(value)
                    if item_id[0] is None:
                        continue
                    labels = value.get("labels", [])
                    if not labels:
                        continue
                    label = labels[0]
                    labels_for_user[item_id] = label
                    all_items.add(item_id)
                    global_label_set.add(label)
                user2labels[uid] = labels_for_user
                annotator_span_counts[uid] += len(labels_for_user)

            all_items = sorted(all_items)
            for item in all_items:
                la = user2labels.get(a_id, {}).get(item, "O")
                lb = user2labels.get(b_id, {}).get(item, "O")
                pair_labels_all[(a_id, b_id)]["a"].append(la)
                pair_labels_all[(a_id, b_id)]["b"].append(lb)

        print(f"=== Span counts (tasks with exactly {desired} annotators) ===")
        for uid, cnt in sorted(annotator_span_counts.items()):
            print(f"Annotator {uid}: {cnt} spans")

        print(f"\n=== Cohen kappa (exactly {desired} annotators) ===")
        for (a_id, b_id), seqs in pair_labels_all.items():
            kappa = cohen_kappa_score(seqs["a"], seqs["b"])
            print(f"{a_id} vs {b_id}: kappa = {kappa:.4f}, on {len(seqs['a'])} items")

            # 按 label 的 kappa
            for lab in sorted(global_label_set):
                bin_a = [1 if x == lab else 0 for x in seqs["a"]]
                bin_b = [1 if x == lab else 0 for x in seqs["b"]]
                # 如果两边全是0，就没必要算
                if all(v == 0 for v in bin_a) and all(v == 0 for v in bin_b):
                    continue
                lk = cohen_kappa_score(bin_a, bin_b)
                print(f"    {a_id} vs {b_id}: [{lab}] kappa = {lk:.4f}, on {len(bin_a)} items")

    # === case 2: >=3 annotators -> Fleiss, and optionally pairwise ===
    else:
        global_label_set = set()
        tasks_filtered = []

        for task in data:
            anns = task.get("annotations", [])
            users = list({ann.get("completed_by") for ann in anns})
            if len(users) != desired:
                continue

            user2labels = {}
            all_items = set()
            for ann in anns:
                uid = ann.get("completed_by")
                labels_for_user = {}
                for res in ann.get("result", []):
                    if res.get("type") != "labels":
                        continue
                    value = res.get("value", {})
                    item_id = make_item_id(value)
                    if item_id[0] is None:
                        continue
                    labels = value.get("labels", [])
                    if not labels:
                        continue
                    label = labels[0]
                    labels_for_user[item_id] = label
                    all_items.add(item_id)
                    global_label_set.add(label)
                user2labels[uid] = labels_for_user
                annotator_span_counts[uid] += len(labels_for_user)

            tasks_filtered.append({
                "users": users,
                "items": sorted(all_items),
                "user2labels": user2labels
            })

        # build label space
        label_list = sorted(global_label_set)
        label2idx = {lab: i for i, lab in enumerate(label_list)}
        if "O" not in label2idx:
            label2idx["O"] = len(label2idx)
            label_list.append("O")

        # Fleiss matrix (overall)
        M = []
        for t in tasks_filtered:
            users = t["users"]
            items = t["items"]
            user2labels = t["user2labels"]
            for item in items:
                row = [0] * len(label_list)
                for uid in users:
                    lab = user2labels.get(uid, {}).get(item, "O")
                    row[label2idx[lab]] += 1
                M.append(row)

        print(f"=== Span counts (tasks with exactly {desired} annotators) ===")
        for uid, cnt in sorted(annotator_span_counts.items()):
            print(f"Annotator {uid}: {cnt} spans")

        kappa_f = fleiss_kappa(M)
        print(f"\n=== Fleiss kappa (exactly {desired} annotators) ===")
        print(f"Fleiss kappa = {kappa_f:.4f} on {len(M)} items, {len(label_list)} labels")

        # Additional: Fleiss by label (treat the label as "yes/no")
        print("\n=== Per-label Fleiss kappa (binary) ===")
        for lab in label_list:
            # Create a new matrix for this label: where label = 1, otherwise = 0.
            M_lab = []
            for t in tasks_filtered:
                users = t["users"]
                items = t["items"]
                user2labels = t["user2labels"]
                for item in items:
                    #
                    row = [0, 0]
                    for uid in users:
                        l = user2labels.get(uid, {}).get(item, "O")
                        if l == lab:
                            row[1] += 1
                        else:
                            row[0] += 1
                    M_lab.append(row)
            k_lab = fleiss_kappa(M_lab)
            print(f"[{lab}] Fleiss kappa = {k_lab:.4f} on {len(M_lab)} items")

        # optional: pairwise inside these tasks
        if args.pairwise:
            pair_labels_all = defaultdict(lambda: {"a": [], "b": []})
            for t in tasks_filtered:
                users = t["users"]
                items = t["items"]
                user2labels = t["user2labels"]
                for u1, u2 in combinations(users, 2):
                    for item in items:
                        l1 = user2labels.get(u1, {}).get(item, "O")
                        l2 = user2labels.get(u2, {}).get(item, "O")
                        pair_labels_all[(u1, u2)]["a"].append(l1)
                        pair_labels_all[(u1, u2)]["b"].append(l2)

            print("\n=== Pairwise Cohen kappa inside these multi-annotator tasks ===")
            for (u1, u2), seqs in pair_labels_all.items():
                kappa = cohen_kappa_score(seqs["a"], seqs["b"])
                print(f"{u1} vs {u2}: kappa = {kappa:.4f}, on {len(seqs['a'])} items")

    # end main


if __name__ == "__main__":
    main()
