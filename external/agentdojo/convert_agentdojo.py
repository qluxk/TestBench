import json, glob, hashlib, statistics, collections, os

REPO = "repo/runs"
OUT = "../labexp/results_local/axis_a_agentdojo_stats.json"

node_type_totals = collections.Counter()
graph_sizes = []
edge_counts = []
per_suite = collections.defaultdict(lambda: collections.Counter())
labels = collections.Counter()
failures = collections.Counter()
thinking_nodes = 0
user_turn_counts = collections.Counter()
n_graphs = 0
n_files = 0

files = sorted(glob.glob(f"{REPO}/*/*/*/*/*.json") + glob.glob(f"{REPO}/*/*/*/*.json"))
seen_hashes = set()
dup = 0

for f in files:
    n_files += 1
    try:
        d = json.load(open(f))
    except Exception:
        failures["unparseable_json"] += 1
        continue
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        failures["empty_or_missing_messages"] += 1
        continue
    if d.get("error"):
        failures["run_error_flag"] += 1
        continue
    # dedupe identical trajectories
    h = hashlib.sha256(json.dumps(msgs, sort_keys=True, default=str).encode()).hexdigest()
    if h in seen_hashes:
        dup += 1
        continue
    seen_hashes.add(h)

    suite = d.get("suite_name", "unknown")
    labels[d.get("security")] += 1
    labels[("suite", suite, d.get("security"))] += 1

    nodes = []          # (type, text_len)
    call_id_to_idx = {} # tool_call_id -> action node idx
    root_set = False
    nu = 0
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        tcs = m.get("tool_calls")
        if role == "system":
            continue  # environment configuration, record-level L1, not an execution turn
        if role == "user":
            nu += 1
            nodes.append(("Problem", len(str(content or ""))))
            continue
        if role == "assistant":
            has_calls = bool(tcs)
            if content:
                nodes.append(("Thinking" if has_calls else "State", len(str(content))))
            if has_calls:
                for t in tcs:
                    fn = t.get("function") if isinstance(t, dict) else None
                    if fn is None:
                        failures["malformed_tool_call"] += 1
                        continue
                    cid = t.get("id") or f"auto_{len(nodes)}"
                    call_id_to_idx[cid] = len(nodes)
                    nodes.append(("Action", len(json.dumps(t.get("args", {}), default=str)) + len(str(fn))))
            if not has_calls and not content:
                failures["empty_assistant_turn"] += 1
            continue
        if role == "tool":
            nodes.append(("State", len(str(content or ""))))
            continue
        failures[f"unknown_role_{role}"] += 1

    user_turn_counts[nu] += 1
    n_graphs += 1
    tc = collections.Counter(t for t, _ in nodes)
    for t, c in tc.items():
        node_type_totals[t] += c
        per_suite[suite][f"n_{t.lower()}"] += c
    per_suite[suite]["graphs"] += 1
    graph_sizes.append(len(nodes))
    thinking_nodes += tc["Thinking"]

    # structural edges: computes_from (call -> its result) + sequential (turn order)
    # pass 2: match tool messages to calls via tool_call_id
    idx = -1
    e_cf = 0
    node_kinds = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            node_kinds.append(("Problem", None)); continue
        if role == "assistant":
            tcs = m.get("tool_calls")
            if m.get("content"):
                node_kinds.append(("Thinking" if tcs else "State", None))
            for t in (tcs or []):
                node_kinds.append(("Action", (t or {}).get("id")))
            continue
        if role == "tool":
            node_kinds.append(("State", m.get("tool_call_id")))
    e_seq = 0
    e_cf = 0
    for i in range(len(node_kinds) - 1):
        e_seq += 1
    kind_only = [k for k, _ in node_kinds]
    # sequential edges should skip nothing; count is len-1 per graph
    for i, (k, cid) in enumerate(node_kinds):
        if k == "Action" and cid:
            for j in range(i + 1, len(node_kinds)):
                kj, cjid = node_kinds[j]
                if kj == "State" and cjid == cid:
                    e_cf += 1
                    break
    edge_counts.append(("computes_from", e_cf))
    edge_counts.append(("sequential", e_seq))

cf_total = sum(c for t, c in edge_counts if t == "computes_from")
seq_total = sum(c for t, c in edge_counts if t == "sequential")
sizes = graph_sizes
out = {
    "n_files": n_files,
    "n_graphs": n_graphs,
    "duplicates_removed": dup,
    "failures": dict(failures),
    "labels_security": {str(k): v for k, v in labels.items() if not isinstance(k, tuple)},
    "labels_by_suite": {f"{k[0]}/{k[1]}": v for k, v in labels.items() if isinstance(k, tuple) and k[0] in ("banking","slack","travel","workspace")},
    "node_type_totals": dict(node_type_totals),
    "graph_nodes_mean": round(statistics.mean(sizes), 2),
    "graph_nodes_median": statistics.median(sizes),
    "graph_nodes_min": min(sizes),
    "graph_nodes_max": max(sizes),
    "graph_nodes_p90": sorted(sizes)[int(0.9 * len(sizes))],
    "thinking_fraction_of_graphs": round(thinking_nodes / max(1, sum(node_type_totals.values())), 4),
    "user_turn_distribution_top": dict(user_turn_counts.most_common(5)),
    "edges_structural": {"computes_from": cf_total, "sequential": seq_total,
                          "edges_per_graph_mean": round((cf_total + seq_total) / max(1, n_graphs), 2)},
    "per_suite": {k: dict(v) for k, v in per_suite.items()},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ["n_files","n_graphs","duplicates_removed","failures","labels_security","node_type_totals","graph_nodes_mean","graph_nodes_median","graph_nodes_p90","edges_structural"]}, indent=1))
