"""Dump AgentDojo runs into canonical TraceCanon graphs (deterministic stage).

Hygiene rules and counters replicate external/agentdojo/convert_agentdojo.py exactly
(same glob, same dedup, same failure taxonomy) so the dump must reproduce
labexp/results_local/axis_a_agentdojo_stats.json. Node raw_content is now the
verbatim extracted text (block-list contents joined) instead of Python repr;
node/edge COUNTS are unchanged relative to the published stats.

Outputs (external/agentdojo/graphs/):
  graphs.jsonl      one canonical graph per accepted run
  index.csv         per-graph metadata index
  dump_summary.json counters + attack-type x security x utility census
"""
import csv, glob, hashlib, json, os, sys, collections

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo", "runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
NTYPES = ("Problem", "Thinking", "Action", "State")


def block_text(content):
    """Verbatim text of an AgentDojo message content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("content", "")))
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return json.dumps(content, default=str)


def action_text(t):
    fn = t.get("function")
    if isinstance(fn, dict):
        name, args = fn.get("name", ""), fn.get("arguments", {})
    else:
        name, args = fn, t.get("args", {})
    try:
        args_s = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        args_s = str(args)
    return json.dumps({"name": name, "arguments": json.loads(args_s) if args_s.startswith("{") else args_s},
                      sort_keys=True, default=str)


def is_dos(attack_type):
    return bool(attack_type) and "dos" in str(attack_type).lower()


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPO, "*/*/*/*/*.json"))) + \
            sorted(glob.glob(os.path.join(REPO, "*/*/*/*.json")))
    seen = set()
    failures = collections.Counter()
    census = collections.Counter()          # (attack_type, security, utility)
    census_dos = collections.Counter()      # (security, utility) for dos runs
    kappa_tally = collections.Counter()     # (kappa, y_lambda)
    n_files = n_graphs = dup = 0
    node_type_totals = collections.Counter()
    per_suite = collections.defaultdict(collections.Counter)
    edge_totals = collections.Counter()

    fp = open(os.path.join(OUT, "graphs.jsonl"), "w")
    ip = open(os.path.join(OUT, "index.csv"), "w", newline="")
    iw = csv.writer(ip)
    iw.writerow(["graph_id", "suite", "user_task_id", "injection_task_id", "attack_type",
                 "pipeline", "y_naive", "y_lambda", "kappa", "n_nodes", "n_edges"])

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
        h = hashlib.sha256(json.dumps(msgs, sort_keys=True, default=str).encode()).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)

        suite = d.get("suite_name", "unknown")
        sec, util = d.get("security"), d.get("utility")
        attack = d.get("attack_type")
        atk_s = str(attack) if attack is not None else "none"
        census[(atk_s, str(sec), str(util))] += 1
        dos = is_dos(attack)
        if dos:
            census_dos[(str(sec), str(util))] += 1

        # ---- pass 1: nodes (verbatim text) ----
        nodes, node_kinds = [], []
        nu = 0
        for m in msgs:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                nu += 1
                nodes.append(("Problem", block_text(m.get("content"))))
                node_kinds.append(("Problem", None))
                continue
            if role == "assistant":
                tcs = m.get("tool_calls") or []
                c = m.get("content")
                if c:
                    txt = block_text(c)
                    nodes.append(("Thinking" if tcs else "State", txt))
                    node_kinds.append(("Thinking" if tcs else "State", None))
                for t in tcs:
                    fn = t.get("function") if isinstance(t, dict) else None
                    if fn is None:
                        failures["malformed_tool_call"] += 1
                        continue
                    cid = t.get("id") or f"auto_{len(nodes)}"
                    nodes.append(("Action", action_text(t)))
                    node_kinds.append(("Action", cid))
                if not tcs and not c:
                    failures["empty_assistant_turn"] += 1
                continue
            if role == "tool":
                nodes.append(("State", block_text(m.get("content"))))
                node_kinds.append(("State", m.get("tool_call_id")))
                continue
            failures[f"unknown_role_{role}"] += 1

        n_nodes = len(nodes)
        if n_nodes == 0:
            failures["no_execution_nodes"] += 1
            continue
        n_graphs += 1
        tc = collections.Counter(t for t, _ in nodes)
        for t, c in tc.items():
            node_type_totals[t] += c
            per_suite[suite][f"n_{t.lower()}"] += c
        per_suite[suite]["graphs"] += 1

        # ---- pass 2: structural edges ----
        edges = []
        for i in range(n_nodes - 1):                     # sequential: turn order
            edges.append(("node_%04d" % (i + 1), "node_%04d" % (i + 2), "sequential"))
        pos_of = {}
        for i, (k, cid) in enumerate(node_kinds):        # computes_from: call -> its result
            if k == "Action" and cid:
                pos_of[i] = cid
        for i, cid in pos_of.items():
            for j in range(i + 1, n_nodes):
                kj, cjid = node_kinds[j]
                if kj == "State" and cjid == cid:
                    edges.append(("node_%04d" % (i + 1), "node_%04d" % (j + 1), "computes_from"))
                    break
        edge_totals["sequential"] += max(0, n_nodes - 1)
        edge_totals["computes_from"] += sum(1 for e in edges if e[2] == "computes_from")

        # ---- labels under Lambda ----
        # AgentDojo native semantics (benchmark.py / task_suite.py): security = the
        # injection task's check PASSED, i.e. the attacker's realized goal; DoS runs
        # override security = not utility. Published TraceCanon convention (tab:convert):
        # y_naive = 1 iff security is True, counted over all runs incl. DoS (8,442);
        # Lambda then quarantines DoS (kappa=disruption_dos, y=None).
        y_naive = None if sec is None else int(bool(sec))       # security=True -> breach realized
        if sec is None:
            kappa, y_lambda = "unlabeled", None
        elif dos:
            kappa, y_lambda = "disruption_dos", None             # quarantined under Lambda
        else:
            kappa = "breach_under_injection" if y_naive == 1 else "benign"
            y_lambda = y_naive
        gid = "ad_%08d" % n_graphs
        kappa_tally[(kappa, str(y_lambda))] += 1
        user_task = d.get("user_task_id")
        inj_task = d.get("injection_task_id")
        first_user = next((block_text(m.get("content")) for m in msgs if m.get("role") == "user"), "")
        g = {
            "graph_id": gid,
            "source_file": os.path.relpath(f, REPO),
            "domain": suite,
            "task": first_user[:300],
            "nodes": [{"id": "node_%04d" % (i + 1), "sequence": i + 1, "type": t,
                       "raw_content": txt,
                       "raw_content_sha256": hashlib.sha256(txt.encode()).hexdigest(),
                       "source_path": "messages", "metadata": {}}
                      for i, (t, txt) in enumerate(nodes)],
            "edges": [{"source": s, "target": tt, "type": ty, "edge_origin": "structural_certified"}
                      for s, tt, ty in edges],
            "graph_label": {"label": y_lambda, "label_naive": y_naive,
                            "policy": "outcome_breach_adapted",
                            "anomaly_kind": kappa,
                            "source": {"security": sec, "utility": util,
                                       "attack_type": atk_s, "user_task_id": user_task,
                                       "injection_task_id": inj_task,
                                       "pipeline": os.path.relpath(f, REPO).split(os.sep)[1]}},
        }
        fp.write(json.dumps(g, ensure_ascii=False, default=str) + "\n")
        iw.writerow([gid, suite, user_task, inj_task, atk_s,
                     g["graph_label"]["source"]["pipeline"],
                     "" if y_naive is None else y_naive,
                     "" if y_lambda is None else y_lambda,
                     kappa, n_nodes, len(edges)])

    fp.close(); ip.close()
    total_nodes = sum(node_type_totals.values())
    summary = {
        "n_files": n_files, "n_graphs": n_graphs, "duplicates_removed": dup,
        "failures": dict(failures),
        "census_attack_security_utility": {f"{a}|sec={s}|util={u}": c
                                           for (a, s, u), c in sorted(census.items())},
        "census_dos": {f"sec={s}|util={u}": c for (s, u), c in sorted(census_dos.items())},
        "labels_lambda": {f"{k}|y={y}": c for (k, y), c in sorted(kappa_tally.items())},
        "node_type_totals": dict(node_type_totals),
        "total_nodes": total_nodes,
        "edges_structural": dict(edge_totals),
        "per_suite": {k: dict(v) for k, v in per_suite.items()},
    }
    json.dump(summary, open(os.path.join(OUT, "dump_summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in
                      ["n_files", "n_graphs", "duplicates_removed", "failures",
                       "census_dos", "node_type_totals", "total_nodes", "edges_structural"]},
                     indent=1))


if __name__ == "__main__":
    main()
