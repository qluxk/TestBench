import json, time, requests

CROSSREF = "https://api.crossref.org/works/"
DATACITE = "https://api.datacite.org/dois/"
MAILTO = {"mailto": "bench@example.org"}
HEADERS = {"User-Agent": "bench-builder/1.0 (mailto:bench@example.org)"}

ENTRIES = [
    ("liu2025wainjectbench", ("arxiv", "2510.01354")),
    ("ma2026autodojo", ("arxiv", "2606.15057")),
    ("yang2026when", ("arxiv", "2606.20023")),
    ("wang2026safety", ("arxiv", "2607.28685")),
    ("sun2026pastabench", ("arxiv", "2609.28197")),
    ("liu2024automatic", ("arxiv", "2403.04957")),
    ("shi2025prompt", ("arxiv", "2504.19793")),
    ("lee2026will", ("arxiv", "2608.30362")),
    ("wang2026landscape", ("arxiv", "2602.10453")),
    ("liu2026passing", ("arxiv", "2610.03448")),
    ("chen2024struq", ("arxiv", "2402.06363")),
    ("chen2024secalign", ("arxiv", "2410.05451")),
    ("mou2026toolsafe", ("arxiv", "2601.10156")),
    ("liu2026agentdog", ("arxiv", "2601.18491")),
    ("hossain2026nexus", ("arxiv", "2607.19356")),
    ("chen2025agentguard", ("arxiv", "2502.09809")),
    ("chen2026tests", ("arxiv", "2603.18245")),
    ("lee2026webmcpphalanx", ("arxiv", "2608.24017")),
    ("liu2022bond", ("arxiv", "2206.10071")),
    ("tang2023gadbench", ("arxiv", "2306.12251")),
    ("liu2022pygod", ("arxiv", "2204.12095")),
    ("zhao2019pyod", ("arxiv", "1901.01588")),
    ("tang2022rethinking", ("arxiv", "2205.15508")),
    ("qiao2025anomalygfm", ("arxiv", "2502.09254")),
    ("zheng2026owleye", ("arxiv", "2601.19102")),
    ("yang2024adllm", ("arxiv", "2412.11142")),
    ("wang2026glass", ("arxiv", "2609.05253")),
    ("qiao2024deep", ("arxiv", "2409.09957")),
    ("hoshen2023representation", ("arxiv", "2307.11085")),
    ("goyal2024orchid", ("arxiv", "2408.13347")),
    ("ju2022blackbox", ("arxiv", "2202.09389")),
    ("tao2022adversarial", ("arxiv", "2208.01819")),
    ("choi2024unveiling", ("arxiv", "2412.18370")),
    ("lyu2025navigating", ("arxiv", "2506.13276")),
    ("ma2025graphtextack", ("arxiv", "2511.12423")),
    ("gao2023addressing", ("doi", "10.1145/3543507.3583268")),
    ("roy2024gadnr", ("doi", "10.1145/3616855.3635767")),
    ("he2024adagad", ("doi", "10.1609/aaai.v38i8.28691")),
    ("dong2025smoothgnn", ("doi", "10.1145/3696410.3714615")),
    ("zheng2025dynamic", ("doi", "10.1609/aaai.v39i12.33464")),
    ("zipperle2022provenancebased", ("doi", "10.1145/3539605")),
    ("goyal2023sometimes", ("doi", "10.14722/ndss.2023.24207")),
    ("xu2024procsage", ("doi", "10.1186/s42400-024-00240-w")),
    ("mohammadi2025evaluation", ("doi", "10.1145/3711896.3736570")),
]


def esc(s):
    return s.replace("\\", "").replace("{", "").replace("}", "")


def cr_entry(key, doi):
    r = requests.get(CROSSREF + doi, params=MAILTO, headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = r.json()["message"]
    typ = m.get("type", "")
    title = esc(m["title"][0])
    auth = " and ".join(
        " ".join(x for x in [a.get("given", ""), a.get("family", "")] if x)
        for a in m.get("author", [])
    )
    year = m["issued"]["date-parts"][0][0]
    ct = (m.get("container-title") or [""])[0]
    vol, issue, pages = m.get("volume", ""), m.get("issue", ""), m.get("page", "")
    pub = esc(m.get("publisher", ""))
    lines = []
    if typ == "journal-article" or "journal" in typ:
        lines.append("@article{%s," % key)
        lines.append("  title = {%s}," % title)
        lines.append("  author = {%s}," % auth)
        lines.append("  year = {%s}," % year)
        if ct:
            lines.append("  journal = {%s}," % esc(ct))
        if vol:
            lines.append("  volume = {%s}," % vol)
        if issue:
            lines.append("  number = {%s}," % issue)
        if pages:
            lines.append("  pages = {%s}," % pages)
    else:
        lines.append("@inproceedings{%s," % key)
        lines.append("  title = {%s}," % title)
        lines.append("  author = {%s}," % auth)
        lines.append("  year = {%s}," % year)
        if ct:
            lines.append("  booktitle = {%s}," % esc(ct))
        if pages:
            lines.append("  pages = {%s}," % pages)
        if pub:
            lines.append("  publisher = {%s}," % pub)
    lines.append("  doi = {%s}," % doi)
    lines.append("}")
    return "\n".join(lines) + "\n"


def dc_entry(key, arxiv_id):
    r = requests.get(DATACITE + "10.48550/arxiv." + arxiv_id, headers=HEADERS, timeout=30)
    r.raise_for_status()
    a = r.json()["data"]["attributes"]
    title = esc(a["titles"][0]["title"])
    auth = " and ".join(esc(c["name"]) for c in a.get("creators", []))
    year = a.get("publicationYear", "")
    lines = [
        "@misc{%s," % key,
        "  title = {%s}," % title,
        "  author = {%s}," % auth,
        "  year = {%s}," % year,
        "  howpublished = {arXiv:%s}," % arxiv_id,
        "  eprint = {%s}," % arxiv_id,
        "  archivePrefix = {arXiv},",
        "  doi = {10.48550/arxiv.%s}," % arxiv_id,
        "  url = {https://arxiv.org/abs/%s}," % arxiv_id,
        "}",
    ]
    return "\n".join(lines) + "\n"


out, fails = [], []
for key, (kind, ident) in ENTRIES:
    try:
        if kind == "doi":
            out.append(cr_entry(key, ident))
        else:
            out.append(dc_entry(key, ident))
        print("OK  ", key)
    except Exception as e:
        fails.append((key, kind, ident, str(e)[:120]))
        print("FAIL", key, str(e)[:120])
    time.sleep(0.4)

with open("refs_built.bib", "w") as f:
    f.write("\n".join(out))
print("\nWrote", len(out), "entries;", len(fails), "failures")
for f_ in fails:
    print("FAILED:", f_)
