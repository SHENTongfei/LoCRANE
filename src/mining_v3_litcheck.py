# -*- coding: utf-8 -*-
"""mining_v3_litcheck.py - literature matrix v3 (pre-registered in MINING_v3_PLAN.md).
Input: runs_v3/mining_v3_M2_M3_genes.csv -> top-30 by gip_all + top-10 z_LM up/down.
Method (PubMed eutils via gateway 1099, abstract-level, zero fabrication):
  for each gene: esearch db=pubmed term=f"{gene} AND (aging OR longevity OR
  centenarian OR 'whole blood' OR 'blood cells' OR transcriptome)" -> top-3
  PMIDs; esummary titles.
Grades (rule, fixed before results):
  A = >=1 title contains the gene AND (aging|longevity|centenarian|aging-*)
  B = title contains gene-family/mechanism keyword (SIRT|FOKO|FOXO|APC|WNT|
      mTOR|AMPK|mitochondri|immuno|inflamm-|senesc)
  C = gene known (>=5 pubmed hits in any topic) but combo novel
  NONE = 0 hits = new-discovery CANDIDATE (NO-FAIL: candidate only, never "discovery")
Output: runs_v3/mining_v3_lits.json + mining_v3_litcheck.csv.
"""
import os, re, json, time, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
PROXY = "http://127.0.0.1:1099"

def curl(url, out):
    subprocess.call(["curl", "-sk", "--max-time", "30", "-x", PROXY, url, "-o", out], cwd=HERE)

def esearch(term, db="pubmed", retmax=5):
    out = os.path.join(HERE, "_lc_probe.xml")
    curl(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={db}&term={term}&retmax={retmax}", out)
    x = open(out, encoding="utf-8", errors="ignore").read() if os.path.exists(out) else ""
    return re.findall(r"<Id>(\d+)</Id>", x)

def esummary(ids):
    if not ids:
        return []
    out = os.path.join(HERE, "_lc_summ.xml")
    curl("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=" + ",".join(ids) + "&retmode=json", out)
    x = open(out, encoding="utf-8", errors="ignore").read() if os.path.exists(out) else "{}"
    try:
        d = json.loads(x)
        res = d.get("result", {})
        return [(res[i]["title"], i) for i in res.get("uids", [])]
    except Exception:
        return []

MECH = re.compile(r"SIRT|FOKO|FOXO|mTOR|AMPK|mitochondri|immuno|inflammat|senesc|APOE|WNT|NAD", re.I)

def main():
    gdf = pd.read_csv(os.path.join(RUNS, "mining_v3_M2_M3_genes.csv"))
    top = gdf.nlargest(30, "gip_all")["gene"].tolist()
    lup = gdf.nlargest(10, "z_LM")["gene"].tolist()
    ldn = gdf.nsmallest(10, "z_LM")["gene"].tolist()
    genes = list(dict.fromkeys(top + lup + ldn))
    out = {}
    for g in genes:
        ids = esearch(f'"{g}" AND (aging OR longevity OR centenarian OR "whole blood" OR transcriptome)')
        titles = esummary(ids[:3])
        txt = " ".join(t for t, _ in titles)
        n_any = len(esearch(g))
        if any(re.search(rf"\b{re.escape(g)}\b", t, re.I) and re.search(r"aging|longevity|centenarian", t, re.I) for t, _ in titles):
            grade = "A"
        elif any(MECH.search(t) for t, _ in titles):
            grade = "B"
        elif n_any >= 5:
            grade = "C"
        else:
            grade = "NONE"
        out[g] = {"grade": grade, "n_pubmed_any": int(n_any),
                  "top_titles": [t[:110] for t, _ in titles],
                  "pmids": [i for _, i in titles]}
        print(f"[litcheck] {g}: {grade} (n={n_any})", flush=True)
        time.sleep(1)
    rows = []
    for g, d in out.items():
        row = gdf[gdf["gene"] == g].iloc[0]
        rows.append({"gene": g, "grade": d["grade"], "n_pubmed_any": d["n_pubmed_any"],
                     "gip_all": row["gip_all"], "z_LM": row["z_LM"], "rank_all": int(row["rank_all"]),
                     "top_title": d["top_titles"][0] if d["top_titles"] else "(no hit)"})
    pd.DataFrame(rows).to_csv(os.path.join(RUNS, "mining_v3_litcheck.csv"), index=False)
    json.dump(out, open(os.path.join(RUNS, "mining_v3_lits.json"), "w"), indent=1)
    none = [g for g, d in out.items() if d["grade"] == "NONE"]
    print(f"[litcheck] DONE. NONE-candidates (new-discovery candidates, NOT claims): {none}", flush=True)

if __name__ == "__main__":
    main()
