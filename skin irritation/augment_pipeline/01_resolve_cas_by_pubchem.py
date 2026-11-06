# -*- coding: utf-8 -*-
"""skin_보강후보_통합.xlsx의 신규 CASRN 98개를 PubChem API로 SMILES 변환.
기존 casrn_smiles_cache.csv와 동일 스키마(casrn, cid, smiles, status)로 저장."""
import os
import re
import sys
import time

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skin irritation/
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def resolve(casrn):
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{casrn}/cids/JSON",
            timeout=10,
        )
        if r.status_code != 200:
            return None, "not_found"
        cids = r.json()["IdentifierList"]["CID"]
        cid = cids[0]
        r2 = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON",
            timeout=10,
        )
        if r2.status_code != 200:
            return cid, "cid_found_no_smiles"
        smiles = r2.json()["PropertyTable"]["Properties"][0]["SMILES"]
        return cid, smiles
    except Exception as e:
        return None, f"error:{e}"


df = pd.read_excel(f"{BASE}\\skin_보강후보_통합.xlsx", sheet_name="성분함량")
cache = pd.read_csv(f"{BASE}\\casrn_smiles_cache.csv")
existing = set(cache["casrn"].dropna())

candidates = sorted(set(df["CAS_원문"].dropna()) - existing)
valid = [c for c in candidates if CAS_RE.match(str(c))]
invalid = [c for c in candidates if not CAS_RE.match(str(c))]
print(f"신규 CASRN {len(candidates)}개 중 형식 유효 {len(valid)}개, 형식 이상(건너뜀) {len(invalid)}개: {invalid}")

rows = []
for i, casrn in enumerate(valid):
    cid, result = resolve(casrn)
    if result == "not_found":
        rows.append({"casrn": casrn, "cid": None, "smiles": None, "status": "not_found"})
    elif isinstance(result, str) and result.startswith("error:"):
        rows.append({"casrn": casrn, "cid": None, "smiles": None, "status": result})
    elif result == "cid_found_no_smiles":
        rows.append({"casrn": casrn, "cid": cid, "smiles": None, "status": "cid_found_no_smiles"})
    else:
        rows.append({"casrn": casrn, "cid": cid, "smiles": result, "status": "found"})
    print(f"[{i+1}/{len(valid)}] {casrn} -> {rows[-1]['status']}")
    time.sleep(0.25)

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}\\casrn_smiles_cache_new98.csv", index=False, encoding="utf-8-sig")
print()
print(out["status"].value_counts())
print(f"저장: {BASE}\\casrn_smiles_cache_new98.csv")
