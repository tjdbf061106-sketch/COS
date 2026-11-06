# -*- coding: utf-8 -*-
"""CAS번호가 없는 성분을 화학물질 '이름'으로 PubChem에서 찾아본다.
'Others', 'Inert Ingredients' 같은 제네릭 용어는 제외하고, 실제 화학물질명으로 보이는 것만 시도."""
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

GENERIC_PATTERNS = [
    r"^other\(?s\)?$", r"^others$", r"^inert ingredient", r"^other ingredient",
    r"^trade secret", r"^proprietary", r"^n/?a$", r"^various", r"^unidentified",
    r"^related compounds$", r"^commercial mixture$", r"^no cas", r"^undisclosed",
]
GENERIC_RE = re.compile("|".join(GENERIC_PATTERNS), re.IGNORECASE)


def is_generic(name):
    name = str(name).strip()
    return bool(GENERIC_RE.match(name)) or len(name) < 3


def resolve_by_name(name):
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON",
            timeout=10,
        )
        if r.status_code != 200:
            return None, "not_found"
        cid = r.json()["IdentifierList"]["CID"][0]
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


comp = pd.read_excel(f"{BASE}\\skin_보강후보_통합.xlsx", sheet_name="성분함량")
missing = comp[comp["CAS_원문"].isna()].copy()
missing["is_generic"] = missing["성분명_원문"].apply(is_generic)
candidates = missing[~missing["is_generic"]]["성분명_원문"].dropna().unique()
print(f"CAS 없는 {len(missing)}개 성분 중 제네릭 제외 {len(candidates)}개 고유 이름 시도")

rows = []
for i, name in enumerate(sorted(candidates)):
    cid, result = resolve_by_name(name)
    if result == "not_found":
        rows.append({"name": name, "cid": None, "smiles": None, "status": "not_found"})
    elif isinstance(result, str) and result.startswith("error:"):
        rows.append({"name": name, "cid": None, "smiles": None, "status": result})
    elif result == "cid_found_no_smiles":
        rows.append({"name": name, "cid": cid, "smiles": None, "status": "cid_found_no_smiles"})
    else:
        rows.append({"name": name, "cid": cid, "smiles": result, "status": "found"})
    print(f"[{i+1}/{len(candidates)}] {name[:50]!r} -> {rows[-1]['status']}")
    time.sleep(0.25)

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}\\name_smiles_cache.csv", index=False, encoding="utf-8-sig")
print()
print(out["status"].value_counts())
print(f"저장: {BASE}\\name_smiles_cache.csv")
