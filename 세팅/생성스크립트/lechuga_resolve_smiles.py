# -*- coding: utf-8 -*-
"""Lechuga 2025 계면활성제 22종 — INCI명 기준 PubChem SMILES 해결 시도 (0단계, 선행 블로커 확인).
재학습/production 변경 없음. 결과만 cosmetic_validation/lechuga_smiles_cache.csv 로 저장.
skin_feature_rebuild.ipynb의 fetch_smiles_by_query 패턴(3회 재시도, 요청간 0.3s)을 그대로 따름.
"""
import json as _json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = r"C:\Users\UserK\학술제"
LECHUGA_CSV = rf"{BASE}\cosmetic_validation\lechuga2025_surfactant_mixture_zein.csv"
OUT_CSV = rf"{BASE}\cosmetic_validation\lechuga_smiles_cache.csv"


def fetch_smiles_by_query(query, by="name"):
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{by}/"
           f"{urllib.parse.quote(query)}/property/SMILES,IUPACName/JSON")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = _json.loads(r.read())
        prop = data["PropertyTable"]["Properties"][0]
        return prop["CID"], prop["SMILES"], prop.get("IUPACName")


def resolve(name, extra_queries=()):
    """name 그대로 시도 후, 실패하면 extra_queries(동의어 후보)를 차례로 시도."""
    queries = [name] + list(extra_queries)
    for q in queries:
        for attempt in range(3):
            try:
                cid, smiles, iupac = fetch_smiles_by_query(q)
                return dict(cid=cid, smiles=smiles, iupac=iupac, status="found", matched_query=q)
            except urllib.error.HTTPError as e:
                status = f"http_error_{e.code}"
                time.sleep(1.5 + attempt)
            except Exception as e:
                status = f"error_{type(e).__name__}"
                time.sleep(1.5 + attempt)
        # 이 query는 실패, 다음 동의어로 넘어가기 전 기록 유지
    return dict(cid=None, smiles=None, iupac=None, status=status, matched_query=None)


# 22종 INCI명 — 일부는 PubChem이 바로 못 알아듣는 표기라 가능한 동의어를 같이 시도.
# UVCB(에톡실화 분포체) 계열은 "대표 단일 호모로그"로 강제 치환하지 않고 원문 그대로만 시도 —
# 실패하면 추측 없이 미해결로 남김(팀 규칙: UVCB 추측 금지).
NAME_SYNONYMS = {
    "Capryleth-6 Carboxylic Acid": [],
    "Laureth-11 Carboxylic Acid": [],
    "Laureth-4 Carboxylic Acid": [],
    "Polyoxyethylene(6) alkyl(C8-12) ethers": [],
    "Polyoxyethylene(11) alkyl(C12-14) ethers": [],
    "Polyoxyethylene(11) alkyl(C16-18) ethers": [],
    "Capryl Glucoside": ["Octyl glucoside", "Caprylyl glucoside"],
    "Lauryl Glucoside": ["Dodecyl glucoside"],
    "Coco Glucoside": [],
    "Glycereth-17 Cocoate": [],
    "Glycereth-7 Cocoate": [],
    "Glycereth-2 Cocoate": [],
    "Glycereth-6 Cocoate": [],
    "Lauramine Oxide": ["Dodecyldimethylamine oxide", "Lauryldimethylamine oxide"],
    "Myristamine Oxide": ["Tetradecyldimethylamine oxide"],
    "Cocamidopropylamine Oxide": ["Cocamidopropylamine N-oxide"],
    "Sodium laureth sulfate": ["Sodium lauryl ether sulfate"],
    "Sodium lauryl sulfate": ["Sodium dodecyl sulfate"],
    "Zinc Coceth Sulfate": [],
    "Zinc Coco-Sulfate: 27% - 30%": ["Zinc Coco-Sulfate"],
    "Disodium Capryloyl Glutamate: 30% - 37%": ["Disodium Capryloyl Glutamate"],
    "Disodium Lauroyl Glutamate": [],
}

df = pd.read_csv(LECHUGA_CSV)
names = sorted(set(df["product1"].dropna()) | set(df["product2"].dropna()))
print(f"총 {len(names)}종 해결 시도")

rows = []
for i, name in enumerate(names):
    syn = NAME_SYNONYMS.get(name, [])
    res = resolve(name, syn)
    res["inci_name"] = name
    rows.append(res)
    mark = "O" if res["status"] == "found" else "X"
    print(f"[{mark}] {name}  ->  {res['status']}"
          + (f" (matched: {res['matched_query']})" if res["matched_query"] and res["matched_query"] != name else ""))
    time.sleep(0.3)

out = pd.DataFrame(rows)[["inci_name", "status", "matched_query", "cid", "smiles", "iupac"]]
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

n_found = (out["status"] == "found").sum()
print()
print(f"해결: {n_found}/{len(out)}종")
print(out["status"].value_counts())
