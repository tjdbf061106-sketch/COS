"""Reproducible formulation review and outcome-free feature enrichment."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "shampoo_physics_20261102"
TYPES = ["anionic", "non-ionic", "amphoteric", "amino-acid", "cationic"]
POLYMERS = ["Luviquat Excellence", "Dehyquart CC6", "Dehyquart CC7 Benz", "Salcare Super 7"]
THICKENERS = ["Arlypon F", "Arlypon TT"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitize(value):
    if isinstance(value, dict): return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [sanitize(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def slug(value): return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def snapshot_inputs():
    mapping = {
        "data/formulations.csv": "formulations_baseline.csv",
        "results/observed_starting_formulations.csv": "selected44_baseline.csv",
        "data/feature_names.json": "stock_feature_names.json",
        "data/ingredient_registry.csv": "ingredient_registry_baseline.csv",
        "sources/LiquidFormulationsDataset_2023.json": "raw_shampoo.json",
        "sources/surfactants_information.csv": "raw_surfactants.csv",
        "sources/ingredient_structures.pptx": "ingredient_structures.pptx",
        "results/outer_folds.csv": "outer_folds_baseline.csv",
    }
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    records = []
    for original, name in mapping.items():
        source, dest = BASE / original, ROOT / "inputs" / name
        if not dest.exists(): shutil.copyfile(source, dest)
        assert digest(source) == digest(dest), f"Input changed: {source}"
        records.append({"baseline_path": original, "snapshot": "inputs/"+name, "sha256": digest(dest)})
    baseline_hashes = {p.relative_to(BASE).as_posix(): digest(p) for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    provenance = ROOT / "input_provenance.json"
    if provenance.exists():
        assert read_json(provenance)["baseline_files"] == baseline_hashes, "Baseline was modified"
    write_json(provenance, {"records": records, "baseline_files": baseline_hashes,
                            "github_commit_reviewed": read_json(ROOT / "review_protocol.json")["reviewed_github_commit"]})


def curate_registry():
    registry = pd.read_csv(ROOT / "inputs/ingredient_registry_baseline.csv", keep_default_na=True)
    raw = pd.read_csv(ROOT / "inputs/raw_surfactants.csv")
    registry["supplier_as_study"] = "BASF"
    registry["experimental_lot_id"] = "not_reported"
    registry["historical_batch_spec_confirmed"] = False
    registry["source_inci"] = registry.inci
    registry["reviewed_inci"] = registry.inci
    registry["inci_review_status"] = "source_preserved"
    registry.loc[registry.ingredient_id == "Dehyquart A-CA", ["reviewed_inci", "inci_review_status"]] = ["Cetrimonium Chloride", "supplier_alias_supported_source_name_retained"]
    registry["source_active_range_basis"] = np.where(registry.role.eq("surfactant"), "source CSV active_matter column; not batch assay", "not_reported")
    registry["catalog_active_min_pct"] = np.nan
    registry["catalog_active_max_pct"] = np.nan
    registry["catalog_range_relation"] = "not_compared"
    refs = {"Dehyton PK 45": (37,41), "Dehyton AB 30": (29,33), "Dehyton MC": (30.5,33.5)}
    for name, (lo, hi) in refs.items():
        mask = registry.ingredient_id.eq(name)
        registry.loc[mask, ["catalog_active_min_pct", "catalog_active_max_pct"]] = [lo, hi]
        a = registry.loc[mask].iloc[0]
        relation = "disjoint_ranges" if hi < a.active_min_pct or lo > a.active_max_pct else "overlap_not_identical"
        registry.loc[mask, "catalog_range_relation"] = relation
    registry["study_mw_reference_g_mol"] = np.nan
    registry["study_mw_range_min_g_mol"] = np.nan
    registry["study_mw_range_max_g_mol"] = np.nan
    registry["study_mw_status"] = "not_used_for_core_features"
    for name, mass in [("Luviquat Excellence",40000),("Dehyquart CC7 Benz",500000)]:
        registry.loc[registry.ingredient_id.eq(name), ["study_mw_reference_g_mol", "study_mw_status"]] = [mass,"approximate_reference_not_exact_batch_MW"]
    registry.loc[registry.ingredient_id.eq("Salcare Super 7"), ["study_mw_range_min_g_mol", "study_mw_range_max_g_mol", "study_mw_status"]] = [100000,200000,"reported_range"]
    registry.loc[registry.ingredient_id.eq("Dehyquart CC6"), "study_mw_status"] = "unknown"
    registry["alkyl_length_source_mean"] = np.nan
    for name in registry.loc[registry.role.eq("surfactant"), "ingredient_id"]:
        match = raw[raw.trade_name.map(slug).eq(slug(name))]
        assert len(match) == 1
        registry.loc[registry.ingredient_id.eq(name), "alkyl_length_source_mean"] = float(match.iloc[0]["(CH2)x"])
    registry["supplier_solids_lower_pct"] = np.nan
    registry["supplier_solids_upper_pct"] = np.nan
    registry.loc[registry.ingredient_id.eq("Luviquat Excellence"), ["supplier_solids_lower_pct", "supplier_solids_upper_pct"]] = [38,42]
    registry["supplier_context_note"] = "Historical study batch equivalence unconfirmed"
    registry.loc[registry.ingredient_id.eq("Luviquat Excellence"), "supplier_context_note"] = "BASF May 2012 p4: solids 38-42%; charge density 6.1 meq/g at pH7; do not equate solids with pure polymer or transfer pH7 charge to pH5.8"
    write_csv(ROOT / "data/ingredient_registry_reviewed.csv", registry)
    return registry


def build_features(stocks, registry, supplier_pk45=False):
    """Reads ONLY stock columns and external ingredient attributes; no fitted statistics."""
    registry = registry.set_index("ingredient_id")
    columns = registry.index.tolist()
    w = stocks[columns].astype(float).copy()
    if not np.isfinite(w.to_numpy()).all() or (w<0).any().any() or (w.sum(axis=1)>100+1e-10).any():
        raise ValueError("Invalid commercial-stock percentages")
    surf = registry[registry.role.eq("surfactant")].index.tolist()
    output = {c: w[c] for c in columns}
    dictionary = [{"feature":c,"family":"original_stock_amounts","units":"mass_percent","definition":"As-supplied commercial grade amount; source stock column preserved"} for c in columns]
    def add(name, value, family, units, definition):
        output[name] = value
        dictionary.append({"feature":name,"family":family,"units":units,"definition":definition})
    add("stock_sum_pct", w.sum(axis=1), "role_totals", "mass_percent", "Sum of the 18 commercial stocks; excludes balance water and titrant")
    add("added_water_balance_pct_estimate",100-w.sum(axis=1),"partial_water_accounting","mass_percent","100 minus commercial-stock sum, pre-titration estimate; excludes water inside stocks")
    s_total = w[surf].sum(axis=1)
    add("designated_surfactant_stock_pct",s_total,"role_totals","mass_percent","Sum of the 12 designated surfactants; thickeners not relabeled as surfactants")
    add("polymer_stock_pct",w[POLYMERS].sum(axis=1),"role_totals","mass_percent","Commercial polymer stock sum, not polymer solids")
    add("thickener_stock_pct",w[THICKENERS].sum(axis=1),"role_totals","mass_percent","Commercial thickener stock sum")
    add("designated_surfactant_count",(w[surf]>0).sum(axis=1),"surfactant_balance","count","Nonzero designated surfactant grades")
    sorted_amounts = np.sort(w[surf].to_numpy(),axis=1)
    major, minor = sorted_amounts[:,-1], sorted_amounts[:,-2]
    positive = s_total.to_numpy()>0
    add("surfactant_major_stock_share",np.divide(major,s_total,out=np.zeros(len(w)),where=positive),"surfactant_balance","fraction","Largest surfactant stock divided by designated surfactant sum; 0 with no surfactant")
    add("surfactant_pair_balance",np.divide(4*major*minor,s_total.to_numpy()**2,out=np.zeros(len(w)),where=positive),"surfactant_balance","dimensionless","4ab/(a+b)^2 when two surfactants; 0 for one/none; stock-basis balance")
    add("no_designated_surfactant",~positive,"surfactant_balance","indicator","Explicit zero denominator indicator")
    active = registry.loc[surf,["active_min_pct","active_max_pct"]].copy()
    if supplier_pk45: active.loc["Dehyton PK 45"] = [37,41]
    type_stock = {}
    for kind in TYPES:
        names = registry[(registry.role=="surfactant") & (registry.surfactant_type==kind)].index.tolist()
        key=slug(kind); type_stock[key]=w[names].sum(axis=1)
        add(f"type_{key}_stock_pct",type_stock[key],"source_type_totals","mass_percent","Source CSV surfactant type; not a measured charge state")
        for end, col in [("lower","active_min_pct"),("upper","active_max_pct")]:
            add(f"type_{key}_active_claim_{end}_pct",w[names].mul(active.loc[names,col]).sum(axis=1)/100,"source_claimed_active_ranges","mass_percent","Weighted source-claimed range endpoint; not batch-measured active concentration")
    for end,col in [("lower","active_min_pct"),("upper","active_max_pct")]:
        add(f"surfactant_active_claim_{end}_pct",w[surf].mul(active[col]).sum(axis=1)/100,"source_claimed_active_ranges","mass_percent","Marginal endpoint for designated surfactants only; no midpoint normalization")
    add("surfactant_active_claim_width_pct",output["surfactant_active_claim_upper_pct"]-output["surfactant_active_claim_lower_pct"],"source_claimed_active_ranges","percentage_points","Upper minus lower; not a probability interval")
    for end,col in [("lower","water_min_pct"),("upper","water_max_pct")]:
        add(f"water_known_partial_{end}_pct",100-w.sum(axis=1)+w[surf].mul(registry.loc[surf,col]).sum(axis=1)/100,"partial_water_accounting","mass_percent","Balance water plus reported water from designated surfactants; excludes unknown water in polymer/thickener and titrants")
    add("stock_without_water_spec_pct",w[POLYMERS+THICKENERS].sum(axis=1),"partial_water_accounting","mass_percent","Uncharacterized commercial stocks, not an estimate of missing water")
    add("declared_nacl_contribution_pct",w["Dehyton AB 30"]*0.07,"declared_additive_contributions","mass_percent","Only source-declared NaCl in AB30; not total salt or ionic strength")
    add("declared_potassium_sorbate_contribution_pct",w["Texapon SB 3 KC"]*0.005,"declared_additive_contributions","mass_percent","Only source-declared contribution; not total preservatives")
    for end,acid,diol in [("lower",0.0035,0.04),("upper",0.0045,0.07)]:
        add(f"declared_citric_acid_{end}_pct",w["Texapon SB 3 KC"]*acid+w["Plantapon LC 7"]*0.07,"declared_additive_contributions","mass_percent","Source stock contribution; excludes unknown pH-adjustment titrant")
        add(f"declared_propanediol_{end}_pct",w["Plantapon ACG 50"]*diol,"declared_additive_contributions","mass_percent","Source stock contribution only")
    add("alkyl_length_stock_weighted_source_mean",np.divide(w[surf].mul(registry.loc[surf,"alkyl_length_source_mean"]).sum(axis=1),s_total,out=np.zeros(len(w)),where=positive),"source_chain_length","source_chain_length_units","Stock-weighted published (CH2)x descriptor; not exact molecular-chain distribution")
    for name in POLYMERS+THICKENERS:
        add("grade_present_"+slug(name),(w[name]>0).astype(int),"commercial_grade_indicators","indicator","Preserve distinct commercial grades, including the two Polyquaternium-7 products")
    core=list(output)
    for a,b in itertools.combinations_with_replacement(type_stock,2):
        add(f"interaction_type_{a}__{b}",type_stock[a]*type_stock[b]/10000,"type_interactions","fraction_squared","Product of stock fractions; prespecified proxy, not measured intermolecular interaction")
    for a in POLYMERS:
        for b in THICKENERS:
            add(f"interaction_grade_{slug(a)}__{slug(b)}",w[a]*w[b]/10000,"grade_interactions","fraction_squared","Polymer-grade x thickener-grade stock fraction")
    for kind, values in type_stock.items():
        for grade in POLYMERS+THICKENERS:
            add(f"interaction_type_grade_{kind}__{slug(grade)}",values*w[grade]/10000,"type_grade_interactions","fraction_squared","Source surfactant type x commercial polymer/thickener stock fraction")
    frame=pd.DataFrame(output,index=w.index).astype(float)
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError("Non-finite features")
    return frame,pd.DataFrame(dictionary),{"stock18":columns,"physics_core":core,"physics_interactions":list(frame)}


def interp_log(x,y,target):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or (x<=0).any() or (y<=0).any(): raise ValueError("Invalid curve")
    order=np.argsort(x);x,y=x[order],y[order]
    if (np.diff(x)<=0).any():raise ValueError("Duplicate shear rate")
    if not x[0]<=target<=x[-1]:return np.nan
    return float(10**np.interp(np.log10(target),np.log10(x),np.log10(y)))


def review_44(data, registry, features):
    selected=pd.read_csv(ROOT/"inputs/selected44_baseline.csv",float_precision="round_trip").sample_id
    raw={r["ID"]:r for r in read_json(ROOT/"inputs/raw_shampoo.json")}
    frame=data[data.sample_id.isin(selected)].copy().sort_values("sample_id")
    assert len(frame)==44 and frame.sample_id.is_unique
    protocol=read_json(ROOT/"review_protocol.json");flags=protocol["descriptive_review_flags"]
    rows,curve_rows,recipe_rows=[],[],[]
    reg=registry.set_index("ingredient_id")
    stocks=read_json(ROOT/"inputs/stock_feature_names.json")
    for row in frame.to_dict("records"):
        sid=int(row["sample_id"]); curve={k:v for part in raw[sid]["Rheology_Data"] for k,v in part.items()}
        x=np.asarray(curve["shear_rate"],float);y=np.asarray(curve["avg_viscosity"],float);sd=np.asarray(curve["std_dev"],float)
        etas={rate:interp_log(x,y,rate) for rate in protocol["curve_rates_s_inv"]}
        ix=int(np.argmin(abs(np.log(x/100))))
        cv=float(sd[ix]/y[ix]);ntu=float(row["turbidity_ntu"]);err=float(row["turbidity_error_as_reported"])
        rheo={"sample_id":sid,"polymer_grade":next(n for n in POLYMERS if row[n]>0),"thickener_grade":next(n for n in THICKENERS if row[n]>0),
              "rheology_type_source":row["rheology_type"],"rheometer":row["rheometer"],"turbidity_ntu":ntu,"turbidity_error_as_reported":err,
              "turbidity_relative_error":err/ntu if ntu>0 else np.nan,"eta10_mPa_s":etas[10],"eta100_mPa_s":etas[100],"eta500_mPa_s":etas[500],
              "eta10_over_eta500":etas[10]/etas[500],"log_eta_slope_10_to_500":np.log(etas[500]/etas[10])/np.log(500/10),
              "nearest100_shear_rate":float(x[ix]),"nearest100_mean_viscosity":float(y[ix]),"nearest100_std_dev":float(sd[ix]),"nearest100_cv":cv,
              "curve_point_count":len(x),"curve_min_rate":float(x.min()),"curve_max_rate":float(x.max()),
              "upward_steps_over_5pct_fraction":float((np.diff(y)/y[:-1]>0.05).mean()),
              "viscosity_margin_fraction":etas[100]/100-1,"turbidity_margin_fraction":1-ntu/100,
              "flag_turbidity_relative_error":err/ntu>=flags["relative_turbidity_error_at_least"],
              "flag_nearest100_cv":cv>=flags["nearest_100_viscosity_cv_at_least"],
              "flag_viscosity_near_target":etas[100]<100*(1+flags["viscosity_lower_margin_fraction"]),
              "flag_turbidity_near_target":ntu>100*(1-flags["turbidity_upper_margin_fraction"]),
              "flag_turbidity_plus_reported_error_crosses_target":ntu+err>100,
              "flag_pk45_active_discrepancy":row["Dehyton PK 45"]>0,
              "flag_aca_inci_alias_review":row["Dehyquart A-CA"]>0,
              "flag_ab30_representative_charge_review":row["Dehyton AB 30"]>0,
              "flag_commercial_thickener_blend":row["Arlypon TT"]>0,
              "experimental_lot_id":"not_reported","measured_final_pH":"not_reported","target_pH":5.8,"target_pH_tolerance":0.2,"rheology_temperature_C":25,
              "phase_stability_scope":"36h post-titration plus initial assessment; not shelf-life", "polymer_thickener_group":row["polymer_thickener_group"],"ingredient_combination":row["ingredient_combination"]}
        for c in ["surfactant_active_claim_lower_pct","surfactant_active_claim_upper_pct","declared_nacl_contribution_pct","water_known_partial_lower_pct","water_known_partial_upper_pct"]:
            rheo[c]=float(features.loc[sid,c])
        rheo["stock_recipe"]="; ".join(f"{n}: {row[n]:g}%" for n in stocks if row[n]>0)
        rheo["added_water_balance_pct_estimate"]=100-sum(row[n] for n in stocks)
        rows.append(rheo)
        for a,b,c in zip(x,y,sd):curve_rows.append({"sample_id":sid,"shear_rate_s_inv":a,"mean_viscosity_mPa_s":b,"std_dev_mPa_s":c,"polymer_thickener_group":row["polymer_thickener_group"]})
        for n in stocks:
            if row[n]<=0:continue
            r=reg.loc[n]
            recipe_rows.append({"sample_id":sid,"ingredient_id":n,"role":r.role,"stock_pct":row[n],"supplier":"BASF (study)","source_inci":r.source_inci,"reviewed_inci":r.reviewed_inci,
                                "source_active_lower_pct_of_stock":r.active_min_pct,"source_active_upper_pct_of_stock":r.active_max_pct,
                                "source_active_lower_pct_of_formula":row[n]*r.active_min_pct/100,"source_active_upper_pct_of_formula":row[n]*r.active_max_pct/100,
                                "experimental_lot_id":"not_reported","catalog_range_relation":r.catalog_range_relation,"source_note":r.source_note})
    review=pd.DataFrame(rows)
    write_csv(ROOT/"review/formulations44_comparison.csv",review)
    write_csv(ROOT/"review/formulations44_stock_amounts.csv",frame[["sample_id"]+stocks+["added_water_balance_pct_estimate"]])
    write_csv(ROOT/"review/formulations44_ingredients_long.csv",pd.DataFrame(recipe_rows))
    write_csv(ROOT/"review/formulations44_flow_curves.csv",pd.DataFrame(curve_rows))
    representatives=review.sort_values(["turbidity_relative_error","nearest100_cv","sample_id"]).drop_duplicates("polymer_thickener_group")
    representatives=representatives.sort_values("polymer_thickener_group")
    write_csv(ROOT/"review/subsystem_review_representatives.csv",representatives)
    pairs=[]
    for combination,g in frame.groupby("ingredient_combination"):
        for i,j in itertools.combinations(g.index,2):
            a,b=frame.loc[i],frame.loc[j]
            changes={n:float(b[n]-a[n]) for n in stocks if abs(b[n]-a[n])>1e-12}
            pairs.append({"sample_a":int(a.sample_id),"sample_b":int(b.sample_id),"same_ingredient_combination":combination,
                          "stock_delta_b_minus_a":json.dumps(changes,ensure_ascii=False),"stock_L1_difference_pct":sum(abs(v) for v in changes.values()),
                          "eta100_ratio_b_over_a":b.viscosity_100_mPa_s/a.viscosity_100_mPa_s,"turbidity_difference_b_minus_a":b.turbidity_ntu-a.turbidity_ntu,
                          "same_rheometer":a.rheometer==b.rheometer,"interpretation":"Observed association only; not a controlled causal comparison or replicate test"})
    write_csv(ROOT/"review/same_combination_pairs.csv",pd.DataFrame(pairs))
    return review,representatives


def main():
    snapshot_inputs();registry=curate_registry()
    data=pd.read_csv(ROOT/"inputs/formulations_baseline.csv",float_precision="round_trip")
    stocks=read_json(ROOT/"inputs/stock_feature_names.json")
    features,dictionary,sets=build_features(data[stocks],registry)
    sensitivity,_,_=build_features(data[stocks],registry,supplier_pk45=True)
    features.index=data.sample_id;sensitivity.index=data.sample_id
    for name,f in [("features_source_reported_812.csv",features),("features_pk45_supplier_sensitivity_812.csv",sensitivity)]:
        write_csv(ROOT/"data"/name,f.reset_index())
    write_csv(ROOT/"data/feature_dictionary.csv",dictionary)
    write_json(ROOT/"data/feature_sets.json",sets)
    targets=data[["sample_id","stable","viscosity_100_mPa_s","log10_viscosity_100","turbidity_ntu","log10p1_turbidity","ingredient_combination","polymer_thickener_group"]].copy()
    write_csv(ROOT/"data/targets_and_groups_812.csv",targets)
    shutil.copyfile(ROOT/"inputs/outer_folds_baseline.csv",ROOT/"data/outer_folds_preserved.csv")
    delta=pd.DataFrame({"sample_id":data.sample_id,"pk45_stock_pct":data["Dehyton PK 45"],
                        "active_claim_lower_source":features.surfactant_active_claim_lower_pct.to_numpy(),"active_claim_upper_source":features.surfactant_active_claim_upper_pct.to_numpy(),
                        "active_claim_lower_supplier_scenario":sensitivity.surfactant_active_claim_lower_pct.to_numpy(),"active_claim_upper_supplier_scenario":sensitivity.surfactant_active_claim_upper_pct.to_numpy()})
    delta["lower_difference_pct"]=delta.active_claim_lower_supplier_scenario-delta.active_claim_lower_source
    delta["upper_difference_pct"]=delta.active_claim_upper_supplier_scenario-delta.active_claim_upper_source
    write_csv(ROOT/"data/pk45_sensitivity_effects_812.csv",delta)
    # Supplier-only context is deliberately separate from primary predictor sets.
    optional=pd.DataFrame({"sample_id":data.sample_id,"luviquat_solids_reference_lower_partial_pct":data["Luviquat Excellence"]*0.38,
                           "luviquat_solids_reference_upper_partial_pct":data["Luviquat Excellence"]*0.42,
                           "polymer_stock_without_confirmed_solids_spec_pct":data[POLYMERS].sum(axis=1),
                           "historical_batch_equivalence_confirmed":False})
    write_csv(ROOT/"data/supplier_context_not_core_features.csv",optional)
    review,representatives=review_44(data,registry,features)
    features_diagnostics=pd.DataFrame({"feature":features.columns,"nonzero_count":(features!=0).sum().to_numpy(),"n_unique":features.nunique().to_numpy(),
                                       "missing_count":features.isna().sum().to_numpy(),"min":features.min().to_numpy(),"max":features.max().to_numpy()})
    write_csv(ROOT/"data/feature_diagnostics_not_selection.csv",features_diagnostics)
    summary={"status":"prepared_not_retrained","review_rows":len(review),"recipe_component_rows":len(pd.read_csv(ROOT/"review/formulations44_ingredients_long.csv")),
             "represented_subsystems":review.polymer_thickener_group.nunique(),"representative_sample_ids":representatives.sample_id.tolist(),
             "rheology_source_counts":review.rheology_type_source.value_counts().to_dict(),"feature_rows":len(features),"feature_count":features.shape[1],
             "feature_set_sizes":{k:len(v) for k,v in sets.items()},"source_metadata_only_feature_construction":True,
             "review_flag_counts":{c:int(review[c].sum()) for c in review if c.startswith("flag_")},
             "pk45_affected_all_rows":int((data["Dehyton PK 45"]>0).sum()),"pk45_affected_review_rows":int(review.flag_pk45_active_discrepancy.sum()),
             "pk45_max_lower_difference_magnitude_pct":float(abs(delta.lower_difference_pct).max()),
             "pk45_max_upper_difference_magnitude_pct":float(abs(delta.upper_difference_pct).max()),
             "model_training_performed":False,"new_performance_claim":False,"skin_eye_predictions_performed":False,
             "scope_alignment":"GitHub README f049106: cosmetic irritation transfer is not performed; physical-property research is separate"}
    write_json(ROOT/"summary.json",summary)
    print(sanitize(summary))


if __name__=="__main__": main()
