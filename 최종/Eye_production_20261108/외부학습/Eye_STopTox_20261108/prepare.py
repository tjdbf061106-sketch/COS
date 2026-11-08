"""Verify provenance, screen identities globally, train a frozen external teacher.

No EPA outcome enters teacher fitting or feature construction. Screening includes
all production rows (also train-only augmentation) and the entire local ICE file.
Source-study independence cannot be proved from Table S4's four columns.
"""
from common import *
import re, time, importlib.metadata, platform
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize as std
from rdkit.Chem import rdFingerprintGenerator as fg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score
import joblib

RDLogger.DisableLog('rdApp.*')
GEN = fg.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)

@lru_cache(maxsize=None)
def identity(s):
    if not isinstance(s,str) or not s.strip():
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    raw = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    parent = std.FragmentParent(std.Cleanup(mol))
    parent = std.Uncharger().uncharge(parent)
    Chem.RemoveStereochemistry(parent)
    parent = std.TautomerEnumerator().Canonicalize(parent)
    smi = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    ik = Chem.MolToInchiKey(mol)
    return {'canonical':raw, 'parent':smi, 'connectivity':ik.split('-')[0]}

def fps(smiles):
    return np.stack([GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])

def teacher():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=6)

def main():
    start = time.time()
    assert sha(E.DATA) == EXPECTED
    config, old_id = old_config()
    df, X, cols = E.load_eye()
    assert len(df)==673 and df.evaluation_role.value_counts().to_dict()=={
        'existing_evaluation_pool':606, 'augmentation_train_only':67}
    assert config['features']['baseline']==cols
    assert config['groups']==E.merged_groups(df,X).tolist()
    assert config['splits']=={str(s):P.production_splits(df,np.array(config['groups']),s) for s in E.SEEDS}
    for name,h in config['code_hashes'].items():
        assert sha(CODE/name)==h, ('production code changed',name)
    assert sha(u.__file__)==config['epa_code_sha256']
    for pkg,ver in config['environment']['packages'].items():
        assert importlib.metadata.version(pkg)==ver
    checks=read(SEARCH/'artifact_checksums.json')
    for name,h in checks.items():
        assert sha(SEARCH/name)==h, name
    # Archive a hash manifest, not copies or modifications of the production tree.
    protected = [E.DATA, ROOT/'production데이터/skin_vectorized.csv',
        CODE.parent/'eye_all_experiments_v2.ipynb', Path(u.__file__)]
    protected += list(CODE.glob('*.py')) + [p for p in OLD.rglob('*') if p.is_file()]
    protected += [p for p in SEARCH.iterdir() if p.is_file()]
    manifest={str(p):sha(p) for p in protected}
    if (HERE/'protected_hashes_before.json').exists():
        assert read(HERE/'protected_hashes_before.json')==manifest
    else:
        write(HERE/'protected_hashes_before.json', manifest)
    comp_path=E.COMPONENTS
    aug_path=ROOT/'준비/테스트/outputs/eye_component_resolution.csv'
    extra_path=ROOT/'준비/테스트/outputs/v4_integration/component_audit.csv'
    comp=pd.read_csv(comp_path,low_memory=False)
    aug=pd.read_csv(aug_path,low_memory=False)
    extra=pd.read_csv(extra_path,low_memory=False)
    local=pd.read_excel('C:/Users/DS/COS/eye_irritation.xlsx',sheet_name='Data')
    st=pd.read_csv(SEARCH/'stoptx_eye_training_table_s4.csv')
    workbook=pd.read_excel(SEARCH/'stoptx_training_all.xlsx',sheet_name='Excel Table S4',header=1)
    pd.testing.assert_frame_equal(st,workbook,check_dtype=False)
    assert len(st)==3545 and st.Call.value_counts().to_dict()=={0:2400,1:1145}
    component_rows=[]
    for i,r in df.iterrows():
        if r.evaluation_role=='existing_evaluation_pool':
            source=comp[comp.form_key.eq(r.source_formulation_id)]
            entries=[dict(smiles=c.canonical_smiles, cas=str(c.casrn),
                pct=float(c.pct_active_ingredient), origin=str(comp_path),
                record_id=str(c.record_id)) for c in source.itertuples()]
        else:
            selected=extra if r.source_formulation_id.startswith('v4:') else aug
            source=selected[selected['후보키'].eq(r.source_formulation_id) & selected.used.eq(True) & selected.endpoint.eq('eye')]
            entries=[dict(smiles=c['canonical_smiles'],cas=str(c.get('resolved_cas',c['CAS_원문'])),
                pct=float(c['parsed_exact_pct']),origin=str(extra_path if selected is extra else aug_path),
                record_id=str(c['component_row'])) for _,c in source.iterrows()]
        assert len(entries)==int(r.n_components_used), (i,r.formulation_id,len(entries),r.n_components_used)
        amounts=np.array([e['pct'] for e in entries])
        if np.isfinite(amounts).all() and (amounts>0).all():
            weights=amounts/amounts.sum()
        else:
            # Unit component weight is not an invented concentration.
            assert r.weight_mode=='equal_fallback_missing_pct' and len(entries)==1
            weights=np.ones(1)
        vectors=fps([e['smiles'] for e in entries])
        np.testing.assert_allclose(weights@vectors,X[i,:2048],rtol=1e-6,atol=1e-7,
            err_msg='Component linkage must reproduce original Morgan input')
        for entry,w in zip(entries,weights):
            key=identity(entry['smiles']); assert key is not None
            component_rows.append(dict(row_index=i,formulation_id=r.formulation_id,
                evaluation_role=r.evaluation_role,weight=float(w),**entry,**key))
    components=pd.DataFrame(component_rows)
    csv(HERE/'production_component_linkage.csv',components)
    # Screen all known ingredients, including unrepresented CAS in partial recipes.
    production_cas=set(re.findall(r'\b\d{2,7}-\d{2}-\d\b',
        ' '.join(df[['component_casrns','identified_casrns','composition_key']].fillna('').astype(str).to_numpy().ravel())))
    local_cas=set(local.CASRN.dropna().astype(str))
    # Include all local ICE structure strings and existing original component provenance;
    # extra candidates widen the exclusion universe without using their labels.
    identity_sources=[('production',components.smiles),('local_ice',local.SMILES),
        ('component_provenance',comp.canonical_smiles),('augmentation_audit',aug.canonical_smiles),
        ('v4_audit',extra.canonical_smiles)]
    universe=[]
    for origin,smiles in identity_sources:
        for s in smiles.dropna().unique():
            key=identity(str(s))
            universe.append(dict(origin=origin,smiles=s,parseable=key is not None,**(key or {})))
    universe=pd.DataFrame(universe)
    csv(HERE/'screening_structure_universe.csv',universe)
    all_keys={k:set(universe[k].dropna()) for k in ['canonical','parent','connectivity']}
    prod_keys={k:set(components[k]) for k in ['canonical','parent','connectivity']}
    st_keys=[]
    for i,r in st.iterrows():
        key=identity(r.SMILES); assert key is not None
        st_keys.append(dict(stoptx_row=i,**r.to_dict(),**key,
            production_cas_overlap=str(r.CASRN) in production_cas,
            local_ice_cas_overlap=str(r.CASRN) in local_cas,
            **{f'production_{k}_overlap':key[k] in prod_keys[k] for k in prod_keys},
            **{f'all_local_{k}_overlap':key[k] in all_keys[k] for k in all_keys},
            provided_inchikey_overlap=str(r.InChIKey).split('-')[0] in all_keys['connectivity']))
    audit=pd.DataFrame(st_keys)
    flags=[c for c in audit if c.endswith('_overlap')]
    audit['excluded_overlap']=audit[flags].any(axis=1)
    inventory=pd.read_csv(HERE/'sources/canada_priority_inventory.csv')
    ambiguous=inventory[inventory['Substance Category'].str.contains('UVCB|Polymer',case=False,na=False)]
    # 68955-19-1 is non-priority on the Canadian list, but explicitly UVCB in
    # OECD's Canadian categorization record. Do not encode it as a pure C15 ester.
    ambiguous_cas=set(ambiguous['CAS No.'].astype(str)) | {'68955-19-1','68391-05-9'}
    audit['excluded_known_uvcb_or_polymer']=audit.CASRN.astype(str).isin(ambiguous_cas)
    names=inventory.set_index('CAS No.')['Chemical Name'].to_dict()
    categories=inventory.set_index('CAS No.')['Substance Category'].to_dict()
    audit['inventory_name']=audit.CASRN.map(names)
    audit['inventory_substance_category']=audit.CASRN.map(categories)
    audit.loc[audit.CASRN.eq('68955-19-1'),'inventory_name']='Sulfuric acid, mono-C12-18-alkyl esters, sodium salts'
    audit.loc[audit.CASRN.eq('68955-19-1'),'inventory_substance_category']='UVCB-organic-metal salt (OECD Canadian Categorization Results)'
    # Extra conservative representation screen also removes distinct homologues
    # whose binary radius-2 fingerprints collide. No target labels are consulted.
    local_fp_hashes=set()
    for s in universe.parent.dropna().unique():
        mol=Chem.MolFromSmiles(s)
        if mol is not None:
            local_fp_hashes.add(hashlib.sha256(GEN.GetFingerprintAsNumPy(mol).tobytes()).hexdigest())
    def fpkey(s):
        mol=Chem.MolFromSmiles(s)
        return hashlib.sha256(GEN.GetFingerprintAsNumPy(mol).tobytes()).hexdigest() if mol is not None else None
    audit['morgan_identity']=audit.parent.map(fpkey)
    audit['excluded_identical_local_morgan']=audit.morgan_identity.isin(local_fp_hashes)
    # Stereo/tautomer/salt collapses are handled conservatively. Remove entire
    # conflicting parents; keep one row for concordant duplicates.
    conflicting=set(audit.groupby('parent').Call.nunique().loc[lambda x:x>1].index)
    audit['excluded_conflicting_parent']=audit.parent.isin(conflicting)
    audit['excluded_invalid_standardized_parent']=audit.parent.map(lambda s:Chem.MolFromSmiles(s) is None)
    valid=(~audit.excluded_overlap & ~audit.excluded_conflicting_parent & ~audit.excluded_invalid_standardized_parent
        & ~audit.excluded_known_uvcb_or_polymer & ~audit.excluded_identical_local_morgan)
    audit['excluded_duplicate_parent']=False
    duplicates=audit[valid].index[audit[valid].duplicated('parent',keep='first')]
    audit.loc[duplicates,'excluded_duplicate_parent']=True
    audit['aux_training_allowed']=valid & ~audit.excluded_duplicate_parent
    audit['independent_external_evaluation_allowed']=False
    csv(HERE/'stoptx_identity_audit.csv',audit)
    train=audit[audit.aux_training_allowed].copy()
    assert not set(train.parent)&all_keys['parent']
    csv(HERE/'stoptx_aux_training.csv',train)
    # Confirm the misleading 'external' set is quarantined, with record overlap.
    external=pd.read_csv(SEARCH/'stoptx_eye_external_observed_only.csv')
    old_ice=pd.read_excel(SEARCH/'eye_irritation_external.xlsx',sheet_name='Data')
    write(HERE/'data_audit.json', {
        'production_rows':len(df),'evaluation_rows':606,'augmentation_train_only':67,
        'component_linked_rows':int(components.row_index.nunique()),'common_evaluation_rows':606,
        'evaluation_excluded_rows':0,'component_fp_reconstruction_passed':True,
        'stoptx_input_rows':len(st),'exclusion_flag_counts_nonexclusive':{f:int(audit[f].sum()) for f in flags},
        'excluded_overlap_union':int(audit.excluded_overlap.sum()),
        'conflicting_parent_rows':int(audit.excluded_conflicting_parent.sum()),
        'concordant_parent_duplicates_removed':len(duplicates),
        'invalid_standardized_parent_rows':int(audit.excluded_invalid_standardized_parent.sum()),
        'known_uvcb_or_polymer_rows_nonexclusive':int(audit.excluded_known_uvcb_or_polymer.sum()),
        'identical_local_morgan_rows_nonexclusive':int(audit.excluded_identical_local_morgan.sum()),
        'excluded_union_all_reasons':int((~audit.aux_training_allowed).sum()),
        'uvcb_screen_scope':'4353 Canada prioritized CAS inventory plus explicit OECD Canada record 68955-19-1. Absence from this inventory does NOT certify a pure discrete substance.',
        'uvcb_inventory_sha256':sha(HERE/'sources/canada_priority_inventory.csv'),
        'training_rows':len(train),'training_label_counts':train.Call.value_counts().to_dict(),
        'unparseable_screening_structures':int((~universe.parseable).sum()),
        'stoptx_external_observed_rows':len(external),
        'stoptx_external_cas_in_local_ice':int(external.CASRN.astype(str).isin(local_cas).sum()),
        'stoptx_external_marked_training':int(external['Included in StopTox Training Set?'].eq(1).sum()),
        'old_ice_unique_record_ids':int(old_ice.Record_ID.nunique()),
        'old_ice_record_ids_in_local':len(set(old_ice.Record_ID)&set(local.Record_ID)),
        'study_linkage_status':'Table S4 has CASRN, SMILES, InChIKey, Call only; no original study identifiers. Study-level independence is NOT proven. Global CAS/structure exclusion is a conservative surrogate, not proof.',
        'synonym_handling':'Known CAS plus canonical, neutral parent/no-stereo/tautomer, connectivity identity across local structures. Unresolved aliases/UVCBs cannot be fully matched.',
        'production_evaluation_source_counts':df[df.evaluation_role.eq('existing_evaluation_pool')].source.value_counts().to_dict(),
        'source_files_sha256':{str(p):sha(p) for p in [comp_path,aug_path,extra_path,Path('C:/Users/DS/COS/eye_irritation.xlsx')]},
        'rdkit':rdBase.rdkitVersion,
    })
    protocol={
        'baseline_identity':old_id,'models':MODELS,'seeds':E.SEEDS,
        'arms':{'control':'unchanged production concentration 2269 features; audited saved predictions',
            'stoptx_aux':'same 2269 plus four frozen external teacher outputs'},
        'teacher':{'type':'new RandomForestClassifier; no existing weights reused',
            'params':teacher().get_params(),'representation':'2048-bit Morgan radius 2 on standardized nonstereo neutral tautomer parent',
            'target':'STopTox GHS 1 through 2B irritant/corrosive vs not classified; never converted to EPA',
            'tuning':'none; fixed before EPA results; trained only on allowed external observations'},
        'pooling':AUX_NAMES,'pooling_weights':'original represented ingredient percentages normalized within disclosed represented ingredients; nine original single-component missing-pct rows keep unit component weight; concentration stays missing with fold-fitted median; no invented water',
        'screening':'global CAS/canonical/neutral-parent/tautomer/connectivity exclusion against all 673 production rows, whole local ICE and ingredient provenance; additionally known UVCB/polymer CAS from government inventory and binary Morgan equality to any local structure, even distinct homologues; stricter than per-fold exclusion',
        'evaluation':'exact original 606 rows, groups and saved 5x3 splits, train-only 67 augmentation with original blocking',
        'tuning':'original full 20-candidate grids, mean inner AUC; no search over teacher/pooling',
        'threshold':'each arm independently uses the SAME original EPA selection rule on original-row inner OOF; numerical cutoffs need not be identical',
        'aggregation':'pool all 606 outer held-out predictions per seed with fold-specific thresholds; report paired MCC deltas and seed sample SD',
        'limitations':['partial production formulations remain partial','unresolved original-study linkage','UVCB registry coverage is incomplete; remaining structures are source-provided representations, not certified pure chemical identities','three downstream model families, not all six','teacher random seed fixed; three seeds vary EPA splits only'],
        'independent_cosmetic_performance':'not yet established; no incompatible source labels are converted',
    }
    write(HERE/'protocol.json',protocol)
    print('AUDIT',len(train),train.Call.value_counts().to_dict(),flush=True)
    Xt=fps(train.parent.tolist()); yt=train.Call.to_numpy(int)
    # Diagnostic only: source-corpus CV is not a new external EPA benchmark.
    oof=np.full(len(train),np.nan)
    for fold,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(Xt,yt,train.morgan_identity)):
        t=teacher(); t.fit(Xt[tr],yt[tr]); oof[va]=t.predict_proba(Xt[va])[:,1]
        print('TEACHER_CV',fold,flush=True)
    csv(HERE/'teacher_source_internal_oof.csv',pd.DataFrame({'stoptx_row':train.stoptx_row,'Call':yt,'score':oof}))
    write(HERE/'teacher_source_internal_metrics.json',{'role':'source internal diagnostic only; NOT independent external EPA/cosmetics validation',
        'n':len(yt),'groups':'identical binary Morgan fingerprints','auc':roc_auc_score(yt,oof),'mcc_at_0_5':matthews_corrcoef(yt,oof>=0.5),
        'balanced_accuracy_at_0_5':balanced_accuracy_score(yt,oof>=0.5)})
    model=teacher();model.fit(Xt,yt)
    joblib.dump(model,HERE/'stoptx_teacher.joblib',compress=3)
    unique=components.parent.drop_duplicates().tolist()
    scores=dict(zip(unique,model.predict_proba(fps(unique))[:,1]))
    components['stoptx_score']=components.parent.map(scores)
    csv(HERE/'production_component_scores.csv',components)
    block=np.full((len(df),4),np.nan)
    for i,g in components.groupby('row_index'):
        s=g.stoptx_score.to_numpy(float); w=g.weight.to_numpy(float);mean=float(w@s)
        block[i]=[mean,float(s.max()),float(s.min()),float(np.sqrt(w@((s-mean)**2)))]
    assert np.isfinite(block).all()
    np.save(HERE/'aux_features.npy',block)
    out=df[['formulation_id','evaluation_role']].copy();out[AUX_NAMES]=block
    csv(HERE/'aux_features.csv',out)
    write(HERE/'prepared.json',{'status':'complete','elapsed_seconds':time.time()-start,
        'teacher_sha256':sha(HERE/'stoptx_teacher.joblib'), 'aux_sha256':sha(HERE/'aux_features.npy'),
        'protocol_sha256':sha(HERE/'protocol.json'),'data_audit_sha256':sha(HERE/'data_audit.json')})
    print('PREPARED',time.time()-start,flush=True)

if __name__=='__main__':
    main()
