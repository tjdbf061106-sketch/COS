from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
PROJECT = STAGE.parent
MAIN = PROJECT / '3. modeling'
VEC = PROJECT / '2. vectorized/최종'
PREP = PROJECT / '16. cosmetics_transfer_preparation/tables'
ARCHIVE = PROJECT / '시행착오/2026-10-31_18단계_수정전'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save(name, frame):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def modules():
    sys.path.insert(0, str(PROJECT / '5. ablation'))
    import run_ablation as ab
    import analyze_ablation as aa
    return ab, aa, ab.base


def inputs():
    import pandas as pd
    _, _, base = modules()
    _, data, features, _ = base.read_inputs(VEC)
    folds = pd.read_csv(MAIN / 'splits/outer_folds.csv').set_index('formulation_id')
    data['outer_fold'] = folds.loc[data.formulation_id, 'outer_fold'].to_numpy(int)
    assert data.groupby('split_group_id').outer_fold.nunique().max() == 1
    return data, features


def freeze():
    """Lock learning inputs and code before fitting; report code may evolve separately."""
    paths = [VEC / f for f in ['eye_vectorized.csv', 'eye_vectorized_multionly.csv', 'formulation_meta_labels.csv',
                               'model_feature_columns.json']]
    paths += [MAIN / 'modeling.py', MAIN / 'splits/outer_folds.csv',
              PROJECT / '5. ablation/run_ablation.py', PROJECT / '5. ablation/analyze_ablation.py']
    paths += sorted((MAIN / 'splits').glob('all_binary_fold*_inner.csv'))
    paths += [ROOT / f for f in ['protocol.json', 'common_v2.py', 'cohort_v2.py', 'train_v2.py']]
    paths += [ROOT / 'tables/cohort_gate_794.csv']
    paths += [STAGE / 'tables/training_cohort_gate_794.csv',
              PROJECT / '15. input_remediation/tables/input_review_33.csv']
    lock = {'sha256': {p.relative_to(PROJECT).as_posix(): sha(p) for p in paths},
            'protocol': read_json(ROOT / 'protocol.json')}
    target = ROOT / 'training_lock.json'
    if target.exists():
        assert read_json(target) == lock, 'Training inputs/code changed: use a new version, never reuse old caches.'
    else:
        dump('training_lock.json', lock)
    return sha(target)
