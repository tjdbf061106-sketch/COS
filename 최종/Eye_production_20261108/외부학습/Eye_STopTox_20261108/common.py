"""Isolated STopTox transfer experiment; existing artifacts are read-only."""
from pathlib import Path
import sys, os, json, hashlib

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = Path('C:/Users/DS/COS/최종')
SEARCH = Path('C:/Users/DS/COS/eye_external_search_20261108T163250')
OLD = CODE / 'results/concentration_production/f91f3c88787bec6b'
sys.path.insert(0, str(CODE))
import experiment_utils as E
import run_concentration_production as P
u = E.u
EXPECTED = '811268d4813fcdd68e85458641bf5823239cae04e1fa871766c77e04f0221e7f'
MODELS = ['lr', 'svm', 'lgbm']
AUX_NAMES = ['stoptx_weighted_mean', 'stoptx_max', 'stoptx_min', 'stoptx_weighted_sd']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=u._plain)

def csv(p, df):
    df.to_csv(p, index=False, encoding='utf-8-sig', mode='x')

def old_config():
    record = read(OLD / 'config.json')
    return u.load_record(OLD / 'config.json', record['identity']), record['identity']
