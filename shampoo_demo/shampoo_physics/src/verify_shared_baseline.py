"""Read-only baseline verification for this nested GitHub share snapshot."""
from pathlib import Path
import argparse
import verify

ROOT = Path(__file__).resolve().parents[1]


def find_repository():
    for parent in ROOT.parents:
        if (parent / 'skin irritation' / 'skin_vectorized.csv').is_file() and (parent / 'eye irritation' / 'eye_vectorized.csv').is_file():
            return parent
    raise FileNotFoundError('The skin and eye source folders are required; clone the complete COS repository.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Also refit the first fold of each baseline model')
    args = parser.parse_args()
    verify.REPO = find_repository()
    verify.write_json = lambda *args, **kwargs: None
    verify.run(args.refit)
