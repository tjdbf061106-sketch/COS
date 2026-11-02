import numpy as np
import pandas as pd
from common_v2 import ROOT, STAGE, PROJECT, inputs, dump, save

RETRACTED = 'Yamaguchi et al. 2016'


def conflict_flags(frame):
    """Compute both endpoint definitions, after removing the retracted source."""
    keep = ~frame.source.eq(RETRACTED)
    retained = frame[keep]
    n_binary = retained.groupby('comp_group_id').binary_label.nunique()
    n_epa = retained.groupby('comp_group_id').final_label.nunique()
    binary = frame.comp_group_id.map(n_binary).fillna(0).gt(1) & keep
    multiclass = frame.comp_group_id.map(n_epa).fillna(0).gt(1) & keep
    return keep, binary, multiclass


def main():
    data, features = inputs()
    keep, binary, multiclass = conflict_flags(data)
    # Independent exact-feature conflicts use the actual numeric feature matrix.
    hashes = pd.util.hash_pandas_object(data[features], index=False)
    exact = pd.DataFrame({'hash': hashes[keep], 'y': data.loc[keep, 'binary_label']})
    bad_hashes = set(exact.groupby('hash').y.nunique().loc[lambda x: x > 1].index)
    exact_flag = hashes.isin(bad_hashes) & keep
    conflict = binary | exact_flag
    columns = ['formulation_id', 'source', 'substance_type', 'comp_count', 'final_label',
               'binary_label', 'split_group_id', 'comp_group_id', 'outer_fold']
    gate = data[columns].copy()
    gate['excluded_retracted'] = ~keep
    gate['binary_composition_conflict'] = binary
    gate['binary_exact_input_conflict'] = exact_flag
    gate['multiclass_composition_disagreement'] = multiclass
    gate['multiclass_only_disagreement'] = multiclass & ~conflict
    gate['in_baseline699'] = keep
    gate['in_exclusion693'] = keep & ~conflict
    gate['training_policy'] = np.select([~keep, conflict], ['excluded_retracted', 'baseline_only_conflict'], default='both')
    old = pd.read_csv(STAGE / 'tables/training_cohort_gate_794.csv').set_index('formulation_id')
    gate['restored_from_old679'] = gate.in_exclusion693 & ~old.loc[gate.formulation_id, 'in_primary_cohort'].to_numpy(bool)
    # Conditions and uncertain quantities remain visible; no ingredient imputation is introduced here.
    review = pd.read_csv(PROJECT / '15. input_remediation/tables/input_review_33.csv').set_index('formulation_id')
    gate['stage15_full_recipe_hold'] = gate.formulation_id.isin(review.index)
    gate['stage15_issue'] = gate.formulation_id.map(review.issue)
    for field in ['undeclared_pct', 'test_concentration', 'has_missing_pct', 'structure_coverage_pct']:
        gate[field] = data[field]
    assert int(gate.in_baseline699.sum()) == 699
    assert int(gate.in_exclusion693.sum()) == 693
    assert int(gate.restored_from_old679.sum()) == 14
    save('tables/cohort_gate_794.csv', gate)
    save('tables/binary_conflicts_6.csv', gate[conflict])
    save('tables/restored_binary_consistent_14.csv', gate[gate.restored_from_old679])
    def profile(mask):
        d = gate[mask]
        return dict(n=len(d), positives=int(d.binary_label.sum()), negatives=int((1-d.binary_label).sum()),
                    split_groups=d.split_group_id.nunique(), sources=d.source.value_counts().to_dict())
    summary = dict(baseline699=profile(gate.in_baseline699), exclusion693=profile(gate.in_exclusion693),
                   binary_conflict_rows=int(conflict.sum()), binary_conflict_groups=gate.loc[conflict, 'comp_group_id'].nunique(),
                   restored_rows=14, conflict_pairs_already_grouped=True,
                   multiclass_only_is_exclusion=False, label_error_claim=False,
                   purpose='partial_observation_binary_baseline_and_conflict_exclusion_sensitivity',
                   original_labels_modified=False, full_recipe_certification=False)
    dump('cohort_summary.json', summary)
    print('COHORT: retained699 / exclude693 / restored14', flush=True)


if __name__ == '__main__':
    main()
