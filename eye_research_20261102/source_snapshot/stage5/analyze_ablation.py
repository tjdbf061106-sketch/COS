"""Paired cluster uncertainty, source-controlled summaries, figures and Korean report."""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_ablation import ROOT, SEED, VIEW_PREFIXES, assert_originals_unchanged, dump, save

VIEWS = list(VIEW_PREFIXES)
LABELS = {
    'composition': '조성·정보량',
    'composition_descriptors': '+ 분자 기술자',
    'composition_descriptors_fingerprints': '+ 구조 지문',
    'full_with_aux': '+ 구조 보조',
}
CONTRASTS = [
    ('composition', 'composition_descriptors', 'add_descriptors'),
    ('composition_descriptors', 'composition_descriptors_fingerprints', 'add_fingerprints'),
    ('composition', 'composition_descriptors_fingerprints', 'add_descriptors_and_fingerprints'),
    ('composition_descriptors_fingerprints', 'full_with_aux', 'add_structure_aux'),
]
METRICS = ['roc_auc', 'average_precision', 'within_source_pair_auc',
           'balanced_accuracy', 'sensitivity', 'specificity', 'f1']
N_BOOT = 2000


def safe_divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b != 0)


def ranking_metrics(y, score, weights):
    """Exact weighted AUC/AP for every weight row, including tied scores."""
    order = np.argsort(-score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
    tp = np.cumsum(weights[:, order] * y[order], axis=1)[:, ends]
    fp = np.cumsum(weights[:, order] * (1 - y[order]), axis=1)[:, ends]
    zero = np.zeros((len(weights), 1))
    prev_tp, prev_fp = np.c_[zero, tp[:, :-1]], np.c_[zero, fp[:, :-1]]
    npos, nneg = tp[:, -1], fp[:, -1]
    auc = safe_divide(((tp + prev_tp) * (fp - prev_fp)).sum(axis=1), 2 * npos * nneg)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    ap = safe_divide(((tp - prev_tp) * precision).sum(axis=1), npos)
    return auc, ap, npos * nneg


def weighted_metrics(frame, weights, pred_column='y_pred_0_5'):
    y = frame.y_true.to_numpy(int)
    score = frame.score_1.to_numpy(float)
    pred = frame[pred_column].to_numpy(int)
    auc, ap, _ = ranking_metrics(y, score, weights)
    tp = weights @ ((y == 1) & (pred == 1))
    fn = weights @ ((y == 1) & (pred == 0))
    tn = weights @ ((y == 0) & (pred == 0))
    fp = weights @ ((y == 0) & (pred == 1))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    within_num = np.zeros(len(weights))
    within_den = np.zeros(len(weights))
    for source in frame.source.unique():
        mask = frame.source.eq(source).to_numpy()
        part_auc, _, pairs = ranking_metrics(y[mask], score[mask], weights[:, mask])
        within_num += np.nan_to_num(part_auc) * pairs
        within_den += pairs
    return dict(roc_auc=auc, average_precision=ap,
                within_source_pair_auc=safe_divide(within_num, within_den),
                balanced_accuracy=(sensitivity + specificity) / 2,
                sensitivity=sensitivity, specificity=specificity,
                f1=np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0))


def bootstrap_weights(frame, repeats=N_BOOT):
    groups, inverse = np.unique(frame.split_group_id.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(SEED + 5100)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repeats)
    return counts[:, inverse].astype(float)


def subsets(frame, scope):
    result = {'all_evaluable': frame}
    if scope == 'all':
        result['mixture_FIFRA'] = frame[frame.substance_type.eq('Mixture')]
        result['multionly'] = frame[frame.comp_count.gt(1)]
        result['source_ICCVAM'] = frame[frame.source.eq('ICCVAM 2006')]
        result['source_Yamaguchi'] = frame[frame.source.str.startswith('Yamaguchi')]
    return result


def ci(values):
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(valid, [.025, .975])
    return float(lo), float(hi), len(valid)


def analyze_predictions(oof, out):
    summaries, intervals, differences, fold_rows = [], [], [], []
    for (scope, model), group in oof.groupby(['scope', 'model'], sort=False):
        for subset, selected in subsets(group, scope).items():
            frames = {view: selected[selected.view.eq(view)].sort_values('formulation_id').reset_index(drop=True)
                      for view in VIEWS}
            reference = frames[VIEWS[0]]
            assert len(reference) > 0
            for frame in frames.values():
                assert frame.formulation_id.tolist() == reference.formulation_id.tolist()
                for col in ['y_true', 'split_group_id', 'outer_fold', 'source']:
                    assert np.array_equal(frame[col].to_numpy(), reference[col].to_numpy())
            weights = np.vstack([np.ones(len(reference)), bootstrap_weights(reference)])
            for policy, pred_col in [('fixed_0.5', 'y_pred_0_5'), ('inner_BA', 'y_pred_inner_BA')]:
                values = {view: weighted_metrics(frame, weights, pred_col) for view, frame in frames.items()}
                keys = dict(scope=scope, model=model, subset=subset, threshold_policy=policy,
                            n=len(reference), n_groups=reference.split_group_id.nunique(),
                            n_positive=int(reference.y_true.sum()))
                for view, results in values.items():
                    summaries.append({**keys, 'view': view, 'n_features': int(frames[view].n_features.iloc[0]),
                                      **{metric: vals[0] for metric, vals in results.items()}})
                    for metric, vals in results.items():
                        lo, hi, valid = ci(vals[1:])
                        intervals.append({**keys, 'view': view, 'metric': metric, 'value': vals[0],
                                          'ci_low': lo, 'ci_high': hi, 'n_bootstrap': N_BOOT, 'n_valid': valid})
                for before, after, contrast in CONTRASTS:
                    for metric in METRICS:
                        delta = values[after][metric] - values[before][metric]
                        lo, hi, valid = ci(delta[1:])
                        differences.append({**keys, 'contrast': contrast, 'before': before, 'after': after,
                                            'metric': metric, 'before_value': values[before][metric][0],
                                            'after_value': values[after][metric][0], 'difference': delta[0],
                                            'ci_low': lo, 'ci_high': hi, 'n_bootstrap': N_BOOT, 'n_valid': valid})
            print('SUMMARIZED', scope, model, subset, len(reference), flush=True)
        for (view, fold), frame in group.groupby(['view', 'outer_fold']):
            for policy, pred_col in [('fixed_0.5', 'y_pred_0_5'), ('inner_BA', 'y_pred_inner_BA')]:
                point = weighted_metrics(frame, np.ones((1, len(frame))), pred_col)
                fold_rows.append(dict(scope=scope, model=model, view=view, outer_fold=fold,
                                      threshold_policy=policy, n=len(frame),
                                      **{key: value[0] for key, value in point.items()}))
    tables = {
        'metrics_summary': pd.DataFrame(summaries),
        'confidence_intervals': pd.DataFrame(intervals),
        'paired_differences': pd.DataFrame(differences),
        'fold_metrics': pd.DataFrame(fold_rows),
    }
    for name, frame in tables.items():
        save(frame, out / 'tables' / f'{name}.csv')
    return tables


def figures(tables, out):
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'figure.dpi': 120, 'savefig.dpi': 180})
    intervals = tables['confidence_intervals']
    panels = [('all', 'all_evaluable', '전체 학습·평가 (794건)'),
              ('all', 'mixture_FIFRA', '전체 학습 → FIFRA 혼합물 (606건)'),
              ('multionly', 'all_evaluable', '다성분 학습·평가 (235건)')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, (scope, subset, title) in zip(axes, panels):
        rows = intervals[(intervals.scope == scope) & (intervals.subset == subset)
                         & (intervals.model == 'rf_leaf5') & (intervals.metric == 'roc_auc')
                         & (intervals.threshold_policy == 'fixed_0.5')].set_index('view').loc[VIEWS]
        y = np.arange(len(VIEWS))
        ax.hlines(y, rows.ci_low, rows.ci_high, color='#246494', linewidth=2)
        ax.plot(rows.value, y, 'o', color='#246494')
        for j, value in enumerate(rows.value):
            ax.annotate(f'{value:.3f}', (value, j), xytext=(0, 9), textcoords='offset points', ha='center')
        ax.set(yticks=y, yticklabels=[LABELS[v] for v in VIEWS], xlim=(.35, .95), xlabel='ROC-AUC', title=title)
        ax.invert_yaxis()
        ax.set_ylim(len(VIEWS) - .5, -.5)
        ax.axvline(.5, color='#999', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('고정 Random Forest: 피처 추가에 따른 평가 성능과 그룹 bootstrap 95% 구간')
    for ext in ['png', 'pdf']:
        fig.savefig(out / 'figures' / f'feature_ablation_auc.{ext}')
    plt.close(fig)
    diffs = tables['paired_differences']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    names = ['add_descriptors', 'add_fingerprints', 'add_descriptors_and_fingerprints']
    labels = ['기술자 추가', '기술자에 지문 추가', '기술자+지문 추가']
    for ax, (scope, subset, title) in zip(axes, panels):
        for offset, model, color, name in [(-.12, 'rf_leaf5', '#246494', 'Random Forest'),
                                          (.12, 'lr_C0.1', '#bd6715', '로지스틱 회귀')]:
            rows = diffs[(diffs.scope == scope) & (diffs.subset == subset) & (diffs.model == model)
                         & (diffs.metric == 'roc_auc') & (diffs.threshold_policy == 'fixed_0.5')]
            rows = rows.set_index('contrast').loc[names]
            y = np.arange(3) + offset
            ax.hlines(y, rows.ci_low, rows.ci_high, color=color, linewidth=2)
            ax.plot(rows.difference, y, 'o', color=color, label=name)
        ax.set(yticks=np.arange(3), yticklabels=labels, xlabel='ROC-AUC 차이 (추가 후 - 전)', title=title)
        ax.invert_yaxis()
        ax.axvline(0, color='#999', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('동일 평가 표본에서의 짝지은 피처 효과: 95% 근사구간')
    for ext in ['png', 'pdf']:
        fig.savefig(out / 'figures' / f'paired_auc_differences.{ext}')
    plt.close(fig)


def interpret(row):
    if row.ci_low > 0:
        return '증가 방향이며 근사구간이 0을 넘음'
    if row.ci_high < 0:
        return '감소 방향이며 근사구간이 0보다 작음'
    return '근사구간이 0을 포함해 개선 방향을 확정하기 어려움'


def reports(tables, out):
    summary, diffs = tables['metrics_summary'], tables['paired_differences']
    s = summary[summary.threshold_policy.eq('fixed_0.5')]
    d = diffs[diffs.threshold_policy.eq('fixed_0.5')]
    key_rows = {}
    for key, scope, subset, contrast in [
        ('all_structure', 'all', 'all_evaluable', 'add_descriptors_and_fingerprints'),
        ('mixture_structure', 'all', 'mixture_FIFRA', 'add_descriptors_and_fingerprints'),
        ('all_fingerprint', 'all', 'all_evaluable', 'add_fingerprints'),
        ('multi_structure', 'multionly', 'all_evaluable', 'add_descriptors_and_fingerprints')]:
        key_rows[key] = d[(d.scope == scope) & (d.subset == subset) & (d.model == 'rf_leaf5')
                          & (d.metric == 'roc_auc') & (d.contrast == contrast)].iloc[0]
    takeaways = []
    for key, label in [('all_structure', '전체 RF에서 기술자·지문 추가'),
                       ('mixture_structure', '같은 FIFRA 혼합물 안에서 기술자·지문 추가'),
                       ('all_fingerprint', '전체 RF에서 기술자에 지문을 추가한 추가 이득'),
                       ('multi_structure', '다성분 전용 RF에서 기술자·지문 추가')]:
        row = key_rows[key]
        takeaways.append(f'{label}: AUC {row.before_value:.3f} → {row.after_value:.3f}, '
                         f'차이 {row.difference:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]. {interpret(row)}.')
    lines = ['분자 구조 피처 절제 실험', '',
             'EPA I-II 대 III-IV 이분류. 기존 5개 그룹 평가 폴드와 내부 3개 폴드 재사용.',
             'RF: 200 trees, leaf=5, max_features=sqrt. 보조 비교: Logistic C=0.1.',
             '모델 설정은 피처 간 고정. 모델 선택을 반복한 기존 nested_selected와 다른 실험.',
             '주 비교는 ROC-AUC/AP와 고정 임계값 0.5. 내부 BA 임계값은 보조 분석.', '',
             '핵심 결과 (고정 RF; 대괄호는 그룹 bootstrap 95% 근사구간)',
             *takeaways,
             '보조 로지스틱 회귀에서는 피처 추가 효과의 크기와 방향이 달라지므로 모델 독립적 효과로 일반화하지 않는다.', '']
    for scope, subset, label in [('all', 'all_evaluable', '전체 학습·평가'),
                                ('all', 'mixture_FIFRA', '전체 학습 → FIFRA 혼합물 평가'),
                                ('all', 'multionly', '전체 학습 → 다성분 평가'),
                                ('multionly', 'all_evaluable', '다성분 학습·평가')]:
        lines.append(label)
        for model in ['rf_leaf5', 'lr_C0.1']:
            rows = s[(s.scope == scope) & (s.subset == subset) & (s.model == model)].set_index('view')
            lines.append(f'  {model}: ' + ' / '.join(f'{LABELS[v]} AUC {rows.loc[v, "roc_auc"]:.3f}, AP {rows.loc[v, "average_precision"]:.3f}' for v in VIEWS))
            for contrast in ['add_descriptors_and_fingerprints', 'add_fingerprints']:
                for metric in ['roc_auc', 'average_precision']:
                    row = d[(d.scope == scope) & (d.subset == subset) & (d.model == model)
                            & (d.contrast == contrast) & (d.metric == metric)].iloc[0]
                    lines.append(f'    {contrast}, {metric}: {row.difference:+.3f} '
                                 f'(95% 근사구간 {row.ci_low:+.3f} ~ {row.ci_high:+.3f}); {interpret(row)}')
        lines.append('')
    lines += [
        '해석 범위',
        '- 조성 12개에는 구조 확보율·누락 정보가 포함됨. 구조 정보가 전혀 없는 기준선이 아님.',
        '- 전체 AUC는 출처 간 구분의 영향을 받음. FIFRA 및 출처별 표, within_source_pair_auc를 함께 볼 것.',
        '- 같은 출처 안의 비교도 출처·물질 유형 효과를 완전히 분리하지 못함.',
        '- RF의 sqrt 정책은 고정했으나 피처 수에 따라 노드당 후보 피처 수가 변함. 모델·표현에 조건부인 효과임.',
        '- 2,000회 그룹 bootstrap은 고정 OOF 예측의 근사구간. 재학습·분할 불확실성 및 다중비교 보정 미포함.',
        '- 이미 확인한 자료와 분할을 재사용한 사후 탐색. 독립 외부 검증 또는 화장품 성능으로 해석하지 않음.',
        '- 기존 모델과 보고서는 보존했으며, 배포 모델을 교체하지 않았음.',
    ]
    (out / 'RESULTS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report_metrics = s[((s.scope == 'all') & s.subset.isin(['all_evaluable', 'mixture_FIFRA', 'multionly']))
                       | ((s.scope == 'multionly') & s.subset.eq('all_evaluable'))]
    report_diffs = d[(d.metric.isin(['roc_auc', 'average_precision']))
                     & d.contrast.isin(['add_descriptors_and_fingerprints', 'add_fingerprints'])
                     & (((d.scope == 'all') & d.subset.isin(['all_evaluable', 'mixture_FIFRA']))
                        | ((d.scope == 'multionly') & d.subset.eq('all_evaluable')))]
    def table(frame):
        return frame.to_html(index=False, float_format=lambda value: f'{value:.3f}', border=0)
    def embedded(name):
        data = base64.b64encode((out / 'figures' / name).read_bytes()).decode('ascii')
        return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{data}">'
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>구조 피처 절제 실험</title><style>
body{{font-family:"Malgun Gothic",sans-serif;max-width:1250px;margin:35px auto;padding:0 24px;color:#202c39;line-height:1.65}}
h1,h2{{color:#1e5278}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}
td,th{{padding:7px;border-bottom:1px solid #d8e1e8;text-align:right}}th{{background:#eef4f8}}
img{{width:100%;height:auto}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;font:13px/1.7 "Malgun Gothic",sans-serif;background:#f5f8fa;padding:18px}}
</style><h1>분자 구조 피처가 추가로 제공하는 예측 정보</h1>
<p>EPA I–II 대 III–IV 이분류 · 기존 그룹 분할을 고정한 사후 절제 실험</p>
<h2>핵심 결과</h2><ul>{''.join('<li>' + html.escape(line) + '</li>' for line in takeaways)}</ul>
<p>아래 결과는 고정 Random Forest의 피처 비교다. 보조 로지스틱 회귀에서 효과의 크기와 방향이
달라지므로 모델 독립적 기여분으로 일반화하지 않는다. 구간은 고정 OOF 예측에 대한 탐색적 근사구간이다.</p>
<h2>비교 설계</h2><p>조성·정보량 12개 → 분자 기술자 포함 229개 → 지문 포함 2,277개 → 구조 보조 포함 2,280개.
Random Forest와 로지스틱 회귀의 설정을 각각 고정했고, 전처리는 훈련 자료 안에서만 수행했다.
전체 학습 794건과 다성분 학습 235건을 구분한다. 다성분 235건은 전체 자료의 부분집합이다.</p>
<p>주 비교는 임계값에 의존하지 않는 ROC-AUC와 AP이다. 분류 지표는 공통 임계값 0.5로 비교했다.
내부 3폴드에서만 임계값을 정한 결과도 CSV에 제공한다. 이전 보고서의 알고리즘 선택 절차와는 다른 고정 모델 실험이다.</p>
<h2>성능과 짝지은 차이</h2>{embedded('feature_ablation_auc.png')}{embedded('paired_auc_differences.png')}
<h2>주요 성능</h2><div class="scroll">{table(report_metrics[['scope','subset','model','view','n','roc_auc','average_precision','within_source_pair_auc','balanced_accuracy','sensitivity','specificity']])}</div>
<h2>동일 표본에서의 변화와 95% 근사구간</h2><div class="scroll">{table(report_diffs[['scope','subset','model','contrast','metric','difference','ci_low','ci_high']])}</div>
<h2>수치 해석</h2><pre>{html.escape(chr(10).join(lines))}</pre>
<h2>파일</h2><p><a href="tables/paired_differences.csv">짝지은 차이</a> ·
<a href="tables/metrics_summary.csv">전체 성능표</a> · <a href="tables/fold_metrics.csv">폴드별 성능</a> ·
<a href="protocol.json">실험 규약</a> · <a href="checks/verification.json">검증 결과</a></p>
</html>'''
    (out / 'report.html').write_text(body, encoding='utf-8')
    dump(out / 'checks' / 'analysis_complete.json', dict(status='PASSED', bootstrap_repeats=N_BOOT,
         n_summary_rows=len(summary), n_difference_rows=len(diffs), tables_generated_from_oof=True))


def run(out):
    oof = pd.read_csv(out / 'predictions' / 'oof_predictions.csv')
    tables = analyze_predictions(oof, out)
    figures(tables, out)
    reports(tables, out)
    assert_originals_unchanged(json.loads((out / 'original_manifest.json').read_text(encoding='utf-8')))
    print('ANALYSIS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    run(parser.parse_args().output_dir.resolve())
