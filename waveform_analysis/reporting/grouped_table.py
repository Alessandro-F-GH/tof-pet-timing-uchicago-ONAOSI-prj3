"""Caption-free grouped comparison tables from saved metrics only."""
from pathlib import Path
import math
from waveform_analysis.reporting.latex_text import escape_latex


def export_grouped_metric_table(path, rows, led):
    """CV CTR uses fold SD; independent-test metrics use bootstrap SD."""
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    def cell(value, error):
        if not finite(value):
            return 'n/a'
        if finite(error) and error >= 0:
            return rf"${value:.1f} \pm {error:.1f}$"
        return f"${value:.1f}$"

    def line(row):
        cells = [cell(row.get(metric), row.get(metric + '_std'))
                 for metric in ('cv_ctr', 'test_ctr', 'test_rmse')]
        return escape_latex(row['code']) + ' & ' + ' & '.join(cells) + r' \\'

    ordered = sorted(rows, key=lambda r: (not finite(r.get('test_ctr')),
                     r['test_ctr'] if finite(r.get('test_ctr')) else math.inf, r['model']))
    lines = [r'\begin{table}[tbp]', r'\centering', r'\scriptsize',
             r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}lccc@{}}',
             r'\toprule', r'Model & CV CTR [ps] & Test CTR [ps] & Test RMSE [ps] \\',
             r'\midrule', line({'code': 'LED', **led}), r'\midrule']
    lines.extend(line(row) for row in ordered)
    lines.extend([r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path
