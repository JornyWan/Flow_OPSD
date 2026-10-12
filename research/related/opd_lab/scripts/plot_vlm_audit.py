"""Plot reference-known audit decisions; abstentions stay visible in denominators."""
import argparse
from pathlib import Path

from opdlab.io import read_json, sha256, write_json


def main(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    run = Path(run)
    summary = read_json(run/'summary.json')
    if sha256(run/'per_image.json') != summary['per_image_sha256']:
        raise ValueError('Report rows changed')
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5))
    colors = ['#51936a', '#c74242', '#e7a13d', '#9ba1aa']
    labels = ['Correct', 'False positive', 'False negative', 'Abstention']
    for i, cohort in enumerate(('development', 'exposed_calibration', 'new_calibration')):
        data = summary['metrics'][cohort]
        for j, clauses in enumerate((['text'], ['count', 'color', 'relation'])):
            ax = axes[i, j]
            for y, name in enumerate(('v1', 'qwen_exact')):
                rows = [data[name][c] for c in clauses if c in data[name]]
                n = sum(r['reference_known'] for r in rows)
                values = [sum(r['TP']+r['TN'] for r in rows), sum(r['FP'] for r in rows),
                          sum(r['FN'] for r in rows), sum(r['abstained_positive']+r['abstained_negative'] for r in rows)]
                if sum(values) != n: raise ValueError('Plot denominator mismatch')
                left = 0
                for count, color in zip(values, colors):
                    ax.barh(y, count/n, left=left, height=.6, color=color)
                    if count: ax.text(left+count/n/2, y, str(count), ha='center', va='center', fontsize=9)
                    left += count/n
                ax.text(1.015, y, f'n={n}', va='center', fontsize=9)
            ax.set(yticks=[0, 1], yticklabels=['v1', 'Qwen'], xlim=(0, 1.13),
                   xticks=[0, .5, 1], xticklabels=['0%', '50%', '100%'])
            ax.invert_yaxis()
            ax.set_title(cohort.replace('_', ' ')+' / '+('OCR exact' if j == 0 else 'scene clauses'), fontsize=11)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
               loc='lower center', ncol=4, bbox_to_anchor=(.5, .04))
    fig.suptitle('Independent-model audit against provisional observations', fontsize=14)
    fig.text(.5, .015, 'Unknown references excluded; numbers are decisions, not independent prompts. No policy efficacy claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .96))
    for ext in ('png', 'pdf'):
        path = run/f'vlm_audit.{ext}'
        if path.exists(): raise FileExistsError('Use fresh report run')
        fig.savefig(path, dpi=170)
    write_json(run/'plot_provenance.json', {'summary_sha256': sha256(run/'summary.json'),
               'source_sha256': sha256(__file__), 'outputs': {ext: sha256(run/f'vlm_audit.{ext}') for ext in ('png', 'pdf')}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run', required=True)
    main(p.parse_args().run)
