"""Binary audit summaries that keep abstentions and unknown references distinct."""
from collections import Counter


def conjunction(values):
    values = list(values)
    if not values or any(type(v) is not bool and v is not None for v in values):
        raise ValueError('Expected nonempty bool/null values')
    if False in values:
        return False
    return None if None in values else True


def confusion(pairs):
    c = Counter(total=0, TP=0, FP=0, TN=0, FN=0, reference_unknown=0,
                prediction_unknown=0, reference_known=0, both_known=0,
                reference_positive=0, reference_negative=0,
                abstained_positive=0, abstained_negative=0, predicted_positive_reference_unknown=0)
    for pred, ref in pairs:
        if any(type(v) is not bool and v is not None for v in (pred, ref)):
            raise ValueError('Labels must be bool/null, never coerced')
        c['total'] += 1
        c['prediction_unknown'] += pred is None
        if ref is None:
            c['reference_unknown'] += 1
            c['predicted_positive_reference_unknown'] += pred is True
            continue
        c['reference_known'] += 1
        c['reference_positive' if ref else 'reference_negative'] += 1
        if pred is None:
            c['abstained_positive' if ref else 'abstained_negative'] += 1
            continue
        c['both_known'] += 1
        c[('T' if pred == ref else 'F') + ('P' if pred else 'N')] += 1
    def ratio(n, d):
        return n / d if d else None
    return {**c, 'coverage_on_reference_known': ratio(c['both_known'], c['reference_known']),
            'precision_known': ratio(c['TP'], c['TP']+c['FP']),
            'recall_both_known': ratio(c['TP'], c['TP']+c['FN']),
            'recall_including_abstentions': ratio(c['TP'], c['reference_positive']),
            'false_positive_rate_including_abstentions': ratio(c['FP'], c['reference_negative'])}
