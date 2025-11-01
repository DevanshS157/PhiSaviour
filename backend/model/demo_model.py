from typing import List, Any
import math


class DemoClassifier:
    """A tiny deterministic classifier that mimics a trained model's interface.

    It expects a pandas.DataFrame with columns matching features.extract_features.
    """
    def __init__(self):
        # no parameters for demo
        pass

    def predict(self, X) -> List[str]:
        preds = []
        for _, row in X.iterrows():
            score = 0.0
            score += float(row.get('suspicious_words', 0.0)) * 2.0
            score += float(row.get('num_hyphens', 0.0)) * 0.5
            score += float(row.get('num_subdomains', 0.0)) * 0.3
            score += float(row.get('has_percent_encoded', 0.0)) * 3.0
            score += (50 - min(50, float(row.get('url_length', 0.0)))) * -0.01
            preds.append('malicious' if score > 1.0 else 'benign')
        return preds

    def predict_proba(self, X) -> List[List[float]]:
        # produce a fake probability distribution [prob_benign, prob_malicious]
        out = []
        for _, row in X.iterrows():
            score = 0.0
            score += float(row.get('suspicious_words', 0.0)) * 2.0
            score += float(row.get('num_hyphens', 0.0)) * 0.5
            score += float(row.get('num_subdomains', 0.0)) * 0.3
            score += float(row.get('has_percent_encoded', 0.0)) * 3.0
            score += (50 - min(50, float(row.get('url_length', 0.0)))) * -0.01
            # apply a slightly steeper sigmoid to amplify signal
            scaled = score * 1.8
            prob_mal = 1.0 / (1.0 + math.exp(-scaled))
            # normalize into [benign, malicious]
            out.append([round(1 - prob_mal, 4), round(prob_mal, 4)])
        return out
