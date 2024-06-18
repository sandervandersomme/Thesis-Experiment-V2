import pandas as pd
import torch

from src.eval.similarity.methods_fidelity import stats, wasserstein_distance, avg_diff_correlations
from src.eval.similarity.methods_temporal_fidelity import avg_diff_ts_distributions, avg_diff_inter_ts_distances, avg_similarity_longshortterm_correlations
from src.eval.diversity.methods_diversity import calculate_diversity_scores
# from src.eval.utility.methods_utility import 
from src.eval.privacy.methods_privacy import calculate_direct_matches, perform_aia, mia_blackbox_attack, mia_whitebox_attack, reidentification_risk

class Evaluator:
    def __init__(self):
        self.results = pd.DataFrame()
        self.syndata = None

    def evaluate(self, syndata: torch.Tensor, dataset_name: str, model_name: str, model_id: int, syndata_id: int):
        self.syndata = syndata
        self.model_info = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "model_id": model_id,
            "syndata_id": syndata_id
        }
        self.model_scores = {}

        self.evaluate_fidelity()    
        self.evaluate_temporal_fidelity()
        self.evaluate_utility()
        self.evaluate_diversity()
        self.evaluate_privacy()    

        # Update results

    def evaluate_fidelity(self):
        self.model_scores.update(
            **stats,
            **avg_diff_correlations,
            **wasserstein_distance
        )

    def evaluate_temporal_fidelity(self):
        self.model_scores.update(
            **avg_diff_ts_distributions,
            **avg_diff_inter_ts_distances,
            **avg_similarity_longshortterm_correlations
        )

    def evaluate_utility(self):
        pass

    def evaluate_diversity(self):
        self.model_scores.update(
            **calculate_diversity_scores
        )

    def evaluate_privacy(self):
        self.model_scores.update(
            **mia_blackbox_attack,
            **mia_whitebox_attack,
            **calculate_direct_matches,
            **reidentification_risk,
            **perform_aia
        )
