# import similarity
from src.eval.similarity.methods_fidelity import stats, kolmogorov_smirnov, differences_variable_correlations, wasserstein_distance_joint
from src.eval.similarity.methods_temporal_fidelity import difference_timestep_distributions, similarities_long_short_term_correlations, difference_inter_timestep_distances
# import utility
# from src.eval.utility

from src.eval.privacy.white_box_mia import mia_whitebox_attack
from src.eval.privacy.black_box_mia import mia_blackbox_attack
from src.eval.privacy.direct_matching import calculate_direct_matches
from src.eval.privacy.reidentification_risk import reidentification_risk
from src.eval.privacy.aia_attack import perform_aia

# import diversity
from src.eval.diversity.methods_diversity import calculate_diversity_scores

import argparse

def evaluate_fidelity(test_data, syn_data, columns):
    results = {}
    
    statistics = stats(test_data, syn_data, columns)
    correlations = differences_variable_correlations(test_data, syn_data, columns)
    distance = wasserstein_distance_joint(test_data, syn_data, columns)
    results.update(statistics)
    results.update(correlations)
    results.update(distance)

    return results

def evaluate_temporal_fidelity(test_data, syn_data, columns):
    results = {}
    
    avg_diff_tsdis = difference_timestep_distributions(test_data, syn_data, columns)
    avg_diff_inter_ts_distances = difference_inter_timestep_distances(test_data, syn_data, columns)
    long_short_simcors = similarities_long_short_term_correlations(test_data, syn_data, columns)
    results.update(avg_diff_tsdis)
    results.update(avg_diff_inter_ts_distances)
    results.update(long_short_simcors)

temporal_fidelity_methods = [
    difference_timestep_distributions,
    difference_inter_timestep_distances,
    similarities_long_short_term_correlations
]

utility_methods = [
    
]

privacy_methods = [
    mia_whitebox_attack,
    mia_blackbox_attack,
    reidentification_risk,
    calculate_direct_matches,
    perform_aia
]

diversity_methods = [
    calculate_diversity_scores
]

