# import similarity
from src.eval.similarity.methods_similarity import stats, kolmogorov_smirnov,differences_variable_correlations,wasserstein_distance,wasserstein_distance_timesteps, differences_timestep_correlations, differences_timestep_distances

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

similarity_methods = [
    stats,
    kolmogorov_smirnov,
    differences_variable_correlations,
    wasserstein_distance,
    wasserstein_distance_timesteps,
    differences_timestep_distances,
    differences_timestep_correlations
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

all_methods = similarity_methods + privacy_methods + utility_methods + diversity_methods

def parse_exp1_arguments():
    parser = argparse.ArgumentParser()
    # Setup experiment
    parser.add_argument('--dataset', type=str, default='cf')

    # Privacy arguments
    parser.add_argument('--k', type=int, help="Number of neighbors in knn", default=1)
    parser.add_argument('--aia_threshold', type=float, help="", default=0.8)
    parser.add_argument('--mia_threshold', type=float, help="", default=0.8)
    parser.add_argument('--reid_threshold', type=float, help="", default=0.8)
    parser.add_argument('--matching_threshold', type=float, help="", default=0.8)
    parser.add_argument('--num_disclosed_attributes', type=int, help="Number of disclosed attributes in attribute inference attack", default=3)
    parser.add_argument('--epochs', type=int, help="number of epochs to train shadow models for mia attacks", default=50)
    
    # Diversity arguments
    parser.add_argument('--n_components', type=int, help="Number of componenents in pca", default=10)
    parser.add_argument('--n_neighbors', type=int, help="Number of neighbors in knn", default=5)
    parser.add_argument('--reshape_method', type=str, help="How to reshape the data?", choices=['sequences', 'events'], default="sequences")

    method_args = vars(parser.parse_args())
    return parser.parse_args(), method_args