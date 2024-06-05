# import similarity
from src.eval.similarity.methods_similarity import stats, kolmogorov_smirnov,differences_variable_correlations,wasserstein_distance,wasserstein_distance_timesteps, differences_timestep_correlations, differences_timestep_distances

# import utility
# from src.eval.utility

from src.eval.privacy.mia_attack import mia_whitebox_attack

# import diversity
from src.eval.diversity.methods_diversity import calculate_diversity_scores

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
    mia_whitebox_attack
]

diversity_methods = [
    calculate_diversity_scores
]
