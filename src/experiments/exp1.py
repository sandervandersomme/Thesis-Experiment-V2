from src.experiments.base_experiment import Classification_Experiment
from src.utilities.eval.metric import Similarity

from models.RGAN import RGAN

if __name__ == "__main__":
    exp1 = Classification_Experiment(models=[RGAN], 
                                     metrics=[Similarity()],
                                     dataset='cf',
                                     exp_id=1)

    exp1.run(num_instances=3,
             num_datasets=3,
             train_size=0.75)
    