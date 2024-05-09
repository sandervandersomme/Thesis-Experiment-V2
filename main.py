from experiment import Experiment

if __name__ == "__main__":
    models = ["RGAN", "RWGAN"]
    datasets = ["cf", "cf_subset_no_time"]

    exp = Experiment(models, datasets, 1)
    exp.run()