import os
import optuna
import pandas as pd

import argparse

def extract_parameters(dir):
    # Path to the folder containing the Optuna study files
    folder_path = f'outputs/{dir}/hyperparams/'
    save_path = f'outputs/{dir}/results/hyperparams'
    os.makedirs(save_path, exist_ok=True)

    # Dictionary to store the results
    all_results = pd.DataFrame()
    scores = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.db'):  # Assuming your studies are stored in SQLite databases
            # Extract the model name from the filename
            model_name = filename.split('.')[0]
            results = {"model": model_name}

            # Load the study
            study_path = f'sqlite:///{os.path.join(folder_path, filename)}'
            study = optuna.load_study(study_name=model_name, storage=study_path)
            
            # Get the best trial
            best_trial = study.best_trial
            
            results.update(best_trial.params)
            results.update({"score": best_trial.value})

            all_results = pd.concat([all_results, pd.Series(results).to_frame().T], axis=0)

    all_results = all_results.set_index("model").T

    save_path = os.path.join(save_path, f'optimal_hyperparameters')

    # Save the DataFrame to a CSV file (if needed)
    all_results.to_csv(f'{save_path}.csv', index=True)

    # Save the DataFrame to a Markdown file
    with open(f'{save_path}.md', 'w') as f:
        f.write(all_results.to_markdown(index=True))

    # Save the DataFrame to a LaTeX file
    with open(f'{save_path}.tex', 'w') as f:
        f.write(all_results.to_latex(index=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()

    extract_parameters(args.dir)

