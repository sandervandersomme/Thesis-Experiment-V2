# Thesis-Experiment-V2


# How to use:

## Dependencies
Running this code requires the following packages to be installed:
- numpy
- tensorflow
- scikit-learn

This code is written in python3.11!

## Datasets
Because of privacy issues, our Cystic Fibrosis and Sepsis datasets cannot be shared outside of the UMCU.
Our script does however provide a function to generate a dummy dataset with similar shape and structure.

## Running the script
Easiest way to run the code (from the terminal/Command prompt):
```
git clone https://github.com/sandervandersomme/Thesis-Experiment-V2
cd Thesis-Experiment-V2
python .\main.py
```

## Arguments
The main.py file can be run with different configurations. You can add the following arguments to the ```python .\main.py``` command:
- Add '--data' followed by 'cf', 'dummy' or 'sepsis' to a specific dataset.
- Add '--model' followed by 'RGAN' to use a specific model.
- Add '--device' followed by 'cpu', 'mps' or 'cuda' to select your device used for processing your data. If left out, the script automatically selects cuda, mps or cpu based on your system.
- Add '--seed' followed by an integer to enable reproducible results.

- Add '--epochs' followed by an integer. This sets the number of epochs used for training your models.
- Add '--lr' followed by a float. This sets the learning rate for training your model.
- Add '--hidden_dim' followed by an integer. This sets the number of dimensions in your hidden layer.

### Use your own data
If you want to use your own data:
1. Convert your data to a 3-dimensional numpy array. (Our sequence data has three dimensions: patients-ids, events per patient, and variables)
2. Save your numpy array as a numpy file
3. Add your numpy file to the '/data/' directory
4. Add the name of your numpy file (without extension) to the 'choices' of the dataset argument in the parse_args() function in utils.py

### Use your own model
if you want to use your own model:
1. Add the script containing your model to the root directory
2. Import your model at the top of the model.py file
3. Add the name of model script (without extension) to the 'choices' of the model argument in the parse_args() function in utils.py
4. Include your model in the if-else statement in the init_model() function in model.py

# Files in this repository