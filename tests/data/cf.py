from src.data.cf_classification import CF_Classification
from src.data.cf_regression import CF_Regression
from src.data.cf import CF

import torch



if __name__ == "__main__":


    # Load cf
    cf = CF()
    print(cf.sequences.shape, cf.columns)

    # Load synthetic
    syndata = torch.load(f"outputs/testing/RGAN/syn.pt")    

    # Create cf classification real
    print("Create cf class real")
    cf_class = CF_Classification(cf.sequences, cf.columns)
    print(cf_class.columns, cf_class.sequences.shape, cf_class.labels.shape)

    # Create cf classification synthetic
    print("Create cf class synthetic")
    cf_class = CF_Classification(syndata, cf.columns)
    print(cf_class.columns, cf_class.sequences.shape, cf_class.labels.shape)

    # Create cf regression real
    print("Create cf regression real")
    cf_reg = CF_Regression(cf.sequences, cf.columns.copy())
    print(cf_reg.columns, cf_reg.sequences.shape, cf_reg.labels.shape)

    # Create cf regression synthetic
    print("Create cf regression synthetic")
    cf_reg = CF_Regression(syndata, cf.columns.copy())
    print(cf_reg.columns, cf_reg.sequences.shape, cf_reg.labels.shape)

    


    


    