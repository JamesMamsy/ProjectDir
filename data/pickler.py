import os
import glob 
import pandas as pd
import pickle

def pickle(filetype, outfile, folder = None):
    if(folder):
        path = folder
    else:
        path = os.getcwd()

    files = glob.glob(os.path.join(path, filetype))

    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)
    df.to_pickle(outfile)
