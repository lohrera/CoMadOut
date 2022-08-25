
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

from src.oab.data.load_dataset import load_dataset
#from src.oab.data.load_dataset import _uncompress_files, load_plain_dataset, load_preprocessed_dataset
#from src.oab.data.unsupervised import UnsupervisedAnomalyDataset
from src.oab.data.classification_dataset import ClassificationDataset
from scipy.io import loadmat
import random
import os

def load_local_mat(datasets_folder, datasetname, add_rand_cols=0, verbose=False):

    folderpath = f"{datasets_folder}/{datasetname}"
    os.makedirs(folderpath, exist_ok=True)

    filepath = f"{folderpath}/{datasetname}.mat"
    filename_in_folder = f"{folderpath}/{datasetname}.csv"
    
    if not os.path.exists(filename_in_folder):
        if datasetname == 'testdata':
            folderpath = f"./src/utils"
            filename_in_folder = f"{folderpath}/{datasetname}.csv"
        else:
            try:
                file = h5py.File(filepath,'r')
            except Exception as e:
                file_content = loadmat(filepath)

            X = file_content['X']
            y = file_content['y']
            all = np.hstack((X, y)) 
            np.savetxt(filename_in_folder, all, delimiter=",")

    df = pd.read_csv(filename_in_folder)    
    vals, labels = df.iloc[:, :-1], df.iloc[:, -1]
    
    # add random noise columns    
    values = vals.values    
    for _i in range(0,add_rand_cols):
        _3d = np.zeros((vals.shape[0],1))
        for i in range(0,vals.shape[0]):
            _3d[i] = random.random() * (vals.values[:,0].max() - vals.values[:,0].min())
        values = np.concatenate((values, _3d), axis=1)
        vals = pd.DataFrame(values)
        
    if verbose:
        print(f"X.shape {vals.shape} y.shape {labels.shape}")
        cols = ['r' if lbl == 1. else 'g' for lbl in labels]
        vals.plot.scatter(x=0, y=1, c=cols)
        plt.show()
    
    return ClassificationDataset(vals.values, labels.values, name=datasetname)


def get_dataset(dataset_descr) -> np.array:
    try:
        dataset = load_dataset(dataset_descr) # dataset_descr (name) as string
    except Exception as e:
        try:
            dataset = load_plain_dataset(dataset_folder='.', dataset_dict=dataset_descr) # dataset_descr (descriptor) as dict
            #dataset = load_preprocessed_dataset(dataset_folder='.', dataset_dict=dataset_descr)
            #dataset = _uncompress_files(...)
        except Exception as e:
            dataset = load_local_mat('./datasets', dataset_descr)
        
        #**dictionary['anomaly_dataset']['arguments']
        arguments = {'normal_labels': [0], 'anomaly_labels': [1]}        
        dataset = UnsupervisedAnomalyDataset(dataset, **arguments) 

    return dataset

def set_seeds(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
