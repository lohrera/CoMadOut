import wget
import yaml
import os
import gzip
import shutil
import h5py

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Union
from zipfile import ZipFile
from scipy.io import loadmat

# example-pkg-jd-kiel==0.0.37 inactive on 20220815 (temporarily only required oab source imported from https://github.com/ISDM-CAU-Kiel/oab instead)
from src.oab.data.abstract_classes import AnomalyDataset
from src.oab.data.classification_dataset import ClassificationDataset
#from src.oab.data.load_image_dataset import _load_image_dataset
from src.oab.data.utils_image import image_datasets
from src.oab.data.utils import _get_dataset_dict

def load_dataset(dataset_name: str, anomaly_dataset: bool = True,
        preprocess_classification_dataset: bool = False,
        semisupervised: bool = False,
        dataset_folder: str = "datasets"):
    """ Loads an anomaly detection or classification dataset as specified by the
    parameters. Defaults to unsupervised anomaly dataset.

    :param dataset_name: Dataset to be loaded
    :param anomaly_dataset: Return an anomaly dataset (if set to False, a
        classification dataset is returned), defaults to True
    :param preprocess_classification_dataset: Indicates if the dataset should
        be preprocessed if a classification dataset is returned. Note that this
        parameter is only important when a classification dataset is returned,
        defaults to False
    :param semisupervised: Type of anomaly dataset is set to semisupervised,
        otherwise unsupervised, defaults to False
    :param dataset_folder: Folder in which a subdirectory for the loaded dataset
        will be created, defaults to "datasets"

    :return: An anomaly detection or a classification dataset
    """
    # load image datasets separately
    if dataset_name in image_datasets:
        return None
        #return _load_image_dataset(dataset_name, anomaly_dataset,
        #    preprocess_classification_dataset, semisupervised, dataset_folder)


    dataset_dict = _get_dataset_dict(dataset_name)
    print(f"Credits: {dataset_dict['credits']}")

    # download the dataset if it is not yet downloaded
    if not _target_file_exists_in_folder(folderpath=Path(dataset_folder) / dataset_dict['foldername'],
        filename=dataset_dict['filename_in_folder']):
        download_dataset(dataset_dict, dataset_folder)
    # download the yaml if it is not yet downloaded
    if not _target_file_exists_in_folder(folderpath=Path(dataset_folder) / dataset_dict['foldername'],
        filename=dataset_dict['destination_yaml']):
        _download_dataset_yaml(dataset_dict, dataset_folder)

    # perprocess dataset according to parameters
    if not anomaly_dataset and not preprocess_classification_dataset:
        return load_plain_dataset(dataset_dict, dataset_folder)
    elif not anomaly_dataset and preprocess_classification_dataset:
        return load_preprocessed_dataset(dataset_dict, dataset_folder,
            make_anomaly_dataset=False, semisupervised=semisupervised)
    else: # anomaly_dataset == True
        return load_preprocessed_dataset(dataset_dict, dataset_folder,
            make_anomaly_dataset=True, semisupervised=semisupervised)


def download_dataset(dataset_dict: Dict, dataset_folder: str = "datasets") -> None:
    """ Download the dataset specified by the corresponding dictionary into
    the specified folder and if necessary unpacks it, so that there eventually
    is a single csv file with the dataset.

    :param dataset_dict: Dictionary that includes information about the dataset,
        including a list with its urls (with key `'urls_dataset'`) and a list
        with the corresponding filenames (with key `'destination_filenames'`) as
        well as those objects needed by _uncompress_files and
        _make_one_file
    :param dataset_folder: Folder in which an extra folder for the dataset is
        made, and the dataset will be written in that second folder, defaults to
        "datasets"
    """
    dest_folder_path = Path(dataset_folder) / dataset_dict['foldername']
    try:
        os.makedirs(dest_folder_path)
    except:
        pass
    for url_dataset, dest_filename in zip(dataset_dict['urls_dataset'],
                                          dataset_dict['destination_filenames']):
        dest_path = dest_folder_path / dest_filename
        wget.download(str(url_dataset), str(dest_path))
    _uncompress_files(dataset_dict, dataset_folder)
    _make_one_file(dataset_dict, dataset_folder)
    return


def load_preprocessed_dataset(dataset_dict: Dict, dataset_folder: str = "datasets",
    make_anomaly_dataset: bool = True, semisupervised: bool = False): #-> Union[ClassificationDataset, AnomalyDataset]:
    """Loads a preprocessed classification dataset or anomaly dataset as
    specified by parameters returned in the dictionary.

    :param dataset_dict: Dictionary that includes information about the dataset
        that is being loaded. This includes information about where to retrieve
        the dataset, what kind of preprocessing is to be done, and how the dataset
        is transformed into an anomaly dataset.
    :param dataset_folder: Folder in which a subdirectory for the loaded dataset
        will be created, defaults to "datasets"
    :param make_anomaly_dataset: Indicate whether or not the dataset should
        be transformed from a classification dataset into an anomaly dataset using
        the labels specified in `dataset_dict`, defaults to True
    :param semisupervised: Indicate whether or not a semisupervised anomaly
        dataset should be created (otherwise an unsupervised anomaly dataset will
        be created), which is only important if an anomaly dataset is created,
        defaults to False
    """
    cd = load_plain_dataset(dataset_dict, dataset_folder)
    path_yaml = Path(dataset_folder) / dataset_dict['foldername'] / dataset_dict['destination_yaml']
    return cd.perform_operations_from_yaml(str(path_yaml),
        make_into_anomaly_dataset=make_anomaly_dataset,
        unsupervised=(not semisupervised), semisupervised=semisupervised)


def load_plain_dataset(dataset_dict: Dict,
    dataset_folder: str = "datasets") -> ClassificationDataset:
    """Loads a dataset as classification dataset without any preprocessing.

    :param dataset_dict: Dictionary that includes information about the dataset
        that is being loaded. This includes information about where to retrieve
        the dataset and where to store the dataset.
    :param dataset_folder: Folder in which a subdirectory for the loaded dataset
        will be created, defaults to "datasets"
    """
    path = Path(dataset_folder) / dataset_dict['foldername'] / dataset_dict['filename_in_folder']
    if 'load_csv_arguments' in dataset_dict:
        df = pd.read_csv(path, **dataset_dict['load_csv_arguments'])
    else:
        df = pd.read_csv(path)
    if dataset_dict['class_labels'] == 'last':
        values, labels = df.iloc[:, :-1], df.iloc[:, -1]
    elif dataset_dict['class_labels'] == 'first':
        values, labels = df.iloc[:, 1:], df.iloc[:, 0]
    else:
        raise NotImplementedError(f"Class labels in position {dataset_dict['class_labels']} is not implemented. Choose 'first' or 'last'.")
    return ClassificationDataset(values.values, labels.values,
        name=dataset_dict['name'])


def _target_file_exists_in_folder(folderpath: str, filename: str) -> bool:
    """ Helper to check if file already exists.

    :param folderpath: Path of the folder in which to check for the file
    :param filename: Name of the file that is to be checked
    """
    try:
        x = filename in os.listdir(folderpath)
    except FileNotFoundError:
        return False
    return x


def _download_dataset_yaml(dataset_dict: Dict, dataset_folder: str = "datasets") -> None:
    """ Helper that downloads the yaml file belonging to the dataset.

    :param dataset_dict: Dictionary that includes information about what the
        name of the dataset is (with key `'foldername'`), which url is to be used
        to retrieve the yaml (with key `'url_yaml'`) and where to store the yaml
        (with key `'destination_yaml'`)
    :param dataset_folder: Name of the folder in which dataset operations
        happen, defaults to "datasets"
    """
    url = dataset_dict['url_yaml']
    dest = Path(dataset_folder) / dataset_dict['foldername'] / dataset_dict['destination_yaml']
    wget.download(str(url), str(dest))
    return


def _uncompress_files(dataset_dict: Dict, dataset_folder: str = "datasets") -> None:
    """ Helper to uncompress files in zip or gz compression.

    :param dataset_dict: Dictionary that includes information about which files
        to unpack (with key `'destination_filenames'`) and what format they are
        compressed in (with key `'dataset_format'`) as well as what the file
        finally should be called (with key `'filename_in_folder'`)
    :param dataset_folder: Name of the folder in which dataset operations
        happen, defaults to "datasets"
    """
    folderpath = Path(dataset_folder) / dataset_dict['foldername']
    if dataset_dict['dataset_format'] == 'csv':
        return
    elif dataset_dict['dataset_format'] == 'zip':
        for file in dataset_dict['destination_filenames']:
            zippath = str(folderpath / file)
            with ZipFile(zippath) as zf:
                zf.extractall(path=folderpath)
        return
    elif dataset_dict['dataset_format'] == 'gz_single_file':
        for file in dataset_dict['destination_filenames']:
            gzpath = str(folderpath / file)
            targetpath = str(folderpath / dataset_dict['filename_in_folder'])
            with gzip.open(gzpath, 'rb') as f_in:
                with open(targetpath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return
    elif dataset_dict['dataset_format'] == 'mat':
        filepath = folderpath / dataset_dict['destination_filenames'][0]
        file = h5py.File(filepath,'r')
        X = np.array(file['X']).transpose() # dim: # observations, # features
        y = np.array(file['y']).transpose() # dim: # observations, 1
        all = np.hstack((X, y)) # labels are 'last'
        np.savetxt(folderpath / dataset_dict['filename_in_folder'], all, delimiter=",")
        return
    elif dataset_dict['dataset_format'] == 'mat_old':
        filepath = folderpath / dataset_dict['destination_filenames'][0]
        file_content = loadmat(filepath)
        X = file_content['X'] # dim: # observations, # features
        y = file_content['y'] # dim: # observations, 1
        all = np.hstack((X, y)) # labels are 'last'
        np.savetxt(folderpath / dataset_dict['filename_in_folder'], all, delimiter=",")
        return
    raise ValueError(f"Format {dataset_dict['dataset_format']} not defined!")


def _make_one_file(dataset_dict: Dict, dataset_folder: str = "datasets") -> None:
    """ Helper that concatenates multiple files in case a dataset consists of
    multiple files, e.g., if there are distinct train and test datasets.

    :param dataset_dict: Dictionary that includes information about which files
        to concatenate (with key `'filenames_to_concatenate'`, if this is `None`,
        nothing is concatenated), optionally which arguments are used to load the
        file into pandas (with key `'load_csv_arguments'`, e.g., if there is a no
        header), and which file to store the final dataset in (with key
        `'filename_in_folder'`)
    :param dataset_folder: Name of the folder in which dataset operations
        happen, defaults to "datasets"
    """
    if dataset_dict['filenames_to_concatenate'] is None:
        return
    else:
        dfs = []
        folderpath = Path(dataset_folder) / dataset_dict['foldername']
        filenames = dataset_dict['filenames_to_concatenate']
        # load dfs into list
        if 'load_csv_arguments' in dataset_dict:
            for filename in filenames:
                df = pd.read_csv(folderpath / filename, **dataset_dict['load_csv_arguments'])
                dfs.append(df)
        else: # no further arguments
            for filename in filenames:
                df = pd.read_csv(folderpath / filename)
                dfs.append(df)

        # reduce to one df
        df = dfs[0]
        for additional_df in dfs[1:]:
            df = df.append(additional_df, ignore_index=True)

        # save into file
        df.to_csv(folderpath / dataset_dict['filename_in_folder'], index=False)
