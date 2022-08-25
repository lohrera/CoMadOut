
import numpy as np
import pandas as pd
import warnings
import random
import glob
import time
import math
import os
np.seterr(invalid='ignore', divide='ignore')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
import torch

from src.comadout import ComadOut
from src.cmoplus import CMOPlus
from src.utils.load_utils import load_local_mat, set_seeds

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from pyod.utils.utility import precision_n_scores 

from src.algorithms import EllipticEnv, MCD, MLE
from src.pca_mad import PCAMAD

from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE

# performances from the original PCA-MAD paper considered as given

pcamadppr = {
    'arrhytmia':0.810, 'cardio': 0.937, 'annthyroid':0.903, 'breastw':0.989, 'letter':0.659, 'thyroid': 0.982, 'mammography':0.884, 
    'pima':0.713, 'musk':1.000, 'optdigits':0.717, 'pendigits':0.883, 'mnist':0.905, 'shuttle':0.997, 'satellite':0.784, 'satimage-2':0.999, 
    'wine':0.936, 'vowels':0.838, 'glass':0.740, 'wbc':0.938
}

pcamadpprpn = {
    'arrhytmia':0.503, 'cardio': 0.579, 'annthyroid':0.520, 'breastw':0.933, 'letter': 0.150, 'thyroid':0.640, 'mammography':0.333, 
    'pima': 0.556, 'musk':1.000, 'optdigits': 0.001, 'pendigits': 0.271, 'mnist':0.535, 'shuttle':0.961, 'satellite':0.641, 'satimage-2':0.929, 
    'wine': 0.410, 'vowels': 0.241, 'glass': 0.123, 'wbc':0.562
}

pcamadr = {
    'arrhytmia':0.813, 'cardio': 0.955, 'annthyroid':0.878, 'breastw':0.990, 'letter':0.655, 'thyroid': 0.982, 'mammography':0.884, 
    'pima':0.713, 'musk':1.000, 'optdigits':0.586, 'pendigits':0.900, 'mnist':0.910, 'shuttle':0.998, 'satellite':0.783, 'satimage-2':0.999, 
    'wine': 0.949, 'vowels': 0.838, 'glass': 0.728, 'wbc':0.938
}

pcamadrpn = {
    'arrhytmia': 0.507, 'cardio': 0.649, 'annthyroid': 0.451, 'breastw':0.933, 'letter':0.154, 'thyroid': 0.595, 'mammography':0.295, 
    'pima': 0.558, 'musk':1.000, 'optdigits': 0.000, 'pendigits': 0.283, 'mnist': 0.562, 'shuttle':0.965, 'satellite':0.640, 'satimage-2': 0.932, 
    'wine': 0.480, 'vowels':0.241, 'glass':  0.123, 'wbc':0.551
}

cmp=None

def init_algos(ratio, seed):
    return {
        "CMO": ComadOut(n_components=ratio, title='CMO', softmax_scoring=False, center_by='median', random_state=seed, verbose=0, fast=False),
        #CMO+ variants and PCA-MAD are added in method run_experiments
        "HBOS": HBOS(),
        "IF": IForest(random_state=seed),
        "PCA": PCA(n_components=ratio, random_state=seed),
        "PCA(NM)": ComadOut(n_components=ratio, title='PCA(NM)', pca=True, softmax_scoring=False, center_by='mean', random_state=seed, verbose=0, fast=False),
        "LOF": LOF(algorithm='ball_tree'), 
        "KNN": KNN(algorithm='ball_tree'),
        "OCSVM": OCSVM(),
        "MCD": MCD(random_state=seed),
        "EllipticEnv": EllipticEnv(random_state=seed),
        "MLE": MLE(random_state=seed),
        "AE": AutoEncoder(random_state=seed,hidden_neurons=[4, 3, 2, 2, 3, 4], epochs=10, batch_size=4, dropout_rate=0.0, l2_regularizer=0.01, verbose=0),
        "VAE": VAE(random_state=seed,encoder_neurons=[4, 3, 2],decoder_neurons=[2, 3, 4],epochs=10, batch_size=4, dropout_rate=0.0,l2_regularizer=0.001,verbose=0),
    }

def write_result_dicts(resultdir, ts, ratio, seed, dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos):

    pd.DataFrame(dict_resultsap).to_csv(f"{resultdir}/{ts}_results_ap_{ratio}_{seed}.csv", index=False)
    pd.DataFrame(dict_resultsro).to_csv(f"{resultdir}/{ts}_results_roc_{ratio}_{seed}.csv", index=False)
    pd.DataFrame(dict_resultsrc).to_csv(f"{resultdir}/{ts}_results_recall_{ratio}_{seed}.csv", index=False)
    pd.DataFrame(dict_resultspn).to_csv(f"{resultdir}/{ts}_results_prec_n_{ratio}_{seed}.csv", index=False)
    pd.DataFrame(dict_resultst).to_csv(f"{resultdir}/{ts}_results_runtime_{ratio}_{seed}.csv", index=False)

def add_dataset(dict_dataset, folderpath, datasetname, verbose=0):
    
    data = load_local_mat(folderpath, datasetname, add_rand_cols=0, verbose=verbose)
    Xdata = RobustScaler(with_centering=True, with_scaling=True, unit_variance=True).fit_transform(data.values)  
    y = data.labels
    
    dict_dataset["dataset"].append(datasetname)
    cls_, cls_ratio = np.unique(y, return_counts=True)
    
    print(f"{datasetname}         & {Xdata.shape[0]}          & {Xdata.shape[1]}         & {cls_ratio[1]} ({round(cls_ratio[1]/cls_ratio.sum()*100, 1)}\\%)                    \\\\ \\hline")
    
    dict_dataset["total"].append(Xdata.shape[0])
    dict_dataset["normal"].append(cls_ratio[0])
    dict_dataset["outlier"].append(cls_ratio[1])
    dict_dataset["dims"].append(Xdata.shape[1])
    dict_dataset["%out"].append(round(cls_ratio[1]/cls_ratio.sum()*100, 1))
    dict_dataset["X"].append(Xdata)
    dict_dataset["y"].append(y)
        
    return dict_dataset


def run_experiments(Xdata, y, datasetname, dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos,
                  pc_ratio=0.8, seed=0, decs={'t':3,'m':9}):

    n_components=math.ceil(pc_ratio*Xdata.shape[1])
    print(f"ratio: {pc_ratio} -> n_components ({n_components}/{Xdata.shape[1]})")

    print(f"{datasetname} -> CMO+,CMO+k,CMO+e,CMO+ke,CMOEns")

    t1 = time.time()
    
    cmp = CMOPlus(n_components=n_components, title='CMO+', random_state=seed, verbose=0, fast=False)
    cmp.fit(Xdata)
    
    #cmpf = CMOPlusFast(n_components=n_components, title='CMO+', random_state=seed, verbose=0)
    #cmpf.fit(Xdata)
    
    t2 = time.time()
    
    dict_resultsro["CMO+"].append(round(roc_auc_score(y, cmp.accum_pc_outlier_scores_rawsc), decs['m']))
    dict_resultsro["CMO+k"].append(round(roc_auc_score(y, cmp.accum_pc_outlier_scores_kurts), decs['m']))
    dict_resultsro["CMO+e"].append(round(roc_auc_score(y, cmp.accum_pc_outlier_scores_evrsc), decs['m']))
    dict_resultsro["CMO+ke"].append(round(roc_auc_score(y, cmp.accum_pc_outlier_scores_kurts_evr), decs['m']))
    dict_resultsro["CMOEns"].append(round(roc_auc_score(y, cmp.accum_pc_outlier_scores_enmed), decs['m']))

    dict_resultspn["CMO+"].append(round(precision_n_scores(y, cmp.accum_pc_outlier_scores_rawsc), decs['m']))
    dict_resultspn["CMO+k"].append(round(precision_n_scores(y, cmp.accum_pc_outlier_scores_kurts), decs['m']))
    dict_resultspn["CMO+e"].append(round(precision_n_scores(y, cmp.accum_pc_outlier_scores_evrsc), decs['m']))
    dict_resultspn["CMO+ke"].append(round(precision_n_scores(y, cmp.accum_pc_outlier_scores_kurts_evr), decs['m']))
    dict_resultspn["CMOEns"].append(round(precision_n_scores(y, cmp.accum_pc_outlier_scores_enmed), decs['m']))

    dict_resultsap["CMO+"].append(round(average_precision_score(y, cmp.accum_pc_outlier_scores_rawsc), decs['m']))
    dict_resultsap["CMO+k"].append(round(average_precision_score(y, cmp.accum_pc_outlier_scores_kurts), decs['m']))
    dict_resultsap["CMO+e"].append(round(average_precision_score(y, cmp.accum_pc_outlier_scores_evrsc), decs['m']))
    dict_resultsap["CMO+ke"].append(round(average_precision_score(y, cmp.accum_pc_outlier_scores_kurts_evr), decs['m']))
    dict_resultsap["CMOEns"].append(round(average_precision_score(y, cmp.accum_pc_outlier_scores_enmed), decs['m']))

    prec1, rc1, _ = precision_recall_curve(y, cmp.accum_pc_outlier_scores_rawsc)
    prec2, rc2, _ = precision_recall_curve(y, cmp.accum_pc_outlier_scores_kurts)
    prec3, rc3, _ = precision_recall_curve(y, cmp.accum_pc_outlier_scores_evrsc)
    prec4, rc4, _ = precision_recall_curve(y, cmp.accum_pc_outlier_scores_kurts_evr)
    prec4_, rc4_, _ = precision_recall_curve(y, cmp.accum_pc_outlier_scores_enmed)

    dict_resultsrc["CMO+"].append(round(auc(rc1, prec1), decs['m']))
    dict_resultsrc["CMO+k"].append(round(auc(rc2, prec2), decs['m']))
    dict_resultsrc["CMO+e"].append(round(auc(rc3, prec3), decs['m']))
    dict_resultsrc["CMO+ke"].append(round(auc(rc4, prec4), decs['m']))
    dict_resultsrc["CMOEns"].append(round(auc(rc4_, prec4_), decs['m']))
    
    dict_resultst["CMO+"].append(round(t2-t1,decs['t']))
    dict_resultst["CMO+k"].append(round(t2-t1,decs['t']))
    dict_resultst["CMO+e"].append(round(t2-t1,decs['t']))
    dict_resultst["CMO+ke"].append(round(t2-t1,decs['t']))
    dict_resultst["CMOEns"].append(round(t2-t1,decs['t']))
    
    
    print(f"{datasetname} -> PCA-MAD++")
    
    t1=time.time()
    
    pcamad = PCAMAD(n_components=n_components, _evalvariant=0, seed=seed)
    pcamad.fit(Xdata)
    
    t2=time.time()
    
    if datasetname in list(pcamadppr.keys()):
        dict_resultsro["PCA-MAD++"].append(pcamadppr[datasetname])
        dict_resultspn["PCA-MAD++"].append(pcamadpprpn[datasetname])
    else:
        dict_resultsro["PCA-MAD++"].append(round(roc_auc_score(y, pcamad.decision_scores_),decs['m']))
        dict_resultspn["PCA-MAD++"].append(round(precision_n_scores(y, pcamad.decision_scores_),decs['m']))
        
    prec, rc, _ = precision_recall_curve(y, pcamad.decision_scores_)        
    dict_resultsap["PCA-MAD++"].append(round(average_precision_score(y, pcamad.decision_scores_),decs['m']))
    dict_resultsrc["PCA-MAD++"].append(round(auc(rc, prec),decs['m']))
    dict_resultst["PCA-MAD++"].append(round(t2-t1,decs['t']))
        
    print(f"{datasetname} -> PCA-MAD")
    
    t1=time.time()
    
    pcamad = PCAMAD(n_components=n_components, _evalvariant=1, seed=seed)
    pcamad.fit(Xdata)
    
    t2=time.time()
    
    if datasetname in list(pcamadr.keys()):
        dict_resultsro["PCA-MAD"].append(pcamadr[datasetname])
        dict_resultspn["PCA-MAD"].append(pcamadrpn[datasetname])
    else:
        dict_resultsro["PCA-MAD"].append(round(roc_auc_score(y, pcamad.decision_scores_),decs['m']))
        dict_resultspn["PCA-MAD"].append(round(precision_n_scores(y, pcamad.decision_scores_),decs['m']))
        
    prec, rc, _ = precision_recall_curve(y, pcamad.decision_scores_)
    dict_resultsap["PCA-MAD"].append(round(average_precision_score(y, pcamad.decision_scores_),decs['m']))
    dict_resultsrc["PCA-MAD"].append(round(auc(rc, prec),decs['m']))
    dict_resultst["PCA-MAD"].append(round(t2-t1,decs['t']))        
        
    for algn, clf in dict_algos.items():
        print(f"{datasetname} -> {algn}")
        t1=time.time()
        clf.fit(Xdata)
        decision_scores_ = clf.decision_function(Xdata)  
        t2=time.time()
        if algn not in list(dict_resultsro.keys()): dict_resultsro[algn] = []
        if algn not in list(dict_resultsap.keys()): dict_resultsap[algn] = []
        if algn not in list(dict_resultsrc.keys()): dict_resultsrc[algn] = []
        if algn not in list(dict_resultspn.keys()): dict_resultspn[algn] = []
        if algn not in list(dict_resultst.keys()): dict_resultst[algn] = []
        prec, rc, _ = precision_recall_curve(y, decision_scores_)
        dict_resultsro[algn].append(round(roc_auc_score(y, decision_scores_),decs['m']))
        dict_resultsap[algn].append(round(average_precision_score(y, decision_scores_),decs['m']))
        dict_resultsrc[algn].append(round(auc(rc, prec),decs['m']))
        dict_resultspn[algn].append(round(precision_n_scores(y, decision_scores_),decs['m']))
        dict_resultst[algn].append(round(t2-t1,decs['t']))
    
    return dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos


def init_run(ratio, seed, dict_datasets):
    
    dict_resultst={"seed":[], "pc":[], "dataset":[], "CMO":[],"CMO+":[],"CMO+k":[],"CMO+e":[],"CMO+ke":[], "CMOEns":[], "PCA-MAD++":[], "PCA-MAD":[]}
    dict_resultsro={"seed":[], "pc":[], "dataset":[],"CMO":[],"CMO+":[],"CMO+k":[],"CMO+e":[],"CMO+ke":[], "CMOEns":[], "PCA-MAD++":[], "PCA-MAD":[]}
    dict_resultsap={"seed":[], "pc":[], "dataset":[],"CMO":[],"CMO+":[],"CMO+k":[],"CMO+e":[],"CMO+ke":[], "CMOEns":[], "PCA-MAD++":[], "PCA-MAD":[]}
    dict_resultsrc={"seed":[], "pc":[], "dataset":[],"CMO":[],"CMO+":[],"CMO+k":[],"CMO+e":[],"CMO+ke":[], "CMOEns":[], "PCA-MAD++":[], "PCA-MAD":[]}
    dict_resultspn={"seed":[], "pc":[], "dataset":[],"CMO":[],"CMO+":[],"CMO+k":[],"CMO+e":[],"CMO+ke":[], "CMOEns":[], "PCA-MAD++":[], "PCA-MAD":[]}
    
    for datasetname in dict_datasets["dataset"]:
        dict_resultsro['dataset'].append(datasetname)
        dict_resultsrc['dataset'].append(datasetname)
        dict_resultspn['dataset'].append(datasetname)
        dict_resultsap['dataset'].append(datasetname)
        dict_resultst['dataset'].append(datasetname)
    
        dict_resultsro['pc'].append(ratio)
        dict_resultsrc['pc'].append(ratio)
        dict_resultspn['pc'].append(ratio)
        dict_resultsap['pc'].append(ratio)
        dict_resultst['pc'].append(ratio)

        dict_resultsro['seed'].append(seed)
        dict_resultsrc['seed'].append(seed)
        dict_resultspn['seed'].append(seed)
        dict_resultsap['seed'].append(seed)
        dict_resultst['seed'].append(seed)
    
    set_seeds(seed)

    dict_algos = init_algos(ratio, seed)
    
    return dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos


def df2dict(df):
    d_ = {}
    d = df.to_dict()
    for k,v in d.items(): d_[k] = list(d[k].values())
    return d_


def get_top_algo_count(dict_row, ddf, cols, decs=3, _max=True):
    dict_cnt = {i: 0 for i in range(0, len(cols))}
    for idx, row in ddf.round(decs).iterrows():
        if _max:
            bestval = row.values[np.argmax(row.values)]
        else:
            bestval = row.values[np.argmin(row.values)]
        bestcols = np.where(row.values == bestval)[0]
        for idx in bestcols: dict_cnt[idx]+=1
    for k,v in dict_cnt.items(): dict_row[cols[k]] = v
    return dict_row

def get_result_stats_df(dict_results, algocols=None, decs=3, _max=True):
    
    if algocols is None:
        dict_algos = init_algos(0.999, 0)
        algocols = ["CMO", "CMO+","CMO+k","CMO+e","CMO+ke","CMOEns","PCA-MAD++","PCA-MAD"] + list(set(list(dict_algos.keys())) - set(["CMO"]))
       
    df_results = pd.DataFrame(dict_results).round(decs)
    df_results = df_results.style.highlight_max(subset=algocols, color = 'lightgreen', axis = 1)

    d = df_results.__dict__['data'].to_dict()
    d_ = {}
    for k,v in d.items():
        d_[k] = list(d[k].values())
    d = d_
    ddf = pd.DataFrame(d)[algocols]
        
    r1={"dataset": "AVG"}
    for c in ddf.columns: r1[c] = np.array(ddf[c].mean(axis=0)).round(decs)
    
    r2 = get_top_algo_count({"dataset": "WIN"}, ddf, algocols, decs=decs, _max=_max)
    
    for k,v in d.items():
        d[k].append(r1[k])
        d[k].append(r2[k])
        
    d["AVG"] = list(ddf.mean(axis=1).round(decs)) # per dataset
    d["AVG"].extend([0.,0.])
    
    if _max:
        df_results_ext = pd.DataFrame(d).style.highlight_max(subset=algocols, color = 'lightgreen', axis = 1)
    else:
        df_results_ext = pd.DataFrame(d).style.highlight_min(subset=algocols, color = 'lightgreen', axis = 1)
        
    return df_results_ext

def get_average_results(resultdir, ts_run, metric, verbose=0):

    df=None
    df_cols=None
    avg_results=None
    datasets=None
    dict_results={}

    runs=0
    for file_path in glob.glob(f"{resultdir}/{ts_run}*_results_{metric}_*"):
        if verbose: print(file_path)
        runs+=1
        df = pd.read_csv(file_path)
        df_cols = df.columns[3:]

        if avg_results is None:
            avg_results = df.iloc[:, 3:].values
        else:
            avg_results = avg_results + df.iloc[:, 3:].values
            
        if datasets is None: datasets = df.iloc[:, 2]

    dict_results['dataset']=list(datasets)    
    avg_results=avg_results / runs
    dict_results.update(df2dict(pd.DataFrame(avg_results, columns=df_cols)))
    
    return dict_results


def evaluate(datasets_dir, resultdir, ts, lst_datasets, ratio=0.8, runs=3, decs={'t':3,'m':9}, verbose=0):

    dict_dataset={"dataset":[], "total":[], "normal":[], "dims":[], "outlier":[], "%out":[], "X":[], "y":[]}
    
    os.makedirs(datasets_dir, mode=0o777, exist_ok=True)
    os.makedirs(resultdir, mode=0o777, exist_ok=True) 
    
    for ds in lst_datasets:
        dict_dataset = add_dataset(dict_dataset, folderpath=datasets_dir, datasetname=ds)

    df_datasets = pd.DataFrame(dict_dataset)[["dataset", "total", "normal", "dims", "outlier", "%out"]]
    df_datasets.to_csv(f"{resultdir}/{ts}_datasets.csv", sep=',', index=False)
    
    for seed in range(0, runs):
        print(f" ###### seed {seed} ######")

        dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos = init_run(ratio, seed, dict_dataset)

        for idx, row in pd.DataFrame(dict_dataset).iterrows():
            print(f"{row.dataset} ...")
            dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos = run_experiments(
                row.X, row.y, row.dataset, dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos,pc_ratio=ratio,seed=seed,decs=decs)

        write_result_dicts(resultdir, ts, ratio, seed, dict_resultst,dict_resultsro,dict_resultsap,dict_resultsrc,dict_resultspn,dict_algos)