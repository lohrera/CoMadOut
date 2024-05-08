# CoMadOut - A Robust Outlier Detection Algorithm based on CoMAD

Unsupervised learning methods are well established in the area of anomaly detection and achieve state of the art performances on outlier data sets. Outliers play a significant role, since they bear the potential to distort the predictions of a machine learning algorithm on a given data set. Especially among PCA-based methods, outliers have an additional destructive potential regarding the result: they may not only distort the orientation and translation of the principal components, they also make it more complicated to detect outliers. To address this problem, we propose the robust outlier detection algorithm CoMadOut, which satisfies two required properties: (1) being robust towards outliers and (2) detecting them. Our outlier detection method using coMAD-PCA defines dependent on its variant an inlier region with a robust noise margin by measures of in-distribution (ID) and out-of-distribution (OOD). These measures allow distribution based outlier scoring for each principal component, and thus, for an appropriate alignment of the decision boundary between normal and abnormal instances. Experiments comparing CoMadOut with traditional, deep and other comparable robust outlier detection methods showed that the performance of the introduced CoMadOut approach is competitive to well established methods related to average precision (AP), recall and area under the receiver operating characteristic (AUROC) curve. In summary our approach can be seen as a robust alternative for outlier detection tasks.

## How to cite (bibtex)

```
@article{CoMadOut,
author={Lohrer, Andreas and Kazempour, Daniyal and H{\"u}nem{\"o}rder, Maximilian and Kr{\"o}ger, Peer},
title={{CoMadOut - {A} Robust Outlier Detection Algorithm based on CoMAD}},
journal={Machine Learning},
year={2024},
month={May},
day={07},
issn={1573-0565},
doi={10.1007/s10994-024-06521-2},
url={https://doi.org/10.1007/s10994-024-06521-2}
}
```

Python implementation of CoMadOut variants.

## Getting Started

For a quick start open CoMadOut-tutorial.ipynb 

For benchmark results open CoMadOut.ipynb 

Install packages with:

```
$ pip install -r requirements.txt
```

## Datasets

``` 
#select datasets in CoMadOut.ipynb Cell 2 among

lst_datasets=['arrhytmia', 'cardio', 'annthyroid', 'breastw', 'letter', 'thyroid', 'mammography', 'pima', 'musk', 'optdigits', 'pendigits', 
              'mnist', 'shuttle', 'satellite', 'satimage-2', 'wine', 'vowels', 'glass', 'wbc', 'boston']

lst_datasets = ['testdata'] 
``` 

## How to run

```
1. set number of runs or ratio of principal components in see CoMadOut.ipynb Cell 2
2. run evaluation with CoMadOut.ipynb Cell 3
3. get results by running remaining cells
```

