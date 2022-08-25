# CoMadOut - A Robust Outlier Detection Algorithm based on CoMAD

Unsupervised learning methods are well established in the area of anomaly detection and achieve state of the art performances on outlier data sets. Outliers play a significant role, since they bear the potential to distort the predictions of a machine learning algorithm on a given data set. Especially among PCA-based methods, outliers have an additional destructive potential regarding the result: they may not only distort the orientation and translation of the principal components, they also make it more complicated to detect outliers. To address this problem, we propose the robust outlier detection algorithm CoMadOut, which satisfies two required properties: (1) being robust towards outliers and (2) detecting them. Our outlier detection method using coMAD-PCA defines dependent on its variant an inlier region with a robust noise margin by measures of in-distribution (ID) and out-of-distribution (OOD). These measures allow distribution based outlier scoring for each principal component, and thus, for an appropriate alignment of the decision boundary between normal and abnormal instances. Experiments comparing CoMadOut with traditional, deep and other comparable robust outlier detection methods showed that the performance of the introduced CoMadOut approach is competitive to well established methods related to average precision (AP), recall and area under the receiver operating characteristic (AUROC) curve. In summary our approach can be seen as a robust alternative for outlier detection tasks.



Python implementation of CoMadOut variants.

## Getting Started

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

## Supplementary Material

In this work the term "score" is used for the final outlier scores. Instances transformed to subspace representations are called projections. Subspace representations projected to subspace axis are called orthogonal projections.
Principal Component (PC)-wise calculated euclidean distances based on the orthogonal projections are weighted according to the related CoMadOut variant and afterwards accumulated for each sample as total outlier score. In case of the CoMadOut variants CMO+ the PC-wise calculated and subsequently accumulated euclidean distances effectively result in variant dependent weighted L1 scores. 
In future work the proposed scoring methods of CMO+ variants will be re-implemented in a more efficient way from partially iterative to matrix and parallel operations which reduced in a first version (see CMOPlusFast) the average runtime from 7.6 to 0.95 seconds and thus being faster than LOF, KNN, OCSVM, MCD, Elliptic Envelope, AE and VAE. 

In order to reveal actual performance differences between the compared methods the metric measurements in the result tables are rounded to 3 decimals.
Due to randomness dependent methods like e.g. IF, VAE, AE, MCD, Elliptic Envelope, etc. the evaluation has been run also for 10 different seeds instead of just for seed zero which lead to further stability in the achieved results. Nevertheless, CoMadOut variants still show competitiveness by demonstrating top 3 average performances across the 20 datasets for AP (Average Precision), AUROC (Area under Receiver Operating Characteristic Curve) and Recall.
Furthermore, CoMadOut variants demonstrate with AUROC performances above average its on par performance also on high-dimensional data sets like e.g. arrhytmia(274), musk(166) or mnist(100).

Ensuring fair comparison between the methods, no hyperparameter-tuning had been conducted so each method had to run with its defaults. 
Hence, method specific advantages or disadvantages are assumed to be neglectable because of the huge variety of different anomaly data sets. 
Only methods assuming standard normal distributed features had been addressed with upfront standardization shown by supplementary material (see folder fig/*_uv.jpg).
Independent of that the overall performance of CoMadOut variants remains constant with only marginal deviations for those variants involving variance weighting. 

## Results

![AP on 100% Components](<./fig/Fig6ap0999_10seeds.jpg> "AP on 100% Components")

![AP on 25% Components](<./fig/Fig7ap025_10seeds.jpg> "AP on 25% Components") 

![AUROC on 100% Components](<./fig/Fig8roc0999_10seeds.jpg> "AUROC on 100% Components")

![AUROC on 25% Components](<./fig/Fig9roc025_10seeds.jpg> "AUROC on 25% Components") 

![Recall on 100% Components](<./fig/Fig10recall0999_10seeds.jpg> "Recall on 100% Components")

![Recall on 25% Components](<./fig/Fig11recall025_10seeds.jpg> "Recall on 25% Components") 

![P@n on 100% Components](<./fig/Fig12precn0999_10seeds.jpg> "P@n on 100% Components")

![P@n on 25% Components](<./fig/Fig13precn025_10seeds.jpg> "P@n on 25% Components") 

![Runtime on 100% Components](<./fig/Fig14runtime0999_10seeds.jpg> "Runtime on 100% Components")

![Runtime on 25% Components](<./fig/Fig15runtime025_10seeds.jpg> "Runtime on 25% Components") 

