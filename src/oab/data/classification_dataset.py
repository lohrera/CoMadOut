import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Union, Optional

# example-pkg-jd-kiel==0.0.37 inactive on 20220815 (temporarily only required oab source imported from https://github.com/ISDM-CAU-Kiel/oab instead)
from src.oab.data.unsupervised import UnsupervisedAnomalyDataset
from src.oab.data.semisupervised import SemisupervisedAnomalyDataset
from src.oab.data.abstract_classes import AbstractClassificationDataset
from src.oab.data.utils import _read_from_yaml, _get_dataset_dict



class ClassificationDataset(AbstractClassificationDataset):
    """This class represents a classification dataset with the observations'
    values, labels, and the dataset name. It provides preprocessing
    functionality, the preprocessing can be exported to a yaml file and
    preprocessing as specified in a yaml file can be performed to allow
    reproducibility.

    :param values: Values of the observations
    :param labels: Labels of the observation
    :param name: Name of the dataset, defaults to None
    """

    def __init__(self, values: np.ndarray, labels: np.ndarray,
        name: Optional[str] = None):
        """Constructor method.
        """
        self.values = values
        self.labels = labels
        self.name = name
        self.operations_performed = []


    def transform_categorical_to_one_hot(self, cols_to_onehot: List[int]) -> None:
        """Transform the columns belonging to the specified indexes to one-hot
        encodings.

        :param cols_to_onehot:  List of indexes specifying the columns to be
            one-hot encoded.
        """
        parameters = {'cols_to_onehot': cols_to_onehot}
        operation_record = (self.transform_categorical_to_one_hot.__name__, parameters)
        self.operations_performed.append(operation_record)

        # sort columns so that the new columns don't get in the old column's way
        cols_to_onehot = sorted(cols_to_onehot, reverse=True)

        for col in cols_to_onehot:
            col_values = self.values[:, col]
            onehot = pd.get_dummies(col_values).values
            self.values = np.hstack((self.values[:, :col], onehot, self.values[:, (col+1):]))

        return


    def transform_categorical_to_idf(self, cols_to_idf: List[int]) -> None:
        """Transform the columns belonging to the specified indexes to IDF encodings.
        IDF(value) = ln(N/f_value) where value is the value to be encoded, f_value
        its frequency (count of occurrences), and N the total number of occurrences
        of all values.

        :param cols_to_idf:  List of indexes specifying the columns to be IDF
            encoded.
        """
        parameters = {'cols_to_idf': cols_to_idf}
        operation_record = (self.transform_categorical_to_idf.__name__, parameters)
        self.operations_performed.append(operation_record)

        cols_to_idf = sorted(cols_to_idf, reverse=True)
        for col in cols_to_idf:
            to_idf = self.values[:, col]
            ct = Counter(to_idf)
            total = len(to_idf)
            idf = {key: np.log(total/n_occs) for key, n_occs in ct.items()}
            new_col_values = np.array([idf[key] for key in to_idf])[:, np.newaxis]
            self.values = np.hstack((self.values[:, :col], new_col_values, self.values[:, (col+1):]))

        return


    def delete_columns(self, cols_to_delete: List[int]) -> None:
        """Delete the columns specified by index.

        :param cols_to_delete: List of indexes specifying the columns to be
            deleted.
        """
        parameters = {'cols_to_delete': cols_to_delete}
        operation_record = (self.delete_columns.__name__, parameters)
        self.operations_performed.append(operation_record)

        cols_to_delete = sorted(cols_to_delete, reverse=True)
        for col in cols_to_delete:
            self.values = np.hstack((self.values[:, :col], self.values[:, (col+1):]))

        return


    def normalize_columns(self,
            cols_to_normalize: Optional[List[int]] = None) -> None:
        """Normalize the columns specified by index, i.e., performs a linear normalization
        to the range [0, 1].

        :param cols_to_normalize: List of indexes specifying the columns to
            normalized or None if all columns should be normalized.
        """
        parameters = {'cols_to_normalize': cols_to_normalize}
        operation_record = (self.normalize_columns.__name__, parameters)
        self.operations_performed.append(operation_record)

        if cols_to_normalize == None or cols_to_normalize == "None":
            cols_to_normalize = list(range(self.values.shape[1]))

        self._columns_preprocessing(cols_to_normalize, MinMaxScaler)


    def standardize_columns(self,
            cols_to_standardize: Union[None, List[int]] = None) -> None:
        """ Standardize the columns specified by index, i.e., transforms the
        values using the mean and variance so that the transformed values have
        mean 0, variance 1.

        :param cols_to_standardize: List of indexes specifying the columns to
            standardized or None if all columns should be standardized.
        """
        parameters = {'cols_to_standardize': cols_to_standardize}
        operation_record = (self.standardize_columns.__name__, parameters)
        self.operations_performed.append(operation_record)

        if cols_to_standardize == None or cols_to_standardize == "None":
            cols_to_standardize = list(range(self.values.shape[1]))

        self._columns_preprocessing(cols_to_standardize, StandardScaler)


    def robust_scale_columns(self,
            cols_to_robust_scale: Union[None, List[int]] = None) -> None:
        """ Scales the columns specified by index using sklearn's RobustScaler.

        :param cols_to_robust_scale: List of indexes specifying the columns to
            standardized or None if all columns should be standardized.
        """
        parameters = {'cols_to_robust_scale': cols_to_robust_scale}
        operation_record = (self.robust_scale_columns.__name__, parameters)
        self.operations_performed.append(operation_record)

        if cols_to_robust_scale == None or cols_to_robust_scale == "None":
            cols_to_robust_scale = list(range(self.values.shape[1]))

        self._columns_preprocessing(cols_to_robust_scale, RobustScaler)


    def _columns_preprocessing(self, cols_to_change: List[int],
            scaler: Union[MinMaxScaler, StandardScaler, RobustScaler]) -> None:
        """ Internal helper to perform scaling operation on columns specified
        by index.

        :param cols_to_change: Indexes of columns that should be changed
        :param scaler: Scaler to be applied
        """
        cols_to_change = sorted(cols_to_change, reverse=True)
        for col in cols_to_change:
            to_change = self.values[:, col]

            new_col_values = scaler().fit_transform(to_change[:, np.newaxis])
            self.values = np.hstack((self.values[:, :col], new_col_values, self.values[:, (col+1):]))

        return


    def scale(self, scaling_factor: float) -> None:
        """ Scale all values by a scaling factor.

        :param scaling_factor: Scaling factor used to scale all values
        """
        parameters = {'scaling_factor': scaling_factor}
        operation_record = (self.scale.__name__, parameters)
        self.operations_performed.append(operation_record)
        self.values = self.values * scaling_factor
        return


    def delete_duplicates(self, debug: bool = False) -> None:
        """ Delete duplicate rows from the values and labels.
        Note: Which element of a set of duplicate values is deleted is decided by order/chance.
        If they have different labels, only one of the labels will be kept.

        :param debug: Print information about how much data is deleted, defaults
            to False
        """
        parameters = {}
        operation_record = (self.delete_duplicates.__name__, parameters)
        self.operations_performed.append(operation_record)

        self.values = self.values.astype('float')

        _, idxs = np.unique(self.values, axis=0, return_index=True)
        sorted_idxs = np.sort(idxs)
        if debug:
            print(f"Deleted {len(self.values) - len(sorted_idxs)} rows.")
        self.values = self.values[sorted_idxs]
        self.labels = self.labels[sorted_idxs]

        return


    def treat_missing_values(self, missing_value: Union[str, float] = "np.nan",
            delete_attributes: bool = True, debug: bool = False):
        """ Routine to treat missing values like in Campos et al. 2016. (DOI HERE).
        If an attribute is missing for 10% of the observations or more, remove the attribute.
        Otherwise, remove the observation with missing values.

        :param missing_value: Value that symbolizes missing values, defaults to
            `'np.nan'`
        :param delete_attributes: If set to False, attributes will not be
            deleted, but instead all observations with a missing value, defaults to True
        :param debug: Print information about how much data is deleted, defaults
            to False
        """
        parameters = {'missing_value': missing_value,
                      'delete_attributes': delete_attributes}
        operation_record = (self.treat_missing_values.__name__, parameters)
        self.operations_performed.append(operation_record)

        # if missing value is np.nan, change from str to np.nan. Else, it's a
        # number/string/..., so there is no change necessary.
        if missing_value == "np.nan":
            missing_value = np.nan

        if delete_attributes:
            # delete columns if too more 10% or more of observations are missing
            if (type(missing_value) == float) and np.isnan(missing_value):
                missing_matrix = np.isnan(self.values)
            else:
                missing_matrix = (self.values == missing_value)
            delete_cols_bools = np.sum(missing_matrix, axis=0) >= (0.1 * self.values.shape[0])
            cols_to_delete = np.where(delete_cols_bools)[0]
            if debug:
                print(f"The following columns are deleted: {cols_to_delete}.")
            self.values = np.delete(self.values, cols_to_delete, axis=1)

        # delete rows if they still have missing values
        if (type(missing_value) == float) and np.isnan(missing_value):
            missing_matrix = np.isnan(self.values)
        else:
            missing_matrix = (self.values == missing_value)
        delete_rows_bools = np.any(missing_matrix, axis=1)
        rows_to_delete = np.where(delete_rows_bools)[0]
        if debug:
            print(f"The following rows are deleted: {rows_to_delete}.")
        self.values = np.delete(self.values, rows_to_delete, axis=0)
        # also delete the corresponding labels
        self.labels = np.delete(self.labels, rows_to_delete, axis=0)
        return


    def transform_labels_iqr(self, factor: float = 2) -> None:
        """Transforms labels into binary (normal/outlier) labels based on the
        IQR of the original numerical labels. The IQR is Q3 - Q1.
        If numerical value is lower than Q1 - factor * IQR or larger than
        Q3 + factor * IQR, the data is an anomaly, which is indicated by 1.
        Else, i.e., if the value lies within the IQR, it is normal, as indicated by 0.

        :param factor: Determines length of "normal" interval, defaults to 2
        """
        q1 = np.quantile(self.labels, .25)
        q3 = np.quantile(self.labels, .75)
        IQR = q3 - q1
        lower_bound = q1 - factor * IQR
        upper_bound = q3 + factor * IQR
        new_labels = (self.labels < lower_bound) | (self.labels > upper_bound)
        self.labels = new_labels.astype('int')
        return


    def write_operations_to_yaml(self, filename: str = "preprocessing.yaml") -> None:
        """Writes operations that have been performed to a yaml file (for better
        reproducibility).

        :param filename: Filename of yaml file, defaults to `'preprocessing.yaml'`
        """
        dictionary = {'standard_functions':
            [{'name': name, 'parameters': parameters}
             for name, parameters in self.operations_performed]}
        with open(filename, "w+") as f:
            yaml.dump(dictionary, f)
        return


    def perform_operations_from_yaml(self, filepath: str = None,
        make_into_anomaly_dataset: bool = False,
        unsupervised: bool = False, semisupervised: bool = False) -> Optional[Union[UnsupervisedAnomalyDataset, SemisupervisedAnomalyDataset]]:
        """Perform perprocessing functions as specified by the yaml file on the
        dataset.

        :param filepath: Name of the yaml file, defaults to `'preprocessing.yaml'`
        :param make_into_anomaly_dataset: If this is set to true, the dataset
            is directly transformed into an anomaly dataset (from a Classification
            Dataset). For this to work, the appropriate information has to be
            contained in the yaml file. Make sure to also specify if a unsupervised
            or semisupervised dataset should be created. Defaults to False
        :param unsupervised: This is only relevant if the dataset is directly
            transformed into an anomaly dataset, i.e., if the flag `make_into_anomaly_dataset`
            is set to True. If this is also set, an anomaly dataset for unsupervised
            anomaly detection is returned. Defaults to False
        :param semisupervised: This is only relevant if the dataset is directly
            transformed into an anomaly dataset, i.e., if the flag `make_into_anomaly_dataset`
            is set to True. If this is also set, an anomaly dataset for semisupervised
            anomaly detection is returend. Note that `unsupervised` has precedence
            over `semisupervised`: If both are set to True, a unsupervised dataset
            is returned. Defaults to False
        """
        if not filepath:
            dataset_dict = _get_dataset_dict(self.name)
            filepath = Path('datasets') / dataset_dict['name'] / dataset_dict['destination_yaml']
            # if this filepath does not exist, value error

        with open(filepath, "r") as f:
            try:
                dictionary = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"{filepath} is not in readable yaml format")

        # First perform standard functions
        if 'standard_functions' in dictionary:
            list_standard_functions = dictionary['standard_functions']
            # print(list_standard_functions)
            for func_dict in list_standard_functions:
                func_name, params = func_dict['name'], func_dict['parameters']
                # print(func_name, params)
                func = getattr(self, func_name)
                func(**params)

        # Then perform customized functions if there are any
        if 'custom_functions' in dictionary:
            import custom_preprocessing as cp
            list_custom_functions = dictionary['custom_functions']
            for func_dict in list_custom_functions:
                func_name, params = func_dict['name'], func_dict['parameters']
                func = getattr(cp, func_name)
                self.values = func(self.values, **params)

        # If specified, make ClassificationDataset into AnomalyDataset
        if make_into_anomaly_dataset:
            if not 'anomaly_dataset' in dictionary:
                raise Error(f"Specify anomaly_dataset in yaml to crate AnomalyDataset.")
            if unsupervised:
                return UnsupervisedAnomalyDataset(self,
                    **dictionary['anomaly_dataset']['arguments'])
            if semisupervised:
                return SemisupervisedAnomalyDataset(self,
                    **dictionary['anomaly_dataset']['arguments'])
            if dictionary['anomaly_dataset']['type'] == 'unsupervised':
                return UnsupervisedAnomalyDataset(self,
                    **dictionary['anomaly_dataset']['arguments'])
            elif dictionary['anomaly_dataset']['type'] == 'semisupervised':
                return SemisupervisedAnomalyDataset(self,
                    **dictionary['anomaly_dataset']['arguments'])
            else:
                raise Error(f"Only unsupervised and semisupervised anomaly datasets supported.")


    def tranform_from_yaml(self, filepath: str = None, unsupervised: bool = False,
        semisupervised: bool = False):
        """Transforms a ClassificationDataset into an AnomalyDataset of specified
        kind based on parameters from the yaml file. Note that unsupervised and
        semisupervised has to be specified, and unsupervised takes precedence
        over semisupervised.

        :param filepath: Path to yaml file
        :param unsupervised: If this is  set, an anomaly dataset for unsupervised
            anomaly detection is returned. Defaults to False
        :param semisupervised: If this is set and `unsupervised` is not set, an
            anomaly dataset for semisupervised anomaly detection is returned.
            Defaults to False
        """
        if not filepath:
            dataset_dict = _get_dataset_dict(self.name)
            filepath = Path('datasets') / dataset_dict['name'] / dataset_dict['destination_yaml']
            # if this filepath does not exist, value error

        yaml_content = _read_from_yaml(filepath, keys=['anomaly_dataset', 'arguments'])
        if unsupervised:
            return UnsupervisedAnomalyDataset(self, **yaml_content)
        elif semisupervised:
            return SemisupervisedAnomalyDataset(self, **yaml_content)
        else:
            raise Error(f"Only unsupervised and semisupervised anomaly datasets supported. Please specify one of them.")


    def sample_multiple_unsupervised_with_different_normal_labels(self,
        n: int, n_steps: int = 10, contamination_rate: float = 0.05,
        normal_labels: Optional[List[str]] = None,
        all_labels_same_anomaly_dataset_name: bool = True,
        shuffle: bool = True, random_seed=42, apply_random_seed: bool = True,
        keep_frequency_ratio_anomalies: bool = False, equal_frequency_anomalies: bool = False,
        include_description: bool = True, yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True
        ):
        """
        Iterates over all specified labels (or all labels if None) to sample
        from resulting dataset with the set parameters. This can be particularly
        useful in multi-classification datasets like MNIST, where one wants
        to assess an algorithms performance on datasets generated by setting
        different labels to be normal data points.
        Note: It returns (x, y), sampling_config, label. The last part of the
        tuple is the label that is currently set as "normal".

        Internally, this function creates multiple anomaly datasets one after
        another. Each time, one label is set as normal label and all other
        labels are anomalous labels. In case more than one label should be
        normal or not all other labels should be anomalous, the anomaly datasets
        have to be specified individually, i.e., this function is not applicable.

        :param n: Number of data points to sample
        :param n_steps: Number of sampled to take, i.e., number of times
            sampling is repeated, defaults to 10
        :param contamination_rate: Contamination rate when sampling, defaults to 0.05
        :param normal_labels: Optional, either None to use all existing labels
            as normal labels or a list of labels that are to be used as
            normal labels (in the specified order), defaults to None
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels, defaults to False
        :include_description: Includes sampling config file, defaults to True
        :param yamlpath_append: Optionally append sampling arguments to a YAML
            if this is not None, the path of the YAML is specified in this
            argument, defaults to None
        :param yamlpath_new: Optionally create a new YAML with the sampling
            arguments if this is not None, the path of the YAML is specified in
            this argument, defaults to None
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :param normal_labels: Optionally specifies which labels are used as
            normal labels. Note that if a list of values is specified, a new
            anomaly dataset is created for each label. As described above, if
            more than one label should be the normal label, this function
            should not be used. Defaults to None, which means that this is filled
            by a list with all labels in the dataset.
        :param all_labels_same_anomaly_dataset_name: Specify if the dataset name
            stored in the returned config file should be the same for all
            samples (if this is set to True) or should also indicate the
            normal label (if set to False).
            If set to False, dataset names will be MNIST1 (MNIST with 1 as
            normal label), ... MNIST9 for example. Setting this influences
            how the results will eventually be evaluated. If set to True, there
            will be one mean over all samples over all normal labels as mean
            metric. If set to False, metrics are reported for each normal label
            individually. Note: If this is set to False, one could still change
            the description/config to also feature the anomaly labelt.
            Defaults to True.
        """
        # if specified, use all labels as normal labels
        if normal_labels == None:
            normal_labels = np.unique(self.labels)
        # get dataset name
        dataset_name = self.name if self.name else ""

        # sample for all labels
        for normal_label in normal_labels:
            ad = UnsupervisedAnomalyDataset(self, normal_labels=[normal_label])
            for (x, y), sample_config in ad.sample_multiple(n=n, n_steps=n_steps,
                contamination_rate=contamination_rate, shuffle=shuffle,
                random_seed=random_seed, apply_random_seed=apply_random_seed,
                keep_frequency_ratio_normals=False,
                equal_frequency_normals=False,
                keep_frequency_ratio_anomalies=keep_frequency_ratio_anomalies,
                equal_frequency_anomalies=equal_frequency_anomalies,
                include_description=include_description,
                yamlpath_append=yamlpath_append, yamlpath_new=yamlpath_new,
                flatten_images=True):

                if not all_labels_same_anomaly_dataset_name:
                    sample_config.name = f"{dataset_name}{normal_label}"

                yield (x, y), sample_config, normal_label


    def __str__(self):
        """String representation
        """
        return f"Dataset {self.name} with {len(self.values)} samples of dimensionality {self.values.shape[1]}."
