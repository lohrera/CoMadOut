import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Optional, Tuple

# example-pkg-jd-kiel==0.0.37 inactive on 20220815 (temporarily only required oab source imported from https://github.com/ISDM-CAU-Kiel/oab instead)
from src.oab.data.utils import _append_to_yaml, _make_yaml, _read_from_yaml, _get_dataset_dict


class AbstractClassificationDataset(ABC):
    pass



class AnomalyDatasetDescription(ABC):
    """This abstract class is a superclass for anomaly dataset descriptions in
    the unsupervised and semisupervised setting. It provides a common interface
    for them and basic common functionality.
    """
    def __init__(self):
        return

    @abstractmethod
    def from_same_dataset(self, other: 'AnomalyDatasetDescription') -> bool:
        """Test if two dataset descriptions come from a similar sampling fo the same anomlay dataset.

        :param other: Other dataset description
        :return: True if the two datasets come from a similar sampling (in terms of contamination_rate and number_instances)
            from the same anomaly dataset (in terms of name, normal_labels, anomaly_labels).
            False otherwise.
        """
        pass


    def print_for_eval_intro(self) -> str:
        """Return string with basic information about the dataset.

        :return: String with basic information about the dataset
        """
        return (f"Evaluation on dataset {self.name} with normal labels " \
                f"{self.normal_labels} and anomaly labels {self.anomaly_labels}.")


    @abstractmethod
    def print_for_eval_specifics(self) -> str:
        """ Return pretty string representation of most important dataset characteristics.

        :return: Pretty string representation of most important dataset characteristics
        """
        pass



class AnomalyDataset(ABC):
    """The abstract AnomalyDataset is a superclass of the unsupervised and
    semisupervised case. It implements some shared functionality and provides
    a unified interface.

    :param classification_dataset: ClassificationDataset that the AnomalyDataset is based on
    :param normal_labels: The label(s) for normal datapoints
    :param anomaly_labels: The label(s) for anomaly datapoints or `None` if all
        other (i.e., non-normal) labels should be used, defaults to None
    """

    def __init__(self, classification_dataset: AbstractClassificationDataset,
        normal_labels: Iterable, anomaly_labels: Optional = None,
        yamlpath_append: Optional = None, yamlpath_new: Optional = None) -> None:
        """Constructor method
        """
        self.classification_dataset = classification_dataset

        # make sure that the normal_labels attribute is iterable
        if isinstance(normal_labels, Iterable):
            self.normal_labels = normal_labels
        else:
            self.normal_labels = [normal_labels]

        self.anomaly_labels = anomaly_labels
        if self.anomaly_labels == None:
            self.anomaly_labels = self._get_all_other_labels()
        # if one anomaly label was specified, this needs to be transformed into a list
        if not isinstance(self.anomaly_labels, Iterable):
            self.anomaly_labels = [self.anomaly_labels]

        self.normal_idxs_individual, self.normal_idxs = self._get_idxs('normals')
        self.anomaly_idxs_individual, self.anomaly_idxs = self._get_idxs('anomalies')

        # store arguments in YAML if specified
        kwargs = {'normal_labels' : normal_labels,
                  'anomaly_labels': anomaly_labels}
        yaml_value = {'arguments': kwargs}
        if not (yamlpath_append is None): # append arguments to YAML
            _append_to_yaml(yamlpath_append, "anomaly_dataset", yaml_value)
        if not (yamlpath_new is None): # create new YAML with arguments
            _make_yaml(yamlpath_new, "anomaly_dataset", yaml_value)


    def sample_from_yaml(self, filepath: str = None,
        type: str = 'unsupervised_multiple'):
        """
        Call `sample_multiple` using arguments specified in a YAML. YAML has to
        have a `sampling` dict where the value is a dictionary with arguments.

        :param filepath: Path to YAML-file
        :param type: Type of sampling, defaults to 'unsupervised_multiple'
        """
        if not filepath:
            dataset_dict = _get_dataset_dict(self.classification_dataset.name)
            filepath = Path('datasets') / dataset_dict['name'] / dataset_dict['destination_yaml']
            # if this filepath does not exist, value error

        kwargs = _read_from_yaml(filepath, ["sampling", type])

        # check that the sampling is actually supported
        if type in ['semisupervised_explicit_numbers_single',
                    'semisupervised_training_split_multiple',
                    'semisupervised_training_split_single']:
            if not hasattr(self, 'sample_with_explicit_numbers'): # only semisupervised has this attribute
                raise ValueError(f"Cannot sample with type {type} from " \
                    f"unsupervised anomaly dataset.")
        # if type is supported, sample

        if type == 'unsupervised_multiple':
            return self.sample_multiple(**kwargs)
        elif type == 'unsupervised_single':
            return self.sample(**kwargs)
        elif type == 'unsupervised_multiple_benchmark':
            return self.sample_multiple_benchmark(**kwargs)
        elif type == 'semisupervised_multiple':
            return self.sample_multiple(**kwargs)
        elif type == 'semisupervised_explicit_numbers_multiple':
            raise NotImplementedError()
        elif type == 'semisupervised_explicit_numbers_single':
            return self.sample_with_explicit_numbers(**kwargs)
        elif type == 'semisupervised_training_split_multiple':
            return self.sample_multiple_with_training_split(**kwargs)
        elif type == 'semisupervised_training_split_single':
            return self.sample_with_training_split(**kwargs)
        else:
            raise NotImplementedError(f"Sampling from yaml with type {type} is not implemented.")


    def get_description_as_series(self):
        description = dict()
        # name
        description['name'] = self.classification_dataset.name
        # n_instances
        description['n'] = len(self.classification_dataset.labels)
        # n_anomalies
        description['n_anomalies'] = len(self.normal_idxs)
        # contamination_rate
        description['contamination_rate'] = description['n_anomalies'] / description['n']
        # n_features
        description['features'] = self.classification_dataset.values.shape[1]
        return pd.Series(description)


    @property
    def n_normals(self):
        return len(self.normal_idxs)


    @property
    def n_anomalies(self):
        return len(self.anomaly_idxs)


    def _get_all_other_labels(self):
        """Returns a list of all labels that are not normal labels.
        """
        all_labels = np.unique(self.classification_dataset.labels)
        return list(set(all_labels) - set(self.normal_labels))


    def _get_idxs(self, kind: str):
        """For either 'normals' or 'anomalies' (as kind): Returns one list of lists of indexes,
        where each label has its own list of indexes, as well as one list of indexes where
        all indexes are combined.
        Example return value:
        [[1, 2, 3], [5, 6, 7]], [1, 2, 3, 5, 6, 7] (all as np.ndarrays). The first normal (anomalous) label is at
        indexes 1, 2, and 3, and the second normal (anomalous) label is at indexes 5, 6, and 7.
        """
        # set list of labels according to 'kind'
        if kind == 'normals':
            labels = self.normal_labels
        elif kind == 'anomalies':
            labels = self.anomaly_labels
        else:
            raise ValueError(f"Kind must be either 'anomalies' or 'normals', not {kind}.")

        # get list of lists of booleans
        list_of_idxs = []
        for label in labels:
            idxs = (self.classification_dataset.labels == label)
            list_of_idxs.append(idxs)

        # combine lists of booleans
        combined = np.full(shape=len(self.classification_dataset.labels), fill_value=0).astype(bool)
        for idxs in list_of_idxs:
            combined = (combined | idxs)

        # transform from bools into actual indexes
        # first to get a list of indexes for each label, second to get an overall list of indexes
        list_of_idxs = [np.arange(len(self.classification_dataset.labels))[mask] for mask in list_of_idxs]
        combined = np.arange(len(self.classification_dataset.labels))[combined]
        return list_of_idxs, combined


    @abstractmethod
    def sample(self) -> Tuple[Tuple[np.ndarray, np.ndarray], AnomalyDatasetDescription]:
        """
        Sample from the anomaly dataset. The parameters are dependent on the
        subclass, i.e., are different in the unsupervised and semisupervised
        case.

        :return: A tuple with a tuple of datapoints and corresponding values as the first
            part and a description as the second part.
        """
        pass


    @abstractmethod
    def sample_multiple(self) -> Iterable:
        """Sample multiple times from the anomaly dataset as an iterator. The
        parameters are dependent on the subclass, i.e., are different in the
        unsupervised and semisupervised case.

        :return: An iterator which iterates through individual samples. A sample
            consists of a tuple with a tuple of datapoints and corresponding values
            as the first part and a description as the second part.
        """
        pass


    def _test_contamination_rate(self, contamination_rate: float) -> bool:
        """Tests whether the contamination rate is valid, i.e., between 0 and 1.
        """
        if contamination_rate < 0 or contamination_rate > 1:
            raise ValueError(f"Contamination rate should be between 0 and 1, but the current value is {contamination_rate}.")


    def _sample_data(self, kind: str, n_samples: int, keep_frequency_ratio: bool,
        equal_frequency: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Internal operation to sample datapoints and their labels. If neither
        keep_frequency_ratio or equal_frequency are True, sampling is at random.

        :param kind: Kind of data that is to be sampled, either 'normals' or
            'anomalies'
        :param n_samples: Number of datapoints to sample
        :param keep_frequency_ratio: If the sampling frequency of different labels should be the same as
            their frequency in the original dataset in their respective class (normal/anomalous).
            Example: If labels 0 and 1 are anomalous and in the original data, there are 400 0s and
            600 1s, in the sampled output we also want a 4 to 6 ratio. Defaults to
            False in prior methods.
        :param equal_frequency: If the sampling frequency of different labels should be equal.
            Example: If labels 0 and 1 are anomalous and in the original data, there are 400 0s and
            600 1s, in the sampled output we want a 1 to 1 ratio. Defaults to False in prior methods.
        :return: A tuple with the following elements:
            - The datapoints (shape (n, d))
            - Their labels in the anomaly detection setting, i.e. 0 or 1 (shape (n))
            - Their original labels in the dataset (shape (n))
        """
        if not (kind == 'normals' or kind == 'anomalies'):
            raise ValueError(f"Kind has to be 'normals' or 'anomalies', cannot be f{kind}")
        if keep_frequency_ratio and equal_frequency:
            raise ValueError(f"Keeping the frequency ratio and from the original dataset "\
                               "and setting the frequency for all labels to the same is " \
                               "mutually exclusive. Set at max one of them, but not both.")

        # set labels (they are the same no matter how the sampling is done)
        # and check if there are enough data points to be sampled
        if kind == 'normals':
            labels = np.zeros(n_samples)
            if self.n_normals < n_samples:
                raise ValueError(f"Cannot sample {n_samples} normals as the " \
                    f"data set only has {self.n_normals} normal data points.")
        else: # kind == 'anomalies'
            labels = np.ones(n_samples)
            if self.n_anomalies < n_samples:
                raise ValueError(f"Cannot sample {n_samples} anomalies as the " \
                    f"data set only has {self.n_anomalies} anomalies.")

        # sample with keeping the frequency ratio -> get indices
        if keep_frequency_ratio:
            idxs = np.zeros(n_samples) # this array will be filled
            total_points = len((self.normal_idxs
                                if kind == 'normals'
                                else self.anomaly_idxs))
            # iterate through all labels and: (1) determine the sampling size,
            # (2) sample accordingly
            enumerate_labels = (self.normal_labels
                                if kind == 'normals'
                                else self.anomaly_labels)
            n_labels = len(enumerate_labels)
            cumulative_counter = 0 # used to make sure idxs is completely filled
            for counter, label in enumerate(enumerate_labels):
                # get the corresponding set of indexes
                if kind == 'normals':
                    label_idxs = self.normal_idxs_individual[counter].copy()
                else: # kind == 'anomalies'
                    label_idxs = self.anomaly_idxs_individual[counter].copy()
                n_points = len(label_idxs)
                # for all but the last label, sample according to relative frequency
                # of that label in the data set
                if counter < n_labels - 1:
                    n_from_label = int((n_points / total_points) * n_samples)
                    print(f"{n_samples}, and {total_points}, {n_points}")
                # for the last label, make sure that in total n_samples are sampled
                else:
                    n_from_label = n_samples - cumulative_counter
                # shuffle and sample
                np.random.shuffle(label_idxs)
                print(f"{label} and {n_from_label} with {n_samples}")
                idxs[cumulative_counter:(cumulative_counter+n_from_label)] = \
                    label_idxs[:n_from_label]
                cumulative_counter += n_from_label

        # sample with an equal frequency of all labels -> get indices
        elif equal_frequency:
            # calculate how many data points to sample from each index
            n_labels = len((self.normal_labels
                            if kind == 'normals'
                            else self.anomaly_labels))
            n_from_each_label = n_samples // n_labels
            idxs = np.zeros(n_samples) # this array will be filled
            # iterate through all labels
            enumerate_labels = (self.normal_labels
                                if kind == 'normals'
                                else self.anomaly_labels)
            for counter, label in enumerate(enumerate_labels):
                # get idxs for that label
                if kind == 'normals':
                    label_idxs = self.normal_idxs_individual[counter].copy()
                if kind == 'anomalies':
                    label_idxs = self.anomaly_idxs_individual[counter].copy()
                # check if there are enough labels to sample from
                if len(label_idxs) < n_from_each_label:
                    raise ValueError(f"Cannot sample {kind} with equal frequency," \
                        f" as there are too few data points with label {label}.")
                # shuffle idxs for that label
                np.random.shuffle(label_idxs)
                # for all but the last label, take n_from_each_label
                if counter < (n_labels-1):
                    idxs[counter*n_from_each_label : (counter+1)*n_from_each_label] = \
                        label_idxs[:n_from_each_label]
                # for the last label, take remaining, which can be n_from_each_label + 1
                else:
                    n_to_sample = n_samples - (n_labels-1) * n_from_each_label
                    if len(label_idxs) < n_to_sample:
                        raise ValueError(f"Cannot sample {kind} with equal frequency," \
                            f" as there are too few data points with label {label}.")
                    idxs[counter*n_from_each_label:] = label_idxs[:n_to_sample]

        # sample randomly -> get indices
        else:
            if kind == 'normals':
                all_possible_idxs = self.normal_idxs.copy()
            if kind == 'anomalies':
                all_possible_idxs = self.anomaly_idxs.copy()
            np.random.shuffle(all_possible_idxs)
            idxs = all_possible_idxs[:n_samples]

        # retrieve values and original labels based on sampled indices
        idxs = idxs.astype('int')
        values = self.classification_dataset.values[idxs]
        original_labels = self.classification_dataset.labels[idxs]

        return values, labels, original_labels
