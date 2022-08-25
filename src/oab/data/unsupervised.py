import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict

# example-pkg-jd-kiel==0.0.37 inactive on 20220815 (temporarily only required oab source imported from https://github.com/ISDM-CAU-Kiel/oab instead)
from src.oab.data.abstract_classes import AnomalyDataset, AnomalyDatasetDescription
from src.oab.data.utils import _append_to_yaml, _make_yaml, _append_sampling_to_yaml
from src.oab.data.utils_image import image_datasets, reshape_images


# TODO - this is a sampling description!
@dataclass
class UnsupervisedAnomalyDatasetDescription(AnomalyDatasetDescription):
    """Description object for a sample from an unsupervised anomaly dataset.

    :param name: Name of the dataset
    :param normal_labels: List of normal labels
    :param anomaly_labels: List of anomalous labels
    :param number_instances: Number of instances in the sample
    :param number_normals: Number of normal data points in the sample
    :param number_anomalies: Number of anomaly data points in the sample
    :param contamination_rate: Contamination rate of the sample
    """
    name: str

    normal_labels: list()
    anomaly_labels: list()

    number_instances: int
    number_normals: int
    number_anomalies: int
    contamination_rate: float

    # TODO: add stuff for equal_frequency_anomalies, etc.

    def from_same_dataset(self, other: 'UnsupervisedAnomalyDatasetDescription') -> bool:
        """
        Test if two dataset descriptions come from a similar sampling fo the same
        anomlay dataset.

        :param other: Other dataset description
        :return: True if the two datasets come from a similar sampling (in terms of contamination_rate and number_instances)
            from the same anomaly dataset (in terms of name, normal_labels, anomaly_labels).
            False otherwise.

        """
        # If we have image datasets, they don't need to have the same anomaly
        # labels (as all labels are anomalie once)
        if self.name in image_datasets:
            self_attributes = (self.name,
                               self.number_instances,
                               )
            other_attributes = (other.name,
                                other.number_instances,
                                )
        else:
            self_attributes = (self.name,
                               set(self.normal_labels),
                               set(self.anomaly_labels),
                               self.number_instances)
            other_attributes = (other.name,
                                set(other.normal_labels),
                                set(other.anomaly_labels),
                                other.number_instances)
        float_comparison = np.isclose(self.contamination_rate, other.contamination_rate)
        return (self_attributes == other_attributes) and float_comparison


    def print_for_eval_specifics(self) -> str:
        """Return pretty string representation of most important dataset characteristics.

        :return: String with the most important dataset characteristics
        """
        return f"{self.number_instances} instances, contamination_rate {self.contamination_rate}."



class UnsupervisedAnomalyDataset(AnomalyDataset):
    """This class represents an unsupervised anomaly dataset, i.e., when
    sampling from the dataset there is no train/test split, but only the data
    and its labels are returned.
    """

    def sample(self, n, contamination_rate: float = 0.05, shuffle: bool = True,
        random_seed: Union[int, float] = 42, apply_random_seed: bool = True,
        keep_frequency_ratio_normals: bool = False, equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False, equal_frequency_anomalies: bool = False,
        include_description: bool = True, yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True):
        """Sample from the anomaly dataset.

        :param n: Number of data points to sample
        :param contamination_rate: Contamination rate when sampling, defaults to 0.05
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels, defaults to False
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

        :return: TODO
        """
        #
        # Returns
        # -------
        # (values, labels), description : (np.ndarray, np.ndarray), TODO
        #     A tuple with a tuple of datapoints and corresponding values as the first
        #     part and a description as the second part.
        # """
        # # TODO: test if n is too large (i.e., cannot sample that many instances given the other arguments)
        kwargs = {'n': n, 'contamination_rate': contamination_rate, 'shuffle': shuffle, 'random_seed': random_seed,
                  'apply_random_seed': apply_random_seed, 'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
                  'equal_frequency_normals': equal_frequency_normals, 'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
                  'equal_frequency_anomalies': equal_frequency_anomalies, 'include_description': include_description,
                  'flatten_images': flatten_images}
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "unsupervised_single", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'unsupervised_single': kwargs})
        self._test_contamination_rate(contamination_rate)
        if apply_random_seed:
            np.random.seed(random_seed)

        n_normals = int(n * (1 - contamination_rate))
        n_anomalies = n - n_normals

        normals_values, normals_labels, normals_original_labels = \
            self._sample_data('normals', n_normals, keep_frequency_ratio_normals, equal_frequency_normals)
        anomalies_values, anomalies_labels, anomalies_original_labels = \
            self._sample_data('anomalies', n_anomalies, keep_frequency_ratio_anomalies, equal_frequency_anomalies)

        values = np.vstack((normals_values, anomalies_values))
        labels = np.hstack((normals_labels, anomalies_labels))

        # data description
        description = UnsupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances=n, number_normals=n_normals, number_anomalies=n_anomalies,
            contamination_rate=contamination_rate)

        if shuffle:
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            values = values[idxs]
            labels = labels[idxs]

        if self.classification_dataset.name in image_datasets:
            values = reshape_images(values, self.classification_dataset.name,
                flatten_images)

        return (values, labels), description


    def sample_multiple(self, n: int, n_steps: int = 10,
        contamination_rate: float = 0.05, shuffle: bool = True, random_seed=42, apply_random_seed: bool = True,
        keep_frequency_ratio_normals: bool = False, equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False, equal_frequency_anomalies: bool = False,
        include_description: bool = True, yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True):
        """
        Sample multiple times from the anomaly dataset as an iterator.

        :param n: Number of data points to sample
        :param n_steps: Number of sampled to take, i.e., number of times
            sampling is repeated, defaults to 10
        :param contamination_rate: Contamination rate when sampling, defaults to 0.05
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels, defaults to False
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

        :return: TODO
        """
        kwargs = {'n': n, 'n_steps': n_steps, 'contamination_rate': contamination_rate, 'shuffle': shuffle, 'random_seed': random_seed,
                  'apply_random_seed': apply_random_seed, 'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
                  'equal_frequency_normals': equal_frequency_normals, 'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
                  'equal_frequency_anomalies': equal_frequency_anomalies, 'include_description': include_description,
                  'flatten_images': flatten_images}
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "unsupervised_multiple", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'unsupervised_multiple': kwargs})
        # n_steps is not used with sample
        del kwargs['n_steps']
        # sample with specified parameters first
        yield self.sample(**kwargs)
        for _ in range(1, n_steps):
            # increase random seed by 1 to make sure sampling is actually the
            # same, even when an algorithm also uses a random call somewhere
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample(**kwargs)


    def get_entire_dataset(self, shuffle: bool = True,
        random_seed: Union[int, float] = 42, apply_random_seed: bool = True) -> Tuple[np.ndarray, np.ndarray, UnsupervisedAnomalyDatasetDescription]:
        """
        Provide entire dataset (i.e., no sampling involved) together with a description.

        :param shuffle: Shuffle the datapoints and labels, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply the random seed to get reproducible results, defaults to True
        :return: A tuple with the following elements:
            - The datapoints (shape (n, d))
            - Their labels in the anomaly detection setting, i.e. 0 or 1 (shape (n))
            - A description of the dataset
        """
        if apply_random_seed:
            np.random.seed(random_seed)

        n = len(self.classification_dataset.values)
        n_normals = np.sum([np.sum(self.classification_dataset.labels == label) for label in self.normal_labels])
        n_anomalies = n - n_normals
        contamination_rate = n_anomalies / n

        # data description
        description = UnsupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances=n, number_normals=n_normals, number_anomalies=n_anomalies,
            contamination_rate=contamination_rate)

        values = self.classification_dataset.values.copy()
        labels = np.zeros(len(self.classification_dataset.labels))
        labels[self.anomaly_idxs] = 1

        if shuffle:
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            values = values[idxs]
            labels = labels[idxs]

        return (values, labels), description



    def get_sampling_parameters(self, contamination_rate: float = 0.05,
        downscaling_factor = 0.9) -> Dict:
        """
        Returns a dictionary with the number of samples to sample and the
        contamination rate according to the OAB paper's unsupervised
        sampling approach.

        :param contamination_rate: Contamination rate in the samples, defaults to 0.05
        :param downscaling_factor: Factor to downscale maximum sampling size
            in order to make sure that variation exists, defaults to 0.9

        :return: Dictionary with data for parameters `n` and `contamination_rate`
        """
        normals_restriction = self.n_normals / (1 - contamination_rate)
        anomalies_restriction = self.n_anomalies / contamination_rate
        n = int(downscaling_factor * min(normals_restriction, anomalies_restriction))
        return {'n': n, 'contamination_rate': contamination_rate}


    def sample_multiple_benchmark(self, contamination_rate: float = 0.05,
        downscaling_factor: float = 0.9, n_steps: int = 10,
        shuffle: bool = True, random_seed=42, apply_random_seed: bool = True,
        keep_frequency_ratio_normals: bool = False, equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False, equal_frequency_anomalies: bool = False,
        include_description: bool = True, yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True):
        """
        TODO

        :param contamination_rate: Contamination rate in the samples, defaults to 0.05
        :param downscaling_factor: Factor to downscale maximum sampling size
            in order to make sure that variation exists, defaults to 0.9
        :param n_steps: Number of sampled to take, i.e., number of times
            sampling is repeated, defaults to 10
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels, defaults to False
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

        :return: TODO
        """
        kwargs = {'contamination_rate': contamination_rate, 'downscaling_factor': downscaling_factor,
                  'n_steps': n_steps, 'shuffle': shuffle, 'random_seed': random_seed,
                  'apply_random_seed': apply_random_seed, 'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
                  'equal_frequency_normals': equal_frequency_normals, 'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
                  'equal_frequency_anomalies': equal_frequency_anomalies, 'include_description': include_description,
                  'flatten_images': flatten_images}
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "unsupervised_multiple_benchmark", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'unsupervised_multiple_benchmark': kwargs})
        # n_steps is not used with sample
        del kwargs['n_steps']
        # adapt kwargs to fit sample -> replace downscaling_factor by n
        kwargs['n'] = self.get_sampling_parameters(contamination_rate=contamination_rate,
            downscaling_factor=downscaling_factor)['n']
        del kwargs['downscaling_factor']
        # sample with specified parameters first
        yield self.sample(**kwargs)
        for _ in range(1, n_steps):
            # increase random seed by 1 to make sure sampling is actually the
            # same, even when an algorithm also uses a random call somewhere
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample(**kwargs)
