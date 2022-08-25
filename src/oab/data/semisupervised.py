import numpy as np
import pandas as pd

from typing import Union, Tuple, Optional
from dataclasses import dataclass

# example-pkg-jd-kiel==0.0.37 inactive on 20220815 (temporarily only required oab source imported from https://github.com/ISDM-CAU-Kiel/oab instead)
from src.oab.data.abstract_classes import AnomalyDataset, AnomalyDatasetDescription
from src.oab.data.utils import _append_to_yaml, _make_yaml
from src.oab.data.utils_image import image_datasets, reshape_images


@dataclass
class SemisupervisedAnomalyDatasetDescription(AnomalyDatasetDescription):
    """Description object for a sample from a semisupervised anomaly dataset.

    :param name: Name of the dataset
    :param normal_labels: List of normal labels
    :param anomaly_labels: List of anomalous labels
    :param number_instances_training: Number of instances in the training set
    :param number_instances_test: Number of instances in the test set
    :param training_number_normals: Number of normal data points in the training set
    :param training_number_anomalies: Number of anomalous data points in the training set
    :param training_contamination_rate: Contamination rate in the training set
    :param test_number_normals: Number of normal data points in the test set
    :param test_number_anomalies: Number of anomalous data points in the test set
    :param test_contamination_rate: Contamination rate in the test set
    """
    name: str

    normal_labels: list()
    anomaly_labels: list()

    number_instances_training: int
    number_instances_test: int

    training_number_normals: int
    training_number_anomalies: int
    training_contamination_rate: float

    test_number_normals: int
    test_number_anomalies: int
    test_contamination_rate: float



    def from_same_dataset(self, other: 'SemisupervisedAnomalyDatasetDescription') -> bool:
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
                               self.number_instances_training,
                               self.number_instances_test,
                               )
            other_attributes = (other.name,
                                other.number_instances_training,
                                other.number_instances_test,
                                )
        else:
            self_attributes = (self.name,
                               self.number_instances_training,
                               self.number_instances_test,
                               set(self.normal_labels),
                               set(self.anomaly_labels))
            other_attributes = (other.name,
                                other.number_instances_training,
                                other.number_instances_test,
                                set(other.normal_labels),
                                set(other.anomaly_labels))
        float_self_attributes = (self.training_contamination_rate,
                                 self.test_contamination_rate)
        float_other_attributes = (other.training_contamination_rate,
                                  other.test_contamination_rate)
        float_comparison = np.all(np.isclose(float_self_attributes, float_other_attributes))
        return (self_attributes == other_attributes) and float_comparison


    def print_for_eval_specifics(self) -> str:
        """Return pretty string representation of most important dataset characteristics.

        :return: String with the most important dataset characteristics
        """
        return (f"{self.number_instances_training} training instances, " \
                f"{self.number_instances_test} test instances, " \
                f"training contamination rate {self.training_contamination_rate}, " \
                f"test contamination rate {self.test_contamination_rate}.")



class SemisupervisedAnomalyDataset(AnomalyDataset):
    """This class represents a semisupervised anomaly dataset, i.e., when
    sampling from the dataset, an array of (clean) values is returned for
    training, and a tuple for testing with values and labels of the observation
    are returned.
    """


    def sample(self, n_training: int, n_test: int,
               training_contamination_rate: float = 0,
               test_contamination_rate: float = 0.2,
               shuffle: bool = True, random_seed: float = 42,
               apply_random_seed: bool =True,
               training_keep_frequency_ratio_normals: bool = False,
               training_equal_frequency_normals: bool = False,
               training_keep_frequency_ratio_anomalies: bool = False,
               training_equal_frequency_anomalies: bool = False,
               test_keep_frequency_ratio_normals: bool = False,
               test_equal_frequency_normals: bool = False,
               test_keep_frequency_ratio_anomalies: bool = False,
               test_equal_frequency_anomalies: bool = False,
               include_description: bool = True, flatten_images: bool = True):
        """Sample from the anomaly dataset.
        Note that this method does not ensure that data points seen during
        training are resampled during testing. If possible, use other methods
        like sample_with_training_split or sample_with_explicit_numbers
        that ensure this.

        :param n_training: Number of training data points to sample
        :param n_test: Number of test data points to sample
        :param training_contamination_rate: Contamination rate when sampling
            training points, defaults to 0
        :param test_contamination_rate: Contamination rate when sampling
            test points, defaults to 0
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param training_keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling for training. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points for training are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param training_equal_frequency_normals: If there are multiple normal labels and
             this is set, training data will be sampled with an equal distribution
             among these normal labels, defaults to False
        :param training_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in training,
            defaults to False
        :param training_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in training,
            defaults to False
        :param test_keep_frequency_ratio_normals: Like parameter
            `training_keep_frequency_ratio_normals` for normals in test data,
            defaults to False
        :param test_equal_frequency_normals: Like parameter
            `training_equal_frequency_normals` for normals in test data,
            defaults to False
        :param test_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in test data,
            defaults to False
        :param test_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in test data,
            defaults to False
        :include_description: Includes sampling config file, defaults to True
        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple (x_train, x_test, y_test), sample_config. y_test and
            sample_config can be used to pass to the EvaluationObject.
        """
        self._test_contamination_rate(training_contamination_rate)
        self._test_contamination_rate(test_contamination_rate)
        if apply_random_seed:
            np.random.seed(random_seed)

        # # training set - clean
        # if np.isclose(0, training_contamination_rate):
        #     training_values, _, training_original_labels = \
        #         self._sample_data('normals', n_training, training_keep_frequency_ratio_normals,
        #                           training_equal_frequency_normals)
        #     n_training_normals = n_training
        #     n_training_anomalies = 0

        # training set
        # else:
        n_training_normals = int(n_training * (1 - training_contamination_rate))
        n_training_anomalies = n_training - n_training_normals

        training_values_normals, _, training_original_labels_normals = \
            self._sample_data('normals', n_training_normals, training_keep_frequency_ratio_normals,
                              training_equal_frequency_normals)
        training_values_anomalies, _, training_oringinal_labels_anomalies = \
            self._sample_data('anomalies', n_training_anomalies, training_keep_frequency_ratio_anomalies,
                              training_equal_frequency_anomalies)

        training_values = np.vstack((training_values_normals, training_values_anomalies))

        # test set
        n_test_normals = round(n_test * (1 - test_contamination_rate))
        n_test_anomalies = n_test - n_test_normals

        test_values_normals, test_labels_normals, test_original_labels_normals = \
            self._sample_data('normals', n_test_normals, test_keep_frequency_ratio_normals,
                              test_equal_frequency_normals)
        test_values_anomalies, test_labels_anomalies, test_original_labels_anomalies = \
            self._sample_data('anomalies', n_test_anomalies, test_keep_frequency_ratio_anomalies,
                              test_equal_frequency_anomalies)

        test_values = np.vstack((test_values_normals, test_values_anomalies))
        test_labels = np.hstack((test_labels_normals, test_labels_anomalies))

        # data description
        description = SemisupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances_training=n_training,
            number_instances_test=n_test,
            training_number_normals=n_training_normals,
            training_number_anomalies=n_training_anomalies,
            training_contamination_rate=training_contamination_rate,
            test_number_normals=n_test_normals, test_number_anomalies=n_test_anomalies,
            test_contamination_rate=test_contamination_rate)

        # shuffle
        if shuffle:
            # training set
            training_idxs = np.arange(n_training)
            np.random.shuffle(training_idxs)
            training_values = training_values[training_idxs]
            # test set
            test_idxs = np.arange(n_test)
            np.random.shuffle(test_idxs)
            test_values = test_values[test_idxs]
            test_labels = test_labels[test_idxs]

        # potentially reshape values if dealing with image data
        if self.classification_dataset.name in image_datasets:
            training_values = reshape_images(training_values,
                self.classification_dataset.name, flatten_images)
            test_values = reshape_images(test_values,
                self.classification_dataset.name, flatten_images)

        return (training_values, test_values, test_labels), description


    def sample_with_explicit_numbers(self, training_normals: int,
            training_anomalies: int, test_normals: int, test_anomalies: int,
            shuffle: bool = True, random_seed : float = 42,
            apply_random_seed: bool = True,
            keep_frequency_ratio_normals: bool = False,
            equal_frequency_normals: bool = False,
            keep_frequency_ratio_anomalies: bool = False,
            equal_frequency_anomalies: bool = False,
            include_description: bool = True,
            yamlpath_append: Optional = None, yamlpath_new: Optional = None,
            flatten_images: bool = True):
        """
        Sample specified number of points from the anomaly dataset.
        Note that both normal and anomaly points cannot occur in both
        training and test data.

        :param training_normals: Number of normal data points in the training set
        :param training_anomalies: Number of anomalies in the training set
        :param test_normals: Number of normal data points in the test set
        :param test_anomalies: Number of anomalies in the test set
        :param shuffle: Shuffle the training and test set, defaults to True
        :param random_seed: Seed for random number generator, defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param include_description: Whether or not to include a description of the
            sampled dataset, defaults to True
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

        :return: A tuple of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """
        kwargs = {'training_normals': training_normals,
            'training_anomalies': training_anomalies,
            'test_normals': test_normals, 'test_anomalies': test_anomalies,
            'shuffle': shuffle, 'random_seed': random_seed,
            'apply_random_seed': apply_random_seed,
            'include_description': include_description,
            'flatten_images': flatten_images
            }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_explicit_numbers_single", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_explicit_numbers_single': kwargs})

        if apply_random_seed:
            np.random.seed(random_seed)

        n_normals = training_normals + test_normals
        n_anomalies = training_anomalies + test_anomalies

        n_training = training_normals + training_anomalies
        n_test = test_normals + test_anomalies

        # sample data
        values_normals, labels_normals, original_labels_normals = \
            self._sample_data('normals', n_normals,
                keep_frequency_ratio=keep_frequency_ratio_normals,
                equal_frequency=equal_frequency_normals)
        values_anomalies, labels_anomalies, oringinal_labels_anomalies = \
            self._sample_data('anomalies', n_anomalies,
                keep_frequency_ratio=keep_frequency_ratio_anomalies,
                equal_frequency=equal_frequency_anomalies)

        # build training and test set
        training_values = np.vstack((values_normals[:training_normals],
                                     values_anomalies[:training_anomalies]))
        test_values = np.vstack((values_normals[training_normals:],
                                 values_anomalies[training_anomalies:]))
        test_labels = np.hstack((labels_normals[training_normals:],
                                 labels_anomalies[training_anomalies:]))


        # data description
        description = SemisupervisedAnomalyDatasetDescription(name=self.classification_dataset.name,
            normal_labels=self.normal_labels, anomaly_labels=self.anomaly_labels,
            number_instances_training=training_normals + training_anomalies,
            number_instances_test=test_normals + test_anomalies,
            training_number_normals=training_normals,
            training_number_anomalies=training_anomalies,
            training_contamination_rate=training_anomalies/n_training,
            test_number_normals=test_normals,
            test_number_anomalies=test_anomalies,
            test_contamination_rate=test_anomalies/n_test)

        # shuffle
        if shuffle:
            # training set
            training_idxs = np.arange(n_training)
            np.random.shuffle(training_idxs)
            training_values = training_values[training_idxs]
            # test set
            test_idxs = np.arange(n_test)
            np.random.shuffle(test_idxs)
            test_values = test_values[test_idxs]
            test_labels = test_labels[test_idxs]

        # potentially reshape values if dealing with image data
        if self.classification_dataset.name in image_datasets:
            training_values = reshape_images(training_values,
                self.classification_dataset.name, flatten_images)
            test_values = reshape_images(test_values,
                self.classification_dataset.name, flatten_images)

        return (training_values, test_values, test_labels), description


    def sample_multiple(self, n_training: int, n_test: int, n_steps: int = 10,
               training_contamination_rate: float = 0,
               test_contamination_rate: float = 0.2,
               shuffle: bool = True, random_seed: float = 42,
               apply_random_seed: bool =True,
               training_keep_frequency_ratio_normals: bool = False,
               training_equal_frequency_normals: bool = False,
               training_keep_frequency_ratio_anomalies: bool = False,
               training_equal_frequency_anomalies: bool = False,
               test_keep_frequency_ratio_normals: bool = False,
               test_equal_frequency_normals: bool = False,
               test_keep_frequency_ratio_anomalies: bool = False,
               test_equal_frequency_anomalies: bool = False,
               include_description: bool = True, yamlpath_append: Optional = None,
               yamlpath_new: Optional = None, flatten_images: bool = True):
        """
        Sample multiple times from the anomaly dataset as an iterator.
        Note that this method does not ensure that data points seen during
        training are resampled during testing. If possible, use other methods
        like sample_with_training_split or sample_with_explicit_numbers
        that ensure this.

        :param n_training: Number of training data points to sample
        :param n_test: Number of test data points to sample
        :param n_steps: Number of sampled to take, i.e., number of times
            sampling is repeated, defaults to 10
        :param training_contamination_rate: Contamination rate when sampling
            training points, defaults to 0
        :param test_contamination_rate: Contamination rate when sampling
            test points, defaults to 0
        :param shuffle: Shuffle dataset, defaults to True
        :param random_seed: Random seed, defaults to 42
        :param apply_random_seed: Apply random seed to make sampling reproducible, defaults to True
        :param training_keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling for training. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points for training are sampled, 200 of them will be from
            label 0 and 100 from label 1. Defaults to False
        :param training_equal_frequency_normals: If there are multiple normal labels and
             this is set, training data will be sampled with an equal distribution
             among these normal labels, defaults to False
        :param training_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in training,
            defaults to False
        :param training_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in training,
            defaults to False
        :param test_keep_frequency_ratio_normals: Like parameter
            `training_keep_frequency_ratio_normals` for normals in test data,
            defaults to False
        :param test_equal_frequency_normals: Like parameter
            `training_equal_frequency_normals` for normals in test data,
            defaults to False
        :param test_keep_frequency_ratio_anomalies: Like parameter
            `training_keep_frequency_ratio_normals` for anomalies in test data,
            defaults to False
        :param test_equal_frequency_anomalies: Like parameter
            `training_equal_frequency_normals` for anomalies in test data,
            defaults to False
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

        :return: An iterator of A tuple (x_train, x_test, y_test), sample_config.
        """
        kwargs = {'n_training': n_training, 'n_test': n_test,
                  'training_contamination_rate': training_contamination_rate,
                  'training_keep_frequency_ratio_normals': training_keep_frequency_ratio_normals,
                  'training_equal_frequency_normals': training_equal_frequency_normals,
                  'training_keep_frequency_ratio_anomalies': training_keep_frequency_ratio_anomalies,
                  'training_equal_frequency_anomalies': training_equal_frequency_anomalies,
                  'test_contamination_rate': test_contamination_rate,
                  'test_keep_frequency_ratio_normals': test_keep_frequency_ratio_normals,
                  'test_equal_frequency_normals': test_equal_frequency_normals,
                  'test_keep_frequency_ratio_anomalies': test_keep_frequency_ratio_anomalies,
                  'test_equal_frequency_anomalies': test_equal_frequency_anomalies,
                  'shuffle': shuffle, 'random_seed': random_seed,
                  'apply_random_seed': apply_random_seed, 'include_description': include_description,
                  'flatten_images': flatten_images
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_multiple", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_multiple': kwargs})
        # sample with specified parameters first
        yield self.sample(**kwargs)
        for _ in range(1, n_steps):
            # increase random seed by 1 to make sure sampling is actually the
            # same, even when an algorithm also uses a random call somewhere
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample(**kwargs)


    def sample_with_training_split(self, training_split: float,
        max_contamination_rate: float, random_seed : float =42,
        apply_random_seed : bool = True,
        keep_frequency_ratio_normals: bool = False,
        equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False,
        equal_frequency_anomalies: bool = False,
        yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True):
        """
        Sample from a semisupervised anomaly dataset by specifying the split
        of normal data points used during training and a maximum contamination
        rate of the test set.

        :param training_split: Specifies the proportion of normal data points
            that will be used during training
        :param max_contamination_rate: Maximum contamination rate of the test
            set. If this is exceeded, not all anomalies that exist are sampled
        :param random_seed: Seed for random number generator, defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
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

        :return: A tuple of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """
        kwargs = {'training_split': training_split,
            'max_contamination_rate': max_contamination_rate,
            'random_seed': random_seed, 'apply_random_seed': apply_random_seed,
            'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
            'equal_frequency_normals': equal_frequency_normals,
            'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
            'equal_frequency_anomalies': equal_frequency_anomalies,
            'flatten_images': flatten_images,
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_training_split_single", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_training_split_single': kwargs})

        training_normals, test_normals, test_anomalies = \
            self._compute_samples_in_training_and_testing(training_split=training_split,
            max_contamination_rate=max_contamination_rate)

        return self.sample_with_explicit_numbers(training_normals=training_normals,
            training_anomalies=0, test_normals=test_normals,
            test_anomalies=test_anomalies, shuffle=True, random_seed=random_seed,
            apply_random_seed=apply_random_seed,
            keep_frequency_ratio_normals=keep_frequency_ratio_normals,
            equal_frequency_normals=equal_frequency_normals,
            keep_frequency_ratio_anomalies=keep_frequency_ratio_anomalies,
            equal_frequency_anomalies=equal_frequency_anomalies,
            include_description=True,
            flatten_images=flatten_images)


    def sample_multiple_with_training_split(self, training_split: float,
        max_contamination_rate: float, n_steps: int =10, random_seed : float =42,
        apply_random_seed : bool = True,
        keep_frequency_ratio_normals: bool = False,
        equal_frequency_normals: bool = False,
        keep_frequency_ratio_anomalies: bool = False,
        equal_frequency_anomalies: bool = False,
        yamlpath_append: Optional = None,
        yamlpath_new: Optional = None, flatten_images: bool = True):
        """
        Sample multiple times from a semisupervised anomaly dataset by specifying the split
        of normal data points used during training and a maximum contamination
        rate of the test set.

        :param training_split: Specifies the proportion of normal data points
            that will be used during training
        :param max_contamination_rate: Maximum contamination rate of the test
            set. If this is exceeded, not all anomalies that exist are sampled
        :param n_steps: Number of samples to be taken
        :param random_seed: Seed for random number generator in the first sample,
            defaults to 42
        :param apply_random_seed: Whether or not to apply random seed, defaults to True
        :param keep_frequency_ratio_normals: If there are multiple normal labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if normal labels are [0, 1] and there
            are 2000 data points with label 0, 1000 data points with label 1 and
            300 normal data points are sampled, 200 of them will be from
            label 0 and 100 from label 1. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_normals: If there are multiple normal labels and
             this is set, data will be sampled with an equal distribution
             among these normal labels. Note that the split of these values
             between train and test set is at random. Defaults to False
        :param keep_frequency_ratio_anomalies: If there are multiple anomaly labels and
            this is set, the frequency ratio of these labels will be maintained
            during sampling. For example, if anomaly labels are [2, 3] and there
            are 2000 data points with label 2, 1000 data points with label 3 and
            300 anomalous data points are sampled, 200 of them will be from
            label 2 and 100 from label 3. Note that the split of these values
            between train and test set is at random. Defaults to False
        :param equal_frequency_anomalies: If there are multiple anomaly labels and
             this is set, data will be sampled with an equal distribution
            among these anomaly labels. Note that the split of these values
            between train and test set is at random. Defaults to False
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

        :return: An iterator of tuples of (training_values, test_values, test_labels), description
            where values and labels are a numpy.ndarray
        """
        kwargs = {'training_split': training_split,
            'max_contamination_rate': max_contamination_rate, 'n_steps': n_steps,
            'random_seed': random_seed, 'apply_random_seed': apply_random_seed,
            'keep_frequency_ratio_normals': keep_frequency_ratio_normals,
            'equal_frequency_normals': equal_frequency_normals,
            'keep_frequency_ratio_anomalies': keep_frequency_ratio_anomalies,
            'equal_frequency_anomalies': equal_frequency_anomalies,
            'flatten_images': flatten_images,
        }
        # store arguments in YAML if specified
        if yamlpath_append: # append arguments to YAML
            _append_sampling_to_yaml(yamlpath_append, "semisupervised_training_split_multiple", kwargs)
        if yamlpath_new: # create new YAML with arguments
            _make_yaml(yamlpath_new, "sampling", {'semisupervised_training_split_multiple': kwargs})

        del kwargs['n_steps']
        yield self.sample_with_training_split(**kwargs)
        for _ in range(1, n_steps):
            kwargs['random_seed'] = kwargs['random_seed'] + 1
            yield self.sample_with_training_split(**kwargs)

    def sample_original_mvtec_split(self, flatten_images: bool = True):
        """
        Samples with the train-test split from the original publication of
        MVTec AD. This is included to allow reproducing experiments that follow
        this original train-test split.

        :param flatten_images: This flag has behavior only if an image dataset
            is loaded. If this is the case and this is set to True, the
            image will be flattened. If it is set to False, the original dimensionality
            of the image will be used, e.g., 32x32x3. Defaults to True

        :return: A tuple (x_train, x_test, y_test), sample_config. y_test and
            sample_config can be used to pass to the EvaluationObject.
        """
        # only mvtec_ad datasets store which splits were originally used
        # in the dataset publication
        if not hasattr(self.classification_dataset, 'x_normal_test_mask'):
            raise Error(f"This dataset is not loaded as MVTec AD dataset "\
                f"and therefore doesn't provide this functionality.")

        x_normal_test_mask = self.classification_dataset.x_normal_test_mask.astype('bool')
        normals = len(x_normal_test_mask)
        x_normal_train_mask = ~x_normal_test_mask
        x_train = self.classification_dataset.values[:normals][x_normal_train_mask]
        x_test_normals = self.classification_dataset.values[:normals][x_normal_test_mask]
        x_test_anomalies = self.classification_dataset.values[normals:]
        x_test = np.vstack((x_test_normals, x_test_anomalies))
        y = np.hstack((np.zeros(len(x_test_normals)), np.ones(len(x_test_anomalies))))

        sampling_config = SemisupervisedAnomalyDatasetDescription(
            name=self.classification_dataset.name,
            normal_labels=[0], anomaly_labels=[1],
            number_instances_training=len(x_train),
            number_instances_test=len(x_test),
            training_number_normals=len(x_train),
            training_number_anomalies=0,
            training_contamination_rate=0,
            test_number_normals=len(x_test_normals),
            test_number_anomalies=len(x_test_anomalies),
            test_contamination_rate=len(x_test_anomalies)/len(x_test),
        )

        # reshape images
        x_train = reshape_images(x_train,
            dataset_name=self.classification_dataset.name,
            flatten_images=flatten_images)
        x_test = reshape_images(x_test,
            dataset_name=self.classification_dataset.name,
            flatten_images=flatten_images)

        return (x_train, x_test, y), sampling_config




    def _compute_samples_in_training_and_testing(self, training_split: float,
        max_contamination_rate: float) -> Tuple[int, int, int]:
        """
        Helper that computes how many normal data points are needed for training
        data and how many normal and anomalous data points are needed for
        testing with the specified parameters of training split and maximum
        contamination rate.
        """
        n_normals = len(self.normal_idxs)
        n_anomalies = len(self.anomaly_idxs)

        training_normals = int(training_split * n_normals)
        test_normals = n_normals - training_normals
        max_test_size = test_normals + n_anomalies # if all anomalies are sampled

        if n_anomalies / max_test_size > max_contamination_rate: # contamination rate too large if all anomalies are sampled
            test_anomalies = int(max_contamination_rate * test_normals / (1 - max_contamination_rate))

        else:
            test_anomalies = n_anomalies

        return (training_normals, test_normals, test_anomalies)
