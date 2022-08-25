import numpy as np

tensorflow_datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
mvtec_ad_suffixes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle',
    'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush',
    'transistor', 'zipper']
mvtec_ad_datasets = ["mvtec_ad_" + suffix for suffix in mvtec_ad_suffixes]
mvtec_ad_bw_datasets = ["mvtec_ad_" + suffix
                        for suffix in ['grid', 'screw', 'zipper']]
mvtec_ad_color_datasets = ["mvtec_ad_" + suffix
                           for suffix in ['carpet', 'leather', 'tile', 'wood',
                               'bottle', 'cable', 'capsule', 'hazelnut',
                               'metal_nut', 'pill', 'toothbrush', 'transistor']]
image_datasets = tensorflow_datasets + mvtec_ad_datasets + ['crack']


url_dict = {
    'carpet': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/carpet.tar.xz',
    'grid': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/grid.tar.xz',
    'leather': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/leather.tar.xz',
    'tile': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/tile.tar.xz',
    'wood': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/wood.tar.xz',
    'bottle': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/bottle.tar.xz',
    'cable': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/cable.tar.xz',
    'capsule': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/capsule.tar.xz',
    'hazelnut': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/hazelnut.tar.xz',
    'metal_nut': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/metal_nut.tar.xz',
    'pill': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/pill.tar.xz',
    'screw': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/screw.tar.xz',
    'toothbrush': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/toothbrush.tar.xz',
    'transistor': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/transistor.tar.xz',
    'zipper': 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/zipper.tar.xz',
}

reshape_dict = {
    'mnist': (28, 28),
    'fashion_mnist': (28, 28),
    'cifar10': (32, 32, 3),
    'cifar100': (32, 32, 3),
    'crack': (128, 128, 3),
    **{mvtec_ad_dataset: (256, 256, 3) for mvtec_ad_dataset in mvtec_ad_datasets},
    # overwrite black-white image dimensionality
    **{mvtec_ad_dataset: (256, 256) for mvtec_ad_dataset in mvtec_ad_bw_datasets}
}

def reshape_images(x: np.ndarray, dataset_name: str,
    flatten_images: bool = True) -> np.ndarray:
    """
    Reshapes images sampled from a dataset. This can either be flattening or
    reshaping them to the original shapes.

    :param x: X to be reshaped, of dimension (num_samples, ...) where ... is either
        flattened or the original shape
    :param dataset_name: Name of the dataset, used to identify its original
        dimensionality
    :param flatten_images: If set, images are flattened, otherwise reshaped to
        their original dimensionality, defaults to True

    :return: Sampled images with the specified shape
    """
    if not dataset_name in image_datasets:
        raise ValueError(f"Cannot reshape images from {dataset_name} as this" \
            " dataset is not stored as image dataset.")

    if flatten_images:
        dim = 1
        for new_dim in reshape_dict[dataset_name]:
            dim *= new_dim
        return x.reshape(x.shape[0], dim)
    else:
        return x.reshape(x.shape[0], *reshape_dict[dataset_name])
