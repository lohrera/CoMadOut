from typing import Iterable, Any, List, Dict
from os.path import isfile
from pathlib import Path
from ruamel.yaml import YAML
import yaml

def _append_to_yaml(filepath: str, key: str, value: Any) -> None:
    """Helper to append a key-value pair to an existing yaml file.
    Note that this helper only adds key-value pairs at the topmost level of
    the yaml file.

    :param filepath: Path to YAML file
    :param key: Key for new entry
    :param value: Value for new entry
    """
    if not isfile(filepath):
        raise ValueError(f"{filepath} does not exist.")

    yaml=YAML(typ='rt')
    yaml_content = yaml.load(Path("./") / filepath)
    yaml_content[key] = value
    yaml.dump(yaml_content, Path("./") / filepath)


def _append_sampling_to_yaml(filepath: str, key: str, value: Any) -> None:
    """ Helper to append a sampling method to an existing yaml file.
    Note that the key is the sampling method, and this method can only be used
    to append something to the 'sampling' dictionary. The final structure is
    therefore yaml_content['sampling'][key] = value.

    :param filepath: Path to YAML file
    :param key: Key for sampling method, e.g., "unsupervised_multiple"
    :param value: Dictionary of sampling parameters
    """
    if not isfile(filepath):
        raise ValueError(f"{filepath} does not exist.")

    yaml=YAML(typ='rt')
    yaml_content = yaml.load(Path("./") / filepath)

    if 'sampling' in yaml_content:
        yaml_content['sampling'][key] = value
    else:
        yaml_content['sampling'] = {key: value}
    yaml.dump(yaml_content, Path("./") / filepath)



def _make_yaml(filepath: str, key: str, value: Any) -> None:
    """Helper to create a new YAML file with key-value pair.

    :param filepath: Path to YAML file
    :param key: Key for new entry
    :param value: Value for new entry
    """
    yaml=YAML(typ='rt')
    yaml_content = dict()
    yaml_content[key] = value
    yaml.dump(yaml_content, Path("./") / filepath)


def _read_from_yaml(filepath: str, keys: Iterable[str]) -> Any:
    """Helper to read from a YAML file.

    :param filepath: Path to YAML file
    :param keys: Iterable of keys applied to YAML file
    """
    if not isfile(filepath):
        raise ValueError(f"{filepath} does not exist.")

    yaml=YAML(typ='rt')
    yaml_content = yaml.load(Path("./") / filepath)
    for key in keys:
        if not (key in yaml_content):
            raise ValueError(f"Iterable of keys {keys} does not exist in {filepath}. More specifically, this occured when looking for {key} among {list(yaml_content.keys())}")
        yaml_content = yaml_content[key]
    yaml_content = dict(yaml_content)
    return yaml_content


def _get_dataset_dict(dataset_name: str) -> Dict:
    """ Load dictinary with information about the dataset from yaml.

    :param dataset_name: Name of the dataset to load information about

    :return: Dictionary with information about the dataset
    """
    with open(Path(__file__).parent / "datasets.yaml", "r") as f:
        yaml_dict = yaml.safe_load(f)
    try:
        return yaml_dict[dataset_name]
    except KeyError:
        raise KeyError(f"Dataset {dataset_name} is not among the available datasets.")
