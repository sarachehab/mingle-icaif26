from typing import Optional
from mingle.utils.runner.loader.accessors.use import use_object
from mingle.utils.runner.loader.experiment import ExperimentLoader


def use_metadata_section(section: Optional[str] = None):
    def get_obj(loader: ExperimentLoader):
        metadata = loader.load_metadata()
        return metadata if section is None else getattr(metadata, section)
    return use_object(get_obj)


use_dataset_metadata = use_metadata_section("dataset")
