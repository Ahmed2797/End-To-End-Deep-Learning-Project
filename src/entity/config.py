from dataclasses import dataclass 
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Dataclass for storing the configuration required to download and extract data.

    Attributes:
        root_dir (Path): Base directory for data artifacts.
        source_url (str): URL from which to download the data.
        local_data_file (Path): Local path where the data will be saved.
        unzip_dir (Path): Directory where the data will be extracted.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

