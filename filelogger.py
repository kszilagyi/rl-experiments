import os
from pathlib import Path
from typing import Dict, Any, List

from environment import LoggerBackend
from logg import logg

logger = logg(__name__)


class FileLogger(LoggerBackend):
    def __init__(self, directory_key, file_key, columns: List[str]):
        self.directory_key = directory_key
        self.file_key = file_key
        self.f = None
        self.columns = columns

    def log(self, data: Dict[str, Any]):
        if self.f is not None:
            dir_path = Path('./' + data[self.directory_key])
            os.mkdir(dir_path)
            path = (dir_path / data[self.file_key]).resolve()
            if path.exists():
                logger.warn(f'Path {path} already exists')
                exit(1)
            self.f = open(path, 'w')
            self.f.write(','.join(self.columns) + '\n')

        padded_columns = [data[c] if c in data else '' for c in self.columns]
        self.f.write(','.join(padded_columns) + '\n')

    def close(self):
        self.f.close()




