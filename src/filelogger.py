import os
from pathlib import Path
from typing import Dict, Any, List

from src.environment import LoggerBackend
from src.logg import logg

logger = logg(__name__)


class FileLogger(LoggerBackend):
    def __init__(self, columns: List[str]):
        self.f = None
        self.columns = columns

    def log(self, data: Dict[str, Any]):
        if self.f is None:
            dir_path = Path('./job_output')
            dir_path.mkdir(exist_ok=True)
            path = (dir_path / 'results.csv').resolve()
            if path.exists():
                logger.warn(f'Path {path} already exists')
                raise FileExistsError(path)
            self.f = open(path, 'w')
            self.f.write(','.join(self.columns) + '\n')

        padded_columns = [str(data[c]) if c in data else '' for c in self.columns]
        self.f.write(','.join(padded_columns) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()




