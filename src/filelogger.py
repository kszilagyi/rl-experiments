import bz2
from typing import Dict, Any, List

from src.environment import LoggerBackend
from src.logg import logg, OUTPUT_DIR

logger = logg(__name__)


class FileLogger(LoggerBackend):
    def __init__(self, columns: List[str]):
        self.f = None
        self.columns = columns

    def log(self, data: Dict[str, Any]):
        if self.f is None:
            path = (OUTPUT_DIR / 'results.csv.bz2').resolve()
            if path.exists():
                logger.warn(f'Path {path} already exists')
                raise FileExistsError(path)
            self.f = bz2.open(path, 'w')
            self.f.write((';'.join(self.columns) + '\n').encode('utf-8'))

        padded_columns = [str(data[c]) if c in data else '' for c in self.columns]
        self.f.write((';'.join(padded_columns) + '\n').encode('utf-8'))
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()




