import json
import traceback
from multiprocessing.pool import ThreadPool
from pathlib import Path

from google.cloud import storage


def download_status(blob):
    if blob.name.endswith('result.json'):
        file = './tmp/' + blob.name
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file, 'wb') as f:
                blob.download_to_file(f)
        except:
            traceback.print_exc()


def main(batch_name):
    storage_client = storage.Client(project='rl-experiments-296208')
    blobs = list(storage_client.list_blobs('rl-experiments', prefix=batch_name))
    with ThreadPool(100) as p:
        for _ in p.imap_unordered(download_status, blobs):
            pass # wait for each element
    success = 0
    for b in blobs:
        if b.name.endswith('result.json'):
            with open('./tmp/' + b.name) as f:
                result = json.load(f)
                if result['status'] != 'SUCCESS':
                    print(b.name)
                else:
                    success += 1

    print(success)

if __name__ == '__main__':
    main('2021-01-03-14-48-52_06-cont-environments.py_9e8726f6e06f')