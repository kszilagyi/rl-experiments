import argparse

from google.cloud import storage

from src.logg import logg

logger = logg(__name__)

def main():
    parser = argparse.ArgumentParser('Run online')
    parser.add_argument('--batch_name', type=str, required=True)
    args = parser.parse_args()
    logger.info(args)
    batch_name = args.batch_name
    storage_client = storage.Client()
    logger.info('Creating job params files')
    for blob in storage_client.list_blobs('rl-experiments', prefix=batch_name):
        if blob.name.endswith('results.csv.bz2'):
            print(blob.name)


if __name__ == '__main__':
    main()