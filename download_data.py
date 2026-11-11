import argparse
import urllib.request
from pathlib import Path

from data_utils import ROOT, metadata, sha256, validate_source


def main():
    parser = argparse.ArgumentParser(description='Download and hash-check the nine original PADRE recordings')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/Range_data')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for record in metadata()['recordings']:
        destination = args.data_dir / record['file']
        if destination.exists():
            if sha256(destination) != record['sha256']:
                raise ValueError(f'Existing file has an unexpected hash: {destination}')
            print('VERIFIED', destination.name, flush=True)
            continue
        url = metadata()['download_base_url'] + record['file']
        temporary = destination.with_suffix('.csv.download')
        print('DOWNLOADING', record['file'], flush=True)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'DSFF-UAV'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha256(temporary) != record['sha256']:
                raise ValueError(f'Downloaded content differs from expected data: {record["file"]}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    validate_source(args.data_dir)
    print('All nine source files verified.')


if __name__ == '__main__':
    main()
