"""Download public example inputs and official weights; never uploads images."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FILES = (
    ('models/vgg19-dcbb9e9d.pth', 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'dcbb9e9d'),
    ('assets/portrait.png', 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png', None),
    ('assets/van_gogh.jpg', 'https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night.jpg', None),
    ('assets/picasso.jpg', 'https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/picasso_selfport1907.jpg', None),
)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    for name, url, prefix in FILES:
        path = ROOT / name
        if path.is_file() and (not prefix or digest(path).startswith(prefix)):
            print(f'Already available: {name}', flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {url}', flush=True)
        try:
            with urlopen(Request(url, headers={'User-Agent': 'AirBoard-style-experiment'}), timeout=60) as response, partial.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if prefix and not digest(partial).startswith(prefix):
                raise ValueError(f'Checksum mismatch: {name}')
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    print('Ready. Run: python experiments/neural_style/transfer.py')


if __name__ == '__main__':
    main()
