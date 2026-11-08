"""Pretrained weight locations and first-run download (Changeling Natio).

Changeling Tiresias and Chronos use the same e4e checkpoint, face detector and classifier, so before downloading a
file this looks in their folders first and uses theirs: 1.3 GB isn't fetched again.
"""
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / 'pretrained'
SIBLINGS = [WEIGHTS_DIR.parent.parent / name / 'pretrained' for name in ('Changeling-Tiresias', 'Changeling-Chronos', 'Gender-Swap')]

FILES = {
    # e4e encoder + StyleGAN2 FFHQ 1024 generator in one checkpoint (Tov et al., 2021).
    'e4e': ('e4e_ffhq_encode.pt',
            'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/'
            'pretrained_models/encoder4editing/e4e_ffhq_encode.pt'),
    # Directions in StyleGAN2 FFHQ W+ space (18x512) from the same set as Tiresias's and Chronos's. Their authors
    # named them race_black / race_white / race_yellow; Natio calls them African, European and East Asian.
    'african': ('race_black.npy', 'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
                'master/latent_directions/race_black.npy'),
    'european': ('race_white.npy', 'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
                 'master/latent_directions/race_white.npy'),
    'east_asian': ('race_yellow.npy', 'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
                   'master/latent_directions/race_yellow.npy'),
    # How open the eyes are: held where the photo had it (the regional directions also learned narrower or wider eyes).
    'eyes_open': ('eyes_open.npy', 'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
                  'master/latent_directions/eyes_open.npy'),
    # Gender direction (the same file Tiresias swaps along): Natio holds the face's gender with it.
    'gender': ('gender.npy',
               'https://raw.githubusercontent.com/a312863063/generators-with-stylegan2/'
               'master/latent_directions/gender.npy'),
    # FairFace gender classifier (the same one Tiresias uses): Natio keeps the face's gender with it.
    'gender_cls': ('fairface_gender_int8.onnx',
                   'https://huggingface.co/onnx-community/fairface_gender_image_detection-ONNX/'
                   'resolve/main/onnx/model_int8.onnx'),
    # OpenCV YuNet face detector with 5-point landmarks.
    'yunet': ('face_detection_yunet_2023mar.onnx',
              'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
              'models/face_detection_yunet/face_detection_yunet_2023mar.onnx'),
}


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'Downloading {dest.name} ...')

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f'\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    tmp.replace(dest)


def ensure_all():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    paths = {}
    for key, (name, url) in FILES.items():
        path = WEIGHTS_DIR / name
        if not path.exists():
            shared = next((d / name for d in SIBLINGS if (d / name).exists()), None)
            if shared is not None:
                print(f'Using {name} from {shared.parent}')
                path = shared
            else:
                _download(url, path)
        paths[key] = path
    return paths
