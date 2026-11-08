"""Changeling Natio: your face as it might look born somewhere else in the world.

    python main.py                                  # the window
    python main.py photo.jpg --towards east_asian   # writes photo_east_asian.jpg
    python main.py photo.jpg --side-by-side         # all three next to you, writes photo_side_by_side.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

QUALITY_STEPS = {'fast': 0, 'balanced': 100, 'best': 250}


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(Path(path).suffix or '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f'Could not write image: {path}')
    buf.tofile(str(path))


def run_cli(args):
    from natio import Natio, REGIONS
    from window import side_by_side

    natio = Natio(args.device)
    steps = QUALITY_STEPS[args.quality] if args.steps is None else args.steps
    keep_gender = None if args.keep_gender == 'off' else args.keep_gender

    def progress(i, total, loss):
        print(f'\r  learning the face {i}/{total}  loss {loss:.4f}', end='', flush=True)

    for path in args.images:
        src = Path(path)
        print(f'{src.name}:')
        if args.side_by_side:
            mixes = [{}] + [{k: args.strength} for k in REGIONS]
            results = natio.change(read_rgb(src), None, steps, progress, mixes=mixes, keep_gender=keep_gender)
            image, tag = side_by_side(results), 'side_by_side'
        else:
            output = 'photo' if (args.whole_photo or args.keep_hair) else 'frame'
            r = natio.change(read_rgb(src), {args.towards: args.strength}, steps, progress, keep_hair=args.keep_hair,
                             output=output, keep_gender=keep_gender)
            image, tag = r.image, args.towards
        if steps:
            print()
        out = Path(args.output) if args.output and len(args.images) == 1 else src.with_name(f'{src.stem}_{tag}{src.suffix}')
        write_rgb(out, image)
        print(f'  saved {out}')


def run_ui(args):
    from natio import Natio
    from window import Window

    Window(Natio, args.device).run()


def main():
    parser = argparse.ArgumentParser(description='Changeling Natio: your face, born somewhere else (StyleGAN2 + e4e + PTI).')
    parser.add_argument('images', nargs='*', help='Input photo(s). Omit to open the window.')
    parser.add_argument('-o', '--output', help='Output path (single input only).')
    parser.add_argument('--towards', choices=['african', 'european', 'east_asian'], default='european')
    parser.add_argument('--strength', type=float, default=1.0, help='0.2 .. 1.5; default 1.')
    parser.add_argument('--side-by-side', action='store_true', help='All three next to the photo instead of one.')
    parser.add_argument('--keep-gender', choices=['auto', 'female', 'male', 'off'], default='auto')
    parser.add_argument('--quality', choices=list(QUALITY_STEPS), default='balanced')
    parser.add_argument('--steps', type=int, help='Override the number of face-learning steps.')
    parser.add_argument('--whole-photo', action='store_true', help='Paste back into the whole photo (default: the face frame).')
    parser.add_argument('--keep-hair', action='store_true', help='Whole photo, but only the face changes.')
    parser.add_argument('--device', help='cuda or cpu (default: cuda if available).')
    parser.add_argument('--download-models', action='store_true', help='Download any missing models and exit.')
    args = parser.parse_args()

    if args.download_models:
        from natio import weights
        weights.ensure_all()
    elif args.images:
        run_cli(args)
    else:
        run_ui(args)


if __name__ == '__main__':
    main()
