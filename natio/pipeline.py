"""Your face as it might look born somewhere else: align -> e4e invert -> (optional) pivotal tuning ->
latent edit along regional directions -> the edited face frame (or pasted back into the photo).

The directions are three coarse ones StyleGAN2 learned from FFHQ (African, European, East Asian; their
authors' file names were race_black / race_white / race_yellow). They are broad and blunt: whole regions of the
world (South Asia, the Middle East, the Americas, the Pacific...) have no direction of their own, and a mix of
the three only goes some way towards them. Each region has its own weight, and they add up.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .align import FaceAligner, paste_back
from .classify import GenderClassifier
from .encoder import Encoder4Editing
from .stylegan2 import Generator
from . import weights

# Facial structure and skin live in the coarse and middle W+ layers; the finest (colour of the light, the
# photo's own grain) stay the photo's.
EDIT_LAYERS = slice(0, 10)
REGIONS = {"african": "African", "european": "European", "east_asian": "East Asian"}
# Where faces sit along each direction (the mean of the W+ layers below, projected on the unit direction),
# measured on 27 generated portraits: MEMBER is about where people of that region sit, OTHER about where
# everyone else does. A change moves the face's own position towards the target's MEMBER value and the
# other two towards OTHER: a face already there barely moves (no overshoot), a face far away moves a lot.
MEMBER = {"african": 0.0, "european": 0.8, "east_asian": 0.3}
OTHER = {"african": -0.75, "european": -0.45, "east_asian": -0.3}
GAIN = 2.5               # the generator needs a bigger move than the spread of real faces to show it
EYES_PER_MOVE = 1.8      # eyes opened per unit moved towards East Asian (see edit())


@dataclass
class NatioResult:
    image: np.ndarray           # the edited face frame, or the whole photo with it pasted back
    aligned: np.ndarray         # aligned 1024x1024 input crop
    reconstruction: np.ndarray  # the generator's reconstruction of the crop (before editing)
    edited: np.ndarray          # aligned 1024x1024 edited crop
    mix: dict = field(default_factory=dict)   # region -> weight


def to_tensor(image, size, device):
    """RGB uint8 HxWx3 -> 1x3xSxS float in [-1, 1]."""
    t = torch.from_numpy(image).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode='area')
    return t


def to_image(tensor):
    """1x3xHxW in [-1, 1] -> RGB uint8 HxWx3."""
    t = tensor[0].detach().clamp(-1, 1).add(1).mul(127.5).round()
    return t.permute(1, 2, 0).byte().cpu().numpy()


class Natio:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        paths = weights.ensure_all()

        self.aligner = FaceAligner(str(paths['yunet']))

        ckpt = torch.load(paths['e4e'], map_location='cpu', weights_only=True)
        state = ckpt['state_dict']
        self.encoder = Encoder4Editing(1024)
        self.encoder.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')})
        self.generator = Generator(1024)
        self.generator.load_state_dict({k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')})
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        del ckpt, state

        # The encoder is ~1 GB of weights and runs once per photo, so it stays on the CPU;
        # the GPU is left for the generator and pivotal tuning (fits in 4 GB).
        self.encoder.eval()
        self.generator.eval().to(self.device).requires_grad_(False)

        self.directions = {}
        for key in REGIONS:
            d = torch.from_numpy(np.load(paths[key])[0]).float().to(self.device)
            self.directions[key] = d / d.norm()
        e = torch.from_numpy(np.load(paths['eyes_open'])[0]).float().to(self.device)
        self.eyes_direction = e / e.norm()
        g = torch.from_numpy(np.load(paths['gender'])[0]).float().to(self.device)
        self.gender_direction = g / g.norm()
        self.classifier = GenderClassifier(paths['gender_cls'])
        self._lpips = None

    SURE = 0.8                  # Auto trusts FairFace's reading of the gender only this sure
    KEEP_GENDER_STEP = 1.2
    KEEP_GENDER_TRIES = 6

    def _p(self, image, gender):
        seen, confidence = self.classifier(image)
        return confidence if seen == gender else 1 - confidence

    @torch.no_grad()
    def _keep_gender(self, w_edit, gender, p_before):
        """Nudge back along the gender direction until FairFace is about as sure of the original gender."""
        sign = -1.0 if gender == 'female' else 1.0
        want = min(0.95, p_before) - 0.05
        for _ in range(self.KEEP_GENDER_TRIES):
            if self._p(to_image(self.generator(w_edit)), gender) >= want:
                break
            w_edit = w_edit.clone()
            w_edit[:, EDIT_LAYERS] += sign * self.KEEP_GENDER_STEP * self.gender_direction
        return w_edit

    @torch.no_grad()
    def invert(self, aligned):
        w = self.encoder(to_tensor(aligned, 256, 'cpu'))
        return w.to(self.device) + self.latent_avg

    def position(self, w):
        """Where w sits along each direction: region -> value."""
        return {k: float((w[0, EDIT_LAYERS] @ d).mean()) for k, d in self.directions.items()}

    def edit(self, w, mix):
        """mix: region -> strength (0 .. 1.5). Each named region's position goes towards MEMBER; when one
        region is chosen, the others go towards OTHER, by the same strength."""
        if not any(mix.values()):
            return w.clone()
        here = self.position(w)
        total = sum(v for v in mix.values() if v)
        w = w.clone()
        moved = {}
        for key, d in self.directions.items():
            share = mix.get(key, 0) or 0
            goal = MEMBER[key] if share else OTHER[key]
            strength = share if share else total / max(1, sum(1 for v in mix.values() if v))
            moved[key] = strength * GAIN * (goal - here[key])
            w[:, EDIT_LAYERS] += moved[key] * d
        # The East Asian direction also learned near-closed eyes: a caricature, not a face. So the further a face
        # moves along it, the more the eyes are opened again (eyes_open's NEGATIVE side opens them; tried
        # by eye on test faces: -4 for a whole move of about 2.2 looks natural, more starts adding glasses).
        if moved.get("east_asian", 0) > 0:
            w[:, EDIT_LAYERS] -= EYES_PER_MOVE * moved["east_asian"] * self.eyes_direction
        return w

    def tune(self, w, aligned, steps, lr=3e-4, progress=None):
        """Pivotal Tuning (Roich et al. 2021): fine-tune the generator so that the e4e latent reproduces this
        exact face, so edits keep the person's identity. change() restores the pretrained weights afterwards."""
        if self._lpips is None:
            import warnings
            import lpips
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._lpips = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        gen = self.generator.requires_grad_(True)
        opt = torch.optim.Adam(gen.parameters(), lr=lr)
        target_256 = F.interpolate(to_tensor(aligned, 1024, self.device), size=(256, 256), mode='area')
        for i in range(steps):
            out_256 = F.interpolate(gen(w), size=(256, 256), mode='area')
            loss = self._lpips(out_256, target_256).mean() + F.mse_loss(out_256, target_256)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if progress:
                progress(i + 1, steps, loss.item())
            if loss.item() < 0.02:
                break
        gen.requires_grad_(False)

    def _align(self, image):
        landmarks = self.aligner.detect(image)
        if landmarks is None:
            raise ValueError('No face found in the image.')
        return self.aligner.align(image, landmarks, 1024)

    def change(self, image, mix, tune_steps=0, progress=None, mixes=None, keep_hair=False, output='frame',
               keep_gender='auto'):
        """The largest face in `image` (RGB uint8) moved by `mix` (region -> weight). With `mixes` (a list of
        mixes), one result per mix from a single tuning (for the side-by-side). output 'frame' (default): the
        edited face frame; 'photo': pasted back into the photo (keep_hair: only the face).
        keep_gender: 'female' / 'male' hold that gender; 'auto' only when FairFace is sure; None doesn't."""
        aligned, matrix = self._align(image)
        w = self.invert(aligned)
        gender, p_gender = self.classifier(aligned)
        if keep_gender in ('female', 'male'):
            if keep_gender != gender:
                gender, p_gender = keep_gender, 0.9
        elif keep_gender == 'auto' and p_gender < self.SURE:
            keep_gender = None

        pretrained = None
        if tune_steps > 0:
            pretrained = {k: v.cpu().clone() for k, v in self.generator.state_dict().items()}
        try:
            if pretrained is not None:
                self.tune(w, aligned, tune_steps, progress=progress)
            with torch.no_grad():
                recon = to_image(self.generator(w))
                results = []
                for m in (mixes if mixes is not None else [mix]):
                    w_edit = self.edit(w, m)
                    if keep_gender and any(m.values()):
                        w_edit = self._keep_gender(w_edit, gender, p_gender)
                    edited = to_image(self.generator(w_edit))
                    results.append(NatioResult(
                        image=edited if output == 'frame' else paste_back(image, edited, matrix, face_only=keep_hair),
                        aligned=aligned, reconstruction=recon, edited=edited, mix=dict(m)))
        finally:
            if pretrained is not None:
                self.generator.load_state_dict(pretrained)
                torch.cuda.empty_cache()
        return results if mixes is not None else results[0]
