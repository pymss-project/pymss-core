"""Golden parity anchors: python tests/parity.py record|verify

Records state_dict + torch output (+ MLX output where a backend exists) for a
minimal build of every model family. Refactors must load the recorded
state_dict with strict=True (param-name/shape frozen surface) and reproduce
outputs. Anchors live in tests/anchors/ (not shipped).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ANCHOR_DIR = Path(__file__).parent / "anchors"


def _roformer_kwargs(**kw):
    base = dict(dim=8, depth=1, stereo=False, num_stems=1, heads=1, dim_head=4, flash_attn=False,
                stft_n_fft=16, stft_hop_length=4, stft_win_length=16, mask_estimator_depth=1, skip_connection=True)
    base.update(kw)
    return base


def _attr(**kw):
    from pymss_core import AttrDict
    return AttrDict({k: AttrDict(v) if isinstance(v, dict) else v for k, v in kw.items()})


def cases():
    from pymss_core.modules.bs_roformer import (BSConformer, BSRoformer, BSRoformerHyperACE, MelBandConformer,
                                                MelBandRoformer)
    from pymss_core.modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
    from pymss_core.modules.bandit_v2.bandit import Bandit
    from pymss_core.modules.look2hear.apollo import Apollo
    from pymss_core.modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
    from pymss_core.modules.demucs4ht import get_model as htdemucs_get_model
    from pymss_core.modules.scnet import SCNet
    from pymss_core.modules import mdx23c_mlx, bandit_mlx, scnet_mlx, apollo_mlx, demucs_mlx
    from pymss_core.modules.bs_roformer import mlx_roformer
    from pymss_core.modules.legacy_demucs import LegacyConvTasNet, LegacyDemucs, LegacyHDemucs, LegacyV3Demucs

    bs_cfg = _roformer_kwargs(freqs_per_bands=(1,) * 9)
    mel_cfg = _roformer_kwargs(num_bands=4, sample_rate=16000)
    conf_cfg = dict(bs_cfg, time_conformer_depth=1, freq_conformer_depth=1)
    mdx_config = _attr(
        audio=dict(n_fft=2048, hop_length=512, dim_f=1024, num_channels=2),
        model=dict(norm="InstanceNorm", act="gelu", num_subbands=4, num_scales=1, scale=[2, 2],
                   num_blocks_per_scale=1, num_channels=8, growth=4, bottleneck_factor=4),
        training=dict(instruments=["vocals"], target_instrument="vocals"),
    )
    ht_config = _attr(
        model="htdemucs",
        training=dict(instruments=["vocals"], channels=2, samplerate=44100, segment=6),
        htdemucs=dict(channels=8, depth=2, nfft=1024, kernel_size=8, stride=4, t_layers=1, t_heads=2,
                      bottom_channels=0, use_train_segment=False),
    )

    def build_hyperace():
        return BSRoformerHyperACE(**bs_cfg)

    out = {
        "bs_roformer": (lambda: BSRoformer(**bs_cfg), (1, 1, 64), mlx_roformer.mlx_forward_roformer_mx),
        "mel_band_roformer": (lambda: MelBandRoformer(**mel_cfg), (1, 1, 64), mlx_roformer.mlx_forward_roformer_mx),
        "bs_conformer": (lambda: BSConformer(**conf_cfg), (1, 1, 64), mlx_roformer.mlx_forward_roformer_mx),
        "mel_band_conformer": (lambda: MelBandConformer(**mel_cfg, time_conformer_depth=1, freq_conformer_depth=1),
                               (1, 1, 64), mlx_roformer.mlx_forward_roformer_mx),
        "bs_roformer_hyperace": (build_hyperace, (1, 1, 64), mlx_roformer.mlx_forward_roformer_mx),
        "bandit": (lambda: MultiMaskMultiSourceBandSplitRNNSimple(
            in_channel=1, stems=["vocals"], band_specs="musical", fs=44100, n_bands=64, n_fft=2048, hop_length=512,
            n_sqm_modules=2, emb_dim=16, rnn_dim=16, mlp_dim=16), (1, 1, 8192), bandit_mlx.mlx_forward_bandit_mx),
        "bandit_v2": (lambda: Bandit(in_channels=1, stems=["vocals"], n_bands=64, fs=44100, n_fft=2048,
                                     hop_length=512, n_sqm_modules=2, emb_dim=16, rnn_dim=16, mlp_dim=16),
                      (1, 1, 8192), bandit_mlx.mlx_forward_bandit_mx),
        "scnet": (lambda: SCNet(sources=["vocals"], num_dplayer=2), (1, 2, 16384), scnet_mlx.mlx_forward_scnet_mx),
        "apollo": (lambda: Apollo(sr=44100, win=32, feature_dim=16, layer=1), (1, 1, 8192),
                   apollo_mlx.mlx_forward_apollo_mx),
        "mdx23c": (lambda: TFC_TDF_net(mdx_config), (1, 2, 7680), mdx23c_mlx.mlx_forward_mdx23c_mx),
        "htdemucs": (lambda: htdemucs_get_model(ht_config), (1, 2, 8192), demucs_mlx.mlx_forward_demucs_mx),
        "legacy_tasnet": (lambda: LegacyConvTasNet(sources=1, N=8, L=40, B=8, H=16, P=3, X=2, R=1, audio_channels=2),
                          (1, 2, 8192), None),
        "legacy_demucs": (lambda: LegacyDemucs(sources=1, channels=8, depth=2, resample=False, lstm_layers=0,
                                               context=1), (1, 2, 8192), None),
        "legacy_v3_demucs": (lambda: LegacyV3Demucs(sources=1, channels=8, depth=2, lstm_layers=0, dconv_depth=1),
                             (1, 2, 10000), None),
        "legacy_hdemucs": (lambda: LegacyHDemucs(sources=["vocals"], channels=8, depth=2, nfft=1024, hybrid=False), (1, 2, 8192), None),
    }
    return out


def _record_one(name, builder, shape, mlx_fn):
    torch.manual_seed(0)
    model = builder().eval()
    x = torch.randn(shape)
    with torch.inference_mode():
        y = model(x).float()
    anchor = {"sd": model.state_dict(), "x": x, "y_torch": y}
    if mlx_fn is not None:
        import mlx.core as mx
        y_mx = mlx_fn(model, mx.array(x.numpy()), torch.float32)
        anchor["y_mlx"] = torch.from_numpy(np.array(y_mx, dtype=np.float32))
        print(f"  mlx-vs-torch max diff: {(y - anchor['y_mlx']).abs().max().item():.2e}")
    torch.save(anchor, ANCHOR_DIR / f"{name}.pt")


def _packed_invariant():
    # forward_packed_estimators must agree with per-estimator stacking (num_stems>1 path)
    from pymss_core.modules.bs_roformer import BSRoformer
    from pymss_core.modules.bs_roformer.bands import MaskEstimator

    torch.manual_seed(1)
    m = BSRoformer(**_roformer_kwargs(freqs_per_bands=(1,) * 9, num_stems=2)).eval()
    x = torch.randn(2, 5, 9, 8)
    with torch.inference_mode():
        packed = MaskEstimator.forward_packed_estimators(tuple(m.mask_estimators), x)
        stacked = torch.stack([fn(x) for fn in m.mask_estimators], dim=1)
    return 0.0 if packed is None else (packed - stacked).abs().max().item()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "record":
        ANCHOR_DIR.mkdir(exist_ok=True)
        for name, (builder, shape, mlx_fn) in cases().items():
            print(f"recording {name}")
            _record_one(name, builder, shape, mlx_fn)
        return
    bad = 0
    for name, (builder, shape, mlx_fn) in cases().items():
        anchor = torch.load(ANCHOR_DIR / f"{name}.pt", weights_only=False)
        model = builder().eval()
        missing, unexpected = model.load_state_dict(anchor["sd"], strict=True)
        with torch.inference_mode():
            y = model(anchor["x"]).float()
        dt = (y - anchor["y_torch"]).abs().max().item()
        status = f"torch diff {dt:.2e}"
        ok = dt < 1e-4
        if "y_mlx" in anchor:
            import mlx.core as mx
            y_mx = torch.from_numpy(np.array(
                mlx_fn(model, mx.array(anchor["x"].numpy()), torch.float32), dtype=np.float32))
            dm = (y_mx - anchor["y_mlx"]).abs().max().item()
            status += f", mlx diff {dm:.2e}"
            ok = ok and dm < 1e-3
        print(f"{name}: {status} {'OK' if ok else 'FAIL'}")
        bad += not ok
    dp = _packed_invariant()
    print(f"packed-estimators invariant: {dp:.2e} {'OK' if dp < 1e-4 else 'FAIL'}")
    bad += not dp < 1e-4
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
