"""
PyTorch port of the time + spectral `fusion_lr1e-4_wd_ls` model (fusion pipeline,
scripts/fusion_model.py: build_encoder, spectral_branch, head, build("fusion")).

  time branch      conv trunk on the jointly standardized 2 x 2001 window:
                   5 x [Conv1d (16/32/64/64/64 filters, k 9/9/7/5/5, 'same', no bias)
                   -> BatchNorm -> ReLU -> MaxPool 2], then global average + global max
                   pooling -> 128-d
  spectral branch  61 bins x 3 channels (spectral.centre_spectrum, standardized):
                   3 x [Conv1d (16/32/32, k 5/5/3, no bias) -> BatchNorm -> ReLU
                   (-> MaxPool 2, first two)] -> Flatten -> Dropout 0.3 -> Dense 64 ReLU
  head             concat 192-d -> Dropout 0.3 -> Dense 32 ReLU -> Dropout 0.3 -> Dense 1

Same layer sizes and Keras defaults: Glorot-uniform kernels, zero biases, BatchNorm
eps 1e-3 / momentum 0.99 (PyTorch momentum 0.01). The network returns a logit;
probability = sigmoid(logit) (Keras' Dense(1, sigmoid) + BCE trains on the same logit).

FusionEnsemble loads models/fusion_lr1e-4_wd_ls_torch.pt (5 seeds + the spectral
standardizer) and scores windows: 5-seed mean probability (Methods S3.7).
"""
import numpy as np
import torch
import torch.nn as nn

from . import spectral as sp

BN = dict(eps=1e-3, momentum=0.01)      # Keras BatchNormalization defaults
TIME_LAYERS = [(16, 9), (32, 9), (64, 7), (64, 5), (64, 5)]
SPEC_LAYERS = [(16, 5, 2), (32, 5, 2), (32, 3, 1)]
FORMAT = "tdifc-fusion-torch-v1"


class TimeEncoder(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        layers, c = [], in_ch
        for filt, k in TIME_LAYERS:
            layers += [nn.Conv1d(c, filt, k, padding=k // 2, bias=False), nn.BatchNorm1d(filt, **BN),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c = filt
        self.body = nn.Sequential(*layers)
        self.out_dim = 2 * c

    def forward(self, x):                       # x: (B, 2, 2001)
        h = self.body(x)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class SpectralBranch(nn.Module):
    def __init__(self, n_bins, n_ch=3):
        super().__init__()
        layers, c, L = [], n_ch, n_bins
        for filt, k, pool in SPEC_LAYERS:
            layers += [nn.Conv1d(c, filt, k, padding=k // 2, bias=False), nn.BatchNorm1d(filt, **BN), nn.ReLU()]
            if pool > 1:
                layers.append(nn.MaxPool1d(pool))
                L //= pool
            c = filt
        self.body = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(c * L, 64), nn.ReLU())
        self.out_dim = 64

    def forward(self, s):                       # s: (B, 3, n_bins)
        return self.fc(self.body(s))


class FusionNet(nn.Module):
    def __init__(self, n_bins=61):
        super().__init__()
        self.n_bins = n_bins
        self.time = TimeEncoder()
        self.spec = SpectralBranch(n_bins)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.time.out_dim + self.spec.out_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Keras defaults: glorot_uniform kernels, zero biases (BatchNorm: gamma 1, beta 0)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_time, x_spec):
        """x_time (B, 2, 2001) standardized window; x_spec (B, n_bins, 3) standardized spectrum.
        Returns the logit (B,)."""
        z = torch.cat([self.time(x_time), self.spec(x_spec.transpose(1, 2))], 1)
        return self.head(z).squeeze(1)

    def kernels(self):
        """Conv / Dense weight matrices: the 'kernel' variables Keras' decoupled weight decay touched."""
        return [m.weight for m in self.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]


# ------------------------------------------------------------------- I/O ---
def save_ensemble(path, nets, standardizer, extra=None):
    torch.save({"format": FORMAT, "n_bins": nets[0].n_bins,
                "state_dicts": [{k: v.detach().cpu() for k, v in n.state_dict().items()} for n in nets],
                "spec_mu": torch.from_numpy(np.ascontiguousarray(standardizer.mu, np.float32)),
                "spec_sd": torch.from_numpy(np.ascontiguousarray(standardizer.sd, np.float32)),
                **(extra or {})}, path)


class FusionEnsemble:
    """The 5 trained seeds + spectral standardizer; .predict(X, M) -> 5-seed mean probability."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=True)
        if ck.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} model file")
        self.device = device
        self.nets = []
        for sd in ck["state_dicts"]:
            n = FusionNet(ck["n_bins"])
            n.load_state_dict(sd)
            self.nets.append(n.to(device).eval())
        self.zc = sp.Standardizer(mu=ck["spec_mu"].numpy(), sd=ck["spec_sd"].numpy())
        self.cut_off = float(ck.get("cut_off", 0.6))
        self.info = {k: v for k, v in ck.items() if k not in ("state_dicts", "spec_mu", "spec_sd")}

    def inputs(self, X, M, als=None, progress=False):
        """Model inputs from jointly standardized windows X (N, 2, 2001) and masks M (N, 2001)."""
        if als is None:
            als = sp.als_scaled(X, progress=progress)
        return X.astype(np.float32), self.zc(sp.centre_spectrum(als, M))

    @torch.no_grad()
    def predict_inputs(self, xt, xs, batch=512, per_seed=False):
        P = []
        for n in self.nets:
            out = []
            for a in range(0, len(xt), batch):
                lt = n(torch.from_numpy(xt[a:a + batch]).to(self.device),
                       torch.from_numpy(xs[a:a + batch]).to(self.device))
                out.append(torch.sigmoid(lt).cpu().numpy())
            P.append(np.concatenate(out) if out else np.empty(0, np.float32))
        P = np.array(P)
        return P if per_seed else P.mean(0)

    def predict(self, X, M, als=None, per_seed=False):
        if len(X) == 0:
            return np.empty((len(self.nets), 0)) if per_seed else np.empty(0)
        return self.predict_inputs(*self.inputs(X, M, als), per_seed=per_seed)
