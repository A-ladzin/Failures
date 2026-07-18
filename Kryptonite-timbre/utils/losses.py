import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace / Additive Angular Margin Softmax (AAM-Softmax).

    SOTA metric-learning loss for speaker verification. Adds a fixed angular
    margin `m` to the ground-truth cosine similarity, forcing the model to
    learn tighter, more discriminative speaker clusters compared to plain CE.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
               Face Recognition", CVPR 2019. https://arxiv.org/abs/1801.07698

    Args:
        embed_dim:    Dimension of the input speaker embeddings.
        num_classes:  Number of training speakers.
        s:            Feature scale / temperature (default 32.0).
        m:            Angular margin in radians (default 0.2 ≈ 11.5°).
        easy_margin:  If True, uses a softer boundary condition for stability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 32.0,
        m: float = 0.2,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2-normalise both embeddings and class-centre weight vectors
        x = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

        cosine = F.linear(x, W).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - cosine ** 2)

        # cos(theta + m) via angle-addition formula
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Numerical stability: fall back to cosine - mm when theta > pi - m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to the ground-truth class logit
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s

        return F.cross_entropy(output, labels)




class SubcenterArcFaceLoss(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 32.0,
        m: float = 0.2,
        K: int = 10,
        drop_start_epoch: int = 12,
        drop_freq: int = 1,
        drop_threshold: float = math.pi / 6,
        drop_noise_std: float = 0.01,
        lambda_anchor = 0.005,
        detach_mean = False
    ):
        super().__init__()
        self.s               = s
        self.m               = m
        self.K               = K
        self.num_classes     = num_classes
        self.drop_start_epoch = drop_start_epoch
        self.drop_freq        = drop_freq
        self.drop_threshold   = drop_threshold
        self.drop_noise_std   = drop_noise_std
        self.lambda_anchor = lambda_anchor
        self.detach_mean = False
 
        # Weight matrix: (K*C, D) — c classes laid out consecutively per subcenter
        # subcenter k for each class owns rows [K*c : K*c+c]
        self.weight = nn.Parameter(torch.empty(K*num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m
 
        # Track current epoch for drop scheduling (set by train loop)
        self.register_buffer("_epoch", torch.tensor(0, dtype=torch.long))
 
    # ── Forward ──────────────────────────────────────────────────────────────

    def compute_reg(self,W_normed):
        W = W_normed.view(self.K, self.num_classes, -1)
 
        # ── Anchor penalty ────────────────────────────────────────────────────
        # Class mean in embedding space: (C, D)
        mean = W.mean(dim=0)                              # (C, D)
        mean_n = F.normalize(mean, dim=1)                 # (C, D) normalised
        if self.detach_mean:
            mean_n = mean_n.detach()

        # Cosine similarity of each sub-center to class mean: (K, C)
        # W is already normalised, mean_n is normalised
        cos_to_mean = (W * mean_n.unsqueeze(0)).sum(dim=2)  # (K, C)

        # Penalty = 1 - cosine (higher = more diverged)
        anchor_reg = (1.0 - cos_to_mean).mean()
        
        return self.lambda_anchor * anchor_reg
 
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

 
        # (B, K*C) → (B, K, C) → max over K gives nearest sub-center per class
        cosine_all = F.linear(x, W).clamp(-1 + 1e-7, 1 - 1e-7)
        cosine     = cosine_all.view(-1, self.K, self.num_classes).max(dim=1).values
 
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)
 
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output  = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s

        loss = F.cross_entropy(output, labels)

        if self.lambda_anchor > 0:
            loss += self.compute_reg(W)

    
 
        return loss
 
    # ── Drop trick ────────────────────────────────────────────────────────────
 
    def step_epoch(self, epoch: int) -> int:
        """
        Call at the start of each epoch. Returns number of sub-centers dropped.
 
        Integrates the drop schedule: does nothing during warmup, then drops
        diverged sub-centers every drop_freq epochs.
        """
        self._epoch.fill_(epoch)
        if epoch < self.drop_start_epoch:
            return 0
        if (epoch - self.drop_start_epoch) % self.drop_freq != 0:
            return 0
        return self.drop_non_dominant(verbose=True)
 
    @torch.no_grad()
    def drop_non_dominant(self, verbose: bool = False) -> int:
        """
        Re-initialise sub-centers that have diverged from the dominant one.
 
        For each class c:
          1. Find the dominant sub-center: the one with the highest L2 norm.
             (Norm correlates with update frequency — heavily-used sub-centers
              accumulate larger gradients and grow in norm.)
          2. Compute angular distance from dominant to each other sub-center:
             θ_k = arccos( dot(w_dom, w_k) / (|w_dom| * |w_k|) )
          3. If θ_k > drop_threshold, re-initialise:
             w_k ← w_dom + N(0, drop_noise_std²)
             The noise breaks symmetry so the sub-center doesn't immediately
             collapse back to the dominant.
 
        Returns:
            Total number of sub-centers re-initialised across all classes.
        """
        W = self.weight.data          # (K*C, D)
        W_3d = W.view(self.K, self.num_classes, -1)  # (K, C, D)
 
        n_dropped = 0
        for c in range(self.num_classes):
            sub = W_3d[:,c]                          # (K, D)
            norms = sub.norm(dim=1)                # (K,)
            dom   = norms.argmax().item()          # index of dominant sub-center
            w_dom = sub[dom]                       # (D,)
            w_dom_n = F.normalize(w_dom.unsqueeze(0), dim=1).squeeze(0)
 
            for k in range(self.K):
                if k == dom:
                    continue
                w_k   = sub[k]
                w_k_n = F.normalize(w_k.unsqueeze(0), dim=1).squeeze(0)
                cos_dist = (w_dom_n * w_k_n).sum().clamp(-1 + 1e-7, 1 - 1e-7)
                angle    = math.acos(float(cos_dist))
 
                if angle > self.drop_threshold:
                    # Re-init near dominant + small noise
                    noise = torch.randn_like(w_dom) * self.drop_noise_std
                    W_3d[k, c] = w_dom + noise
                    n_dropped += 1
 
        if verbose and n_dropped > 0:
            pct = 100.0 * n_dropped / (self.num_classes * (self.K - 1))
            print(f"  [Drop trick] Re-initialised {n_dropped} sub-centers "
                  f"({pct:.1f}% of non-dominant)")
        elif verbose:
            print(f"  [Drop trick] No sub-centers diverged beyond threshold "
                  f"({math.degrees(self.drop_threshold):.0f}°) — skipping")
 
        return n_dropped