# kl_loss.py
"""Exact forward-KL over a top-K + tail-bucket partition of teacher logits."""
import torch

@torch.no_grad()
def topk_teacher(logits, K):
    """fp32 (idx [T,K] long, vals [T,K], tail_logsumexp [T])."""
    lg = logits.float()
    vals, idx = lg.topk(K, dim=-1)
    full_lse = torch.logsumexp(lg, -1)
    top_mass = (vals - full_lse.unsqueeze(-1)).exp().sum(-1).clamp_max(1 - 1e-7)
    tail = full_lse + torch.log1p(-top_mass)
    return idx, vals, tail

def _kl_topk_per_tok(student_logits, idx, vals, tail):
    """Per-token forward-KL contributions on the K+1 bucket partition ([T])."""
    s = student_logits.float()
    s_top = s.gather(-1, idx)
    full_lse = torch.logsumexp(s, -1, keepdim=True)
    s_tail = full_lse + torch.log1p(
        -(s_top - full_lse).exp().sum(-1, keepdim=True).clamp_max(1 - 1e-7))
    s_buckets = torch.cat([s_top, s_tail], -1)
    t_buckets = torch.cat([vals, tail.unsqueeze(-1)], -1)
    logp_t = torch.log_softmax(t_buckets, -1)
    logq_s = torch.log_softmax(s_buckets, -1)
    p_t = logp_t.exp()
    contrib = p_t * (logp_t - logq_s)
    # When K==V the tail bucket has p_t==0 (teacher tail logit = -inf); zero its
    # non-finite contribution instead of propagating NaN.
    contrib = torch.where(p_t > 0, contrib, torch.zeros_like(contrib))
    return contrib.sum(-1)


def kl_topk(student_logits, idx, vals, tail, mask=None, chunk=512):
    """Forward KL(teacher||student) on the K+1 bucket partition, token-mean.

    To bound peak memory, the [T] token axis is processed in `chunk`-token slabs:
    each slab's student logits / teacher top-K buckets materialize their softmax
    chains independently, are reduced to a scalar numerator + denominator, and are
    freed before the next slab. The result is the EXACT same mask-weighted (or
    plain) token mean as a single-shot call — accumulation is over per-token
    contributions, so chunking changes nothing but the working-set size. Set
    chunk<=0 (or T<=chunk) to process the whole sequence in one slab.
    """
    T = student_logits.shape[0]
    if chunk is None or chunk <= 0:
        chunk = T
    num = student_logits.new_zeros((), dtype=torch.float32)
    den = student_logits.new_zeros((), dtype=torch.float32)
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        per_tok = _kl_topk_per_tok(
            student_logits[c0:c1], idx[c0:c1], vals[c0:c1], tail[c0:c1])
        if mask is not None:
            mc = mask[c0:c1]
            num = num + (per_tok * mc).sum()
            den = den + mc.sum()
        else:
            num = num + per_tok.sum()
            den = den + per_tok.shape[0]
    if mask is not None:
        return num / den.clamp_min(1)
    return num / max(int(den), 1)
