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

def kl_topk(student_logits, idx, vals, tail, mask=None):
    """Forward KL(teacher||student) on the K+1 bucket partition, token-mean."""
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
    per_tok = contrib.sum(-1)
    if mask is not None:
        return (per_tok * mask).sum() / mask.sum().clamp_min(1)
    return per_tok.mean()
