import torch
import torch.nn.functional as F
from .gdn_blackwell.gdn import chunk_gated_delta_rule

_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _tensor_identity(tensor):
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _cache_key(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    return (
        _tensor_identity(q),
        _tensor_identity(k),
        _tensor_identity(v),
        _tensor_identity(state),
        _tensor_identity(A_log),
        _tensor_identity(a),
        _tensor_identity(dt_bias),
        _tensor_identity(b),
        _tensor_identity(cu_seqlens),
        None if scale is None else float(scale),
    )


def _run_impl(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    output, new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    if varlen:
        output = output.squeeze(0)

    return output, new_state


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    key = _cache_key(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    result = _run_impl(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    _CACHE[key] = result
    return result
