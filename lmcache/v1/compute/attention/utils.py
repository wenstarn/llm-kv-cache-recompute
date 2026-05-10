# SPDX-License-Identifier: Apache-2.0
def infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
    attn_name = type(vllm_attn.impl).__name__
    if attn_name == "FlashInferImpl" and enable_sparse:
        # Local
        from .flash_infer_sparse import LMCFlashInferSparseBackend

        return LMCFlashInferSparseBackend(vllm_attn)
    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        # Local
        from .flash_attn import LMCFlashAttnBackend

        return LMCFlashAttnBackend(vllm_attn)
    else:
        raise ValueError(f"Attention backend {attn_name} is not supported in LMCache.")

