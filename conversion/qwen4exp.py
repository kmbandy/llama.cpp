from __future__ import annotations

from typing import Callable, Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import LazyTorchTensor, ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # MTP EXPORT IS ENABLED HERE, deliberately diverging from upstream.
    #
    # Upstream set supports_mtp_export = False / no_mtp = True with the note
    # "the MTP block is a separate draft head; vLLM drops it too". The effect is
    # that EVERY GGUF built from upstream's converter silently lacks the head --
    # including Unsloth's UD-* quants -- so speculative decode is unavailable to
    # anyone using them, and the tensors cannot be recovered from the quantized
    # file afterwards.
    #
    # The weights do exist upstream: Qwen/Qwen3.8-Flash-Next ships 31 mtp.*
    # tensors and text_config.mtp_num_hidden_layers = 1. _QwenMtpMixin is
    # already in this class's MRO (via _LinearAttentionVReorderBase ->
    # Qwen3NextModel), so dropping the two overrides is all that is needed to
    # reach the standard --mtp / --no-mtp handling in convert_hf_to_gguf.py,
    # which is gated on supports_mtp_export and defaults to False in base.py.
    #
    # Note this MTP block carries its OWN 512-expert MoE (mtp.layers.0.mlp.
    # experts.gate_up_proj is [512, 1280, 2560]), so the draft is ~4B and is
    # itself sparse -- it is not a cheap dense head like most NextN blocks.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        # The reference ADDS the two MTP input projections, and
        #     A*e + B*h == [A|B] * concat(e, h)
        # so they join into the single eh_proj the tensor map already knows and the
        # graph consumes as one matmul. Exact, not an approximation.
        # ref: ggml-org#27739; conversion/deepseek.py joins the DeepSeek-V4 pair the same way.
        e_name = "mtp.fc_embedding.weight"
        h_name = "mtp.fc_hidden.weight"

        have_e = e_name in self.model_tensors
        have_h = h_name in self.model_tensors
        if not have_e and not have_h:
            return
        if not have_e or not have_h:
            raise KeyError(f"unpaired MTP input projection: need both {e_name} and {h_name}")

        e = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
        h = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                                       self.hparams["num_hidden_layers"]),
               torch.cat([e, h], dim=1).contiguous())

        del self.model_tensors[e_name]
        del self.model_tensors[h_name]

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]

        # The MTP block brings its OWN hyper-connection mixer, which takes the
        # model-level slot in an MTP-only file (this arch has no output_norm -- the
        # final mixer carries it). Rename before _QwenMtpMixin drops it as a
        # non-MTP tensor. ref: ggml-org#27739
        if name.startswith("mtp.hyper_connection_mixer."):
            return None if cls.no_mtp else (name.replace("mtp.", "model.", 1), item[1])

        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        # The loader reads this array at n_layer_all, which INCLUDES the MTP block, so
        # it must be block_count long or llama_model_loader rejects the file with
        # "wrong array length; expected 49, got 48". self.block_count is n_layer plus
        # the MTP layers (_QwenMtpMixin), and equals n_layer when --no-mtp, so a plain
        # target file is unchanged. The MTP block gets 0: its attention runs DENSE, and
        # 0 is exactly how this array spells "not a QSA layer".
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
            + [0] * (self.block_count - n_layer)
        )

        # The MTP block reads the target's hyper-connection STREAMS, not the collapsed
        # hidden state, so BOTH files declare the wider row: the target for the nextn
        # read-back and the draft for common/speculative.cpp. Same as DeepSeek-V4.
        self.gguf_writer.add_embedding_length_out(hp["hc_count"] * hp["hidden_size"])

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones.
        # An MTP-only file has no PLE layer at all (the reference clears
        # ple_layer_ids for the MTP block), so skip the whole group there.
        if self.mtp_only:
            return
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
