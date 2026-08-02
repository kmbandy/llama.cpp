"""Generic DSpark draft-model conversion helpers.

DSpark = DFlash block-diffusion draft + Markov / confidence heads. The HF
checkpoint ships as mtp.{stage}.* tensors (no embed/head — shared with the
target at runtime). Almost none of the conversion logic is backbone-specific;
backbone subclasses only supply architecture-specific hparam overrides.

See docs/dev/2026-07-31-design-dspark-conversion.md.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger


class DSparkDraftMixin:
    """Generic DSpark draft conversion (architecture-agnostic).

    Compose with a backbone TextModel subclass. The subclass must:
      - set ``model_arch = gguf.MODEL_ARCH.DFLASH``
      - call ``self.dspark_init()`` at the end of ``__init__`` after
        ``model_tensors`` is populated
      - override backbone-specific hparams (e.g. compress_ratios) either in
        ``dspark_backbone_hparams()`` or after ``dspark_init()``

    A second backbone (e.g. Ternary-Bonsai) should be a thin subclass that
    only supplies its own hparam overrides — do not fork this mixin blind.
    """

    # Root (unprefixed) draft tensors after rekey. Mapped to GGUF MODEL_TENSOR.
    DSPARK_ROOT_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
        "main_proj.weight": (gguf.MODEL_TENSOR.FC, ".weight"),
        "main_norm.weight": (gguf.MODEL_TENSOR.ENC_OUTPUT_NORM, ".weight"),
        "markov_head.markov_w1.weight": (gguf.MODEL_TENSOR.DSPARK_MARKOV_W1, ".weight"),
        "markov_head.markov_w2.weight": (gguf.MODEL_TENSOR.DSPARK_MARKOV_W2, ".weight"),
        "confidence_head.proj.weight": (gguf.MODEL_TENSOR.DSPARK_CONF_PROJ, ".weight"),
    }

    # Names that stay unprefixed after mtp.{stage}. rekey (not layers.{stage}.).
    # Includes root-map keys plus residual/scale tensors that are stage-local
    # only for stage 0 in current DeepSeek checkpoints but must not collide.
    _DSPARK_UNPREFIXED_REST: frozenset[str] = frozenset({
        *DSPARK_ROOT_MAP.keys(),
        "main_proj.scale",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    })

    # Populated by dspark_init for subclasses that need the derived stage count.
    dspark_block_count: int = 0

    def dspark_init(self) -> None:
        """Derive block_count from stages actually present and rebuild tensor_map.

        MUST be called after model_tensors is populated (i.e. after ModelBase
        __init__). Never uses the target's num_hidden_layers.
        """
        stages = [
            int(m.group(1))
            for name in self.model_tensors
            if (m := re.match(r"layers\.(\d+)\.", name))
        ]
        if not stages:
            raise ValueError(
                "DSpark draft: no layers.{stage}.* tensors found after rekey. "
                "Expected mtp.{stage}.* tensors in the checkpoint shards."
            )
        self.dspark_block_count = 1 + max(stages)
        self.block_count = self.dspark_block_count
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        # Required DSpark hparams (flat DeepSpec schema on the draft config).
        for key in ("dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids"):
            if key not in self.hparams:
                raise ValueError(
                    f"DSpark draft config.json is missing required key {key!r}. "
                    "Download the draft / MTP config, not the full target config alone."
                )

        self.dspark_backbone_hparams()

        logger.info(
            "DSpark: block_count=%d (from stages present), block_size=%s, "
            "target_layer_ids=%s, noise_token_id=%s",
            self.block_count,
            self.hparams["dspark_block_size"],
            self.hparams["dspark_target_layer_ids"],
            self.hparams["dspark_noise_token_id"],
        )

    def dspark_backbone_hparams(self) -> None:
        """Override in backbone subclasses for architecture-specific hparams."""
        return

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # The drafter only ships a subset of the target's shards;
        # model.safetensors.index.json still lists the full target weight map,
        # so read the shards actually present on disk instead of trusting it
        # (filter_tensors then drops anything that isn't an mtp.* tensor).
        if remote_hf_model_id is not None:
            raise ValueError(
                "DSpark draft conversion does not support remote HF model ids; "
                "download the MTP shards locally and point --outdir / model dir at them."
            )

        import torch
        from .base import LazyTorchTensor

        tensors: dict[str, Callable[[], Tensor]] = {}
        for part_name in ModelBase.get_model_part_names(self.dir_model, "model", ".safetensors"):
            logger.info(f"gguf: indexing model part '{part_name}'")
            with gguf.utility.SafetensorsLocal(self.dir_model / part_name) as model_part:
                for name in model_part.keys():
                    data: gguf.utility.LocalTensor = model_part[name]
                    if self.lazy:
                        data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                    else:
                        dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                        data_gen = lambda data=data, dtype=dtype: torch.from_numpy(  # noqa: E731
                            data.mmap_bytes()
                        ).view(dtype).reshape(data.shape)
                    if titem := self.filter_tensors((name, data_gen)):
                        tname, tgen = titem
                        if tname in tensors:
                            raise ValueError(
                                f"DSpark draft: duplicate rekeyed tensor name {tname!r} "
                                f"(from {name!r}); stages may share unprefixed roots."
                            )
                        tensors[tname] = tgen
        if not tensors:
            raise ValueError(
                f"DSpark draft: no mtp.* tensors found under {self.dir_model}. "
                "Expected model-*-of-*.safetensors shards containing mtp.0/1/2.* weights."
            )
        return tensors

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # Only mtp.* tensors are shipped in these shards; the index also lists
        # the (absent) target backbone tensors — silently drop anything else.
        if not name.startswith("mtp."):
            return None
        rekeyed = cls.dspark_rekey_mtp_tensor_name(name)
        # Chain to the next filter in the MRO (backbone / TextModel).
        return super().filter_tensors((rekeyed, gen))  # type: ignore[misc]

    @classmethod
    def dspark_rekey_mtp_tensor_name(cls, name: str) -> str:
        match = re.match(r"mtp\.(\d+)\.(.+)$", name)
        if match is None:
            raise ValueError(f"Unexpected DSpark tensor {name!r}")
        stage, rest = match.group(1), match.group(2)
        if rest in cls._DSPARK_UNPREFIXED_REST:
            return rest
        return f"layers.{stage}.{rest}"

    def dspark_map_root_tensor(self, name: str) -> tuple[gguf.MODEL_TENSOR, str] | None:
        return self.DSPARK_ROOT_MAP.get(name)

    def dspark_set_vocab(self, set_vocab_fn: Callable[[], None]) -> None:
        if self.target_model_dir is None:
            raise ValueError(
                "DSpark draft model requires --target-model-dir to be specified. "
                "Please provide the path to the target model directory containing the tokenizer."
            )
        logger.info(f"DSpark: Using tokenizer from target model: {self.target_model_dir}")
        original_dir = self.dir_model
        self.dir_model = self.target_model_dir
        try:
            set_vocab_fn()
        finally:
            self.dir_model = original_dir

        self.gguf_writer.add_mask_token_id(int(self.hparams["dspark_noise_token_id"]))

    def dspark_set_gguf_parameters(self) -> None:
        """Emit DSpark / DFlash metadata shared by every backbone.

        Call after the backbone's set_gguf_parameters() so arch-specific keys land
        first; this writes block_size, target_layers, and dflash.hc_mult.
        """
        self.gguf_writer.add_block_size(int(self.hparams["dspark_block_size"]))
        extract_layer_ids = [int(i) + 1 for i in self.hparams["dspark_target_layer_ids"]]
        self.gguf_writer.add_target_layers(extract_layer_ids)

        # DFlash loader requires dflash.hc_mult for n_embd_inp_enc; backbone
        # may also write hyper_connection_count under a different key.
        hc = self.hparams.get("hc_mult", 1)
        self.gguf_writer.add_dflash_hc_mult(int(hc))
