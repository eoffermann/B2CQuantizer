# Faithful, byte-for-byte snapshot of the `MmprojModel` class body from the
# real llama.cpp `conversion/base.py` at our pinned commit
# 720d7fa4097f76e5d0eade5a92c1df87c1faf9d9 (see scripts/build_llama_cpp.sh /
# LLAMA_CPP_COMMIT). Extracted verbatim (lines 2313-2489 of the upstream
# convert_hf_to_gguf.py source tree's conversion/base.py) for
# tests/unit/test_mmproj_patch.py to exercise the REAL patch regex against --
# not a hand-invented stand-in. Free names (ModelBase, gguf, Tensor, Callable,
# Any, Iterable, json, copy, _MISTRAL_COMMON_DATASET_MEAN/STD) are
# intentionally left undefined: this file is never imported/executed, only
# regex-patched and ast.parse()'d, so only syntactic validity matters.
#
# DO NOT hand-edit the class body below -- if the upstream source changes,
# re-extract it from a fresh checkout of the pinned commit.

class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth", "layers", "encoder_layers", "vt_num_hidden_layers"]

    has_vision_encoder: bool = True # by default
    has_audio_encoder: bool = False

    # for models having multiple encoders, we need to separate their hparams
    hparams_vision: dict[str, Any] | None = None
    hparams_audio: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_arch != gguf.MODEL_ARCH.MMPROJ:
            raise TypeError("MmprojModel must be subclassed with model_arch = gguf.MODEL_ARCH.MMPROJ")

        # get n_embd of the text model
        if not self.is_mistral_format:
            if "text_config" not in self.hparams:
                self.hparams["text_config"] = {}
            if "audio_config" not in self.hparams:
                self.hparams["audio_config"] = {}
            text_config = {**self.hparams, **self.hparams["text_config"]}
            self.n_embd_text = text_config.get("hidden_size", text_config.get("n_embd", 0))
        else:
            text_config = {
                k: v for k, v in self.hparams.items() if k not in ["vision_encoder", "audio_encoder"]
            }
            # mistral native params.json: "dim" is the text hidden size ("hidden_dim" is the FFN intermediate size)
            self.n_embd_text = text_config.get("dim", 0)

        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy
        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()
        self.hparams_audio = self.get_audio_config()

        if self.hparams_vision is None and self.hparams_audio is None:
            raise ValueError("vision_config / audio_config not found in hparams")

        # for compat with vision-only models
        self.hparams = self.hparams_vision or self.hparams_audio or self.hparams

        # TODO @ngxson : this is a hack to support both vision and audio encoders
        have_multiple_encoders = self.has_audio_encoder and self.has_vision_encoder
        self.block_count = 128 if have_multiple_encoders else self.find_hparam(self.n_block_keys, True)
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

        # load preprocessor config
        self.preprocessor_config = {}

        # prefer preprocessor_config.json if possible
        preprocessor_config_path = self.dir_model / "preprocessor_config.json"
        if preprocessor_config_path.is_file():
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move media_proc_cfg to root level for compat
                if "media_proc_cfg" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["media_proc_cfg"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

        # prefer processor_config.json if possible
        processor_config_path = self.dir_model / "processor_config.json"
        if processor_config_path.is_file():
            with open(processor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move image_processor to root level for compat
                if "image_processor" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["image_processor"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip non-multimodal tensors
        if "language_model." in name:
            return None

        return super().filter_tensors(item)

    def get_vision_config(self) -> dict[str, Any] | None:
        config_name = "vision_config" if not self.is_mistral_format else "vision_encoder"
        return self.global_config.get(config_name)

    def get_audio_config(self) -> dict[str, Any] | None:
        mm_config_key = "whisper_config" if "whisper_config" in self.hparams else "audio_config"
        return self.global_config.get(mm_config_key)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        output_type: str = self.ftype.name.partition("_")[2]

        if self.fname_out.is_dir():
            fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=output_type, model_type=None)
            self.fname_out = self.fname_out / f"mmproj-{fname_default}.gguf"
        else:
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if self.has_vision_encoder:
            self.gguf_writer.add_clip_has_vision_encoder(True)
            self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

            # vision config
            self.image_size = self.find_vparam(["image_size"])
            self.gguf_writer.add_vision_image_size(self.image_size)
            self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
            self.gguf_writer.add_vision_embedding_length(self.find_vparam(["hidden_size", "width", "vt_hidden_size"]))
            self.gguf_writer.add_vision_feed_forward_length(self.find_vparam(["intermediate_size", "vt_intermediate_size"]))
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(self.find_vparam(["num_attention_heads", "num_heads", "heads", "vt_num_attention_heads"]))

            # preprocessor config
            image_mean = _MISTRAL_COMMON_DATASET_MEAN if self.is_mistral_format else self.preprocessor_config["image_mean"]
            image_std = _MISTRAL_COMMON_DATASET_STD if self.is_mistral_format else self.preprocessor_config["image_std"]

            self.gguf_writer.add_vision_image_mean(image_mean)
            self.gguf_writer.add_vision_image_std(image_std)

        if self.has_audio_encoder:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

            # audio config
            self.gguf_writer.add_audio_embedding_length(self.find_aparam(["hidden_size"]))
            self.gguf_writer.add_audio_feed_forward_length(self.find_aparam(["intermediate_size"]))
            self.gguf_writer.add_audio_block_count(self.find_aparam(self.n_block_keys))
            self.gguf_writer.add_audio_head_count(self.find_aparam(["num_attention_heads"]))

        if not self.has_vision_encoder and not self.has_audio_encoder:
            raise ValueError("MmprojModel must have either vision or audio encoder")

    def write_vocab(self):
        raise ValueError("MmprojModel does not support vocab writing")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def find_aparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_audio is not None
        return self._find_param(self.hparams_audio, keys, optional)

    def _find_param(self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".patch_embd.weight" in new_name or ".patch_merger.weight" in new_name:
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
