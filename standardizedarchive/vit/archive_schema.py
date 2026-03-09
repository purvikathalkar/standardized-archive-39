import zarr
import numpy as np


def create_archive(path, model_config):
    root = zarr.open(path, mode="w")

    # -------------------
    # Metadata
    # -------------------

    meta = root.create_group("metadata")

    meta.attrs["model_name"] = model_config._name_or_path
    meta.attrs["model_type"] = "vision_transformer"
    meta.attrs["hidden_dim"] = model_config.hidden_size
    meta.attrs["num_layers"] = model_config.num_hidden_layers
    meta.attrs["num_heads"] = model_config.num_attention_heads
    meta.attrs["patch_size"] = model_config.patch_size
    meta.attrs["image_resolution"] = model_config.image_size

    seq_len = (model_config.image_size // model_config.patch_size) ** 2 + 1

    meta.attrs["sequence_length"] = seq_len

    # -------------------
    # Core groups
    # -------------------

    root.create_group("inputs")
    root.create_group("tokens")
    root.create_group("activations")
    root.create_group("attention")
    root.create_group("outputs")

    return root