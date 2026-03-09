import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

from archive_schema import create_archive


def trace_vit(image, archive_path):

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="eager")

    model.eval()

    inputs = processor(images=image, return_tensors="pt")

    archive = create_archive(archive_path, model.config)

    # -------------------
    # Store inputs
    # -------------------

    archive["inputs"].create_dataset(
        "processed_image",
        data=inputs["pixel_values"].numpy(),
        chunks=True,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # -------------------
    # Store activations
    # -------------------

    act_group = archive["activations"]

    for i, hidden in enumerate(outputs.hidden_states):

        layer_group = act_group.require_group(f"layer_{i}")

        layer_group.create_dataset(
            "hidden_states",
            data=hidden.detach().cpu().numpy(),
            chunks=True
        )

    # -------------------
    # Store attention
    # -------------------

    print("Number of attention layers:", len(outputs.attentions))
    attn_group = archive["attention"]

    for i, attn in enumerate(outputs.attentions):

        attn_group.create_dataset(
            f"layer_{i}",
            data=attn.detach().cpu().numpy(),
            chunks=True
        )

    # -------------------
    # Outputs
    # -------------------

    logits = outputs.logits.detach().cpu().numpy()

    archive["outputs"].create_dataset("logits", data=logits)

    archive["outputs"].attrs["predicted_class"] = logits.argmax(-1).item()

    return archive