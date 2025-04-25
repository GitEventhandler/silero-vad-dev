if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))

import torch
import os
from typing import Tuple
from model.silero_vad import SileroVADNet
from model.silero_vad_params import silero_8k_params, silero_16k_params

"""
Weight Convert Utils
"""

jit2pt_layer_mapping = {
    "encoder.0.reparam_conv.weight": "encoder.conv.0.weight",
    "encoder.0.reparam_conv.bias": "encoder.conv.0.bias",
    "encoder.1.reparam_conv.weight": "encoder.conv.2.weight",
    "encoder.1.reparam_conv.bias": "encoder.conv.2.bias",
    "encoder.2.reparam_conv.weight": "encoder.conv.4.weight",
    "encoder.2.reparam_conv.bias": "encoder.conv.4.bias",
    "encoder.3.reparam_conv.weight": "encoder.conv.6.weight",
    "encoder.3.reparam_conv.bias": "encoder.conv.6.bias",
    "decoder.rnn.weight_ih": "decoder.lstm.weight_ih",
    "decoder.rnn.weight_hh": "decoder.lstm.weight_hh",
    "decoder.rnn.bias_ih": "decoder.lstm.bias_ih",
    "decoder.rnn.bias_hh": "decoder.lstm.bias_hh",
    "decoder.decoder.2.weight": "decoder.conv.2.weight",
    "decoder.decoder.2.bias": "decoder.conv.2.bias"
}


def _mapping_state_dict(src_model, dst_model, key_mapping: dict):
    """Map and transfer parameters between models using predefined key mappings.

    This function copies model parameters from source to destination model based on provided key mappings.
    Parameters are only copied when:
        - Both keys exist in their respective state_dicts
        - The parameter tensors have matching shapes

    Args:
        src_model (torch.nn.Module): Source model containing original weights
        dst_model (torch.nn.Module): Target model to receive remapped parameters
        key_mapping (dict): Dictionary of key mappings between source and target models

    Returns:
        None

    Note: Differences/corruptions are reported via prints but execution continues.
    """
    src_model_dict = src_model.state_dict()
    dst_model_dict = dst_model.state_dict()

    # Iterate through all defined key mappings
    for src_key, dst_key in key_mapping.items():
        # Check key existence and tensor shape compatibility
        if (
                src_key in src_model_dict and
                dst_key in dst_model_dict and
                src_model_dict[src_key].shape == dst_model_dict[dst_key].shape
        ):
            dst_model_dict[dst_key] = src_model_dict[src_key]
        else:
            if src_key not in src_model_dict:
                print(f"Source key '{src_key}' not found")
            elif dst_key not in dst_model_dict:
                print(f"Destination key '{dst_key}' not found")
            else:
                print(
                    f"Shape mismatch for '{dst_key}': {src_model_dict[src_key].shape} vs {dst_model_dict[dst_key].shape}")

    # Apply the updated state dict to target model
    dst_model.load_state_dict(dst_model_dict)


def convert_weight_from_jit(src_model: torch.jit.RecursiveScriptModule) -> Tuple[SileroVADNet, SileroVADNet]:
    model_8k, model_16k = SileroVADNet(silero_8k_params), SileroVADNet(silero_16k_params)
    src_model_8k, src_model_16k = src_model._model_8k, src_model._model

    # STFT layer requires no conversion as it contains pure numerical parameters
    _mapping_state_dict(src_model_8k, model_8k, jit2pt_layer_mapping)
    _mapping_state_dict(src_model_16k, model_16k, jit2pt_layer_mapping)

    return model_8k, model_16k


def convert_weight_to_jit(
        model_8k: torch.nn.Module,
        model_16k: torch.nn.Module,
        model_jit: torch.jit.RecursiveScriptModule
):
    reversed_mapping = {v: k for k, v in jit2pt_layer_mapping.items()}
    if model_8k is not None:
        _mapping_state_dict(model_8k, model_jit._model_8k, reversed_mapping)
    if model_16k is not None:
        _mapping_state_dict(model_16k, model_jit._model, reversed_mapping)
    return model_jit


def _convert_from_jit(input_path: str, output_dir: str):
    # Verify and prepare output directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_16k = os.path.join(output_dir, "silero_vad_16k.pt")
    output_8k = os.path.join(output_dir, "silero_vad_8k.pt")

    # Check for existing files to prevent overwrites
    existing_files = []
    if os.path.exists(output_16k):
        existing_files.append(output_16k)
    if os.path.exists(output_8k):
        existing_files.append(output_8k)

    if existing_files:
        parser.error(f"File(s) already exist: {', '.join(existing_files)}")

    print("Loading original TorchScript model...")
    jit_model = torch.jit.load(input_path)

    print("Converting model weights...")
    model_8k, model_16k = convert_weight_from_jit(jit_model)

    print("Saving converted models:")
    print(f"- 16kHz model: → {output_16k}")
    print(f"-  8kHz model: → {output_8k}")

    torch.save(model_16k, output_16k)
    torch.save(model_8k, output_8k)
    print("Conversion completed successfully")


def _convert_to_jit(
        model_8k_path: str,
        model_16k_path: str,
        template_jit_path: str,
        output_path: str
):
    # Define output paths with constant file names
    output_path = output_path if output_path.endswith(".jit") else f"{output_path}.jit"

    if os.path.exists(output_path):
        parser.error(f"File already exist: {output_path}")

    print("Loading PyTorch model...")
    model_8k = None if model_8k_path is None else torch.load(model_8k_path)
    model_16k = None if model_16k_path is None else torch.load(model_16k_path)

    print("Loading original TorchScript model...")
    model_jit_template = torch.jit.load(template_jit_path)

    print("Converting model weights...")
    jit_model = convert_weight_to_jit(model_8k, model_16k, model_jit_template)

    print("Saving converted models:")
    torch.jit.save(jit_model, output_path)
    print("Conversion completed successfully")


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Silero VAD Model Conversion Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Convert JIT to PT Subcommand
    parser_jit_to_pt = subparsers.add_parser('jit2pt', help='Convert TorchScript (JIT) to PyTorch format')
    parser_jit_to_pt.add_argument('--input', required=True, help='Original JIT model path')
    parser_jit_to_pt.add_argument('--output', required=True, help='Directory for saved PT models')
    # Convert PT to JIT Subcommand
    parser_pt_to_jit = subparsers.add_parser('pt2jit', help='Convert PyTorch models back to TorchScript')
    parser_pt_to_jit.add_argument('--input_8k', required=False, default=None, help='Path to 8kHz PyTorch model')
    parser_pt_to_jit.add_argument('--input_16k', required=False, default=None, help='Path to 16kHz PyTorch model')
    parser_pt_to_jit.add_argument('--template_jit', required=True, help='Path to structural JIT template model')
    parser_pt_to_jit.add_argument('--output', required=True, help='Output JIT model path')
    args = parser.parse_args()

    if args.command == "jit2pt":
        _convert_from_jit(args.input, args.output)
    elif args.command == "pt2jit":
        _convert_to_jit(args.input_8k, args.input_16k, args.template_jit, args.output)
