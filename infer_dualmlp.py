"""Backward-compatible entry point for Contriever + dual-MLP inference.

The shared inference implementation auto-detects dual-MLP checkpoints and keeps
tokenization, truncation, placeholder validation, and checkpoint handling
identical across model variants.
"""

from infer import build_parser, main


if __name__ == "__main__":
    main(build_parser(default_encoder_type="contriever").parse_args())
