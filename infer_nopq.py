"""Backward-compatible entry point for the CURP no-PQ ablation."""

from infer import build_parser, main


if __name__ == "__main__":
    main(build_parser(default_encoder_type="contriever").parse_args())
