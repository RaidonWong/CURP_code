"""Backward-compatible entry point for RoBERTa-based CURP inference."""

from infer import build_parser, main


if __name__ == "__main__":
    main(build_parser(default_encoder_type="roberta").parse_args())
