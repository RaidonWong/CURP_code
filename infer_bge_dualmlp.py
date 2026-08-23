"""Backward-compatible entry point for BGE + dual-MLP inference."""

from infer import build_parser, main


if __name__ == "__main__":
    main(build_parser(default_encoder_type="bge").parse_args())
