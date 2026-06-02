from __future__ import annotations


def test_serve_parser_accepts_missing_model():
    from pymss.cli import build_parser

    args = build_parser().parse_args(["serve"])

    assert args.model is None
