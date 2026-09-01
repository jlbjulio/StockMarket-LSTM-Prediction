from stock_lstm.cli import build_parser


def test_train_command_has_beginner_friendly_defaults() -> None:
    args = build_parser().parse_args(["train"])
    assert args.ticker == "GOOGL"
    assert args.start == "2015-01-01"
    assert args.epochs == 40
    assert args.output_dir == "artifacts"


def test_forecast_supports_machine_readable_output() -> None:
    args = build_parser().parse_args(
        ["forecast", "--run-dir", "artifacts/example", "--json"]
    )
    assert args.run_dir == "artifacts/example"
    assert args.json is True
