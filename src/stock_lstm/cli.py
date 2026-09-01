from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stock_lstm.config import ExperimentConfig
from stock_lstm.data import download_market_data, generate_demo_data, load_csv
from stock_lstm.pipeline import forecast_next_session, run_training


def _data_from_args(args: argparse.Namespace, config: ExperimentConfig):
    if getattr(args, "demo", False):
        print("Creating the offline demo dataset...")
        return generate_demo_data(rows=args.demo_rows, seed=config.seed)
    if args.data:
        print(f"Loading market data from {args.data}...")
        return load_csv(args.data)
    print(f"Loading {config.ticker} market data...")
    return download_market_data(
        ticker=config.ticker,
        start=config.start,
        end=config.end,
        refresh=getattr(args, "refresh", False),
    )


def _train(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        ticker="DEMO" if args.demo else args.ticker.upper(),
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    data = _data_from_args(args, config)
    print(
        f"Loaded {len(data):,} daily observations "
        f"from {data.index.min().date()} to {data.index.max().date()}."
    )
    run_name = args.run_name or f"{config.ticker}_{datetime.now():%Y%m%d_%H%M%S}"
    result = run_training(
        data,
        config,
        Path(args.output_dir) / run_name,
        verbose=True,
    )
    baseline = result.metrics["persistence_baseline"]["rmse"]
    model_rmse = result.metrics["lstm"]["rmse"]
    change = (baseline - model_rmse) / baseline * 100

    print("\nTraining complete")
    print(f"  Saved experiment: {result.output_dir}")
    print(f"  Best epoch:      {result.best_epoch}")
    print(f"  Device used:     {result.device}")
    print("\nTest results")
    print(f"  {'Metric':<24}{'LSTM':>12}{'Baseline':>12}")
    print(
        f"  {'MAE (lower is better)':<24}"
        f"{result.metrics['lstm']['mae']:>12.4f}"
        f"{result.metrics['persistence_baseline']['mae']:>12.4f}"
    )
    print(
        f"  {'RMSE (lower is better)':<24}"
        f"{model_rmse:>12.4f}{baseline:>12.4f}"
    )
    print(
        f"  {'Direction accuracy':<24}"
        f"{result.metrics['lstm']['directional_accuracy_percent']:>11.2f}%"
        f"{'n/a':>12}"
    )
    if change > 0:
        print(f"\nThe LSTM improved RMSE over the baseline by {change:.2f}%.")
    else:
        print(f"\nThe LSTM did not beat the baseline in this run ({change:.2f}%).")
        print("That is a valid result; try another period or tune using validation data.")
    print("Open test_predictions.png inside the experiment folder to see the comparison.")
    return 0


def _forecast(args: argparse.Namespace) -> int:
    metadata = json.loads((Path(args.run_dir) / "run.json").read_text(encoding="utf-8"))
    stored = ExperimentConfig.from_dict(metadata["config"])
    if args.demo:
        if not args.json:
            print("Creating the offline demo dataset...")
        data = generate_demo_data(rows=args.demo_rows, seed=stored.seed)
    elif args.data:
        if not args.json:
            print(f"Loading market data from {args.data}...")
        data = load_csv(args.data)
    else:
        if not args.json:
            print(f"Loading the latest {(args.ticker or stored.ticker).upper()} market data...")
        data = download_market_data(
            ticker=args.ticker or stored.ticker,
            start=args.start or stored.start,
            end=args.end,
            refresh=args.refresh,
        )
    if not args.json:
        print(f"Loading the trained model from {args.run_dir}...")
    prediction = forecast_next_session(data, args.run_dir, args.device)
    if args.json:
        print(json.dumps(prediction, indent=2))
        return 0

    percentage_change = (
        float(prediction["predicted_close"]) / float(prediction["previous_close"]) - 1
    ) * 100
    print("\nNext-session forecast")
    print(f"  Latest observation:  {prediction['last_observation']}")
    print(f"  Latest close:        {float(prediction['previous_close']):.2f}")
    print(f"  Expected session:    {prediction['forecast_date']}")
    print(f"  Estimated close:     {float(prediction['predicted_close']):.2f}")
    print(f"  Estimated change:    {percentage_change:+.2f}%")
    print("\nThis experimental forecast is not financial advice.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-lstm",
        description="Train an LSTM on stock prices and estimate the next closing price.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Train and evaluate a new model",
        description="Download stock data (or load a CSV), train the LSTM, and save the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_data = train.add_argument_group("data")
    train_data.add_argument("--ticker", default="GOOGL", help="Stock symbol to download")
    train_data.add_argument("--start", default="2015-01-01", help="First date, YYYY-MM-DD")
    train_data.add_argument("--end", help="Optional final date, YYYY-MM-DD")
    train_data.add_argument(
        "--data", help="Use a local CSV with date/open/high/low/close/volume"
    )
    train_data.add_argument(
        "--refresh", action="store_true", help="Download fresh data instead of using the cache"
    )
    train_data.add_argument(
        "--demo", action="store_true", help="Use the offline demo instead of real market data"
    )
    train_data.add_argument(
        "--demo-rows", type=int, default=1_000, help="Number of rows in the offline demo"
    )

    training = train.add_argument_group("training")
    training.add_argument(
        "--lookback", type=int, default=40, help="Previous sessions shown to the model"
    )
    training.add_argument("--epochs", type=int, default=40, help="Maximum training cycles")
    training.add_argument("--batch-size", type=int, default=64, help="Sequences per update")
    training.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden units")
    training.add_argument("--layers", type=int, default=2, help="Number of LSTM layers")
    training.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    training.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer step size")
    training.add_argument(
        "--patience", type=int, default=10, help="Epochs to wait before early stopping"
    )
    training.add_argument("--seed", type=int, default=42, help="Random seed")
    training.add_argument(
        "--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:0"
    )

    train_output = train.add_argument_group("output")
    train_output.add_argument(
        "--output-dir", default="artifacts", help="Parent folder for experiment results"
    )
    train_output.add_argument("--run-name", help="Optional name for the experiment folder")
    train.set_defaults(handler=_train)

    forecast = subparsers.add_parser(
        "forecast",
        help="Predict with a saved model",
        description="Load a trained experiment and estimate the next closing price.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    forecast.add_argument(
        "--run-dir", required=True, help="Experiment folder created by the train command"
    )
    forecast.add_argument("--ticker", help="Override the stock symbol saved in the experiment")
    forecast.add_argument("--start", help="Override the first download date")
    forecast.add_argument("--end", help="Optional final download date")
    forecast.add_argument("--data", help="Use a local OHLCV CSV instead of downloading")
    forecast.add_argument(
        "--refresh", action="store_true", help="Download fresh data instead of using the cache"
    )
    forecast.add_argument(
        "--demo", action="store_true", help="Use offline demo data for a demo-trained model"
    )
    forecast.add_argument(
        "--demo-rows", type=int, default=1_000, help="Rows used by the offline demo"
    )
    forecast.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda")
    forecast.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    forecast.set_defaults(handler=_forecast)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
