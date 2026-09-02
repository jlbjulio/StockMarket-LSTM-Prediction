# Stock Market LSTM Prediction

A Machine Learning project that uses an LSTM neural network to estimate the closing
price of a stock for the next trading session.

The project can download market data automatically, train the model, compare its results with
a simple reference prediction, create charts, and save everything needed to make another
prediction later. You do not need to edit the source code to use it.

> This project is for learning and experimentation. Its predictions are not financial advice.

## Quick start

### 1. Download the project

```bash
git clone https://github.com/jlbjulio/StockMarket-LSTM-Prediction.git
cd StockMarket-LSTM-Prediction
```

### 2. Create a Python environment

Python 3.10 or newer is recommended.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e ".[dev]"
```

### 3. Try the offline demo

This is the fastest way to check that everything works. It does not download anything from
the internet and usually finishes in a short time.

```bash
stock-lstm train --demo --epochs 5 --run-name demo
```

The results will be saved in `artifacts/demo/`.

## Train with a real stock

The following command downloads GOOGL daily prices and trains the model:

```bash
stock-lstm train --ticker GOOGL --start 2015-01-01
```

You can replace `GOOGL` with another ticker such as `AAPL`, `MSFT`, `NVDA`, or `AMZN`.

For a shorter first experiment, use fewer epochs:

```bash
stock-lstm train --ticker AAPL --start 2020-01-01 --epochs 10 --run-name apple-test
```

Market downloads are stored in `data/raw/`, so they do not need to be downloaded again on
every run. Use `--refresh` when you explicitly want to update the cached data.

## Understanding the results

At the end of training, the terminal displays:

- where the experiment was saved;
- the best training epoch;
- the model's error on data it did not see during training;
- the result of a simple reference prediction;
- whether the LSTM improved on that reference.

Each experiment has its own folder:

| File | What it contains |
| --- | --- |
| `test_predictions.png` | Chart comparing real prices, LSTM predictions, and the reference |
| `training_history.png` | Training and validation loss by epoch |
| `metrics.json` | Complete numerical results |
| `predictions.csv` | Every date, real value, and predicted value from the test period |
| `run.json` | Settings and data range used for the experiment |
| `model_state.pt` | Trained LSTM model |
| `feature_scaler.joblib` | Saved feature transformation |
| `target_scaler.joblib` | Saved prediction transformation |

The most useful first comparison is **LSTM RMSE vs. baseline RMSE**. Lower is better. If the
LSTM does not beat the reference prediction, the project reports that honestly instead of
presenting the result as a successful forecast.

## Make a new prediction

After training, pass the experiment folder to the `forecast` command:

```bash
stock-lstm forecast --run-dir artifacts/GOOGL_YYYYMMDD_HHMMSS
```

Replace the example folder with the path printed at the end of your training run. The command
downloads the latest prices, loads the saved model, and displays:

- the date and closing price of the latest observation;
- the next expected trading date;
- the estimated next closing price;
- the estimated percentage change.

The displayed date is the next business day and may need adjustment for market holidays.

## Use your own CSV file

You can train without downloading data:

```bash
stock-lstm train --data path/to/prices.csv --ticker MY_STOCK
```

The CSV must contain these columns:

```text
date, open, high, low, close, volume
```

Column names are not case-sensitive. Dates are sorted automatically, duplicated rows are
removed, and invalid prices are ignored.

## Use the Jupyter notebook

The notebook provides a visual walkthrough of the same workflow:

```bash
jupyter lab notebooks/lstm_walkthrough.ipynb
```

Open [notebooks/lstm_walkthrough.ipynb](notebooks/lstm_walkthrough.ipynb) if you prefer to
explore the data and charts one step at a time. The notebook uses the reusable Python package,
so the training code is kept in one place and behaves the same from the notebook or terminal.

## How the model works

In simple terms, the project follows these steps:

1. Load daily opening, high, low, closing, and volume data.
2. Create indicators describing recent returns, momentum, volatility, volume, and RSI.
3. Keep the oldest observations for training and the newest observations for testing.
4. Show the LSTM a window of previous trading sessions.
5. Estimate the return and closing price for the following session.
6. Compare the prediction with the real price and a simple baseline.

The data is never randomly shuffled. Transformations are learned from the training period
only, which prevents the model from seeing future information during training.

## Useful options

View every available command:

```bash
stock-lstm --help
stock-lstm train --help
stock-lstm forecast --help
```

Common training options:

| Option | Example | Purpose |
| --- | --- | --- |
| `--ticker` | `--ticker MSFT` | Stock symbol to download |
| `--start` | `--start 2018-01-01` | First date used |
| `--end` | `--end 2025-01-01` | Optional final date |
| `--epochs` | `--epochs 20` | Maximum training cycles |
| `--lookback` | `--lookback 40` | Previous sessions shown to the model |
| `--run-name` | `--run-name msft-test` | Easy-to-recognize output folder |
| `--device` | `--device cpu` | Force CPU training |
| `--refresh` | `--refresh` | Download market data again |

## Project layout

```text
src/stock_lstm/       Main data, model, training, evaluation, and CLI code
notebooks/            Jupyter walkthrough
tests/                Automated tests
artifacts/            Generated models, charts, and results
data/raw/             Cached market data
```

## Run the checks

```bash
ruff check .
pytest
```

GitHub Actions runs the same checks automatically on pushes and pull requests.

## Important limitations

Stock prices are affected by news, economic events, market conditions, and many factors that
are not present in historical price data. A good result in one period may not continue in
another period. The reported metrics also do not include transaction fees or trading spread.

The project is therefore best understood as a clean demonstration of time-series modeling,
evaluation, and deployment rather than a guaranteed trading system.

## License

MIT
