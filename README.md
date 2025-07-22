# StockMarket-LSTM-Prediction

This project focuses on stock market prediction using LSTM (Long Short-Term Memory) neural networks, applied to time series of financial data. It was developed as part of an academic project for the course **Aprendizaje Automático** at the **Universidad Tecnológica de Panamá (UTP)**.

## Description

The goal of this project is to build a model capable of predicting future stock market prices based on historical data. Deep Learning techniques are used, especially LSTM networks, due to their ability to capture sequential patterns and long-term dependencies in time series.

The notebook includes:
- Data preparation (normalization, sequence creation, train-test split).
- LSTM model construction.
- Training and evaluation.
- Prediction visualization.
- Performance metrics calculation.

## Technologies and Tools

- Python 3.x  
- Jupyter Notebook  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  

## Project Contents

- `ML_Project_LSTM.ipynb`: Main notebook containing the full project workflow.
- This `README.md` with full documentation.
- The data can be loaded from external sources (CSV, financial APIs) or simulated within the notebook.


## Requirements

Install the following dependencies before running the project:

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
```
Or in a single line:
```
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Execution Instructions

1. Clone this repository:
```
git clone https://github.com/tu-usuario/StockMarket-LSTM-Prediction.git
```
2. Navigate to the project directory:
```
cd StockMarket-LSTM-Prediction
```
3. Install dependencies (if not done yet):
```
pip install numpy pandas matplotlib scikit-learn tensorflow
```
4. Open the notebook:
```
jupyter notebook ML_Project_LSTM.ipynb
```


## Results

The notebook generates predictions of future prices and compares them with actual values. It includes visualizations and metrics such as:

- RMSE (Root Mean Square Error)  
- R² (Coefficient of Determination)  
- Directional Accuracy (correctly predicting price movement up or down)

## Possible Improvements

- Include more variables (volume, technical indicators, news sentiment).  
- Tune hyperparameters (layers, neurons, learning rate, epochs).  
- Test hybrid architectures (CNN+LSTM, Transformer for time series).  
- Implement temporal cross-validation.

## Disclaimer

This project is for academic and educational purposes only. It does not constitute financial advice or investment recommendation.

## Author

This repository was developed as part of the course **Aprendizaje Automático** during the 2025 semester by **Julio Lara**, from the **Ingeniería en Sistemas/Tecnología** program at the **Universidad Tecnológica de Panamá (UTP)**.

