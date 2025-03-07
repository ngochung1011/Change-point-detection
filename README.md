# Changepoint Detection for Vietnamese Financial Market

## 📌 Introduction
This project focuses on detecting changepoints (anomalies) in the **Vietnamese financial market**, particularly in the **VN30 index**. Changepoint detection is crucial for identifying significant shifts in market trends, helping investors make informed decisions. 

The study refines the **MOIRAI** model, a transformer-based time series forecasting approach, and applies it to financial data to detect anomalies based on prediction errors.

---

## 📑 Project Overview
### Objectives:
- **Collect** and preprocess VN30 stock price data.
- **Fine-tune** the MOIRAI model for time-series forecasting.
- **Detect changepoints** based on prediction errors using Mean Absolute Error (MAE).
- **Evaluate model performance** and compare with traditional methods.

### Key Features:
- Uses **deep learning-based probabilistic forecasting**.
- Implements **MOIRAI fine-tuning** for anomaly detection.
- Compares **changepoint-based investment strategy** with traditional methods.

---

## 📂 Project Structure
```
├── data/                      # Raw and processed stock data
│   ├── vn30_data.csv          # Original dataset
│   ├── processed_vn30_data.csv # Preprocessed data
│
├── models/                    # Trained models
│   ├── moirai_finetuned.pth   # Fine-tuned MOIRAI model
│
├── src/                       # Source code
│   ├── data_preprocessing.py  # Preprocesses VN30 data
│   ├── moirai_finetuning.py   # Fine-tunes MOIRAI model
│   ├── changepoint_detection.py # Detects anomalies
│   ├── evaluation.py          # Evaluates model performance
│
├── results/                   # Stores evaluation results
│   ├── detected_anomalies.txt # Detected changepoints
│   ├── performance_metrics.txt # Model performance summary
│
├── README.md                  # Project documentation
```

---

## ⚙️ Installation
To run this project, install the required dependencies:
```sh
pip install numpy pandas torch transformers scikit-learn
```
For a virtual environment:
```sh
python -m venv env
source env/bin/activate  # On Mac/Linux
# OR
env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 1️⃣ Data Preprocessing
```python
from src.data_preprocessing import load_data, normalize_data
data = load_data("data/vn30_data.csv")
norm_data, scaler = normalize_data(data)
norm_data.to_csv("data/processed_vn30_data.csv")
```

### 2️⃣ Fine-Tune the MOIRAI Model
```python
from src.moirai_finetuning import finetune_moirai
train_data = torch.tensor(np.load("data/processed_vn30_data.npy"))
finetune_moirai(train_data, epochs=10)
```

### 3️⃣ Detect Changepoints
```python
from src.changepoint_detection import detect_anomalies
predictions = np.load("results/predictions.npy")
actual_values = np.load("results/actual_values.npy")
anomalies = detect_anomalies(predictions, actual_values, k=10)
print("Detected anomalies:", anomalies)
```

### 4️⃣ Evaluate Model Performance
```python
from src.evaluation import evaluate_model
mae, mse = evaluate_model(predictions, actual_values)
print(f"MAE: {mae}, MSE: {mse}")
```

---

## 📊 Model Performance
| Model          | MSE   | MAE   | MASE  | MAPE  |
|---------------|------ |------ |------ |------ |
| MOIRAI-Base (Pretrained) | 0.0211 | 0.1161 | 3.3849 | 0.2888 |
| MOIRAI-Base (Fine-Tuned) | 0.0177 | 0.1048 | 3.0538 | 0.2862 |

### Comparison with Traditional Investment Strategies
| Strategy | Avg. Return (%) |
|----------|---------------|
| Bollinger Bands | 5.2% |
| Moving Average Crossover | 6.8% |
| Changepoint-Based Strategy | **9.3%** |

---

## 🔥 Future Improvements
- **Integrate additional financial indicators** like trading volume and news sentiment analysis.
- **Optimize hyperparameters** for MOIRAI fine-tuning.
- **Deploy a real-time anomaly detection API** for stock market insights.

---

## 🤝 Contributors
📌 **Project Lead:** Đặng Ngọc Hưng  
📌 **Supervisors:** TS. Ngô Minh Mẫn, TS. Vũ Đức Thịnh

If you find this project useful, feel free to fork and contribute! 🚀
