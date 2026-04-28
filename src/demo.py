import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TCN(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, kernel_size=2, output_size=1):
        super(TCN, self).__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size-1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size-1),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y.squeeze()

def load_data(gage_file, discharge_file):
    gage = pd.read_csv(gage_file, parse_dates=["time"])
    discharge = pd.read_csv(discharge_file, parse_dates=["time"])

    gage = gage[["time", "value"]].rename(columns={"value": "gage_height"})
    discharge = discharge[["time", "value"]].rename(columns={"value": "discharge"})

    gage = gage.sort_values("time")
    discharge = discharge.sort_values("time")

    data = pd.merge_asof(gage, discharge, on="time").reset_index(drop=True)

    data["gage_diff"] = data["gage_height"].diff().fillna(0)

    return data

def main():

    if len(sys.argv) != 3:
        print("Usage: python demo.py <gage_csv> <discharge_csv>")
        sys.exit(1)

    print("\n--- LOADING MODEL ---")

    model_data = torch.load(
        "flood_model.pth",
        map_location="cpu",
        weights_only=False
    )

    model = TCN()
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    mean = model_data["mean"]
    std = model_data["std"]
    SEQ_LEN = model_data["seq_len"]

    print("Model loaded successfully")

    print("\n--- LOADING DATA ---")

    data = load_data(sys.argv[1], sys.argv[2])

    features = data[["gage_height", "discharge", "gage_diff"]].values.astype(np.float32)

    features = (features - mean) / std

    sequences = []

    for i in range(len(features) - SEQ_LEN):
        sequences.append(features[i:i+SEQ_LEN])

    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)

    print("\n--- RUNNING MODEL ---")

    all_preds = []

    with torch.no_grad():
        for i in range(0, len(sequences), 32):
            batch = sequences[i:i+32]
            preds = model(batch)
            all_preds.extend(preds.numpy())

    all_preds = np.array(all_preds)

    THRESHOLD = 2.20

    alert_probs = 1 / (1 + np.exp(-5 * (all_preds - THRESHOLD)))
    alert_probs_percent = alert_probs * 100

    print("\nPredicted flash flood probabilities (first 10):")
    for p in alert_probs_percent[:10]:
        print(f"{p:.6f}%")

    print("\nPredicted heights (first 10):")
    for p in all_preds[:10]:
        print(f"{p:.3f}")

    latest = alert_probs[-1]

    print("\n--- FLOOD RISK STATUS ---")
    if latest > 0.7:
        print("HIGH RISK: Flash flood likely!")
    elif latest > 0.4:
        print("MODERATE RISK: Stay alert")
    else:
        print("LOW RISK: Conditions normal")

    print(f"Latest probability: {latest * 100:.6f}%")

    time_axis = (data["time"].iloc[SEQ_LEN:] - data["time"].iloc[SEQ_LEN]).dt.total_seconds() / 60
    xlabel = "Time (relative to dataset start)"

    plt.figure(figsize=(12, 6))

    plt.plot(time_axis, all_preds, label="Predicted Gage Height")

    plt.axhline(y=THRESHOLD, color='r', linestyle='--', label="Flood Threshold")

    plt.title("Flash Flood Prediction Over Time")
    plt.xlabel(xlabel)
    plt.ylabel("Gage Height")
    plt.legend()

    plt.show()

    plt.figure(figsize=(12, 6))

    plt.plot(time_axis, alert_probs * 100, label="Flood Probability (%)")

    plt.axhline(y=70, color='r', linestyle='--', label="High Risk (70%)")
    plt.axhline(y=40, color='orange', linestyle='--', label="Moderate Risk (40%)")

    plt.title("Flash Flood Risk Probability Over Time")
    plt.xlabel(xlabel)
    plt.ylabel("Probability (%)")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()