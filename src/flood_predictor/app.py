from logging import root
import sys
import numpy as np
import pandas as pd
import torch
import tkinter as tk

from flood_predictor.__main__ import TCN, load_and_merge

def predict(gage_csv, discharge_csv):
    model_file = "src/hour_prediction_model.pth"

    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    seq_len = checkpoint["seq_len"]
    threshold = checkpoint["threshold"]

    data = load_and_merge(gage_csv, discharge_csv)

    data["gage_diff"] = data["gage_height"].diff().fillna(0)
    features = data[["gage_height", "discharge", "gage_diff"]].values.astype(np.float32)

    recent = features[-seq_len:]
    recent = (recent - mean) / std

    x = torch.tensor(recent).unsqueeze(0)

    model = TCN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predicted_height = model(x).item()

    flood_prob = 1 / (1 + np.exp(-5 * (predicted_height - threshold)))
    flood_prob_percent = flood_prob * 100

    if flood_prob_percent < 10:
        risk_level = "Very Low"
    elif flood_prob_percent < 30:
        risk_level = "Low"
    elif flood_prob_percent < 60:
        risk_level = "Moderate"
    elif flood_prob_percent < 85:
        risk_level = "High"
    else:
        risk_level = "Severe"

    return flood_prob_percent, risk_level

def main():
    if len(sys.argv) != 3:
        print("Usage: flood-predictor-app <gage_height_csv> <discharge_csv>")
        return 1

    gage_csv = sys.argv[1]
    discharge_csv = sys.argv[2]

    flood_prob_percent, risk_level = predict(gage_csv, discharge_csv)

    print(f"Predicted flash flood probability 1 hour from now: {flood_prob_percent:.3f}%")
    print(f"Risk Level: {risk_level}")

    root = tk.Tk()
    root.title("Flood Predictor")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width // 2) - (800 // 2)
    y = (screen_height // 2) - (500 // 2)

    root.geometry(f"800x500+{x}+{y}")

    text = (
        f"Flash flood probability in the next hour: {flood_prob_percent:.3f}%\n"
        f"Risk Level: {risk_level}"
    )

    label = tk.Label(root, text=text, font=("Arial", 24), justify="center")
    label.pack(expand=True)

    root.mainloop()

    return 0

if __name__ == "__main__":
    sys.exit(main())