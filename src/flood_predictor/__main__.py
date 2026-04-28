import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F

def load_and_merge(gage_file, discharge_file):
    gage = pd.read_csv(gage_file, parse_dates=["time"])
    discharge = pd.read_csv(discharge_file, parse_dates=["time"])
    gage = gage[["time", "value"]].rename(columns={"value": "gage_height"})
    discharge = discharge[["time", "value"]].rename(columns={"value": "discharge"})
    gage = gage.sort_values("time")
    discharge = discharge.sort_values("time")
    data = pd.merge_asof(gage, discharge, on="time").reset_index(drop=True)

    return data

class FlashFloodDataset(Dataset):
    def __init__(self, df, seq_len, horizon):
        self.seq_len = seq_len
        self.horizon = horizon

        df = df.copy()
        df["gage_diff"] = df["gage_height"].diff().fillna(0)

        self.data = df[["gage_height", "discharge", "gage_diff"]].values.astype(np.float32)
        self.labels = df["gage_height"].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_len]
        y = self.labels[index+self.seq_len+self.horizon]
        return torch.tensor(x), torch.tensor(y)
    
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)

class TCN(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, kernel_size=2, output_size=1):
        super(TCN, self).__init__()

        self.tcn = nn.Sequential(
            CausalConv1d(input_size, hidden_size, kernel_size, dilation=1),
            nn.ReLU(),

            ResidualBlock(hidden_size, kernel_size, dilation=1),
            ResidualBlock(hidden_size, kernel_size, dilation=2),
            ResidualBlock(hidden_size, kernel_size, dilation=4),
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y.squeeze()


def main() -> int:
    start_time = time.time()

    if len(sys.argv) != 3:
        print("Usage: python flood_predictor.py <gage_csv> <discharge_csv>")
        sys.exit(1)

    print("---DATA STEP---")
    gage_csv = sys.argv[1]
    discharge_csv = sys.argv[2]

    data = load_and_merge(gage_csv, discharge_csv)

    print("---TRAIN/TEST SPLIT STEP---")
    SEQ_LEN = 72 # number of past timestamps to look at
    THRESHOLD = 2.20 # hardcoded flash flood threshold
    HORIZON = 12 # 12 * 5 minutes = 1 hour

    dataset = FlashFloodDataset(data, SEQ_LEN, HORIZON)

    train_size = int(0.8 * len(dataset))

    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

    train_data = dataset.data[:train_size]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8

    dataset.data = (dataset.data - mean) / std

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    print("---MODEL CREATION STEP---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device)

    model = TCN().to(device)

    print("---OPTIMIZER AND LOSS FUNCTION STEP---")
    optimizer = Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    print("---TRAINING STEP---")

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    print("---EVALUATION STEP---")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y_pred = model(x)
            all_preds.extend(y_pred.cpu().numpy())

    all_preds = np.array(all_preds)

    alert_probs = 1 / (1 + np.exp(-5 * (all_preds - THRESHOLD)))

    alert_probs_percent = np.round(alert_probs * 100, 3)

    print("Predicted flash flood probabilities (first 10):",
          [f"{p:.3f}%" for p in alert_probs_percent[:10]])

    print("Predicted heights (first 10):", np.round(all_preds[:10], 3))

    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "threshold": THRESHOLD
    }, "flood_predictor_model.pth")

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")

    return 0

if __name__ == "__main__":
    sys.exit(main())