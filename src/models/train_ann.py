import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

# Optuna-found defaults:
DEFAULT_ANN_CONFIG = {
    "hidden_sizes": [428, 442, 220],
    "dropout": 0.2835776221997114,
    "epochs": 829,
    "lr": 0.0011676487575205433,
    "weight_decay": 6.370451204388144e-05
}

def load_params(json_path):
    if json_path is None:
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64,32], dropout=0.2):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_ann(csv_path, out_dir, cfg=None, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y_raw = df['PGA']
    y = np.log1p(y_raw).values.reshape(-1,1)

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = RobustScaler()
    selector = SelectKBest(score_func=f_regression, k='all')

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    X_sel = selector.fit_transform(X_scaled, y.flatten())

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=test_size, random_state=random_state)

    # config resolution: use defaults then override with cfg if provided
    config = DEFAULT_ANN_CONFIG.copy()
    if cfg:
        # map potential Optuna keys to our config if present
        # If cfg contains 'num_layers' and 'n_units_l{i}', we build hidden_sizes from that.
        if 'num_layers' in cfg:
            num_layers = int(cfg['num_layers'])
            hidden = []
            for i in range(num_layers):
                k = f'n_units_l{i}'
                if k in cfg:
                    hidden.append(int(cfg[k]))
            if len(hidden) == num_layers:
                config['hidden_sizes'] = hidden
        # override scalar keys if present
        for key in ['dropout','epochs','lr','weight_decay','hidden_sizes']:
            if key in cfg:
                config[key] = cfg[key]

    hidden_sizes = config['hidden_sizes']
    dropout = float(config['dropout'])
    epochs = int(config['epochs'])
    lr = float(config['lr'])
    weight_decay = float(config['weight_decay'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(X_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        # optional: you can print or log loss every N epochs

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    joblib.dump(selector, os.path.join(out_dir, 'selector.joblib'))
    # Save model artifact dict expected by inference loader
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[1],
        'hidden_sizes': hidden_sizes,
        'dropout': dropout
    }, os.path.join(out_dir, 'torch_ann.pt'))

    print(f"✅ Saved ANN artifact to {out_dir} (epochs={epochs}, lr={lr}, dropout={dropout}, hidden={hidden_sizes})")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to features CSV (with PGA column).")
    parser.add_argument("--out", default="models", help="Output directory to save artifacts.")
    parser.add_argument("--params", default=None, help="Optional JSON path with ANN/Optuna params to override defaults.")
    args = parser.parse_args()

    cfg = load_params(args.params) if args.params else None
    train_ann(args.csv, args.out, cfg=cfg)
