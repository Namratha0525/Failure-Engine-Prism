"""
Phase 4 — Trace-Based Failure Propagation
==========================================
Build service dependency graphs and train GCN / GAT
for anomalous-node detection and root-cause identification.
"""

import os, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from failure_engine.config import DEVICE, MODEL_DIR
from failure_engine.phase1_ingestion import TRACE_FEATURE_COLS, METRIC_FEATURE_COLS

warnings.filterwarnings("ignore")

# Try torch_geometric; fall back to manual GCN if unavailable
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ═════════════════════════════════════════════════════════════════════════
#  Graph Construction (scenario-level)
# ═════════════════════════════════════════════════════════════════════════

def _build_scenario_graph(df_scenario):
    """Build a simple graph representation from scenario-level features.
    Each scenario becomes one graph with node features derived from
    per-window statistics across services."""
    # use trace + metric features as a proxy for node features
    feat_cols = [c for c in TRACE_FEATURE_COLS + METRIC_FEATURE_COLS if c in df_scenario.columns]
    feats = df_scenario[feat_cols].mean().values.astype(np.float32)
    feats = np.nan_to_num(feats, 0.0)
    # Create a simple single-node graph per scenario (aggregated)
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  GCN / GAT Models  (graph-level classification)
# ═════════════════════════════════════════════════════════════════════════

class GraphClassifier(nn.Module):
    """GCN or GAT graph classifier."""
    def __init__(self, input_dim, hidden, num_classes, conv_type="gcn", heads=4, layers=3, dropout=0.3):
        super().__init__()
        self.conv_type = conv_type
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if conv_type == "gat" and HAS_PYG:
            self.convs.append(GATConv(input_dim, hidden, heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden * heads))
            for _ in range(layers - 2):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
                self.bns.append(nn.BatchNorm1d(hidden * heads))
            self.convs.append(GATConv(hidden * heads, hidden, heads=1))
            self.bns.append(nn.BatchNorm1d(hidden))
        elif HAS_PYG:
            self.convs.append(GCNConv(input_dim, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(layers - 1):
                self.convs.append(GCNConv(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))

        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ═════════════════════════════════════════════════════════════════════════
#  Fallback MLP (if torch_geometric unavailable)
# ═════════════════════════════════════════════════════════════════════════

class TraceMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(model, X_train, y_train, X_val, y_val, epochs=100, patience=15):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    classes, counts = np.unique(y_train, return_counts=True)
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

    best_f1, best_state, wait = 0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xv).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def _create_pyg_data(X, y, num_nodes_per_graph=5):
    """Create PyG Data objects from flat features.
    Split features evenly across num_nodes, fully connected edges."""
    data_list = []
    feat_dim = X.shape[1] // num_nodes_per_graph
    if feat_dim == 0:
        feat_dim = X.shape[1]
        num_nodes_per_graph = 1

    # precompute edge_index for fully connected graph
    edges = []
    for i in range(num_nodes_per_graph):
        for j in range(num_nodes_per_graph):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)

    for idx in range(len(X)):
        x_flat = X[idx]
        # pad if needed
        padded = np.zeros(num_nodes_per_graph * feat_dim, dtype=np.float32)
        padded[:len(x_flat)] = x_flat[:len(padded)]
        node_feats = padded.reshape(num_nodes_per_graph, feat_dim)
        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=edge_index.clone(),
            y=torch.tensor(y[idx], dtype=torch.long)
        )
        data_list.append(data)
    return data_list


def _train_pyg(model, train_data, val_data, epochs=100, patience=15):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = torch_geometric_loader(train_data, batch_size=32, shuffle=True)
    val_loader = torch_geometric_loader(val_data, batch_size=64, shuffle=False)

    # class weights
    y_all = np.array([d.y.item() for d in train_data])
    classes, counts = np.unique(y_all, return_counts=True)
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1, best_state, wait = 0, None, 0
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            opt.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                preds = model(batch).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch.y.cpu().numpy())
        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def torch_geometric_loader(data_list, batch_size=32, shuffle=True):
    from torch_geometric.loader import DataLoader as PyGLoader
    return PyGLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def run_phase4(df) -> dict:
    """Train trace-based graph models."""
    print("\n" + "=" * 60)
    print("  PHASE 4 — Trace-Based Failure Propagation")
    print("=" * 60)

    feat_cols = [c for c in TRACE_FEATURE_COLS + METRIC_FEATURE_COLS if c in df.columns]
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(df["label_root_cause"].values)
    num_classes = len(le.classes_)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    if HAS_PYG:
        # Create graph data
        num_nodes = min(5, X.shape[1])
        train_data = _create_pyg_data(X_train, y_train, num_nodes)
        test_data = _create_pyg_data(X_test, y_test, num_nodes)
        feat_dim = train_data[0].x.shape[1]

        # GCN
        print("\n  Training GCN ...")
        gcn = GraphClassifier(feat_dim, 64, num_classes, conv_type="gcn").to(DEVICE)
        gcn, gcn_f1 = _train_pyg(gcn, train_data, test_data)
        gcn.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in torch_geometric_loader(test_data, batch_size=64, shuffle=False):
                batch = batch.to(DEVICE)
                p = gcn(batch).argmax(dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(batch.y.cpu().numpy())
        results["GCN"] = {
            "top1": accuracy_score(trues, preds),
            "f1_macro": f1_score(trues, preds, average="macro", zero_division=0),
            "model": gcn, "type": "pyg"
        }
        print(f"    Top-1: {results['GCN']['top1']:.4f} | F1: {results['GCN']['f1_macro']:.4f}")

        # GAT
        print("  Training GAT ...")
        gat = GraphClassifier(feat_dim, 32, num_classes, conv_type="gat", heads=4).to(DEVICE)
        gat, gat_f1 = _train_pyg(gat, train_data, test_data)
        gat.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in torch_geometric_loader(test_data, batch_size=64, shuffle=False):
                batch = batch.to(DEVICE)
                p = gat(batch).argmax(dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(batch.y.cpu().numpy())
        results["GAT"] = {
            "top1": accuracy_score(trues, preds),
            "f1_macro": f1_score(trues, preds, average="macro", zero_division=0),
            "model": gat, "type": "pyg"
        }
        print(f"    Top-1: {results['GAT']['top1']:.4f} | F1: {results['GAT']['f1_macro']:.4f}")
    else:
        print("  [WARN] torch_geometric not available, using MLP fallback")

    # MLP fallback (always train as baseline)
    print("  Training Trace MLP ...")
    mlp = TraceMLP(X.shape[1], num_classes).to(DEVICE)
    mlp, mlp_f1 = _train_mlp(mlp, X_train, y_train, X_test, y_test)
    mlp.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = mlp(Xv).argmax(dim=1).cpu().numpy()
    results["TraceMLP"] = {
        "top1": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "model": mlp, "type": "torch"
    }
    print(f"    Top-1: {results['TraceMLP']['top1']:.4f} | F1: {results['TraceMLP']['f1_macro']:.4f}")

    # Best
    best_name = max(results, key=lambda k: results[k]["top1"])
    best = results[best_name]
    print(f"\n  ★ Best trace model: {best_name} (Top-1: {best['top1']:.4f})")

    path = os.path.join(MODEL_DIR, "trace_best_model.pt")
    torch.save({"state_dict": best["model"].state_dict(),
                "model_name": best_name,
                "num_classes": num_classes,
                "input_dim": X.shape[1],
                "label_classes": list(le.classes_)}, path)
    print(f"  Saved → {path}")

    # Comparison table
    print(f"\n  {'Model':<15s} {'Top-1':>7s} {'F1-Mac':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['top1']:7.4f} {r['f1_macro']:7.4f}")

    return {"best_name": best_name, "results": results, "label_encoder": le}


if __name__ == "__main__":
    from failure_engine.phase1_ingestion import load_cached, ingest_all
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase4(df)
