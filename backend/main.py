"""
main.py — FastAPI Backend
  GET  /           → health check
  POST /predict    → predict cluster for income + spending_score
  GET  /data       → all training points + cluster_map (for chart)
  GET  /app        → serves the frontend UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import pickle, os

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Segmentation API",
    description="K-Means clustering for mall customer segmentation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.join(os.path.dirname(__file__), '..')
LABELS_PATH  = os.path.join(ROOT_DIR, 'cluster_labels.pkl')
DATA_PATH    = os.path.join(ROOT_DIR, 'mall_customers.csv')

# ── Serve Frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/app", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

# ── Load cluster_labels.pkl + CSV → recompute centroids ───────────────────────
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load original labels from user's pkl
with open(LABELS_PATH, 'rb') as f:
    y_train = pickle.load(f)          # np.ndarray of ints, shape (200,)

# 2. Load training features from CSV
df       = pd.read_csv(DATA_PATH)
X_train  = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. Fit scaler on training data (same as the notebook)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 4. Recompute centroids as cluster means (no re-training needed)
K         = int(y_train.max()) + 1
centroids = np.array([
    X_scaled[y_train == i].mean(axis=0)
    for i in range(K)
])

# 5. Build cluster_map: assign unique personas by ranking income & spending
centroids_orig = scaler.inverse_transform(centroids)
_inc = centroids_orig[:, 0]
_spd = centroids_orig[:, 1]

# Rank clusters (argsort gives ascending; [-1] = highest)
inc_rank = np.argsort(_inc)   # [lowest_inc … highest_inc]
spd_rank = np.argsort(_spd)   # [lowest_spd … highest_spd]

# Build a combined score: high income + high spending = VIP
score = _inc / _inc.max() + _spd / _spd.max()
order = np.argsort(score)     # ascending overall score

_PERSONA_ORDERED = [
    ("Low Income, Low Spending",        "💤 Conservative Customers — Value-sensitive shoppers. Target with discounts and combo offers."),
    ("Low Income, High Spending",       "⚠️  Impulsive Buyers — Spend beyond means. Offer budget deals and flexible EMI plans."),
    ("Average Income, Average Spending","📊 Standard Customers — Balanced profile. Suitable for general promotions and seasonal sales."),
    ("High Income, Low Spending",       "⭐ Cautious Spenders — High earners who save. Target with luxury campaigns and exclusive incentives."),
    ("High Income, High Spending",      "💎 VIP Customers — Premium spenders. Offer exclusive loyalty programs and premium products."),
]

cluster_map = {}
for rank, cid in enumerate(order):
    cluster_map[int(cid)] = _PERSONA_ORDERED[rank]

print(f"[OK] Loaded cluster_labels.pkl — {K} clusters, {len(y_train)} training points.")

# ── Schemas ───────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    income:         float = Field(..., ge=0,  le=500, description="Annual Income in k$",    json_schema_extra={"example": 70})
    spending_score: float = Field(..., ge=1,  le=100, description="Spending Score 1–100",   json_schema_extra={"example": 80})

class PredictResponse(BaseModel):
    cluster:     int
    label:       str
    description: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "Customer Segmentation API is running 🚀"}


@app.post("/predict", response_model=PredictResponse, tags=["ML"])
def predict(data: CustomerInput):
    """Predict the cluster for a new customer."""
    X_in    = np.array([[data.income, data.spending_score]])
    X_sc    = scaler.transform(X_in)
    dists   = np.linalg.norm(X_sc[:, None] - centroids, axis=2)
    cluster = int(np.argmin(dists, axis=1)[0])

    label, description = cluster_map[cluster]
    return PredictResponse(cluster=cluster, label=label, description=description)


@app.get("/data", tags=["ML"])
def get_training_data():
    """Return all training points with cluster assignments — used for scatter plot."""
    points = [
        {
            "income":   float(X_train[i, 0]),
            "spending": float(X_train[i, 1]),
            "cluster":  int(y_train[i])
        }
        for i in range(len(X_train))
    ]
    personas = {
        str(cid): {"label": v[0], "description": v[1]}
        for cid, v in cluster_map.items()
    }
    return {"points": points, "cluster_map": personas}


# -- Entrypoint ---------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
