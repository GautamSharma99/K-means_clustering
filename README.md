# 🛒 Customer Segmentation using K-Means Clustering

An end-to-end Machine Learning application that segments mall customers using **K-Means clustering implemented from scratch**, exposed via a **FastAPI** backend and visualized through a premium **interactive dashboard**.

---

## 🎯 Project Overview

This project applies unsupervised learning to segment 200 mall customers into 5 distinct groups based on their **Annual Income** and **Spending Score**, enabling targeted marketing strategies.

| Cluster | Segment | Description |
|---------|---------|-------------|
| 💎 | High Income, High Spending | VIP Customers — offer loyalty programs |
| ⭐ | High Income, Low Spending | Cautious Spenders — target with luxury campaigns |
| ⚠️ | Low Income, High Spending | Impulsive Buyers — offer EMI plans |
| 💤 | Low Income, Low Spending | Conservative Customers — offer discounts |
| 📊 | Average Income, Average Spending | Standard Customers — general promotions |

---

## 🏗️ Architecture

```
Frontend (HTML + Plotly.js)
        ↓  POST /predict
FastAPI Backend (Python)
        ↓
cluster_labels.pkl + mall_customers.csv
        ↓
K-Means Centroids (recomputed from labels)
        ↓
Cluster ID + Persona → JSON Response
```

---

## 📁 Project Structure

```
K-means_clustering/
│
├── backend/
│   ├── main.py              # FastAPI app (API + static serving)
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   └── index.html           # Interactive dashboard (Plotly.js)
│
├── mall_customers.csv        # Dataset (200 customers, 5 features)
├── cluster_labels.pkl        # Pre-trained cluster assignments
├── Untitled15.ipynb          # Original K-Means notebook (from scratch)
└── implementation.md         # Full project specification
```

---

## 🚀 How to Run

### 1. Set up virtual environment
```bash
python -m venv venv
venv\scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the backend
```bash
python -m uvicorn main:app --reload
```

### 4. Open the dashboard
Navigate to **http://127.0.0.1:8000/app** in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict cluster for a new customer |
| `GET` | `/data` | Get all training points + cluster map |
| `GET` | `/app` | Serve the frontend UI |
| `GET` | `/docs` | Interactive API documentation |

### Example — Predict Cluster
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 90, "spending_score": 85}'
```

**Response:**
```json
{
  "cluster": 3,
  "label": "High Income, High Spending",
  "description": "💎 VIP Customers — Premium spenders..."
}
```

---

## 🧠 K-Means Algorithm (from scratch)

Implemented in `Untitled15.ipynb` using **NumPy only** — no `sklearn.KMeans`:

```python
for _ in range(300):
    # 1. Compute Euclidean distances to each centroid
    distances = np.linalg.norm(X_scaled[:, None] - centroids, axis=2)

    # 2. Assign each point to the nearest centroid
    labels = np.argmin(distances, axis=1)

    # 3. Recompute centroids as cluster means
    new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(k)])

    # 4. Check convergence
    if np.allclose(centroids, new_centroids): break
    centroids = new_centroids
```

**Optimal K = 5** determined via the **Elbow Method** (WCSS vs K).

---

## 📊 Dataset

**Mall Customers Dataset** — `mall_customers.csv`

| Feature | Description |
|---------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Customer age (18–69) |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1–100) | Mall-assigned spending score |

- **200 rows**, **5 columns**
- No missing values

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Algorithm | K-Means from scratch (NumPy) |
| Backend | FastAPI + Uvicorn |
| Data Processing | Pandas, Scikit-learn (StandardScaler only) |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Visualization | Plotly.js |
| Language | Python 3.10+ |
