# 🚀 Customer Segmentation Visualizer (K-Means)

## 📌 Overview

This project is an end-to-end Machine Learning application that performs **customer segmentation using K-Means clustering**.

It allows users to:

* Input customer data (Income, Spending Score)
* Predict their cluster using a trained model (`.pkl`)
* Visualize the result on a cluster graph

---

## 🎯 Objective

Build an interactive system that:

* Uses a trained K-Means model
* Exposes predictions via FastAPI
* Provides a frontend UI for input and visualization

---

## 👥 Target Users

* Students (academic projects)
* Beginners in ML
* Demo users for clustering concepts

---

## 🧩 Features

### ✅ 1. Input Form

* Annual Income (k$)
* Spending Score (1–100)

### ✅ 2. Prediction API

* Accepts input
* Returns:

  * Cluster ID
  * Cluster Persona

### ✅ 3. Visualization

* Scatter plot of clusters
* Highlights user point

### ✅ 4. Cluster Interpretation

* Maps cluster → business persona

---

## 🏗️ System Architecture

```
Frontend (HTML/React)
        ↓
FastAPI Backend
        ↓
KMeans Model (.pkl)
        ↓
Prediction + Response
```

---

## ⚙️ Backend (FastAPI)

### 📁 Folder Structure

```
backend/
│
├── main.py
├── model.pkl
├── scaler.pkl
├── utils.py
└── requirements.txt
```

---

### 🚀 API Endpoints

#### 1. Health Check

```
GET /
```

#### 2. Predict Cluster

```
POST /predict
```

### Request:

```json
{
  "income": 70,
  "spending_score": 80
}
```

### Response:

```json
{
  "cluster": 1,
  "persona": "High Income, High Spending"
}
```

---

### 🧠 Backend Logic

1. Load model (`.pkl`)
2. Load scaler (if used)
3. Transform input
4. Predict cluster
5. Map cluster → persona
6. Return JSON

---

### 🔥 Cluster Mapping

```python
cluster_map = {
    0: "Low Income, Low Spending",
    1: "High Income, High Spending",
    2: "Low Income, High Spending",
    3: "High Income, Low Spending",
    4: "Average Customers"
}
```

---

## 🎨 Frontend

### 🖥️ Tech Options

* HTML + CSS + JS (Recommended)
* React + Tailwind (Advanced)

---

### 📄 UI Layout

#### 1. Title

```
Customer Segmentation Dashboard
```

#### 2. Input Form

* Income input
* Spending Score input
* Predict button

#### 3. Result Display

* Cluster ID
* Persona description

#### 4. Visualization

* Scatter plot
* Cluster colors
* Highlighted user point

---

## 📊 Visualization

### Tools

* Chart.js OR Plotly

### Graph Details

* X-axis → Income
* Y-axis → Spending Score
* Color → Cluster

---

## 🔌 API Flow

1. User enters data
2. Clicks “Predict”
3. Frontend sends POST request
4. Backend processes input
5. Returns cluster + persona
6. UI updates with result + visualization

---

## ⚠️ Edge Cases

* Empty input
* Invalid values
* Out-of-range inputs

---

## 📦 Deployment

### Backend

* Render / Railway

### Frontend

* Netlify / Vercel

---

## 🧠 Future Enhancements

* Upload CSV for bulk clustering
* Real-time clustering animation
* 3D clustering (add Age)
* Cluster comparison dashboard

---

## 🗣️ Viva Explanation

> This project implements an end-to-end machine learning pipeline where a K-Means clustering model is trained, deployed using FastAPI, and integrated with a frontend interface for real-time prediction and visualization of customer segments.

---

## 🔧 Tech Stack

* Python
* NumPy, Pandas
* FastAPI
* Scikit-learn (for scaling)
* Chart.js / Plotly
* HTML/CSS/JS or React

---

## 📁 How to Run

### Backend

```bash
uvicorn main:app --reload
```

### Frontend

* Open `index.html`
  OR
* Run React app

---

## ✅ Deliverables

* Trained K-Means model (`.pkl`)
* FastAPI backend
* Interactive frontend
* Visualization dashboard

---
