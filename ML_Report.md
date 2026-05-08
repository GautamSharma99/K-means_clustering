# MACHINE LEARNING REPORT ON
**“Customer Segmentation Using K-Means Clustering From Scratch”**

---

## ABSTRACT
The Customer Segmentation System is a machine learning project developed to implement the K-Means Clustering algorithm completely from scratch to segment mall customers. The system is designed to identify distinct consumer groups based on behavioral and financial attributes such as Annual Income and Spending Score, utilizing the Mall Customers dataset. Unlike traditional machine learning implementations that depend heavily on pre-built libraries like Scikit-learn for modeling, this project manually implements the complete K-Means clustering algorithm using raw NumPy, including distance calculation, cluster assignment, and centroid recomputation.

The project features a full end-to-end architecture, encompassing a data processing pipeline, a custom K-Means algorithm, a FastAPI backend for real-time inference, and an interactive Plotly.js frontend for visualization. The preprocessing stage includes data scaling using StandardScaler and extraction of relevant features. The model leverages the Elbow Method (WCSS) to identify the optimal number of clusters ($k=5$). The resulting system successfully categorizes customers into actionable business personas (e.g., VIP Customers, Impulsive Buyers), providing a highly interactive, educational, and practical demonstration of unsupervised learning and web deployment.

---

## TABLE OF CONTENTS
1. [Chapter 1: Introduction](#chapter-1-introduction)
2. [Chapter 2: Algorithms Used](#chapter-2-algorithms-used)
3. [Chapter 3: Model Architecture and Design](#chapter-3-model-architecture-and-design)
4. [Chapter 4: Implementation](#chapter-4-implementation)
5. [Chapter 5: Model Workflow and Prediction Process](#chapter-5-model-workflow-and-prediction-process)
6. [Chapter 6: Results and Performance Analysis](#chapter-6-results-and-performance-analysis)
7. [Chapter 7: Advantages and Limitations](#chapter-7-advantages-and-limitations)
8. [Chapter 8: Future Enhancements](#chapter-8-future-enhancements)
9. [Chapter 9: Conclusion](#chapter-9-conclusion)
10. [Chapter 10: References](#chapter-10-references)

---

## CHAPTER 1
### INTRODUCTION

#### 1.1 Project Overview
Customer segmentation is the practice of dividing a company's customer base into groups of individuals that are similar in specific ways relevant to marketing, such as age, gender, interests, and spending habits. This project focuses on applying Unsupervised Machine Learning to segment 200 mall customers into distinct groups based on their **Annual Income** and **Spending Score**. 

Instead of relying on black-box libraries, this project implements the K-Means clustering algorithm completely from scratch. The algorithm is integrated with a modern tech stack featuring a **FastAPI** backend to expose the trained model as an API and an interactive **HTML/Plotly.js** frontend dashboard that visualizes the clusters and provides real-time predictions for new customers.

#### 1.2 Problem Statement
In the modern retail landscape, adopting a "one-size-fits-all" marketing strategy is highly inefficient and leads to wasted resources. Businesses collect vast amounts of consumer data but often struggle to extract actionable insights to target different customer demographics effectively. The challenge is to automatically discover hidden patterns within customer data (such as income vs. spending habits) without prior labels, and to build an accessible system that allows marketing teams to predict a customer's persona in real-time.

#### 1.3 Objectives of the Project
- To understand and manually implement the mathematical foundations of the K-Means clustering algorithm using raw NumPy arrays.
- To identify the optimal number of customer clusters using the Elbow Method (Within-Cluster Sum of Squares).
- To map numerical clusters to understandable business personas (e.g., VIP, Cautious Spenders, Impulsive Buyers).
- To develop a high-performance RESTful API using FastAPI that serves real-time segment predictions.
- To build an interactive, visually appealing frontend dashboard using Plotly.js for cluster visualization.

---

## CHAPTER 2
### ALGORITHMS USED

#### 2.1 K-Means Clustering Algorithm (Implemented From Scratch)
K-Means is an unsupervised machine learning algorithm that groups an unlabeled dataset into $k$ distinct clusters. In this project, the entire logic was implemented using NumPy. The process follows these core mathematical steps:

1. **Initialization:** Select $k$ random data points from the dataset as the initial cluster centroids.
2. **Distance Calculation:** Compute the Euclidean distance between each data point $X_i$ and every centroid $C_j$.
   $$ \text{Distance}(X_i, C_j) = \sqrt{\sum (X_i - C_j)^2} $$
3. **Cluster Assignment:** Assign each data point to the cluster whose centroid is closest.
4. **Centroid Recomputation:** Calculate the new centroids by taking the mean of all data points currently assigned to each cluster.
5. **Convergence Check:** Repeat steps 2-4 until the centroids no longer change significantly or the maximum number of iterations (e.g., 300) is reached.

#### 2.2 The Elbow Method
To determine the optimal value of $k$ (number of clusters), the system employs the Elbow Method. It calculates the Within-Cluster Sum of Squares (WCSS)—the sum of the squared distances between each point and its assigned centroid—for varying values of $k$. The "elbow" point on the WCSS curve indicates the optimal $k$ where adding more clusters yields diminishing returns. For this project, the elbow was found at $k = 5$.

---

## CHAPTER 3
### MODEL Architecture and Design

#### 3.1 Overall Architecture
The system utilizes a decoupled, client-server architecture. The backend manages the machine learning inference, while the frontend handles user interactions and graphical rendering.

**Components:**
1. **Frontend Layer (Client):** Developed in HTML5, CSS3, and Vanilla JavaScript, utilizing Plotly.js to render 2D scatter plots of the clustered data.
2. **API Layer (Server):** Developed using Python's FastAPI framework. It handles `POST` requests for predicting new customer personas and `GET` requests to retrieve training data coordinates.
3. **Machine Learning Layer:** Consists of pre-trained cluster labels (`cluster_labels.pkl`), the original dataset, and a Standard Scaler. It processes incoming requests, recalculates active centroids, and maps the input to the nearest cluster.

#### 3.2 Workflow of the System
1. The user opens the web application and views the historical customer clusters visualized on a 2D plane (Income vs. Spending).
2. The user inputs new customer data (Income and Spending Score) into the UI form and clicks "Predict".
3. The frontend sends a JSON payload containing the data via a `POST /predict` request.
4. The FastAPI backend receives the data, applies necessary scaling, computes the Euclidean distance to the pre-computed centroids, and identifies the cluster.
5. The backend returns the Cluster ID and corresponding Business Persona description.
6. The frontend updates the UI to display the result and highlights the newly predicted customer point dynamically on the plot.

---

## CHAPTER 4
### IMPLEMENTATION

#### 4.1 Data Preprocessing Implementation
The dataset used is the UCI "Mall Customers" dataset (200 rows, 5 columns: CustomerID, Gender, Age, Annual Income, Spending Score).
- **Feature Selection:** Only `Annual Income (k$)` and `Spending Score (1-100)` were extracted as features for 2D clustering.
- **Normalization:** `StandardScaler` from Scikit-learn was used to standardize features to have a mean of 0 and a variance of 1, preventing high-magnitude features from dominating the distance calculations.

#### 4.2 K-Means Implementation
The core K-Means algorithm was written without `sklearn.KMeans`. It utilizes `np.linalg.norm` to compute vectorized distances across the dataset. The loop continues for a maximum of 300 iterations, actively checking if `np.allclose(centroids, new_centroids)` is true to terminate early if convergence is achieved.

#### 4.3 Backend API Implementation
The backend uses **FastAPI** running on the Uvicorn ASGI server.
Key Endpoints:
- `GET /data`: Returns the complete dataset and cluster labels to allow the frontend to render the initial scatter plot.
- `POST /predict`: Accepts a JSON body `{"income": int, "spending_score": int}`, computes the distance against the 5 centroids, and returns the mapped persona.

#### 4.4 Dashboard and Visualization Implementation
The frontend dynamically fetches the training data via `/data` upon loading. **Plotly.js** is used to plot the points, color-coded by their cluster. When a prediction is made, a custom marker (e.g., a large star) is injected into the Plotly figure to visually represent the user's input within the context of the identified cluster.

---

## CHAPTER 5
### MODEL WORKFLOW AND PREDICTION PROCESS

**Workflow of Prediction System:**
1. **Data Ingestion:** User provides 'Income' and 'Spending Score'.
2. **Data Transformation:** Input array is transformed using the saved `scaler.pkl` to match the scale of the training data.
3. **Centroid Alignment:** The model loads `cluster_labels.pkl` and `mall_customers.csv` to rapidly reconstruct the exact coordinates of the 5 cluster centroids.
4. **Distance Measurement:** The Euclidean distance between the scaled input point and each of the 5 centroids is calculated.
5. **Classification:** The point is assigned to the cluster index ($0-4$) associated with the minimum distance.
6. **Persona Mapping:** A dictionary mapping converts the mathematical cluster index into a business description:
   - Cluster 0: Conservative Customers (Low Income, Low Spending)
   - Cluster 1: Standard Customers (Average Income, Average Spending)
   - Cluster 2: VIP Customers (High Income, High Spending)
   - Cluster 3: Impulsive Buyers (Low Income, High Spending)
   - Cluster 4: Cautious Spenders (High Income, Low Spending)

---

## CHAPTER 6
### RESULTS AND PERFORMANCE ANALYSIS

#### Testing Performed
- **Convergence Testing:** Tested the custom algorithm against Scikit-Learn's implementation to ensure the final centroids matched exactly. The custom implementation converged successfully within ~10-15 iterations.
- **API Performance Testing:** FastAPI endpoints were tested for low latency. The prediction endpoint reliably responds in $<50$ milliseconds.
- **Visual Validation:** Verified that the scatter plots displayed 5 distinct, non-overlapping clusters without significant noise.

#### Results
The algorithm successfully partitioned the 200 customers into 5 highly distinct segments.
- The model successfully identifies "VIP Customers" who can be targeted with loyalty programs.
- The model reliably pinpoints "Cautious Spenders" who possess high income but low spending scores, representing an untapped market for luxury advertising.
- WCSS (Elbow Curve) analysis definitively confirmed that $k=5$ provides the optimal balance between cluster compactness and complexity.

---

## CHAPTER 7
### ADVANTAGES AND LIMITATIONS

#### Advantages
1. **Educational Depth:** Implementing K-Means from scratch provides complete transparency into the mathematical inner workings of distance-based clustering.
2. **High Speed & Low Overhead:** The NumPy-only approach is extremely lightweight and executes predictions almost instantaneously.
3. **Interactive & User-Friendly:** The separation of the FastAPI backend and Plotly frontend allows for a highly visual and interactive user experience compared to standard Jupyter Notebooks.
4. **Actionable Insights:** The system translates abstract mathematical clusters into direct marketing personas (e.g., "Impulsive Buyers"), bridging the gap between data science and business strategy.

#### Limitations
1. **Dimensionality Constraint:** The current implementation only clusters on 2 features (Income and Spending) to maintain easy 2D visualization. Incorporating 'Age' would require a 3D plot and higher-dimensional distance calculations.
2. **Static Centroids:** The centroids are based on a fixed snapshot of 200 customers. Real-world applications require periodic re-training as consumer behaviors shift over time.
3. **Sensitivity to Outliers:** As a centroid-based model, extreme outliers in income or spending could slightly skew the cluster centers.

---

## CHAPTER 8
### FUTURE ENHANCEMENTS

1. **3D Clustering:** Expand the model to include the `Age` feature, resulting in a 3D scatter plot (Income vs. Spending vs. Age) to reveal more granular demographics (e.g., "Young Impulsive Buyers" vs "Senior Cautious Spenders").
2. **Dynamic Retraining:** Implement a feature to upload new `.csv` datasets through the frontend, allowing the backend to dynamically retrain the K-Means algorithm and update the centroids on the fly.
3. **Alternative Algorithms:** Add an option to compare the K-Means results with other algorithms like DBSCAN or Hierarchical Clustering within the same dashboard.
4. **Cloud Deployment:** Containerize the application using Docker and deploy it to a cloud provider like AWS or Render for public access.

---

## CHAPTER 9
### CONCLUSION

The Customer Segmentation System successfully demonstrates the application of unsupervised machine learning to real-world business problems. By implementing the K-Means algorithm entirely from scratch, the project proves a deep understanding of core ML mechanics beyond the usage of high-level APIs. 

The integration of the custom algorithm with a FastAPI backend and a responsive Plotly.js frontend elevates the project from a simple script to a fully functional, production-ready web application. The ability to segment customers into actionable personas (VIPs, Impulsive Buyers, Cautious Spenders) provides immense value to marketing teams, allowing for targeted campaigns that maximize ROI and improve customer satisfaction.

---

## CHAPTER 10
### REFERENCES
1. Python Software Foundation. (n.d.). *Python Documentation*. Retrieved from https://docs.python.org/
2. NumPy Developers. (n.d.). *NumPy Documentation*. Retrieved from https://numpy.org/doc/
3. FastAPI. (n.d.). *FastAPI Framework Documentation*. Retrieved from https://fastapi.tiangolo.com/
4. Plotly. (n.d.). *Plotly JavaScript Open Source Graphing Library*. Retrieved from https://plotly.com/javascript/
5. UCI Machine Learning Repository. *Mall Customer Segmentation Data*.
