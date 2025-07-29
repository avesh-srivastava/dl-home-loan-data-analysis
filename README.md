# Cohorts of Songs

## Project Overview

In today's digital landscape, personalized experiences are a key driver of user engagement. Whether it's on an e-commerce platform or a music streaming service like **Spotify**, users expect content recommendations tailored to their preferences.

This project aims to perform **unsupervised learning (clustering)** on Spotify song data to group similar songs into **cohorts** based on their acoustic and musical features. These clusters can be used to improve music recommendation engines by suggesting songs that belong to the same cohort a user prefers.

---

## Problem Statement

Spotify, a leading audio streaming platform with over **456 million monthly users**, wants to **enhance its recommendation engine** by identifying natural groupings of songs. 

**Objective:**  
As a data scientist, your goal is to:
- Perform **Exploratory Data Analysis (EDA)**
- Apply **K-Means Clustering**
- Use **PCA for Visualization**
- Derive insights about different song cohorts

---

## Dataset

The dataset contains various features describing audio characteristics of songs. Some key features include:

- `danceability`: How suitable a track is for dancing.
- `energy`: Intensity and activity level of the track.
- `tempo`: The overall tempo in BPM (beats per minute).
- `loudness`, `speechiness`, `valence`, `instrumentalness`, `acousticness`, etc.

> A full **Dataset Dictionary** is provided in the [`data/`](./data/) folder as an Excel file:  
> [`dataset_dictionary.xlsx`](./data/dataset_dictionary.xlsx)

---

## Technologies & Tools Used

- Python 3.x
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for clustering and PCA)

---

## Methodology

1. **Data Cleaning**
   - Removed irrelevant columns
   - Handled null/missing values
   - Scaled features using MinMaxScaler

2. **EDA (Exploratory Data Analysis)**
   - Visualized feature distributions
   - Correlation heatmaps
   - Pairplots of key audio features

3. **Dimensionality Reduction**
   - Applied **PCA** to reduce to 2D space
   - Visualized clusters with scatter plots

4. **Clustering**
   - Applied **K-Means** clustering algorithm
   - Used **Elbow Method** to identify optimal `k`
   - Evaluated clustering performance visually

5. **Result Interpretation**
   - Analyzed characteristics of each cluster
   - Identified potential patterns in tempo, energy, and valence across cohorts

---

## Key Results

- Optimal clusters: **(e.g., k=4 based on elbow curve)**  
- Clusters clearly segregated based on energy, tempo, and valence
- Songs with similar danceability and energy naturally grouped together
- PCA visualization helped validate meaningful clustering

---

## Installation & Usage

### Requirements
- Python 3.8+
- Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- Scikit-learn (for clustering and PCA)

### Run the notebook
> git clone https://github.com/avesh-srivastava/ml-unsupervised-song-clustering.git
> cd ml-unsupervised-song-clustering
> pip install -r requirements.txt
> jupyter notebook notebooks/Cohorts_of_Songs.ipynb

---

## Folder Structure

```
ml-unsupervised-song-clustering/
├── data/
│ └── dataset_dictionary.xlsx
│ └── rolling_stones_spotify.csv
├── notebooks/
│ └──Cohorts_of_Songs.ipynb
├── README.md
└── requirements.txt (optional)

```

---

## Future Improvements

- Try DBSCAN or Hierarchical Clustering for more dynamic cohort discovery
- Integrate with Spotify API for real-time recommendations
- Use t-SNE or UMAP for more granular cluster visualization

--- 

## License

This project is for educational purposes only. Feel free to reuse the code with proper attribution.