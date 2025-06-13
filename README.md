# AI-ML-Internship-tasks
# Task 1: Iris Dataset Exploration

## Objective
Perform exploratory data analysis (EDA) on the Iris dataset to understand feature distributions, relationships between variables, and identify outliers.

## Dataset Used
**Iris Dataset** (included in Seaborn)
- 150 samples of iris flowers (50 per species)
- Features:
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm)
  - `petal_width` (cm)
- Target:
  - `species` (setosa, versicolor, virginica)

## Methods Applied
1. **Data Exploration**
   - Statistical summaries (`.describe()`)
   - Missing value checks
2. **Visualizations**
   - Pairplot (scatter matrix)
   - Histograms
   - Box plots
3. **Tools**
   - Python (Pandas, Seaborn, Matplotlib)
   - Jupyter Notebook

## Key Findings
1. **Species Separation**:
   - Setosa is easily distinguishable by petal measurements
   - Virginica has the largest flowers on average

2. **Feature Importance**:
   - Petal measurements are better classifiers than sepal measurements
   - Petal length shows the clearest separation between species

3. **Outliers**:
   - Few outliers in sepal_width for setosa
   - Virginica shows the widest range in petal sizes

## Sample Visualization
![pairplot](https://github.com/user-attachments/assets/eee79a2a-915f-4119-9a72-a0dbe9b705db)
)

## How to Run
1. Install requirements:
   ```bash
   pip install pandas seaborn matplotlib jupyter
