# Machine Learning from Scratch

NumPy implementations of classic machine learning algorithms, trained and evaluated on the [UCI Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) dataset.

Each notebook loads the data with scikit-learn, then fits a model written from scratch (no `sklearn` estimators). Scaling, train/test splits, and metrics use scikit-learn helpers so the focus stays on the algorithms.

## Notebooks

| Notebook | Algorithm |
| --- | --- |
| `Least_Squares.ipynb` | Least squares classification (one-hot targets, closed-form weights) |
| `Logistic_Regression.ipynb` | Multiclass logistic regression |
| `Naive_Bayes.ipynb` | Gaussian Naive Bayes |
| `K-Nearest_Neighbors.ipynb` | k-nearest neighbors |
| `Support_Vector_Machines.ipynb` | Support vector machines |
| `Multilayer_Perceptron.ipynb` | Multilayer perceptron |
| `K-Means.ipynb` | k-means clustering |
| `PCA.ipynb` | Principal component analysis |

## Setup

```bash
pip install numpy scikit-learn matplotlib
```

Open any notebook and run the cells in order. The Wine split uses `random_state=42` and an 80/20 train/test split where a split applies.
