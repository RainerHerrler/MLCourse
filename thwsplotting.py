import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state, shuffle

def make_blobs(centers=2, random_state=4, n_samples=30):
    g = check_random_state(random_state)

    # fixed setup (2 centers, 2 features, std=1)
    centers = g.uniform(-10, 10, size=(centers, centers))
    neg = int(n_samples/2)
    pos = n_samples-neg
    n = [neg, pos]  # 30 samples split evenly

    X = np.vstack([centers[i] + g.normal(scale=1.0, size=(n[i], 2)) for i in range(2)])
    y = np.array([i for i in range(2) for _ in range(n[i])])

    X, y = shuffle(X, y, random_state=g)
    return X, y

def plotScatter(X, y, figsize=(6,5)):
    fig, axes = plt.subplots(1, 1, figsize=figsize) 
    discreteScatter(X,y,ax=axes)
    plt.legend(loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    return plt

def discreteScatter(X,y,ax):
    markers = ['o', '^']
    for i, class_value in enumerate(np.unique(y)):
        ax.scatter(
            X[y == class_value, 0],
            X[y == class_value, 1],
            marker=markers[i % len(markers)],
            s=100,                    # 👈 größere Punkte
            edgecolor='black',        # 👈 schwarzer Rand
            label=f"Class {class_value}"
        )