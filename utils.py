import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_moons
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc

import seaborn as sns


def get_colors(y):
    colors = ('red', 'blue', 'green', 'gray', 'cyan', 'black', 'purple')
    cmap = ListedColormap(colors[:len(set(y))])
    return cmap


def close_enough(x, y):
    return np.allclose(x, y)


def get_accuracy(model, x, y):
    pred = model.predict_classes(x)
    return accuracy_score(y, pred)

def get_scores(model, x, y, use_predict = False):
    if use_predict:
        y_pred = model.predict(x)
    else:
        y_pred = model.predict_classes(x)
    return {'accuracy score': accuracy_score(y, y_pred),
            'precision score': precision_score(y, y_pred),
            'recall score' : recall_score(y, y_pred)}


def plot_confusion_matrix(y_cv, y_pred):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_cv, y_pred) #confusion matrix

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g')  #heatmap, fmt='g' converts scientific to float
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    #last version of seaborn is messing the heatmap when working with matplotlib
    #so we have to adjust these lines
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    
def plot_roc_curve(y: pd.DataFrame, y_hat: pd.DataFrame, title="Receiver operating characteristic"):
    """
    Function to plot the roc curve
    """
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (area = {auc(fpr, tpr):>.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def lenses_dataset():
    n_samples=200
    noise=0.1
    seed = 23
    return make_moons(n_samples=n_samples, noise=noise,
                               random_state=seed)


def make_classification_dataset(n_samples, n_features,
                                n_classes=2, n_informative=None,
                                noise=0.01, seed=None):
    if n_informative is None:
        n_informative = n_features
    return make_classification(n_samples=n_samples, n_features=n_features,
                               n_redundant=0, n_repeated=0,
                               n_informative=n_informative, n_classes=n_classes,
                               flip_y=noise, shuffle=True, random_state=seed)


def visualize_2d_classification(model, x, y, h=0.05):
    # set up plotting grid
    xx1, xx2 = np.meshgrid(np.arange(x[:,0].min(), x[:,0].max(), h),
                           np.arange(x[:,1].min(), x[:,1].max(), h))
    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = model.predict_classes(grid)
    Z = np.array(Z)
    Z = 1-Z.reshape(xx1.shape)  # for display purposes
    cmap = get_colors(y)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    f = plt.figure()
    # Decision boundary drawn on training set
    plt.contourf(xx1, xx2, Z, alpha=0.8)
    plt.scatter(x[:,0], x[:,1],
                c=y, cmap=cmap)

    plt.tight_layout()
    ax = f.get_axes()
    return f, ax[0]


def create_and_write_datasets(n_features, noise=0.01, seed=None):
    x, y = make_classification_dataset(
        200, n_features, n_classes=2, n_informative=None,
        noise=noise, seed=seed)

    df = pd.DataFrame(x, columns=["x{}".format(i+1) for i in range(n_features)])
    df["y"] = y
    df[:100].to_csv("train_{}features.csv".format(n_features), index=False)
    df[100:].to_csv("test_{}features.csv".format(n_features), index=False)
    return df


def create_datasets():
    # df = create_and_write_datasets(2, noise=0.4, seed=20)
    # df = create_and_write_datasets(50, noise=0.1, seed=40)
    # test_df = pd.read_csv("test_50features.csv")
    # test_df.drop("y", axis=1).to_csv("test_50features.csv", index=False)
    pass

def check_score():
    pass
