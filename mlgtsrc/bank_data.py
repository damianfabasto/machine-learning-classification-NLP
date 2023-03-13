import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.base import clone
from copy import deepcopy
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

import pdb

# Example from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
        scoring = "roc_auc"
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring = scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return axes

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report_randomized_search(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")

# From https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
def plot_roc_curve(fpr, tpr, ax, label=None, color = 'blue'):
    ax.plot(fpr, tpr, linewidth=2, label=label, color = color)
    ax.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    ax.axis([0, 1, 0, 1])                                    # Not shown in the book
    ax.set_xlabel('False Positive Rate (Fall-Out)', fontsize=10) # Not shown
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=10)    # Not shown
    ax.legend(loc = 'best')
    ax.grid(True)                                            # Not shown


def plot_ml_plots(
    estimator_base,
    estimator_tuned,
    mlAlgorithm,
    X,
    y,
    figure=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
        scoring = "roc_auc"
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if figure is None:
        figure = plt.figure(figsize=(20, 5))

    numPlots = 4


    # Plot the learning curve for the base algorithm
    ax = figure.add_subplot(2, 2, 1)
    train_sizes, train_scores_base, test_scores_base, _, _ = learning_curve(
        estimator_base,
        X,
        y,
        cv=KFold(n_splits=cv),
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores_base, axis=1)
    train_scores_std = np.std(train_scores_base, axis=1)
    test_scores_mean = np.mean(test_scores_base, axis=1)
    test_scores_std = np.std(test_scores_base, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    ax.set_title(mlAlgorithm + "\n base", size = 15)



    # Plot the learning curve for the base algorithm
    ax = figure.add_subplot(2, 2, 2)
    train_sizes, train_scores_tuned, test_scores_tuned, _, _ = learning_curve(
        estimator_tuned,
        X,
        y,
        cv=KFold(n_splits=cv),
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores_tuned, axis=1)
    train_scores_std = np.std(train_scores_tuned, axis=1)
    test_scores_mean = np.mean(test_scores_tuned, axis=1)
    test_scores_std = np.std(test_scores_tuned, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    ax.set_title(mlAlgorithm + "\n after hyperparameter search", size = 15)


    # Plot the confusion matrix for the tuned model
    ###############################################
    ax = figure.add_subplot(2, 2, 3)
    y_train_pred = cross_val_predict(estimator_tuned, X, y, cv=cv)
    y_train      = y
    conf_matrix = confusion_matrix(y_train, y_train_pred)

    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=15)

    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(["Refused T. Deposit (predicted)", "Bought T. Deposit (predicted)"], fontsize=10, rotation=45)
    ax.set_yticklabels(["Refused T. Deposit (actual)", "Bought T. Deposit (actual)"], fontsize=10, rotation=360)



    # Plot the ROC curves
    ###############################################
    ax = figure.add_subplot(2, 2, 4)

    # # ROC of base model
    # y_probas = cross_val_predict(estimator_base, X, y, cv=cv,
    #                              method="predict_proba")
    # y_scores = y_probas[:, 1]  # score = proba of positive class
    # fpr, tpr, thresholds = roc_curve(y, y_scores)
    # plot_roc_curve(fpr, tpr, ax, label='Base', color='brown')

    # ROC of tuned model
    y_probas = cross_val_predict(estimator_tuned, X, y, cv=cv,
                                        method="predict_proba")
    y_scores = y_probas[:, 1]  # score = proba of positive class

    fpr, tpr, thresholds = roc_curve(y, y_scores)
    plot_roc_curve(fpr, tpr, ax, label='Tuned model', color='blue')


    ax.set_title("ROC curve", fontsize=15)
    figure.tight_layout()

    return figure



def plot_ml_plots_2(
    estimator_base,
    estimator_tuned,
    mlAlgorithm,
    X,
    y,
    param_name,
    param_range,
    figure=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
        scoring = "roc_auc"
):
    """
    Generate 4 plots:
    1) Learning curve for base estimator
    2) Learning curve for tuned estimator
    3) Validation curve across a given hyperparameter
    4) Confusion matrix


    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if figure is None:
        figure = plt.figure(figsize=(20, 5))

    numPlots = 4


    # Plot the learning curve for the base algorithm
    ax = figure.add_subplot(2, 2, 1)
    train_sizes, train_scores_base, test_scores_base, _, _ = learning_curve(
        estimator_base,
        X,
        y,
        cv=KFold(n_splits=cv),
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores_base, axis=1)
    train_scores_std = np.std(train_scores_base, axis=1)
    test_scores_mean = np.mean(test_scores_base, axis=1)
    test_scores_std = np.std(test_scores_base, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    ax.set_title(mlAlgorithm + "\n base", size = 15)



    # Plot the learning curve for the tuned algorithm
    ax = figure.add_subplot(2, 2, 2)
    train_sizes, train_scores_tuned, test_scores_tuned, _, _ = learning_curve(
        estimator_tuned,
        X,
        y,
        cv=KFold(n_splits=cv),
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores_tuned, axis=1)
    train_scores_std = np.std(train_scores_tuned, axis=1)
    test_scores_mean = np.mean(test_scores_tuned, axis=1)
    test_scores_std = np.std(test_scores_tuned, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    ax.set_title(mlAlgorithm + "\n after hyperparameter search", size = 15)


    # Plot the Validation curve, based on a given hyperparameter
    #############################################################
    ax = figure.add_subplot(2, 2, 3)

    train_scores, test_scores = validation_curve(
        estimator_tuned,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    # Plot learning curve
    ax.grid()
    ax.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        param_range, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        param_range, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")
    ax.set_title(mlAlgorithm + "\n after hyperparameter search", size = 15)




    # Plot the confusion matrix for the tuned model
    ###############################################
    ax = figure.add_subplot(2, 2, 4)
    y_train_pred = cross_val_predict(estimator_tuned, X, y, cv=cv)
    y_train      = y
    conf_matrix = confusion_matrix(y_train, y_train_pred)

    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=15)

    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(["Refused T. Deposit (predicted)", "Bought T. Deposit (predicted)"], fontsize=10, rotation=45)
    ax.set_yticklabels(["Refused T. Deposit (actual)", "Bought T. Deposit (actual)"], fontsize=10, rotation=45)

    return figure


class TreePruning():
    """
    Implement tree pruning and test it against a validation set
    """
    def __init__(self, treeclf, X_val, y_val):
        self.tree = treeclf
        self.X_val = deepcopy(X_val)
        self.y_val = deepcopy(y_val)

        self.indicestoprune = self.indicesToPrune()

    def indicesToPrune(self):
        # Get the indices to prune where both left and right children are not -1
        indicesToPrune = []
        for idx, (left_child, right_child) in enumerate(zip(self.tree.tree_.children_left, self.tree.tree_.children_left)):
            if left_child!=TREE_LEAF and right_child !=TREE_LEAF:
                indicesToPrune.append(idx)
        # Verse the list, since we want to start from the terminal nodes
        indicesToPrune.reverse()
        return indicesToPrune

    def prune(self):
        """
        Iteratively start pruning the tree, starting from the terminal leaves and going up
        :return:
        """
        for idx in self.indicestoprune:
            # Each node sequentially into leaf
            self.tree.tree_.children_left[idx] = TREE_LEAF
            self.tree.tree_.children_right[idx] = TREE_LEAF
            self.tree.tree_.feature[idx] = TREE_UNDEFINED
            yield self.tree

    def evaluatePruningThroughValidation(self, X_train, y_train):
        prunedTrees = []
        for prunedTree in self.prune():
            # Evaluate the performance on the validation set
            y_val_pred = prunedTree.predict_proba(self.X_val)
            y_val_scores = y_val_pred[:, 1]  # score = proba of positive class
            auc = roc_auc_score(self.y_val, y_val_scores)
            # Clone model and re-fit
            clonedCopy = clone(prunedTree)
            clonedCopy.fit(X_train, y_train)
            prunedTrees.append( (auc, clonedCopy) )
        prunedTrees = sorted(prunedTrees, key = lambda x: x[0], reverse = True)
        return prunedTrees



class PrunedTree(DecisionTreeClassifier):

    def __init__(self, *,
        criterion = "gini",
        splitter = "best",
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.,
        max_features = None,
        random_state = None,
        max_leaf_nodes = None,
        min_impurity_decrease = 0.,
        min_impurity_split = None,
        class_weight = None,
        presort = 'deprecated',
        ccp_alpha = 0.0,
        pruneLevel = None):

        super().__init__(
            criterion = criterion,
            splitter = splitter,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            random_state=random_state,
            max_leaf_nodes = max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight = class_weight,
            presort = presort,
            ccp_alpha = ccp_alpha)

        self.pruneLevel = pruneLevel

    def indicesToPrune(self):
        # Get the indices to prune where both left and right children are not -1
        indicesToPrune = []
        for idx, (left_child, right_child) in enumerate(zip(self.tree_.children_left, self.tree_.children_left)):
            if left_child!=TREE_LEAF and right_child !=TREE_LEAF:
                indicesToPrune.append(idx)
        # Verse the list, since we want to start from the terminal nodes
        indicesToPrune.reverse()
        return indicesToPrune

    def prune(self, indicestoprune):
        """
        Iteratively start pruning the tree, starting from the terminal leaves and going up
        :return:
        """
        for idx in indicestoprune:
            # Each node sequentially into leaf
            self.tree_.children_left[idx] = TREE_LEAF
            self.tree_.children_right[idx] = TREE_LEAF
            self.tree_.feature[idx] = TREE_UNDEFINED


    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        # Call parent class
        DecisionTreeClassifier.fit(self, X, y, sample_weight=sample_weight, check_input=True,
            X_idx_sorted=X_idx_sorted)
        if self.pruneLevel is not None:
            # Get the indices that can be pruned
            prunable_nodes = self.indicesToPrune()
            # Prune as many as the number of self.pruneLevel
            indicestoprune = prunable_nodes[:self.pruneLevel]
            self.prune(indicestoprune)
        return self






