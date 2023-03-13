import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plotValidationCurve(param_range, train_scores, test_scores, ax, title, xlabel, ylabel = 'score', xscale = ""):

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
    ax.set_title(title)
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)
    if xscale =='log':
        ax.set_xscale('log')


def plotLearningCurve(train_sizes, train_scores, test_scores, ax, title, xlabel = 'Sample size', ylabel = 'score'):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

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

    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)

    ax.set_title(title)



    return ax

def plotScoreVsFitTime(fit_times, test_scores, ax, title, xlabel = 'In sample fit time', ylabel = "Validation set score"):
    fit_times_mean = np.mean(fit_times, axis=1)


    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    ax.grid()
    ax.plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    ax.fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)


def plotSampleSizeVsFitTime(train_sizes, fit_times, ax, title, xlabel = 'Sample size', ylabel = "Fit time"):

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot fit_time vs sample size
    ax.grid()
    ax.plot(train_sizes, fit_times_mean, "o-")
    ax.fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)

    ax.set_title(title)
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)



def plot_learning_validation(resultsDict, algoName, figure):
    """
    Plot learning curve, validation curve and running time
    """
    ################################
    # Learning cuve
    ################################
    ax = figure.add_subplot(1, 3, 1)

    train_sizes, train_scores, test_scores, fit_times, _ = resultsDict[algoName]['learning']


    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)


def plot_confusion_matrix(y_pred, y_test, ax, title, xlabel = 'Sample size', ylabel = "Fit time"):
    # Plot the confusion matrix for the test set
    ###############################################

    conf_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_title(title, fontsize=15)

    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(["Predicted\nNegative Label", "Predicted\nPositive Label"], fontsize=10, rotation=45)
    ax.set_yticklabels(["Actual\nNegative Label", "Actual\nPositive Label"], fontsize=10, rotation=45)

# From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plotLearningCurveBeforeTunning(train_sizes, train_scores, test_scores, ax, title, xlabel = 'Sample size', ylabel = 'score'):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    # ax.fill_between(
    #     train_sizes,
    #     train_scores_mean - train_scores_std,
    #     train_scores_mean + train_scores_std,
    #     alpha=0.1,
    #     color="r",
    # )
    # ax.fill_between(
    #     train_sizes,
    #     test_scores_mean - test_scores_std,
    #     test_scores_mean + test_scores_std,
    #     alpha=0.1,
    #     color="g",
    # )
    ax.plot(
        train_sizes, train_scores_mean, marker = "x", linestyle = "-.", color=adjust_lightness("r", 1.5)
    )
    ax.plot(
        train_sizes, test_scores_mean, marker = "x", linestyle = "-.", color=adjust_lightness("g", 1.5)
    )


    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)

    ax.set_title(title)



    return ax