import matplotlib.pyplot as plt
from matplotlib.colors import *
import seaborn as sns
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def function_print_binary(y_test,y_pred):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    (tn, fp, fn, tp) = confusion_matrix.ravel()
    accuracy = (tn + tp) / (tp + tn + fp + fn)
    names=["L","H"]
    print(classification_report(y_true=y_test, y_pred=y_pred,target_names=names))
    print("-Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("-Precision element y:", metrics.precision_score(y_test, y_pred, average =
None))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("-Recall:", metrics.recall_score(y_test, y_pred, average =
None))
    probs = y_pred
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    print("show information in roc with the information:")
    print('-AUC: %.2f' % auc)
    plot_roc_curve(fpr, tpr)


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(8, 6)):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(dpi=128, figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


