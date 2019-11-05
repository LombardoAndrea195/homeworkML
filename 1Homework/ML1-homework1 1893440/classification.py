
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import  tree
from sklearn.utils.multiclass import unique_labels

from measurement import function_print_binary, plot_confusion_matrix


def classificator3(vector,target,test,randoms,Kind,Kind_Target):
    print("The target is doing a classification " + " " + Kind_Target + " " + "using a sklearn funtion" + " " + Kind + "")
    X_train, X_test, y_train, y_test = train_test_split(vector, target, test_size=test,
                                                        random_state=randoms)  # 100%-test( training) and test% test
    if (Kind == "MultinomialNB" and Kind_Target == "Multiclasse"):
        clf = OneVsRestClassifier(MultinomialNB())
    elif (Kind == "GaussianNB" and Kind_Target == "Multiclasse"):
         clf=OneVsRestClassifier(GaussianNB())
    elif (Kind == "DecisionTreeClassifier"and  Kind_Target == "Multiclasse"):
        clf = OneVsRestClassifier(tree.DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    s=confusion_matrix(y_test, y_pred)

    names=["gcc","clang","icc"]
    print(classification_report(y_test, y_pred,
                          target_names=names))
    print("-accuracy: ", accuracy_score(y_test, y_pred))
    print("-recall: ", recall_score(y_test, y_pred, average=
    None))
    print("-precision: ", precision_score(y_test, y_pred, average=
    None))

    classes = [0,1,2]
    plot_confusion_matrix(y_test, y_pred,classes)


def classificator2(vector,target,test,randoms,Kind,Kind_Target):

    print("The target is doing a classification " + " " + Kind_Target + " " + "using a sklearn funtion" + " " + Kind + "")
    X_train, X_test, y_train, y_test = train_test_split(vector, target, test_size=test,
                                                        random_state=randoms)  # 100%-test( training) and test% test
    if (Kind == "MultinomialNB" and Kind_Target == "Multiclasse"):
        clf=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    elif(Kind=="GaussianNB" and  Kind_Target == "Multiclasse" ):
        clf=GaussianNB()
    elif (Kind == "DecisionTreeClassifier"and  Kind_Target == "Multiclasse"):
        clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    s=confusion_matrix(y_test, y_pred)
    names=["gcc","clang","icc"]
    print(classification_report(y_test, y_pred,
                          target_names=names))

    print("-accuracy: ", accuracy_score(y_test, y_pred))
    print("-recall: ",recall_score(y_test, y_pred, average =
    None))
    print("-precision: ",precision_score(y_test, y_pred, average =
    None))

    classes = [0,1,2]
    plot_confusion_matrix(y_test, y_pred,classes)

def classificator(vector,target,test,randoms,Kind,kind_Target):

    print("The target is doing a classification " + " " + kind_Target + " " + "using a sklearn funtion" + " " + Kind + "")
    X_train, X_test, y_train, y_test = train_test_split(vector, target,test_size=test,random_state=randoms)  # 100%-test( training) and test% test
    if(Kind=="BernulliNB"and kind_Target=="Binario"):
        clf=BernoulliNB()
    elif(Kind=="GaussianNB"):
        clf=GaussianNB()
    elif(Kind=="DecisionTreeClassifier"):
        clf=tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)  # Predict probabilities for the test data
    function_print_binary(y_test, y_pred)
    classes = target[unique_labels(y_test, y_pred)]
    plot_confusion_matrix(y_test, y_pred,classes)
