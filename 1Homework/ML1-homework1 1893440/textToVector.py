
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.tree import tree


def ManageContentForBlindSetToVectBinary(Train_dataset,test_blind):
    Vectorized = CountVectorizer()
    Vectorized.fit(Train_dataset['instructions'])
    vectorTrain_ = Vectorized.transform(Train_dataset['instructions'])
    vector_Blind = Vectorized.transform(test_blind)
    vector1 = vectorTrain_.toarray()
    vector2 = vector_Blind.toarray()
    clf = tree.DecisionTreeClassifier()
    clf.fit(vector1, Train_dataset['opt'])
    y_pred = clf.predict(vector2)  # Predict probabilities for the test data
    print(y_pred)
    return y_pred
def ManageContentForBlindSetToVectMulticlass(Train_dataset,test_blind):
    Vectorized = CountVectorizer()
    Vectorized.fit(Train_dataset['instructions'])
    vectorTrain_ = Vectorized.transform(Train_dataset['instructions'])
    vector_Blind = Vectorized.transform(test_blind)
    vector1 = vectorTrain_.toarray()
    vector2 = vector_Blind.toarray()
    clf = tree.DecisionTreeClassifier()
    clf.fit(vector1, Train_dataset['compiler'])
    y_pred = clf.predict(vector2)  # Predict probabilities for the test data
    print(y_pred)
    return y_pred


def ManageContentToVector(select,content) :
    if(select==1) :
        vectorcase = HashingVectorizer(content)

    elif(select==2):
        vectorcase = text2tfidf(content)
    else:
       vectorcase=text2vector(content)
    return vectorcase
def text2vector(text):
    vector=CountVectorizer().fit(text)
    print(vector.vocabulary_)

    vector2=vector.transform(text)
    print(vector2.shape)
   # print(type(vector2))
    #print(vector2.toarray())
    return vector2.toarray()


def text2tfidf(text_p):

        # TF-IDF version
        ifidf = TfidfVectorizer()
        ifidf.fit(text_p)

        print(ifidf.vocabulary_)

        vector = ifidf.transform(text_p)

        return vector.toarray()