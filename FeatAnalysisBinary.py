import numpy as np
from sklearn.svm import LinearSVC, SVC
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from Simple5FoldClassifier import GetXYVocab
from copy import deepcopy
from scipy.sparse import csr_matrix
from sklearn import grid_search



def AnalyseClassifierFeats(Classifier, Vocab, TopN=20):
    W = deepcopy(Classifier.coef_[0,:])
    FeatsAndVocab = zip(W.tolist(), Vocab)
    FeatsAndVocab.sort()

    TopNeg = [WF[1] for WF in FeatsAndVocab[:TopN]]
    TopPos = [WF[1] for WF in FeatsAndVocab[-TopN:]]

    return TopPos, TopNeg, W.tolist()


def AnalyseSampleFeats(X, Y, Classifier, Vocab, Sentences):
    line_ind = None
    for Index in xrange(X.shape[0]):
        if line_ind:
            Index = int(line_ind) - 2
        Sent = Sentences[Index]
        Sample = X[Index, :]
        PredLabel = Classifier.predict(Sample)[0]
        if PredLabel != Y[Index]:
            continue
        ChosenW = csr_matrix(Classifier.coef_[0,:])
        FeatsTimesW = ChosenW.multiply(Sample)
        FeatsTimesW = FeatsTimesW.todense().T.tolist()
        FeatsTimesW = [Val for List in FeatsTimesW for Val in List]
        FeatsTimesWAndVocab = zip(FeatsTimesW, Vocab)
        FeatsTimesWAndVocab.sort()
        if 1 == PredLabel:
            FeatsTimesWAndVocab.reverse()

        print '*' * 80
        print 'Ind: {}  Sentence: {}    actual label:{},    pred label: {} '.format(Index, Sent, Y[Index], PredLabel)
        print '-' * 80
        print 'Top 20 feats: '
        pprint(FeatsTimesWAndVocab[:20])
        print '*' * 80
        line_ind = raw_input('Please input the sentence index:')


NumSamples = -1
X, Y, Vocab, Sentences = GetXYVocab(NumSamples)
# Sentences = [l.strip() for l in open ('ForNRCFeats(492).csv').xreadlines()][:NumSamples]
Classifier = LinearSVC(C=10)
Classifier.fit(X, Y)
Preds = Classifier.predict(X)
Acc = accuracy_score(y_true=Y, y_pred=Preds)
print classification_report(Y, Preds)

X = csr_matrix(X)
TopN = 20
TopPos, TopNeg, W  = AnalyseClassifierFeats(Classifier, Vocab, TopN)
print '*' * 80
print 'top {} pos feats: '.format(TopN);pprint (TopPos);print '*'*80
print 'top {} neg feats: '.format(TopN);pprint(TopNeg);print '*'*80
print '*' * 80

AnalyseSampleFeats(X, Y, Classifier, Vocab, Sentences)
