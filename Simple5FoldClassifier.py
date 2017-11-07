import os, sys, json, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC,SVC
from pprint import pprint
from sklearn.cross_validation import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import grid_search
from random import randint
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectFromModel, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from LexFeatsProcessor import LoadStemmedLex, GetLexFeats
from imblearn.combine import SMOTEENN, SMOTETomek


def SpaceTokenizer (Str):
    return [l.strip().split() for l in Str.split('\n') if l]

def GetYFromStringLabels (Labels):
    Y = []
    # Y = Labels
    for L in Labels:
        if '1' == L:
            Y.append(1)
        elif '-1' == L:
            Y.append(-1)
        elif '0' == L:
            Y.append(0)
        # elif 'conflict' == L:
        #     Y.append(2)
        else:
            # print 'error case'
            Y.append(0)
    return Y

def GetXYVocab (NumpSamples=-1):
    StemmedLexicons = LoadStemmedLex()
    Lines = [l.strip() for l in open ('AllNRCFeats.txt').readlines()][:NumpSamples]
    Sentences = [''.join(l.strip().split('^')[:-2]) for l in open ('ForNRCFeats.csv').xreadlines()][:NumpSamples]

    Labels = [L.split(';')[-1] for L in Lines]
    YY = GetYFromStringLabels(Labels)
    new_Lines=[]
    new_Sentences=[]
    for i,y in enumerate(YY):
        if y != 0:
            new_Lines.append(Lines[i])
            new_Sentences.append(Sentences[i])

    LexFeats = [GetLexFeats(Sent, StemmedLexicons) for Sent in new_Sentences]
    LexFeats = np.array (LexFeats)
    LexFeats = csr_matrix (LexFeats)
    print 'loaded lexicon features of shape', LexFeats.shape

    Samples, new_Labels = zip(*[tuple(L.split(';')) for L in new_Lines])
    Y = GetYFromStringLabels(new_Labels)
    print 'loaded {} samples'.format(len(Samples))
    print 'Label dist: ', Counter(Y)

    CountVecter = CountVectorizer(lowercase=True,dtype=np.float64,encoding='utf-8', tokenizer=None,binary=False)#,max_df=0.95)
    X = CountVecter.fit_transform(Samples)
    X = Normalizer().fit_transform(X)
    print 'shape of X matrix before adding lex feats', X.shape
    # Select_Feats = SelectKBest(f_classif, k=10000)
    Select_Feats = SelectPercentile(f_classif, percentile=9)
    X_new = Select_Feats.fit_transform(X, Y)

    # lsvc = LinearSVC(C=100, penalty="l1", dual=False).fit(X, Y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_new = model.transform(X)

    print 'shape of X matrix after selecting KBest feats', X_new.shape
    X = hstack([X_new,LexFeats])
    print 'shape of X matrix after adding lex feats',X.shape

    print '*'*80
    feature_names = CountVecter.get_feature_names()
    print 'number of features before selection', len(feature_names)
    mask = Select_Feats.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    print 'number of features after selection', len(new_features)

    Vocab = new_features + ['HLPos', 'HLNeg', 'HLSum',
                            'NrcPos', 'NrcNeg', 'NrcSum',
                            'SubjPos', 'SubjNeg', 'SubjSum']
    print 'number of vocabulary (adding lex feats)', len(Vocab)
    return X, Y, Vocab,new_Sentences

def SavetoCSV(Preds):
    d = {'pred': pd.Series(Preds)}
    df = pd.DataFrame(d)
    df.to_csv('Preds.csv', sep='^', header=False, index=False)

def PredProba(Classifier, X_test):
    Preds = Classifier.predict_proba(X_test)
    Pred_polarity = []
    p_class = [-1, 0, 1]
    for l in Preds:
        p = max(l)*p_class[list(l).index(max(l))]
        Pred_polarity.append(p)
    return Pred_polarity

def Main ():
    X, Y, Vocab,_ = GetXYVocab()
    Predictions = []
    Accs = [];Ps = [];Rs = [];Fs = []
    print 'FVs prepared of shape',X.shape
    for i in xrange (5):
        print 'run ',i+1
        X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2,random_state=randint(0,100))
        print 'train and test shapes', X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape
        Params = {'C':[0.001,0.01,0.1,1,10,100,1000]}
        Classifier = grid_search.GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True), Params,n_jobs=-1,cv=3)
        # Classifier = grid_search.GridSearchCV(SVC(), Params,n_jobs=-1,cv=3)
        # Classifier = LinearSVC(C=0.1,class_weight='balanced')
        # X_train = X_train.toarray()
        # sm = SMOTETomek()
        # X_resampled, y_resampled = sm.fit_sample(X_train, np.array(y_train))
        Classifier.fit(X_train, y_train)
        print 'best estimator after 5 fold CV: ', Classifier.best_estimator_

        # PerformFeatAnalysis (Classifier, X_train, Y, Vocab)
        Polarity = PredProba(Classifier, X_test)
        Preds = Classifier.predict(X_test)
        Predictions.extend(Polarity)
        Acc = accuracy_score(y_true = y_test, y_pred=Preds)
        P = precision_score(y_true = y_test, y_pred=Preds, average='weighted')
        R = recall_score(y_true = y_test, y_pred=Preds, average='weighted')
        F = f1_score(y_true = y_test, y_pred=Preds, average='weighted')
        Accs.append(Acc); Ps.append(P);Rs.append(R);Fs.append(F)
        print (Acc,P,R,F)
        print classification_report (y_test, Preds)

    SavetoCSV(Predictions)
    Accs = np.array (Accs)
    Ps = np.array (Ps)
    Rs = np.array (Rs)
    Fs = np.array (Fs)
    MeanA = np.mean(Accs); StdA = np.std(Accs)
    MeanP = np.mean(Ps); StdP = np.std(Ps)
    MeanR = np.mean(Rs); StdR = np.std(Rs)
    MeanF = np.mean(Fs); StdF = np.std(Fs)

    print 'Average Acc: {}, Prec: {}, Recall: {}, F1: {}'.format(MeanA, MeanP, MeanR, MeanF)
    print 'Std Acc: {}, Prec: {}, Recall: {}, F1: {}'.format(StdA, StdP, StdR, StdF)


if __name__ == '__main__':
    Main()