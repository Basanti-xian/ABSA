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

    CountVecter = CountVectorizer(lowercase=True,dtype=np.float64,encoding='utf-8', tokenizer=None, max_features=20000)#,max_df=0.95)
    X = CountVecter.fit_transform(Samples)
    X = Normalizer().fit_transform(X)
    print 'shape of X matrix before adding lex feats', X.shape
    # Select_Feats = SelectPercentile(f_classif, percentile=9)
    # X_new = Select_Feats.fit_transform(X, Y)
    # print 'shape of X matrix after selecting KBest feats', X_new.shape

    X = hstack([X,LexFeats])
    print 'shape of X matrix after adding lex feats',X.shape

    print '*'*80
    feature_names = CountVecter.get_feature_names()
    # print 'number of features before selection', len(feature_names)
    # mask = Select_Feats.get_support()  # list of booleans
    # new_features = []  # The list of your K best features
    # for bool, feature in zip(mask, feature_names):
    #     if bool:
    #         new_features.append(feature)
    # print 'number of features after selection', len(new_features)

    Vocab = feature_names + ['HLPos', 'HLNeg', 'HLSum',
                            'NrcPos', 'NrcNeg', 'NrcSum',
                            'SubjPos', 'SubjNeg', 'SubjSum']
    print 'number of vocabulary (adding lex feats)', len(Vocab)
    print '*' * 80
    return X, Y, Vocab, CountVecter

def GetTestXY (fname, CountVecter):
    StemmedLexicons = LoadStemmedLex()
    Lines = [l.strip() for l in open (fname).readlines()]
    Sentences = [''.join(l.strip().split('^')[:-2]) for l in open (fname).xreadlines()]

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
    print '#'*20
    print fname
    print 'loaded {} samples from {}'.format(len(Samples),fname)
    print 'Label dist: ', Counter(Y)

    X = CountVecter.transform(Samples)
    X = Normalizer().fit_transform(X)
    X = hstack([X, LexFeats])
    return X, Y


def SavetoCSV(Preds, name):
    d = {'pred': pd.Series(Preds)}
    df = pd.DataFrame(d)
    fname = 'Preds({}).csv'.format(name)
    df.to_csv(fname, sep='^', header=False, index=False)

def PredProba(Classifier, X_test):
    Preds = Classifier.predict_proba(X_test)
    Pred_polarity = []
    p_class = [-1, 1]
    for l in Preds:
        p = max(l)*p_class[list(l).index(max(l))]
        Pred_polarity.append(p)
    return Pred_polarity

def Main ():
    X_train, Y_train, Vocab, CountVecter = GetXYVocab()
    print 'FVs prepared of shape',X_train.shape
    Classifier = SVC(C = 10, kernel='linear', class_weight='balanced', probability=True)
    Classifier.fit(X_train, Y_train)
    # PerformFeatAnalysis (Classifier, X, Y, Vocab)

    filenames = ['cara_v7.csv', 'alan_82_v4.csv', 'tim_52_v4.csv', 'Others1_v6.csv', 'Others2_v4.csv']
    names = [f.split('_')[0] for f in filenames]
    Fnames = ['AllNRCFeats({}).txt'.format(n) for n in names]

    for ind, fname in enumerate(Fnames):
        X_test, y_test = GetTestXY(fname, CountVecter)
        Polarity = PredProba(Classifier, X_test)
        SavetoCSV(Polarity, names[ind])
        Preds = Classifier.predict(X_test)
        Acc = accuracy_score(y_true = y_test, y_pred=Preds)
        P = precision_score(y_true = y_test, y_pred=Preds, average='weighted')
        R = recall_score(y_true = y_test, y_pred=Preds, average='weighted')
        F = f1_score(y_true = y_test, y_pred=Preds, average='weighted')
        print (Acc,P,R,F)
        print classification_report (y_test, Preds)


if __name__ == '__main__':
    Main()