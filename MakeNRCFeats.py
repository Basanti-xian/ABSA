import nltk,string,sys
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from time import time
from nltk.util import ngrams
from nltk.corpus import stopwords
from copy import deepcopy
from pprint import pprint
from itertools import chain, combinations
import copy
from joblib import Parallel, delayed

reload(sys)
sys.setdefaultencoding('utf8')


stop = ['a','is','be','are','my','myself','our','ourselves','they','her',
        'herself','him','himself','you','your','yours','yourself','yourselves',
        'this','that','the','and','in','at','to','about']
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if item not in stop:
            stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return stems


def pad_sequence(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    if pad_left:
        sequence = chain((pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n-1))
    return sequence

def skipngrams(sequence, n, k, pad_left=False, pad_right=False, pad_symbol=None):
    sequence_length = len(sequence)
    sequence = iter(sequence)
    sequence = pad_sequence(sequence, n, pad_left, pad_right, pad_symbol)

    if sequence_length + pad_left + pad_right < k:
        # raise Exception("The length of sentence + padding(s) < skip")
        yield []

    if n < k:
        raise Exception("Degree of Ngrams (n) needs to be bigger than skip (k)")

    history = []
    nk = n+k

    # Return point for recursion.
    if nk < 1:
        return
    # If n+k longer than sequence, reduce k by 1 and recur
    elif nk > sequence_length:
        for ng in skipngrams(list(sequence), n, k-1):
            yield ng

    while nk > 1: # Collects the first instance of n+k length history
        history.append(next(sequence))
        nk -= 1

    # Iterative drop first item in history and picks up the next
    # while yielding skipgrams for each iteration.
    for item in sequence:
        history.append(item)
        current_token = history.pop(0)
        # Iterates through the rest of the history and
        # pick out all combinations the n-1grams
        for idx in list(combinations(range(len(history)), n-1)):
            ng = [current_token]
            for _id in idx:
                ng.append(history[_id])
            yield tuple(ng)

    # Recursively yield the skigrams for the rest of seqeunce where
    # len(sequence) < n+k
    for ng in list(skipngrams(history, n, k-1)):
        yield ng

def GetAllNRCFeats (Sent, SentIndex, Cat):
    T0 = time()
    try:
        Tokens = tokenize(Sent.lower())
        Chars = list(''.join(Tokens))

        # POS tag features
        WordPOSTags = [Item[1] for Item in nltk.pos_tag(Tokens)]

        # word ngram features
        WordUnigrams = deepcopy(Tokens)
        WordBigrams = ['_'.join(list(Item)) for Item in list(ngrams(Tokens, n=2))]
        WordTrigrams = ['_'.join(list(Item)) for Item in list(ngrams(Tokens, n=3))]

        # character ngram features
        CharTrigrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=3))]
        CharFourgrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=4))]
        CharFivegrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=5))]
        CharSixgrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=6))]

        # category specific word ngram features
        # change '+' to '_'
        WordUnigramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in WordUnigrams]
        WordBigramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in WordBigrams]
        WordTrigramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in WordTrigrams]

        # category specific char ngram features
        # change '+' to '_'
        CharTrigramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in CharTrigrams]
        CharFourgramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in CharFourgrams]
        CharFivegramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in CharFivegrams]
        CharSixgramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in CharSixgrams]

        # category specific POS tag features
        # change '+' to '_'
        WordPOSTagsCat = [Item + '_' + str(Cat[SentIndex]) for Item in WordPOSTags]

        # non continous ngram features
        TwoSkipBiGrams = list(skipngrams(WordUnigrams, n=2, k=2))
        TwoSkipTriGrams = list(skipngrams(WordUnigrams, n=3, k=2))
        ThreeSkipTriGrams = list(skipngrams(WordUnigrams, n=3, k=3))
        ThreeSkipFourGrams = list(skipngrams(WordUnigrams, n=4, k=3))
        TwoSkipBiGrams = ['_'.join(list(Item)) for Item in TwoSkipBiGrams]
        TwoSkipTriGrams = ['_'.join(list(Item)) for Item in TwoSkipTriGrams]
        ThreeSkipTriGrams = ['_'.join(list(Item)) for Item in ThreeSkipTriGrams]
        ThreeSkipFourGrams = ['_'.join(list(Item)) for Item in ThreeSkipFourGrams]

        # category specific non continous ngram features
        # change '+' to '_'
        TwoSkipBiGramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in TwoSkipBiGrams]
        TwoSkipTriGramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in TwoSkipTriGrams]
        ThreeSkipTriGramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in ThreeSkipTriGrams]
        ThreeSkipFourGramsCat = [Item + '_' + str(Cat[SentIndex]) for Item in ThreeSkipFourGrams]

        AllFeats = WordUnigrams + WordUnigramsCat + WordBigrams + WordBigramsCat +  \
                   TwoSkipBiGrams + TwoSkipTriGrams + TwoSkipBiGramsCat + TwoSkipTriGramsCat +  \
                   CharFourgrams + CharFourgramsCat + CharFivegrams + CharFivegramsCat + CharSixgrams + CharSixgramsCat \
                   + WordTrigrams + WordTrigramsCat + ThreeSkipTriGrams
                   # + CharTrigrams + CharTrigramsCat + ThreeSkipFourGrams + ThreeSkipTriGramsCat + ThreeSkipFourGramsCat\
                   # + WordPOSTags + WordPOSTagsCat
        AllFeats = ' '.join(AllFeats)
        print 'processed sentence :{} in {} sec'.format(SentIndex, time() - T0)
    except Exception as e:
        AllFeats = ''
        print e
        raw_input()
    # print Sent
    # pprint(AllFeats)
    return AllFeats

def Main (FName, Output_f, NumSentToProc=-1, NumCpu=8):
    T0 = time()
    Sentences = [''.join(l.strip().split('^')[:-2]) for l in open (FName).xreadlines()][:NumSentToProc]
    # Cat = [l.strip().split('^')[-2] for l in open (FName).xreadlines()][:NumSentToProc]
    Cat = [l.strip().split('^')[-2].replace(' ', '_') for l in open(FName).xreadlines()][:NumSentToProc]
    Label = [l.strip().split('^')[-1] for l in open (FName).xreadlines()][:NumSentToProc]
    print 'loaded {} sents, {} cats and {} pol from {}'.format(len(Sentences), len(Cat), len(Label),FName)
    raw_input('hit any key...')
    AllFeatsExpSentences = Parallel(n_jobs=NumCpu)(delayed(GetAllNRCFeats)(Sent, SentIndex, Cat) for SentIndex, Sent in enumerate(Sentences))

    # with open ('AllNRCFeats.txt','w') as FH:
    with open(Output_f, 'w') as FH:
        for Index, Item in enumerate(AllFeatsExpSentences):
            print >>FH, str(Item)+';'+Label[Index]

    print 'processed {} sentences in a total of {} sec. with 8 cpu'.format(len(AllFeatsExpSentences), round (time()-T0,2))

if __name__ == '__main__':
    filenames = ['cara_v7.csv', 'alan_82_v4.csv', 'tim_52_v4.csv', 'Others1_v6.csv', 'Others2_v4.csv']
    names = [f.split('_')[0] for f in filenames]
    Fnames = ['ForNRCFeats({})'.format(n) for n in names]
    Output_fnames = ['AllNRCFeats({}).txt'.format(n) for n in names]
    # FName = 'ForNRCFeats.csv'
    NumSent = 800
    for ind,f in enumerate(Fnames):
        Main(FName=f, Output_f = Output_fnames[ind], NumSentToProc=NumSent, NumCpu=4)
