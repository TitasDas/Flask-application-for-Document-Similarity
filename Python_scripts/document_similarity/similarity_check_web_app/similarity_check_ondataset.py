import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text

data = pd.read_csv("new_benchmark.csv")
data = data.drop(columns=['id','multi-label'])

nlp = spacy.load("en_core_web_sm")
eng_contractions = ["ain't", "amn't", "aren't", "can't", "could've", "couldn't",
                    "daresn't", "didn't", "doesn't", "don't", "gonna", "gotta", 
                    "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd",
                    "how'll", "how's", "I'd", "I'll", "I'm", "I've", "isn't", "it'd",
                    "it'll", "it's", "let's", "mayn't", "may've", "mightn't", 
                    "might've", "mustn't", "must've", "needn't", "o'clock", "ol'",
                    "oughtn't", "shan't", "she'd", "she'll", "she's", "should've",
                    "shouldn't", "somebody's", "someone's", "something's", "that'll",
                    "that're", "that's", "that'd", "there'd", "there're", "there's", 
                    "these're", "they'd", "they'll", "they're", "they've", "this's",
                    "those're", "tis", "twas", "twasn't", "wasn't", "we'd", "we'd've",
                    "we'll", "we're", "we've", "weren't", "what'd", "what'll", 
                    "what're", "what's", "what've", "when's", "where'd", "where're",
                    "where's", "where've", "which's", "who'd", "who'd've", "who'll",
                    "who're", "who's", "who've", "why'd", "why're", "why's", "won't",
                    "would've", "wouldn't", "y'all", "you'd", "you'll", "you're", 
                    "you've", "'s", "s","d","m", "abov", 'afterward', 'ai', 'alon',
                    'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 
                    'becam', 'becaus', 'becom', 'befor', 'besid', 'ca', 'cri', 'dare',
                    'describ', 'did', 'doe', 'dure', 'els', 'elsewher', 'empti', 'everi',
                    'everyon', 'everyth', 'everywher', 'fifti', 'forti', 'gon', 'got', 
                    'henc', 'hereaft', 'herebi', 'howev', 'hundr', 'inde', 'let', 'll',
                    'mani', 'meanwhil', 'moreov', "n't", 'na', 'need', 'nobodi',
                    'noon', 'noth', 'nowher', 'ol', 'onc', 'onli', 'otherwis', 'ought',
                    'ourselv', 'perhap', 'pleas', 'sever', 'sha', 'sinc', 'sincer', 'sixti',
                    'somebodi', 'someon', 'someth', 'sometim', 'somewher', 'ta', 'themselv',
                    'thenc', 'thereaft', 'therebi', 'therefor', 'togeth', 'twelv', 'twenti',
                    've', 'veri', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi',
                    'wherev', 'whi', 'wo', 'yourselv',"'d", "'m", 'anywh', 'el', 'elsewh', 'everywh',
                    'ind', 'otherwi', 'plea', 'somewh'
                    ]

stemmer = SnowballStemmer("english")
custom_stopwords = text.ENGLISH_STOP_WORDS.union(eng_contractions)

def spacy_similarity(x, y):
    if x is not None and y is not None:
    	return x.similarity(y)
    return 0

def tokens_and_stem(x, do_stem=True):
    tokens = [word.lower() for sent in nltk.sent_tokenize(x) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    if do_stem:
        return stems
    else:
        return filtered_tokens

def cosine_sim_score(x, y):
    cosine_score = 0
    tfidf_vectorizer = TfidfVectorizer(max_df=1, max_features=2000,
                                    min_df=0.0, stop_words=custom_stopwords,
                                 use_idf=True, tokenizer=tokens_and_stem, ngram_range=(1,3))
    train_tfidf = tfidf_vectorizer.fit_transform([y])
    new_tfidf = tfidf_vectorizer.transform([x])
    cosine_score = cosine_similarity(new_tfidf, train_tfidf)
    return cosine_score

#calculating the spacy_similarity and cosine similarity and comparing average with a threshold of 0.7
data['spacy_sim']=data.apply(lambda row: spacy_similarity(nlp(row.content_x),nlp(row.content_y)), axis=1)
data['cosine_sim']=data.apply(lambda row: cosine_sim_score(row.content_x, row.content_y), axis=1)
data['mean_score']=data[['spacy_sim','cosine_sim']].mean(axis=1)
data['result']=data['mean_score']>0.7
data['new_binary_label']=data[['result']].astype(int)

data = data.drop(columns=['spacy_sim','cosine_sim','mean_score','result'])

def precision(actual, prediction):
    fp = 0.
    tp = 0.
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == 1 and prediction[index][i] == 1):
                tp += 1
            elif (actual[index][i] == 0 and prediction[index][i] == 1):
                fp += 1
    print("tp: " + str(tp))
    print("fp: " + str(fp))
    return tp/(tp + fp)
def recall(actual, prediction):
    fn = 0.
    tp = 0.
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == 1 and prediction[index][i] == 1):
                tp += 1
            elif (actual[index][i] == 1 and prediction[index][i] == 0):
                fn += 1
    print("tp: " + str(tp))
    print("fn: " + str(fn))
    return tp/(tp + fn)

def accuracy(actual, prediction):
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    
    for (index, val) in enumerate(actual):
        
        for (i,v) in enumerate(val):
            if (actual[index][i] == 1 and prediction[index][i] == 1):
                tp += 1
            elif (actual[index][i] == 1 and prediction[index][i] == 0):
                fn += 1
            elif (actual[index][i] == 0 and prediction[index][i] == 1):
                fp += 1
            elif (actual[index][i] == 0 and prediction[index][i] == 0):
                tn += 1
    print("tp: " + str(tp))
    print("tn: " + str(tn))
    print("fp: " + str(fp))
    print("fn: " + str(fn))
    acc = (tp+tn)/(tp+tn+fp+fn)   
    return acc

prec_score = precision(data[['binary-label']].values, data[['new_binary_label']].values)
recall_score = recall(data[['binary-label']].values, data[['new_binary_label']].values)
accuracy = accuracy(data[['binary-label']].values, data[['new_binary_label']].values)
print("accuracy :" + str(accuracy) + "precision :" + str(prec_score) + "recall :" + str(recall_score))