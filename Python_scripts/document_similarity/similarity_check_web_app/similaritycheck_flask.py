from flask import Flask
from flask import request
from flask import redirect

import spacy
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
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
nltk.download('stopwords')
nltk.download('punkt')
custom_stopwords = text.ENGLISH_STOP_WORDS.union(eng_contractions)

#gets the similarity between Named entity tags and Parts of speech tags
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
    tfidf_vectorizer = TfidfVectorizer(max_features=2000,
                                     stop_words=custom_stopwords,
                                 use_idf=True, tokenizer=tokens_and_stem, ngram_range=(1,3))
    train_tfidf = tfidf_vectorizer.fit_transform([y])
    new_tfidf = tfidf_vectorizer.transform([x])
    cosine_score = cosine_similarity(new_tfidf, train_tfidf)
    return cosine_score

def get_index():
    return "<form action=\"/eval\" method=\"get\" id=\"eval\">     <textarea name=\"x\" rows=\"30\" cols=\"100\" form=\"eval\">Enter content X here...</textarea>       <textarea name=\"y\" rows=\"30\" cols=\"100\" form=\"eval\">Enter content Y here...</textarea>       <input type=\"Submit\">       </form>"

@app.route("/")
def init():
    return get_index()

@app.route("/eval")
def eval():
    x =request.args.get('x')
    y =request.args.get('y')
    spacy_sim = spacy_similarity(nlp(x), nlp(y))
    cosine_score = cosine_sim_score(x,y)
    total_score = float(spacy_sim) + float(cosine_score)
    threshold_score = float(total_score/2)
    verdict = threshold_check(threshold_score)
    result = "<br>Threshold score : "+str(threshold_score)+"<br>Verdict : "+str(verdict)
    return result

def threshold_check(threshold):
    if(threshold>0.7):
        return "Similar"
    else:
        return "Dissimilar"

if __name__ == "__main__":
   app.run(host='0.0.0.0')