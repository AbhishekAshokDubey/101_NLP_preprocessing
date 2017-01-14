# TODO: check for all the nltk download files (stopword, stemmer), before using and
# prompt for installing if missing.
# TODO: Lower case a separate function, inside a must run function, or at every code place?

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import gensim
from autocorrect import spell
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

try:
    import hunspell_cffi as hunspell
except:
    print("hunspell is not installed. Please install: hunspell_cffi")

#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')
# pip install Hunspell-CFFI

def readfile(path, column_info=0):
    file_ext = os.path.splitext(path)[-1]
    # TODO: column_info can be list or single value
    if file_ext == '.csv':
            #return pd.read_csv(path,usecols=[column_info])
            return list(pd.read_csv(path,usecols=[column_info]).values.flatten())
    elif file_ext == '.txt':
        with open(path) as f:
            return f.readlines()

def get_iterable_labeledsentences(text_list):
    data = []
    for line_no, line in enumerate(text_list):
        data.append(LabeledSentence(words= line.split(), tags=['SENT_'+str(line_no)]))
    return data

#def get_iterable_labeledsentences(text_list):
#    for line_no, line in enumerate(text_list):
#        yield LabeledSentence(words= line.split(), tags=['SENT_'+str(line_no)])

# TODO: train in loop
def doc2vec(text_list):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025,min_count=1,)
    sentences = get_iterable_labeledsentences(text_list)
    model.build_vocab(sentences)
    model.train(sentences)
    return model
    #model.docvecs.doctags
    #model.docvecs.similarity('SENT_1','SENT_2')
    #model.docvecs.doctag_syn0
    #model.docvecs.most_similar("SENT_1")
    #model.docvecs[1]
    #model.infer_vector(["hey",'we', 'the'])
    

def word2vec_model(train_textcorpus, word2vec_algo='gensim'):
    text_list_word_token = [];
    if type(train_textcorpus) == list:
        for line in train_textcorpus:
            text_list_word_token.append(word_tokenize(line))
    elif type(train_textcorpus) is str:
        if train_textcorpus == 'brown':
            from nltk.corpus import brown
            text_list_word_token = brown.sents()
        elif train_textcorpus == 'movie_reviews':
            from nltk.corpus import movie_reviews
            text_list_word_token = movie_reviews.sents()
        elif train_textcorpus == 'treebank':
            from nltk.corpus import treebank
            text_list_word_token = treebank.sents()
    # model.most_similar('man', topn=3)
    # model.most_similar('man', topn=3)
    # model.most_similar('man', topn=3)
    return gensim.models.Word2Vec(text_list_word_token)

def postagging(text_list, tagger = 'nltk'):
    tagged_text_list = [];
    for line in text_list:
        tagged_text_list.append(nltk.pos_tag(word_tokenize(line)))
    return tagged_text_list;


def removestopword(text_list, stopword_list = stopwords.words('english')):
    clean_text_list = [];
    for line in text_list:
        new_line = ' '.join(filter(lambda x: x.lower() not in stopword_list, line.split()))
        clean_text_list.append(new_line.replace("\n"," "))
    return clean_text_list;


def stem(text_list, stemmertype = 'porter'):
    if stemmertype.lower() == 'lancaster':
        stemmer = nltk.LancasterStemmer()
    else:
        stemmer = nltk.PorterStemmer()
    clean_text_list = [];
    for line in text_list:
        clean_text_list.append(' '.join([stemmer.stem(x) for x in word_tokenize(line)]))
    return clean_text_list;
    

def lemmatize(text_list, lemmatizertype = 'wordnet'):
    # for better lemmatization results, use the commented code below
    # import wordnet
    # pos_dict = {'NN':wordnet.NOUN, 'NNS':wordnet.NOUN, 'NNP':wordnet.NOUN, 'NNPS':wordnet.NOUN, 'VB':wordnet.VERB, 'VBD':wordnet.VERB, 'VBG':wordnet.VERB, 'VBN':wordnet.VERB, 'VBP':wordnet.VERB, 'VBZ':wordnet.VERB,'RB':wordnet.ADV, 'RBR':wordnet.ADV, 'RBS':wordnet.ADV,'JJ':wordnet.ADJ, 'JJR':wordnet.ADJ, 'JJS':wordnet.ADJ}
    # [lemmatizer.lemmatize(w,pos=pos_dict.get(t,wordnet.NOUN)) for (w,t) in p(word_tokenize(line))]
    lemmatizer = nltk.WordNetLemmatizer()
    clean_text_list = [];
    for line in text_list:
        clean_text_list.append(' '.join([lemmatizer.lemmatize(x) for x in word_tokenize(line)]))
    return clean_text_list;


def dict_replace(text_list, regex_dict_path = r"C:\Users\Adubey4\Desktop\text_preProces\preprocesstext\data\regex_dict.csv", word_dict_path=r"C:\Users\Adubey4\Desktop\text_preProces\preprocesstext\data\word_dict.csv"):

    regex_file = pd.read_csv(regex_dict_path,usecols=['search_for','replace_with'])
    pattern_list = [];
    for i in range(regex_file.shape[0]):
        pattern_list.append('(?P<'+regex_file.iloc[i]["replace_with"].strip()+'>'+regex_file.iloc[i]["search_for"].strip()+')')
    pattern_str = "|".join(pattern_list) 
    pattern = re.compile(pattern_str)
    text_str = pattern.sub(lambda m: m.lastgroup, "\n".join(text_list).lower())

    word_file = pd.read_csv(word_dict_path,usecols=['search_for','replace_with'])
    pattern_list = [];
    for i in range(word_file.shape[0]):
        word_or_str = "|".join(["("+x.strip().lower()+")" for x in word_file.iloc[i]["search_for"].split(';')])
        pattern_list.append('(?P<'+word_file.iloc[i]["replace_with"].strip()+'>'+ word_or_str +')')
    pattern_str = "|".join(pattern_list)
    pattern = re.compile(pattern_str)
    return pattern.sub(lambda m: m.lastgroup, text_str).split('\n')

# TODO: Include all good spell checkers, Pychant/ PyEnchant etc
# http://pythonhosted.org/pyenchant/download.html    
# https://pypi.python.org/pypi/autocorrect/0.1.0    
# pip install autocorrect
# http://stackoverflow.com/questions/4500752/python-check-whether-a-word-is-spelled-correctly
# https://pypi.python.org/pypi/hunspell/0.2
# https://github.com/hunspell/hunspell
# https://github.com/Sentynel/hunspell-cffi
# pip install Hunspell-CFFI
# http://extensions.openoffice.org/en/project/english-dictionaries-apache-openoffice
def spellcorrection(text_list, spell_corrector = 'autocorrect', dictpath=r"C:\Users\Adubey4\Desktop\pychant\dict-en"):
    clean_text_list = [];
    if spell_corrector.lower() == 'autocorrect':
        for line in text_list:
            clean_text_list.append(' '.join([spell(x) for x in line.split()]))
        return clean_text_list;

    elif spell_corrector.lower() == 'hunspell':
        spellchecker = hunspell.Hunspell(path=dictpath)
        for line in text_list:
            clean_text_list.append(' '.join([x if spellchecker.check(x) else spellchecker.suggest(x)[0] for x in line.split()]))
        return clean_text_list;
