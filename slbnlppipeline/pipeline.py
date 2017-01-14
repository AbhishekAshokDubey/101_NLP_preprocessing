# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 04:44:48 2017

@author: ADubey4
"""

try:
    __import__('imp').find_module('nltk')
    __import__('imp').find_module('gensim')
    __import__('imp').find_module('autocorrect')
except ImportError:
    raise ImportError("Please install nltk, gensim & other libraries below (using following commands)\n"
    "pip install nltk\n"
    "pip install gensim\n"
    "pip install autocorrect\n"
    "pip install Hunspell-CFFI\n"
    "also install following packages for nltk (use following commands inside python):\n"
    "nltk.download('wordnet')\n"
    "nltk.download('averaged_perceptron_tagger')\n"
    "nltk.download('stopwords')\n"
    "nltk.download('punkt')")

import yaml
import operator
from nltk.corpus import stopwords
import processtext
import summarizer

def pipeline(config_file_path, sequence=[]):
    with open(config_file_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    sorted_operation_order = sorted(cfg['pipeline'].items(), key=operator.itemgetter(1))
    text_list = processtext.readfile(cfg['readfile']['path'], cfg['readfile']['column_info'])

    for (operation,rank) in sorted_operation_order:
        if rank > 0:
            print(operation + "...")
            if operation == 'stopword_removal':
                if cfg['stopword_removal']['stopwords_file_path'].strip() == '':
                    text_list = processtext.removestopword(text_list, stopword_list = stopwords.words('english'))
                else:
                    path = cfg['stopword_removal']['stopwords_file_path'].strip()
                    with open(path) as f:
                        stopwords_list = f.readlines()
                    stopwords_list = [x.strip() for x in stopwords_list]
                    if cfg['stopword_removal']['append_to_default_list']:
                        stopwords_list += stopwords.words('english')
                    text_list = processtext.removestopword(text_list, stopwords_list)
                    
            if operation == 'dict_replacement':
                text_list = processtext.dict_replace(text_list, regex_dict_path=cfg['dict_replacement']['regex_dict_path'].strip(), word_dict_path=cfg['dict_replacement']['word_dict_path'].strip())
            if operation == 'spell_correction':
                text_list = processtext.spellcorrection(text_list, spell_corrector = cfg['spell_correction']['spell_corrector_type'].strip(), dictpath=cfg['spell_correction']['spell_list_hunspell'].strip())
            if operation == 'stemming':
                text_list = processtext.stem(text_list, stemmertype = cfg['stemming']['stemmertype'])
            if operation == 'lemmatizer':
                text_list = processtext.lemmatize(text_list, lemmatizertype = cfg['lemmatizer']['lemmatizertype'])
            if operation == 'postagging':                
                text_list = processtext.postagging(text_list, tagger = cfg['postagging']['tagger'])
            if operation == 'word2vec_model':
                if cfg['word2vec_model']['train_textcorpus'] == 'samedata':
                    return processtext.word2vec_model(text_list, word2vec_algo=cfg['word2vec_model']['word2vec_algo'])
                else:
                    return processtext.word2vec_model(cfg['word2vec_model']['train_textcorpus'], word2vec_algo=cfg['word2vec_model']['word2vec_algo'])
            if operation == 'summarization':                    
                if cfg['summarization']['type'] == 'doc_vec':
                    smz = summarizer.textSummarizer(summarizer_type = 'doc_vec', doc_vec_train_list = text_list)
                else:
                    smz = summarizer.textSummarizer(summarizer_type = 'word_freq', min_cut=0.1, max_cut=0.9)
                text_list = smz.summarize_list(text_list, cfg['summarization']['sent_len'])

    return text_list

#config_file_path = r"C:\Users\Adubey4\Desktop\git\101_NLP_preprocessing\resources\sample_nlp_config_pipeline.yaml"
#a = pipeline(config_file_path)