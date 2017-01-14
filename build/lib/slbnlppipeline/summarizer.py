#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:12:08 2017

@author: ADubey4
"""

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from scipy import spatial
from processtext import doc2vec

class docvec_summarizers:
    def __init__(self, model = None , train_doc_list = []):
        if model:
            self._model = model
        elif train_doc_list:
            self._model = doc2vec(train_doc_list)

    def summarize(self, doc, sent_count):
        sents = sent_tokenize(doc)
        if sent_count >= len(sents):
            return sents
        doc_vec = self._model.infer_vector(word_tokenize(doc))
        sent_vec_similarity_list = defaultdict(int)
        
        for i,sent in enumerate(sents):
            sent_vec = self._model.infer_vector(word_tokenize(sent))
            sent_cosine_dist = spatial.distance.cosine(doc_vec,sent_vec)
            sent_vec_similarity_list[i] = 1 - sent_cosine_dist
            
        sents_idx = nlargest(sent_count, sent_vec_similarity_list, key=sent_vec_similarity_list.get)
        return [sents[j] for j in sents_idx]
            
    def summarize_list(self, doc_list , sent_count):
        doc_summ_list = []
        for doc in doc_list:
            doc_summ_list.append(self.summarize(doc, sent_count))
        return doc_summ_list


# frequency_summarizer from: http://glowingpython.blogspot.in
# http://glowingpython.blogspot.in/2014/09/text-summarization-with-nltk.html
class frequency_summarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english')
                              + list(punctuation))

    def _word_frequencies(self, word_sent):
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        m = float(max(freq.values()))
        for w in list(freq):
            freq[w] = freq[w] / m
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, doc, sent_count):
        sents = sent_tokenize(doc)
        if sent_count >= len(sents):
            return sents
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._word_frequencies(word_sent)
        ranking = defaultdict(int)
        for (i, sent) in enumerate(word_sent):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sents_idx = nlargest(sent_count, ranking, key=ranking.get)
        return [sents[j] for j in sents_idx]
    
    def summarize_list(self, doc_list , sent_count):
        doc_summ_list = []
        for doc in doc_list:
            doc_summ_list.append(self.summarize(doc, sent_count))
        return doc_summ_list


class textSummarizer:
    def __init__(self, summarizer_type = 'word_freq', min_cut=0.1, max_cut=0.9, doc_vec_model = None , doc_vec_train_list = []):
        self.summarizer_type = summarizer_type
        if self.summarizer_type == 'word_freq':
            self._summarizer = frequency_summarizer(min_cut, max_cut)
        else:
            self._summarizer = docvec_summarizers(model = doc_vec_model, train_doc_list = doc_vec_train_list)
    
    def summarize(self, doc, sent_count):
        return self._summarizer.summarize(doc, sent_count)
    
    def summarize_list(self, doc_list , sent_count):
        return self._summarizer.summarize_list(doc_list, sent_count)

#text = "The target of the automatic text summarization is to reduce a textual document to a summary that retains the pivotal points of the original document. The research about text summarization is very active and during the last years many summarization algorithms have been proposed. In this post we will see how to implement a simple text summarizer using the NLTK library (which we also used in a previous post) and how to apply it to some articles extracted from the BBC news feed. The algorithm that we are going to see tries to extract one or more sentences that cover the main topics of the original document using the idea that, if a sentences contains the most recurrent words in the text, it probably covers most of the topics of the text. Here's the Python class that implements the algorithm: "
#text_list = processtext.readfile('datasets/text_emotion.csv',column_info='content')
#
#s1 = textSummarizer()
#s2 = textSummarizer(summarizer_type = 'doc_vec', doc_vec_train_list = text_list)
#
#print(s1.summarize(text, 2))
#print(s2.summarize(text, 2))