# 101_NLP_preprocessing


Author: Abhishek Dubey<br/>
Version: 0.1


### Installation prerequisites 
Please install nltk, gensim & other libraries below (using following commands)

pip install nltk<br/>
pip install gensim<br/>
pip install autocorrect<br/>
pip install Hunspell-CFFI<br/>

Please also install following packages/ corpus for nltk (use following commands inside python):

import nltk<br/>
nltk.download('wordnet')<br/>
nltk.download('averaged_perceptron_tagger')<br/>
nltk.download('stopwords')<br/>
nltk.download('punkt')<br/>


### Installation steps<br/>
1. cd 'path to repo'
2. python setup.py install


### Sample code
```
config_file_path = "path to the yaml file"
pipeout = processtext.pipeline(config_file_path)
```

OR

```
text_list1 = processtext.readfile('mydata.csv',column_info='text')
text_list1 = processtext.readfile('mydata.csv',column_info=1)
text_list1 = list(text_list1.values.flatten())
text_list2 = processtext.readfile('mydata.txt',column_info=1)

text_list_stpwrd = processtext.removestopword(text_list)
text_list_dict = processtext.dict_replace(text_list1, regex_dict_path = r"path to regex file")

text_list_spell1 = processtext.spellcorrection(text_list)
text_list_spell2 = processtext.spellcorrection(text_list,spell_corrector = 'hunspell', dictpath=r"C:\Users\Adubey4\Desktop\pychant\dict-en")

text_list_stemp = processtext.stem(text_list, stemmertype = 'porter')
text_list_steml = processtext.stem(text_list, stemmertype = 'lancaster')
```

##Note:
For the yaml & other sample resource files, referred below, check the "Resources" folder:

A sample YAML file for the pipeline:
```
readfile:
    path: 'path to main text file'
    column_info: text # column name/ number
stopword_removal:
    stopwords_file_path: 'path for the stop words file'
    append_to_default_list: False
spell_correction:
    spell_corrector_type : autocorrect # options: autocorrect, hunspell
    spell_list_hunspell : 'C:\Users\Adubey4\Desktop\pychant\dict-en'
dict_replacement:
    regex_dict_path : 'path to regex file'
    word_dict_path : 'path for dict replacement file'
stemming:
    stemmertype: porter # options: porter, lancaster
lemmatizer:
    lemmatizertype : wordnet # options: wordnet
postagging:
    tagger: nltk
word2vec_model:
    word2vec_algo: gensim
    train_textcorpus: brown # options: 'samedata', 'brown', 'movie_reviews', 'treebank' corpus from nltk corpus
pipeline:
    stopword_removal: 1 # No operation is 0, First operation is 1, Second operation is 2 and so on
    spell_correction: 0
    dict_replacement: 2
    stemming: 0
    lemmatizer: 3
    postagging: 0
    word2vec_model: 4
    doc2vec_model: 0
```
