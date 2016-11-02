# 101_NLP_preprocessing


Author: Abhishek Dubey
Version: 0.1

##### Installation prerequisites 
"Please install nltk, gensim & other libraries below (using following commands)"

pip install nltk

pip install gensim

pip install autocorrect

pip install Hunspell-CFFI

Please also install following packages/ corpus for nltk (use following commands inside python):

import nltk

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')

nltk.download('stopwords')

nltk.download('punkt')

##### Sample code
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

#Note:
For the yaml & other sample resource files check the "Resources" folder:
