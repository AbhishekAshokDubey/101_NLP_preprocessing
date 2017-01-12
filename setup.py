import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#def read(fname):
#    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "slbnlppipeline",
    version = "0.1",
    author = "Abhishek Dubey",
    author_email = "adubey40@gmail.com",
    description = ("NLP pipeline"),
    license = "BSD",
    keywords = "nlp",
    packages=['slbnlppipeline'],
    #package_dir = {'':'slbnlppipeline'},
    install_requires=[
    "nltk",
    "gensim",
    "autocorrect",
	"scipy"
    ],
    #package_data={'slbnlppipeline': ['data/*.csv']},
#    data_files=[('slbnlppipeline', ['nlp_pipeline_config.yaml']),
#                  #('data', ['regex_dict.csv','word_dict.csv'])
#                  ],
    long_description="A nlp pipeline For & By ADS at SLB pune, build over NLTL & Gensim",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)