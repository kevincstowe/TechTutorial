# import modules & set up logging
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Defining a helper function that will print out some results for any given model
def evaluate_model(model):
    print(model)
    print(model['first'])
    print("Most similar to 'first' : " + str(model.most_similar(['first'])))    # Looks at the most similar words, based on cosine similarity
    print("Most similar to 'python' : " + str(model.most_similar(['python'])))  # Sometimes the results are weird, especially for rarer words

    # This is the classic example : king - man + woman should equal queen.
    print("king - man + woman : " + str(model.most_similar(positive=['king', 'woman'], negative=[
        'man'])))


''' 
Starting tiny : two sentences, each a list of words 
'''

from gensim.models import Word2Vec
sentences = [['this', 'is', 'the', 'first', 'sentence'], ['the', 'second', 'sentence', 'is', 'here']]

# Train the model. "min_count=1" makes it include all words, even if they only occur once
model = Word2Vec(sentences, min_count=1)

# Run the helper function defined above:
evaluate_model(model)


''' 
Train a larger model! Let's look at some Wiki data 
'''

# Load each line as a sentence, then split that line into words based on white space
# If you'd like to use any other file that is one sentence per line, just replace 'wiki_sents.txt'
sentences = [line.split() for line in open("wiki_sents.txt", encoding="utf-8").readlines()]
print ("This data has : " + str(len(sentences)) + " sentences")

# Train and evaluate the model
model = Word2Vec(sentences, min_count=3)
evaluate_model(model)

# A lot better! But still not perfect, as the dataset is too small, even at +1 million sents


''' 
Finally, a huge model : the GoogleNews dataset. This takes forever to train, but you can download and use a pre-trained version. 
'''

from gensim.models import KeyedVectors

# You'll have to download this file and put it somewhere ( available at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/, or search for it)
model = KeyedVectors.load_word2vec_format('C:/Users/Kevin/PycharmProjects/vectors/models/GoogleNews-vectors-negative300.bin', binary=True)
evaluate_model(model)

# Much better! The bigger the data, the better.