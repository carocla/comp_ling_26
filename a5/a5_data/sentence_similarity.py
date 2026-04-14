import csv, string
import math
import numpy as np
from scipy.stats import spearmanr
from embeddings import Embeddings
from nltk import pos_tag
from nltk.tokenize import word_tokenize



def read_sts(infile = '/projects/e31408/data/a5/sts-dev.csv'):
#def read_sts(infile = 'sts-dev.csv'):
    sts = {}
    for row in csv.reader(open(infile), delimiter='\t'):
        if len(row) < 7: continue
        val = float(row[4])
        s1, s2 = row[5], row[6]
        sts[s1, s2] = val / 5.0
    return sts

def calculate_sentence_embedding(embeddings, sent, weighted = False):
    """
    Calculate a sentence embedding vector.

    If weighted is False, this is the elementwise sum of the constituent word vectors.
    If weighted is True, multiply each vector by a scalar calculated
    by taking the log of its word_rank. The word_rank value is available
    via a dictionary on the Embeddings class, e.g.:
       embeddings.word_rank['the'] # returns 1

    In either case, tokenize the sentence with the `word_tokenize` function,
    lowercase the tokens, and ignore any words for which we don't have word vectors. 

    Parameters
    ----------
    sent : str
        A sentence for which to calculate an embedding.

    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    np.array of floats
        Embedding vector for the sentence.
    
    """
    # >>> YOUR ANSWER HERE
    tokens = word_tokenize(sent)
    results = []
    for token in tokens:
        token = token.lower()
        if embeddings.__contains__(token):
            scalar = 1
            if weighted:
                wordRank = embeddings.word_rank[token]
                scalar = math.log(wordRank)
            results.append(embeddings.__getitem__(token)*scalar)
    return np.sum(results,axis = 0)
    # >>> END YOUR ANSWER



def score_sentence_dataset(embeddings, dataset, weighted = False):
    """
    Calculate the correlation between human judgments of sentence similarity
    and the scores given by using sentence embeddings.

    Parameters
    ----------
    dataset : dictionary of the form { (sentence, sentence) : similarity_value }
        Dataset of sentence pairs and human similarity judgments.
    
    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    float
        The Spearman's Rho ranked correlation coefficient between
        the sentence emedding similarities and the human judgments.     
    """
    # >>> YOUR ANSWER HERE
    emedScores=[]
    humanScores=[]
    for key,val in dataset.items():
        s1,s2=key
        s1_embedding = calculate_sentence_embedding(embeddings, s1, weighted)
        s2_embedding = calculate_sentence_embedding(embeddings, s2, weighted)
        emedScore=embeddings.cosine_similarity(s1_embedding,s2_embedding)
        emedScores.append(emedScore)
        humanScores.append(val)
    res = spearmanr(emedScores,humanScores)
    return res.statistic
    # >>> END YOUR ANSWER

if __name__ == '__main__':
    embeddings = Embeddings()
    sts = read_sts()
    
    print('STS-B score without weighting:', score_sentence_dataset(embeddings, sts))
    print('STS-B score with weighting:', score_sentence_dataset(embeddings, sts, True))
