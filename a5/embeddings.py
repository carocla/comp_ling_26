"""
Note: Since this assignment consists of multiple files, the usual flags will
only occur in this file but represent your responses for the entire assignment. 
Please adjust them accordingly (i.e. only flip 'completed' to True when you have
finished working on all of the assignment files)
"""
completed                = False         # Change this flag to True when you've completed the assignment.
expected_completion_date = '02/26/2024'  # If your assignment is late, change this date to your expected completion date.
questions_or_comments    = ""            # Fill in this string with any questions or comments you have; leave empty if none.
extensions               = False         # Change this flag to True if you completed any extensions for this assignment.
extensions_description   = ""            # If you did any extensions, briefly explain what you did and where we should look for it.

import math
import numpy as np

class Embeddings:

    def __init__(self, glove_file = '/projects/e31408/data/a5/glove_top50k_50d.txt'):
        self.embeddings = {}
        self.word_rank = {}
        for idx, line in enumerate(open(glove_file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1

    def __getitem__(self, word):
        return self.embeddings[word]

    def __contains__(self, word):
        return word in self.embeddings

    def vector_norm(self, vec):
        """
        Calculate the vector norm (aka length) of a vector.

        This is given in SLP Ch. 6, equation 6.8. For more information:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Parameters
        ----------
        vec : np.array
            An embedding vector.

        Returns
        -------
        float
            The length (L2 norm, Euclidean norm) of the input vector.
        """
        # >>> YOUR ANSWER HERE
        return 1.0
        # >>> END YOUR ANSWER

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        # >>> YOUR ANSWER HERE
        return 1.0
        # >>> END YOUR ANSWER

    def most_similar(self, vec, n = 5, exclude = []):
        """
        Return the most similar words to `vec` and their similarities. 
        As in the cosine similarity function, allow words or embeddings as input.


        Parameters
        ----------
        vec : str or np.array
            Input to calculate similarity against.

        n : int
            Number of results to return. Defaults to 5.

        exclude : list of str
            Do not include any words in this list in what you return.

        Returns
        -------
        list of ('word', similarity_score) tuples
            The top n results.        
        """
        # >>> YOUR ANSWER HERE
        return [('cat', 987654321)]
        # >>> END YOUR ANSWER

if __name__ == '__main__':
    
    embeddings = Embeddings()
    word = 'lemon'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])
