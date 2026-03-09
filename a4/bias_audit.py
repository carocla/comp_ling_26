completed                = False         # Change this flag to True when you've completed the assignment.
expected_completion_date = '02/16/2024'  # If your assignment is late, change this date to your expected completion date.
questions_or_comments    = ""            # Fill in this string with any questions or comments you have; leave empty if none.
extensions               = False         # Change this flag to True if you completed any extensions for this assignment.
extensions_description   = ""            # If you did any extensions, briefly explain what you did and where we should look for it.                                                                                


import json, math
from collections import defaultdict

class PMICalculator:
    """Code to read the SNLI corpus and calculate PMI association metrics on it.
    """
    
    def __init__(self, infile = '/projects/e31408/data/a4/snli_1.0/snli_1.0_dev.jsonl', label_filter=None):
        self.infile = infile
        self.label_filter = label_filter # restricts the set of examples to read

        # mappings of words to indices of documents in which they appear
        self.premise_vocab_to_docs = defaultdict(set) 
        self.hypothesis_vocab_to_docs = defaultdict(set)
        self.n_docs = 0
        self.COUNT_THRESHOLD = 10
        
    
    def preprocess(self):
        """
        Read in the SNLI corpus and accumulate word-document counts to later calculate PMI.

        Your first task will be to look at the corpus and figure out how to read in its format.
        One hint - each line in the '.jsonl' files is a json object that can be read into a python
        dictionary with: json.loads(line)

        The corpus provides pre-tokenized and parsed versions of the sentences; you should use this
        existing tokenization, but it will require getting one of the parse representations and
        manipulating it to get just the tokens out. I recommend using the _binary_parse one.
        Remember to lowercase the tokens.

        As described in the assignment, instead of raw counts we will use binary counts per-document
        (e.g., ignore multiple occurrences of the same word in the document). This works well
        in short documents like the SNLI sentences.

        To make the necessary PMI calculations in a computationally efficient manner, the code is set up
        so that you do this slightly backwards - instead of accumulating counts of words, for each word
        we accumulate a set of indices (or other unique identifiers) for the documents in which it appears.
        This way we can quickly see, for instance, how many times two words co-occur in documents by 
        intersecting their sets. Document identifiers can be whatever you want; I recommend simply 
        keeping an index of the line number in the file with `enumerate` and using this.

        You can choose to modify this setup and do the counts for PMI some other way, but I do recommend
        going with the way it is.

        When loading the data, use the self.label_filter variable to restrict the data you look at:
        only process those examples for which the 'gold_label' key matches self.label_filter. 
        If self.label_filter is None, include all examples.        

        Finally, once you've loaded everything in, remove all words that don't appear in at least 
        self.COUNT_THRESHOLD times in the hypothesis documents.
        

        Parameters
        ----------
        None

        Returns
        -------
        None (modifies self.premise_vocab_to_docs, self.hypothesis_vocab_to_docs, and self.n_docs)

        """
        # >>> YOUR ANSWER HERE
        pass
        # >>> END YOUR ANSWER

        
    def pmi(self, word1, word2, cross_analysis=True):
        """
        Calculate the PMI between word1 and word1. the cross_analysis argument determines
        whether we look for word1 in the premise (True) or in the hypothesis (False).
        In either case we look for word2 in the hypothesis.        

        Since we are using binary counts per document, the PMI calculation is simplified.
        The numerator will be the number of total number of documents times the number
        of times word1 and word2 appear together. The denominator will be the number
        of times word1 appears total times the number of times word2 appears total.

        Do this using set operations on the document ids (values in the self.*_vocab_to_docs
        dictionaries). If either the numerator or denominator is 0 (e.g., any of the counts 
        are zero), return 0.

        Parameters
        ----------
        word1 : str
            The first word in the PMI calculation. In the 'cross' analysis type,
            this refers to the word from the premise.
        word2 : str
            The second word in the PMI calculation. In both analysis types, this
            is a word from the hypothesis documents.
        cross_analysis : bool
            Determines where to look up the document counts for word1;
            if True we look in the premise, if False we look in the hypothesis.

        Returns
        -------
        float
            The pointwise mutual information between word1 and word2.
        
        """

        # >>> YOUR ANSWER HERE
        return 0.0
        # >>> END YOUR ANSWER 

    def print_top_associations(self, target, n=10, cross_analysis=True):
        """
        Function to print the top associations by PMI across the corpus with
        a given target word. This is for qualitative use and 

        Since `word2` in the PMI calculation will always use counts in the 
        hypothesis, you'll want to loop over all words in the hypothesis vocab.

        Calculate PMI for each relative to the target, and print out the top n
        words with the highest values.
        """
        # >>> YOUR ANSWER HERE
        pass
        # >>> END YOUR ANSWER 
    

if __name__ == '__main__':
    all_labels = PMICalculator()
    all_labels.preprocess()


    # Below add whatever code you wish to do further qualitative analyses,
    # or make a separate script for that which loads your PMICalculator class.
    #
    # >>> YOUR ANSWER HERE
    pass
    # >>> END YOUR ANSWER 
