completed                = False         # Change this flag to True when you've completed the assignment.
expected_completion_date = '02/02/2024'  # If your assignment is late, change this date to your expected completion date.
questions_or_comments    = ""            # Fill in this string with any questions or comments you have; leave empty if none.
extensions               = False         # Change this flag to True if you completed any extensions for this assignment.
extensions_description   = ""            # If you did any extensions, briefly explain what you did and where we should look for it.


import os, math
from collections import Counter

class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """

    def __init__(self, train_dir='haiti/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        self.vocabulary = set([])
        self.logprior = {}
        self.loglikelihood = {} # keys should be tuples in the form (w, c)


    def train(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of SLP Chapter 4.

        Note that self.train_data contains the paths to training data files.
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()

        Like in A2, you can assume they are pre-tokenized so you can get words with
        simply `words = doc.split()`

        Remember to account for whether the self.REMOVE_STOPWORDS flag is set or not;
        if it is True then the stopwords in self.stopwords should be removed whenever
        they appear.

        When converting from the pseudocode, consider how many loops over the data you
        will need to properly estimate the parameters of the model, and what intermediary
        variables you will need in this function to store the results of your computations.

        Parameters
        ----------
        None (reads training data from self.train_data)

        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # each line = one document!!!
        doc_counts = {}
        wc_by_class = {}
        total_tokens = {}
        total_docs = 0

        for c in self.classes:
            wc_by_class[c] = Counter()
            total_tokens[c] = 0

            c_docs = open(self.train_data[c], encoding="utf-8").readlines()
            doc_counts[c] = len(c_docs)
            total_docs += doc_counts[c]

            for doc in c_docs:
                words = doc.split()
                if self.REMOVE_STOPWORDS:
                    words = [w for w in words if w not in self.stopwords]

                self.vocabulary.update(words)
                wc_by_class[c].update(words)
                total_tokens[c] += len(words)

        for c in self.classes:
            self.logprior[c] = math.log(doc_counts[c] / total_docs)

        V = len(self.vocabulary)
        for c in self.classes:
            denom = total_tokens[c] + V
            for w in self.vocabulary:
                num = wc_by_class[c][w] + 1
                self.loglikelihood[w, c] = math.log(num / denom)

        
    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier. 

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4.

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """        
        sum = self.logprior[c]

        words = doc.split()
        if self.REMOVE_STOPWORDS:
            words = [w for w in words if w not in self.stopwords]

        for word in words:
            if word in self.vocabulary:
                sum += self.loglikelihood[word, c]
        return sum
                
    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.

        Consider using the `max` built-in function. There are a number of ways to do this:
           https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

        Parameters
        ----------
        doc : str
            A text representation of a document to score.
        
        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        chances = {}
        for c in self.classes:
            chances[c] = self.score(doc, c)
        return max(chances, key=chances.get)


    def evaluate(self, test_dir='haiti/test', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Not the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to. 

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """        
        test_data = {c: os.path.join(test_dir, c) for c in self.classes}
        if not target in test_data:
            print('Error: target class does not exist in test data.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for c in self.classes:
            with open(test_data[c], 'r') as f:
                for doc in f:
                    doc = doc.strip()
                    if not doc:
                        continue
                    pred = self.predict(doc)
                    if c == target:
                        if pred == target:
                            outcomes['TP'] += 1
                        else:
                            outcomes['FN'] += 1
                    else:
                        if pred == target:
                            outcomes['FP'] += 1
                        else:
                            outcomes['TN'] += 1

        precision = outcomes['TP'] / (outcomes['TP'] + outcomes['FP'])
        recall = outcomes['TP'] / (outcomes['TP'] + outcomes['FN'])
        f1_score = (2*precision*recall) / (precision + recall)
        return (precision, recall, f1_score)


    def print_top_features(self, k=10):
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp( self.loglikelihood[w, c] - min(self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c) )
                results[c][w] = ratio

        for c in self.classes:
            print(f'Top features for class <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key = lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')
            
            
if __name__ == '__main__':
    target = 'relevant'

    clf = NaiveBayesClassifier(train_dir = 'haiti/train')
    clf.train()
    print(f'Performance on class <{target.upper()}>, keeping stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')
    
    clf = NaiveBayesClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Performance on class <{target.upper()}>, removing stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')


    clf.print_top_features()


