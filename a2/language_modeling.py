completed                = False         # Change this flag to True when you've completed the assignment.
expected_completion_date = '01/24/2024'  # If your assignment is late, change this date to your expected completion date.
questions_or_comments    = ""            # Fill in this string with any questions or comments you have; leave empty if none.
extensions               = False         # Change this flag to True if you completed any extensions for this assignment.
extensions_description   = ""            # If you did any extensions, briefly explain what you did and where we should look for it.

import math
from collections import defaultdict

class NgramLanguageModel:
    """
    Class with code to train and run n-gram language models with add-k smoothing.
    """

    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)

        self.k = 0.01

    def train(self, infile='samiam.train'):
        """Trains the language models by calculating n-gram counts from the corpus
        at the path given in the variable `infile`.

        These counts should be accumulated on the unigram_counts and bigram_counts
        objects. Note that these must be referenced with a prefix of 'self.', e.g.:
            self.unigram_counts

        You can assume the training corpus contains one sentence per line, which is
        already tokenized for you and thus can be tokenized with simply sentence.split().

        Remember that you have to add a special '<s>' token to the beginning
        and '</s>' token to the end of each sentence to correctly estimate the
        probabilities.

        To run properly with the autograder, make keys for your bigram_counts dict
        by joining with an underscore (e.g, 'GREEN_EGGS').

        Parameters
        ----------
        infile : str (defaults to 'brown_news.train')
            File path to the training corpus.

        Returns
        -------
        None (updates class attributes self.*_counts)
        """
        with open(infile, "r", encoding="utf-8") as f:
            for sentence in f:
                tokens = ["<s>"] + sentence.split() + ["</s>"]

                for token in tokens:
                    self.unigram_counts[token] += 1

                for i in range(len(tokens) - 1): #each adj. word pair
                    temp = tokens[i] + "_" + tokens[i + 1]
                    self.bigram_counts[temp] += 1
        pass


    def predict_unigram(self, sentence):
        """Calculates the log probability of the given sentence using a unigram LM.

        You can assume the sentence can be tokenized with simply sentence.split(),
        but remember the special sentence-end token, which must match what you used in
        training. Do not include a probability for the sentence-start token.

        As we talked about in class, we use log probability to avoid floating point
        underflow. This allows us to add log probabilities to one another, which
        is equivalent to multiplying regular probabilities.

        I suggest creating a float variable initialized to 0.0, and as you go
        through the words in the sentence, do += math.log(prob) for each word.
        At the end, simply return this accumulated sum - it is the log probability.

        The book focuses on the bigram model. The MLE probability for the unigram model
        is a slightly special case, where the numerator is what we would expect (the
        counts of that unigram), but the denominator is just the count of all tokens in
        the training corpus. Convince yourself why this must be if it's not clear.

        Reminder that you should augment your probability estimates with add-k smoothing,
        referencing the value of k with `self.k`; the bigram version is equation 3.25 in
        SLP, Ch 3. In this case we're arbitrarily choosing k to be 0.01, which is already
        set in the constructor for this class (the __init__ function).

        Parameters
        ----------
        sentence : str
            A sentence for which to calculate the probability.

        Returns
        -------
        float
            The log probability of the sentence.
        """
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        N = sum(self.unigram_counts.values())
        V = len(self.unigram_counts)
        logp = 0.0
        for w in tokens[1:]:
            c = self.unigram_counts[w]
            prob = (c + self.k) / (N + self.k*V)
            logp += math.log(prob)
        return logp


    def predict_bigram(self, sentence):
        """Calculates the log probability of the given sentence using a bigram LM.

        Analogous to predict_unigram, but uses a bigram model instead.

        Reminder to incorporate sentence-start and sentence-end tokens that match what
        you used in training.

        Parameters
        ----------
        sentence : str
            A sentence for which to calculate the probability.

        Returns
        -------
        float
            The log probability of the sentence.
        """
        tokens = ["<s>"] + sentence.split() + ["</s>"]
        V = len(self.unigram_counts)
        logp = 0.0
        for n in range(0, len(tokens) - 1):
            prev, cur = tokens[n], tokens[n+1]
            key = prev + "_" + cur
            num = self.bigram_counts[key] + self.k
            denom = self.unigram_counts[prev]+ self.k*V
            logp += math.log(num/denom)
        return logp


    def test_perplexity(self, test_file, ngram_size='unigram'):
        """Calculate the perplexity of the trained LM on a test corpus.

        This seems complicated, but is actually quite simple.

        First we need to calculate the total probability of the test corpus.
        We can do this by summing the log probabiities of each sentence in the corpus.

        Then we need to normalize (e.g., divide) this summed log probability by the
        total number of tokens in the test corpus. The one tricky bit here is we need
        to augment this count of the total number of tokens by one for each sentence,
        since we're including the sentence-end token in these probability estimates.

        Finally, to convert this result back to a perplexity, we need to multiply it
        by negative one, and exponentiate it - e.g., if we have the result of the above
        in a variable called 'val', we will return math.exp(val).

        This log-space calculation of perplexity is not super directly explained in
        the main text of the chapter, but it is related to information theory math
        that is described in section 3.7.

        Another nice explanation of this, for your reference, is here:
        https://towardsdatascience.com/perplexity-in-language-models-87a196019a94

        Parameters
        -------
        test_file : str
            File path to a test corpus.
            (assumed pre-tokenized, whitespace-separated, one line per sentence)

        ngram_size : str
            A string ('unigram' or 'bigram') specifying which model to use.
            Use this variable in an if/else statement.

        Returns
        -------
        float
            The perplexity of the corpus (normalized total log probability).
        """
        total_logp = 0.0
        token_count = 0
        with open(test_file, "r", encoding="utf-8") as f:
            for sentence in f:
                token_count += len(sentence.split()) + 1
                if ngram_size == 'unigram':
                    total_logp += self.predict_unigram(sentence)
                else:
                    total_logp += self.predict_bigram(sentence)
            N = len(self.unigram_counts)
            perplex = math.exp(-1 * total_logp / token_count)
            # N is total number of words seen
            return perplex


if __name__ == '__main__':
    ngram_lm = NgramLanguageModel()
    ngram_lm.train('samiam.train')
    print('Training perplexity, unigram:\t', ngram_lm.test_perplexity('samiam.train'))
    print('Training perplexity, bigram:\t', ngram_lm.test_perplexity('samiam.train','bigram'))
    print('Test perplexity, unigram:\t', ngram_lm.test_perplexity('samiam.test'))
    print('Test perplexity, bigram:\t', ngram_lm.test_perplexity('samiam.test','bigram'))
