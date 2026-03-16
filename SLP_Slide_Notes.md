# Speech and Language Processing (Daniel Jurafsky & James H. Martin)

# Chapter 2: Words & Tokens

## Words

* **Tokenization**: the task of separating out or tokenizing words and word parts from running text  
* **Morpheme**: the meaningful subpart of words (like the morpheme \-er in the word longer)  
  * 每个汉子是一个 morpheme, single unit of meaning  
* **Byte-Pair Encoding (BPE):** an algorithm that automatically breaks up input text into tokens. This algorithm uses simple statistics of letter sequences to induce a vocabulary of subword tokens  
* **Edit Distance**: measures how similar two words or strings are based on the number of edits (insertions, deletions, substitutions) it takes to change one string into the other.  
* Punctuation is counted as words when relevant to the task  
  * For finding boundaries of things (commas, periods, colons) and for identifying some aspects of meaning (question marks, exclamation marks, quotation marks)  
  * LLMs generally count punctuation as separate words  
* **Word types** are the number of distinct words in a corpus  
  * if the set of words in the vocabulary word instance is V, the number of types is the vocabulary size |V|  
* **Word instances** are the total number N of running words  
* **Orthographic Words**: words based on our English writing system.   
* Herdan’s Law: The relationship between the number of types |V| and number of instances. k and β are positive constants, and 0 \< β \< 1\.  
  * |V| \= kN^β  
* **Function words:** the grammatical words like English a and of, that tend not to grow indefinitely  
* **Content words:** nouns, adjectives and verbs that tend to have meanings about people and places and events.   
  * Nouns, and especially particular nouns like names and technical terms do tend to grow indefinitely.  
* Computational models constantly see words it has never seen before

## Morphemes: Part of Words

* Two broad classes of morphemes  
1. **Roots**: the central morpheme of the word, supplies main meaning  
2. **Affixes**: adding additional meanings of various kinds  
   Bonus. **Clitics**: a morpheme that acts synactically like a word but is reduced in form and   
    attached to another word  
   * i.e. *I’ve* 的 *‘ve* / *doctor’s* 的 *‘s*  
     * clitics can’t stand on their own  
* **Inflectional morphemes:** grammatical morphemes that tend to play a syntactic role, such as marking agreement  
  * Productive, predictable (*\-s*/-*es*, \-*ed*)  
* **Derivational morphemes:** more idiosyncratic in their application and meaning  
  * Difficult to predict exact meaning (i.e. *carefully* 的 \-*ful* / *\-ly)*  
* **Morphological typology:** The study of how languages vary in their morphology, i.e., how words break up into their parts  
* Two dimensions are particularly relevant for computational word tokenization  
1. morphemes per word  
* **Isolating** languages: each word usually has one morpheme (i.e. Cantonese, Vietnamese)  
* **Synthetic** langauges: a single word may have many morphemes   
* **Polysynthetic** languages: a single word may have veryyy many morphemes (i.e. Koryak)

2. The degree to which morphemes are easily segmentable  
* **Agglutinative** languages: morphemes have relatively clean boundaries (Turkish)  
* **Fusion** languages: a single affix may have multiple morphemes (Russian)  
  * English suffix \-s in *She reads the article* is fusion, it means third person singular but also means present tense  
* Languages dont necessarily fit neatly into those boxes, they are more general tendencies

## Unicode

* **Unicode** standard is a method for representing text written using any character in any script of the languages of the world  
  * Assigns a unique id, a **code point** for each character in unicode (150k of them\!)  
    * a code point does not specify the **glyph**, the visual rep. of a char  
      * glyphs are stored in **fonts**  
  * **code points** are the abstract unicode rep of the char, when we need to represent a character in a text string, we write an **encoding** of the character  
    * **UTF-8** is the most common encoding standard  
      * represents characters efficiently, writes some chars w less bytes and some with more \-\> **variable-length encoding**  
* **ASCII** (American Standard Code for Information Interchange). ASCII represented each character with a single byte

## Subword Tokenization: Byte-Pair Encoding

* Three candidates for tokens: words, morphemes, and characters  
* Why tokenize?   
  * Makes it so different systems/algos can agree, standardization is essential for replicability.  
  * Smaller tokens for morphemes & letters eliminate the issue of **unknown words**  
    * modern tokenizers have sets of tokens with **subwords** including arbitrary substrings or meaning-bearing units like the morphemes \-est / \-er  
* Two tokenization algorithms that are widely used in modern LMs  
1. **byte-pair encoding** / **BPE** \- has two parts: **trainer** and **encoder**  
   * Used in large systems like OpenAI GPT4o  
     * iteratively merges frequent neighboring tokens to create longer and longer tokens  
* starts with a vocabulary that is the set of all individual characters  
2. **unigram** **language modeling**  
* We normally run BPE on the individual bytes of UTF-8-encoded text.  
  * we take a Unicode representations of text as a series of code points  
  * encode it in bytes using UTF-8  
  * we treat each of these individual bytes as the input to BPE  
* Language models usually create their tokens in a **pretokenization** stage   
  * this stage first segments the input using regular expressions  
    * for example: breaking the input at spaces and punctuation, stripping off clitics, and breaking numbers into sets of digits.   
  * Pretokenization can be made to allow BPE tokens to span multiple word  
    * **SuperBPE** and **BoundlessBPE** algos  
1. first induce regular BPE subword tokens by enforcing pretokenization  
2. then run a second stage of BPE allowing merges across spaces and punctuation  
* Many tokenizers used in practice for large language models are multilingual, trained on many languages  
  * Training data is still largely dominated by English text  
    * tokenizers do better on English as a result  
    * other languages get their words split up into shorter tokens

## Corpora

* When developing computational models for language processing from a corpus, it’s important to consider who produced the language, in what context, for what purpose  
* Corpus creators can build a **datasheet** or **data** **statement,** specifying properties like:  
  * Motivation  
    * Why was the corpus collected? By whom?  
    * Who funded it?  
  * Context  
    * When and in what situation was the text written/spoken?   
    * Was it spoken conversation, edited text, social media communication, monologue vs. dialogue?  
  * Language variety  
    * What language/dialect was it in?  
  * Speaker demographics  
    * Age/gender of text’s authors?  
  * Collection process  
    * How big is the data? If it is a subsample, how was it sampled?  
    * Was the data collected consensually?  
    * How was the data pre-processed?

## Regex

* re.search(pattern, string) \= python regex search function  
  * r”regex” is python syntax for pattern	  
  * \[^Ss\] \= neither ‘S’ nor ‘s’  
  * \* \= 0+ occurrences  
  * \+ \= 1+ occurrences  
  * A{n} \= n occurences of A  
  * \\b \= word boundary  
  * \\B \= non-word boundary  
* re.sub(pattern, repl, string) \= python regexsubstitution function  
  * re.sub(r"(\\d{2})/(\\d{2})/(\\d{4})", r"\\2-\\1-\\3", string)}  
    * 10/15/2011 \-\> 15-10-2011  
    * \\1 is first group stored digits  
* ?: \= **non-capturing group**   
  * r"(?:\\d\\d/\\d\\d/\\d\\d\\d\\d\\s+){14}(\\d\\d/\\d\\d/\\d\\d\\d\\d)" \= stores only 15th date  
* (?= pattern) is true if pattern occurs but is **zero-width**,  
* print(re.compile(*pattern*).findall(*text*)) \= pre-tokenization  
* python also has an external regex library, better than the internal re library  
  * in regex library it has special \\p and \\P operators   
    * \\p{L} \= any Unicode letter  
    * \\P{L} \= any non-letter  
    * \\p{N} \= any number  
    * \\P{N} \= any non-number

## Rule-based Tokenization

* In rule-based tokenization, we pre-define a standard and implement rules to implement that kind of tokenization  
* A rule-based tokenizer can also be used to expand **clitic** contractions that are marked by apostrophes  
  * ex: converting *what’re* to *what are* or *we’re* to *we are*.  
* **Named entity recognition**: the task of detecting names, dates, and organizations  
* Tokenization needs to be very fast, for rule-based tokenization we generally use deterministic algorithms based on Regex  
  * must be carefully designed to deal with ambiguities  
* Sentence segmentation   
  * Optional  
  * Important when detecting structure (ie parse structure)  
  * Depends on the language & genre

## Minimum Edit Distance

* **Minimum Edit Distance:** the minimum number of editing operations needed to transform one string into another. (substitution, insertion, deletion)  
* an **Alignment**: a correspondence between the substrings of the two sequences  
  * Beneath aligned strings is an **operator list**   
    * **d** for deletion, **s** for substitution, **i** for insertion  
* Could weight/cost factor these operations  
  * **Levenshtein** distance between two sequences is the simplest weighting factor in which each of the three operations has a cost of 1  
* Dynamic programming: applies a table-driven method to solve problems through combining solutions to subproblems

# Chapter 3: N-gram Language Models

* A language model is a machine learning LM model that predicts upcoming words.   
  * More formally, a LM assigns a probability to each possible next word, or gives a probability distribution over possible next words.  
* **Augmentiative and Alternative Communication (AAC)**  
* **n-gram**:  a probabilistic model that can estimate the probability of a word given the n-1 previous words, and thereby also to assign probabilities to entire sequences.  
  * a 2-gram / **bigram**, is a two-word sequence like “*The water”*  
  * 3-gram / **trigram**, is a three-word sequence of words like “*The water of”*

## N-Grams

* **bigram** model, for example, approximates the probability of a word given all the previous words P(wn|w1:n-1) by using only the conditional probability given the preceding word P(wn|wn−1).   
* **Markov** assumption: the assumption that the probability of a word depends only on the previous word  
* **Maximum Likelihood Estimation / MLE:** we get maximum likelihood estimation the MLE estimate for the parameters of an n-gram model by getting counts from a corpus, and normalizing the counts so that they lie between 0 and 1\.   
  *  In MLE, the resulting parameter set maximizes the likelihood of the training set T given the model M 	\-\> P(T|M)

  (3.11) Bigram probability of word wn given a previous word wn-1 

P(wn|wn-1) \=C(wn-1wn)C(wn-1)  
	(3.12) General case of MLE n-gram parameter estimation  
P(wn|wn-N+1:n-1) \= C(wn-N+1:n-1wn)C(wn-N+1:n-1)

* **Relative Frequency**: observed frequency of a sequence / observed frequency of a prefix  
* LM probabilities are stored and computed in log space as **log probabilities**  
  * Because:  
    * probabilities are \>=1, the more we multiply the smaller the product  
    * adding in log space is equivalent to multiplying in linear space, so we combine log probabilities through addition  
    * get bigger results\!   
    * if needed, convert back to probabilities by taking exp of the logprob g

## Evaluating LMs: Training and Test Sets

* **extrinsic evaluation**: end-to-end evaluation, embed the LM in an applicaion and measure improvement  
* **intrinsic evaluation**: measurees the quality of a model independent of any application  
* **perplexity:** sttandard intrinsic metric for measuring LM performance  
* to evaluate any ML model you need  
  * training set \- corpus on which n-gram LM is built  
  * development set  
  * test set \- reflect the language we want to use the model for  
    * must not be in training set \- thats **data contamination**  
    * only run on test set at very end  
* whichever LM assigns a higher probability to the test set is a better model

## Evaluating LMs: Perplexity

* **perplexity** is a metric that is per-word, normalized by length so we can compare probability of test sets across texts of different lengths  
* The **perplexity** (PP or PPL) of a language model on a test set is the inverse probability of the test set (one over the probability of the test set), normalized by the number of words (or tokens).  
  * sometimes called the per-word or per-token perplexity  
  * \*\*The lower the perplexity of a model on the data, the better the model  
  * \*\*The weighted average branching factor of a language  
  * Minimizing perplexity \== maximizing test set probability  
* An (intrinsic) improvement in perplexity does not guarantee an (extrinsic) improvement in the performance of a language processing task like speech recognition or machine translation

(3.16) Perplexity of W with a unigram language model  
perplexity(W) \=Ni=1N1P(Wi)

* **Sampling** from a distribution means to choose random points according to their likelihood  
  * Sampling from a language model means to generate some sentences, choosing each sentence according to its likelihood as defined by the model

## Generalizing vs. Overfitting the Training Set

* Avoid generalization tactics  
  * Use a training corpus that has a similar **genre** to whatever task you’re trying to accomplish  
  * Get training data in the appropriate dialect / variety

## Smoothing, Interpolation, and Backoff

* Problem with using MLE for probabilities: any finite training corpus will be missing some acceptable word sequences.   
  * Known as unseen sequences or **zeros** \- sequences that don’t occur in the training set but do occur in the test set  
  * They’re a problem because:  
    * their presence means we are underestimating the probability of word sequences that might occur, which hurts the performance of any application we want to run on this data  
    * if the probability of any word in the test set is 0, the probability of the whole test set is 0  
* Perplexity is defined based on the inverse prob. of the test set. If some words in context have zero probability, we can’t divide by zero, so we can’t compute perplexity at all.  
* The standard way to deal with zero probability n-grams is called **smoothing** or **discounting**  
* **Laplace/add-one smoothing**: The simplest way to do smoothing, add one to all the n-gram counts before we normalize them into probabilities  
  * practical for text classification but generally not optimal

(3.28) Add-k smoothing (\* \= adjusted)  
P\*Add-k(wn|wn-1)=C(wn-1wn)+kC(wn-1)+kV

* Sometimes using less context can help us generalize more for contexts that the model hasn’t learned much about  
* **interpolation**: computing a new probability by interpolating (weighting and combining) the trigram, bigram, and unigram probabilities.  
  * we mix together the probabilities, each weighted by a lambda λ.  
* **held-out corpus**: an additional training corpus, so-called because we hold it out from the training data, used to set the λ values.  
  * fix n-gram probabilities and then search for the λ values that give the highest probability of the held-out set  
* optimal set of λ’s can be found other ways  
  * **EM** algo: an iterative learning algo that converges on locally optimal λ’s  
* **backoff**: if the n-gram we need has zero counts, we approximate it by backing off to the (n-1)-gram. We continue backing off until we reach a history with counts.  
  * we have to **discount** the higher-order n-grams to save some probability mass for the lower order n-grams  
* **stupid backoff** is much simpler. If a higher-order n-gram has a zero count, we backoff to a lower order n-gram, weighed by a fixed weight.

## Advanced: Perplexity’s Relation to Entropy

* **Entropy** is a measure of information  
  * One intuitive way to think about entropy is as a *lower bound* on the number of bits it would take to encode a certain decision or piece of information in the optimal coding scheme

# Chapter 4: Logistic Regression and Text Classification

* **language modeling** can also be viewed as classification: each word can be thought of as a class, and so predicting the next word is classifying the context-so-far into a class for each next word  
* **Supervised ML:** in addition to the input and the set of output classes, we have a **labeled training set** and a **learning algorithm**  
* **Probabilistic classifiers** give the probability of the observation being in the class in addition to giving an answer (the class this observation is in).  
  * ie logistic regression  
* Probabilistic classifiers have 4 components:  
1. A **feature representation** of the input. For each input observation x(i) , this will be a vector of features \[x1, x2, ..., xn\]   
2. A classification function that computes the estimated class by computing the probability P(y=yi|x) for each output class yi  
   1. **sigmoid** and **softmax** are tools for classification  
3. An **objective function** that we want to optimize for learning, usually involving minimizing a loss function corresponding to error on training examples.  
   1. **cross-entropy loss function**  
4. An algorithm for optimizing the objective function  
   1. **stochastic gradient descent**

* Probabilistic ML classifiers have 2 phases  
  * **training**: We train the system (inlogistic regression \= training the weights w and b) using stochastic gradient descent and the cross-entropy loss.  
  * **test:** Given a test example x we compute the probability P(y=yi|x) for each output class yi. Then, given this vector of probabilities, we return the higher probability label y \= 1 or y \= 0\.

## The sigmoid function

* To make a decision on a test instance, after learning weights in training, the logistic regression classifier multiplies each input feature (xi) by its weight wi, sums up the weighted features and adds the bias term b. Resulting single \# z is the weighted sum of the evidence for the class. Pass sum through **sigmoid function** to generate a probability. 

(4.2)   
z=(i=1nwixi)+b

* **Logit:** input to the sigmoid function (the score **z \= w\*x+b**), inverse of the sigmoid  
* Logistic regression can be used with two classes (e.g., positive and negative sentiment) or with multiple classes (**multinomial logistic regression**, for example for n-ary text classification, part-of-speech labeling, etc.).  
* Multinomial logistic regression uses the softmax function to compute probabilities  
* The weights (vector w and bias b) are learned from a labeled training set via a loss function, such as the cross-entropy loss, that must be minimized.  
* Minimizing this loss function is a convex optimization problem, and iterative algorithms like gradient descent are used to find the optimal weights.  
* **period disambiguation:** deciding if a period period disambiguation is the end of a sentence or part of a word, by classifying each period into one of two classes, end-of-sentence and not-EOS.  
* **Regularization** is used to avoid overfitting.

# 

# Rob Voigt Slides

### Linguistic Structure, NLP “Tasks”, and Annotation

**Traditional Levels of Structure**  
**Phonetics** \= sounds 

* The physical production and perception of speech sounds   
* Unit of analysis: speech sound  
* *NLP Tasks:* Speech synthesis, Automated transcription

**Phonology** \= ordering of sounds 

* The systematic organization of speech sounds   
* Unit of analysis: phoneme

**Morphology** \= words and word parts 

* The structure and constituent parts of words   
* Unit of analysis: morpheme  
* *NLP Tasks:* \*Morphological Segmentation\*, Lemmatization and Inflection

**Syntax** \= ordering of words 

* The systematicity of word orderings  
* *NLP Tasks:* Syntactic Parsing, Downstream applications (e.g. Machine Translation or Semantic Similarity)

**Semantics** \= propositional meaning 

* The propositional (e.g., literal) meanings of words and larger units (frequently sentences)  
* *NLP Tasks:* Recognizing Textual Entailment, Semantic Parsing

**Pragmatics** \= non-propositional meaning

* The beyond-propositional meanings of words and larger units

		**More levels of Structure..**  
**Reference** \= pointing out things with words 

* What entity in the world does a linguistic expression point out? Includes pronouns, honorifics, naming and nicknaming  
* **NLP Tasks**: Coreference Resolution, Named Entity Recognition

**Prosody** \= suprasegmental sounds like pitch  
**Discourse** \= sequences between large units 

* The relations between clauses and propositions  
* NLP Tasks: Discourse Parsing, Argumentation Mining

**Social Meaning** \= social implicature of variation

* Many sorts of complex socially enmeshed meaning-making:   
  * Sentiment and stance   
  * Regional variation   
  * Identity performance   
  * Memes and spread of ideas

* Linguistic categories are purely abstract human creations\! There is no ground truth.  
  * So we usually evaluate with Inter-Annotator Agreement  
  * Have some proportion of the data annotated by multiple people  
  * Obtain a measurement of consistency \- how often do people make the same judgment?

(Inner-Annotator Agreement: Common \- Cohen’s Kapps)   
(po=probability of observed agreement, pe= probability of expected agreement)  
k=po-pe1-pe

* What counts as good? Different opinions, task dependent. .75-.8 seems fair

### Linguistic Structure, NLP “Tasks”, and Annotation

* Words seem to vary along 3 affective dimensions; so the connotation of a word is a vector in 3-space   
  * valence: the pleasantness of the stimulus   
  * arousal: the intensity of emotion provoked by the stimulus   
  * dominance: the degree of control exerted by the stimulus  
* Every modern NLP algorithm uses embeddings as the representation of word meaning  
  * embeddings make it so you can generalize to other words positionally instead of needing exact same word in training or test  
* tf-idf Embeddings  
  * Information Retrieval workhorse\!   
  * A common baseline model   
  * Sparse vectors   
  * Words are represented by (a simple function of) the counts of nearby words  
* Word2vec Embeddings   
* Dense vectors   
* Representation is created by training a classifier to predict whether a word is likely to appear nearby

# Chapter 6: Neural Networks

* **Feedforward Network:** a network where computation proceeds iteratively from one layer of units to the next
