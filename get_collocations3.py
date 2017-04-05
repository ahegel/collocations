import string
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.corpus import stopwords


# find collocations for each word
def get_collocations(corpus, windowsize=10, numresults=10):
    '''This function uses the Natural Language Toolkit to find the top collocations in a corpus.
    It takes as an argument a string that contains the corpus you want to
    find collocations from. It prints the top collocations it finds.
    '''
    # convert the corpus (a string) into  a list of words
    tokens = word_tokenize(corpus)
    # initialize the bigram association measures object to score each collocation
    bigram_measures = BigramAssocMeasures()
    # initialize the bigram collocation finder object to find and rank collocations
    finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
    # apply a series of filters to narrow down the collocation results
    ignored_words = stopwords.words('english')
    finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in ignored_words)
    finder.apply_freq_filter(1)
    # calculate the top results by T-score
    # list of all possible measures: .raw_freq, .pmi, .likelihood_ratio, .chi_sq, .phi_sq, .fisher, .student_t, .mi_like, .poisson_stirling, .jaccard, .dice
    results = finder.nbest(bigram_measures.student_t, numresults)
    # print the results
    print("Top ", str(numresults), " collocations:")
    for k, v in results:
        print(str(k), ", ", str(v))


def get_keyword_collocations(corpus, keyword, windowsize=10, numresults=10):
    '''This function uses the Natural Language Toolkit to find collocations
    for a specific keyword in a corpus. It takes as an argument a string that
    contains the corpus you want to find collocations from. It prints the top
    collocations it finds for each keyword.
    '''
    # convert the corpus (a string) into  a list of words
    tokens = word_tokenize(corpus)
    # initialize the bigram association measures object to score each collocation
    bigram_measures = BigramAssocMeasures()
    # initialize the bigram collocation finder object to find and rank collocations
    finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
    # initialize a function that will narrow down collocates that don't contain the keyword
    keyword_filter = lambda *w: keyword not in w
    # apply a series of filters to narrow down the collocation results
    ignored_words = stopwords.words('english')
    finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in ignored_words)
    finder.apply_freq_filter(1)
    finder.apply_ngram_filter(keyword_filter)
    # calculate the top results by T-score
    # list of all possible measures: .raw_freq, .pmi, .likelihood_ratio, .chi_sq, .phi_sq, .fisher, .student_t, .mi_like, .poisson_stirling, .jaccard, .dice
    results = finder.nbest(bigram_measures.student_t, numresults)
    # print the results
    print("Top collocations for ", str(keyword), ":")
    collocations = ''
    for k, v in results:
        if k != keyword:
            collocations += k + ' '
        else:
            collocations += v + ' '
    print(collocations, '\n')


# Replace this with your filename
infile = "sample_corpus.txt"

# Read in the corpus you want to find collocations from
with open(infile) as tmpfile:  
    data = tmpfile.read()

# Clean the data
data = data.translate(None, string.punctuation)  # remove punctuation
data = "".join(i for i in data if ord(i) < 128)  # remove non-ascii characters

# Get the top collocations for the entire corpus
get_collocations(data)
print(' ')

# Replace this with a list of keywords you want to find collocations for
words_of_interest = ["love", "death"]

# Get the top collocations for each keyword in the list above
for word in words_of_interest:
    get_keyword_collocations(data, word)
