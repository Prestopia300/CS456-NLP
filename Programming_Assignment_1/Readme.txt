Unigram perplexity= infinity
Bigram perplexity = infinity
smoothed bigram model = 2.552947671716129 x 10^61


o Which model performed worst and why might you have expected that model to have performed worst?
    Both Unigram and Bigram tied for worse, because they both are infinity. This is becuase atleast one sentence in each's probability = 0


o Did smoothing help or hurt the model’s ‘performance’ when evaluated on this corpus? Why might that be?
    Smoothing helped the models performance, because it results in a perplexity that isn't infinity. If smoothing is not used, then it will likely end up being infinity.