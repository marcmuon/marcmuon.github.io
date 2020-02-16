---
layout: post
title: Using LDA Topic Models as a Classification Model Input
---

*This article was originally published on Medium [here](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28).*

Topic Modeling Overview
=======================

Topic Modeling in NLP seeks to find hidden semantic structure in documents. They are probabilistic models that can help you comb through massive amounts of raw text and cluster similar groups of documents together in an unsupervised way.

This post specifically focuses on Latent Dirichlet Allocation (LDA), which was a technique proposed in 2000 for population genetics and re-discovered independently by ML-hero Andrew Ng et al. in 2003. LDA states that each document in a corpus is a combination of a fixed number of topics. A topic has a probability of generating various words, where the words are all the observed words in the corpus. These ‘hidden’ topics are then surfaced based on the likelihood of word co-occurrence. Formally, this is Bayesian Inference problem \[1\].

LDA Output
==========

Once LDA topic modeling is applied to set of documents, you‘re able to see the words that make up each hidden topic. In my case, I took 100,000 reviews from Yelp Restaurants in 2016 using the Yelp dataset \[2\]. Here are two examples of topics discovered via LDA:

![Image]({{site.url}}/images/lda_img1.png)

You can see the first topic group seems to have identified word co-occurrences for negative burger reviews, and the second topic group seems to have identified positive Italian restaurant experiences. The third topic isn’t as clear-cut, but generally seems to touch on terrible, dry salty food.

Converting Unsupervised Output to a Supervised Problem
======================================================

I was more interested to see if this hidden semantic structure (generated unsupervised) could be converted to be used in a supervised classification problem. Assume for a minute that I had only trained a LDA model to find 3 topics as above. After training, I could then take all 100,000 reviews and see the _distribution of topics_ for every review. In other words, some documents might be 100% Topic 1, others might be 33%/33%/33% of Topic 1/2/3, etc. That output is just a vector for every review showing the distribution. The idea here is to test whether the distribution per review of hidden semantic information could predict positive and negative sentiment.

Project Goal
============

With that intro out of the way, here was my goal:

![Image]({{site.url}}/images/lda_img2.png)

Specifically:

1.  Train LDA Model on 100,000 Restaurant Reviews from 2016
2.  Grab Topic distributions for every review using the LDA Model
3.  Use Topic Distributions directly as feature vectors in supervised classification models (Logistic Regression, SVC, etc) and get F1-score.
4.  Use the same 2016 LDA model to get topic distributions from 2017 (**the LDA model did not see this data!**)
5.  Run supervised classification models again on the 2017 vectors and see if this generalizes.

> If the supervised F1-scores on the unseen data generalizes, then we can posit that the 2016 topic model has identified latent semantic structure that persists over time in this restaurant review domain.

Data Prep
=========

UPDATE (9/23/19): I’ve added a README to [the Repo](https://github.com/marcmuon/nlp_yelp_review_unsupervised) which shows how to create a MongoDB using the source data. I’ve also included a [preprocessing script](https://github.com/marcmuon/nlp_yelp_review_unsupervised/blob/master/preprocess.py) which will allow you to create the exact training and test DataFrames I use below. However, I realize that might be a lot of work, so I also included pickle files of my Train and Test DataFrames in the directory [here](https://github.com/marcmuon/nlp_yelp_review_unsupervised/tree/master/data). This will allow you to follow along with the notebooks in the repo directly, namely, [here](https://github.com/marcmuon/nlp_yelp_review_unsupervised/blob/master/notebooks/2-train_corpus_prep_and_LDA_train.ipynb) and then [here](https://github.com/marcmuon/nlp_yelp_review_unsupervised/blob/master/notebooks/3-test_corpus_prep_and_apply_LDA_get_vectors.ipynb). **If you’d rather just get the highlights/takeaways, I point out all the key bits in the rest of this blog post below with code snippets.**

LDA Pre-Processing
==================

I used the truly wonderful gensim library to create bi-gram representations of the reviews and to run LDA. Gensim’s LDA implementation needs reviews as a sparse vector. Conveniently, gensim also provides convenience utilities to convert NumPy dense matrices or scipy sparse matrices into the required form.

I’ll show how I got to the requisite representation using gensim functions. I started with a pandas DataFrame containing the text of every review in a column named`'text’`, which can be extracted to a list of list of strings, where each list represents a review. This is the object named`words` in my example below:

```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['come','order','try','go','get','make','drink','plate','dish','restaurant','place',
                  'would','really','like','great','service','came','got'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
    
def get_corpus(df):
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram

train_corpus, train_id2word, bigram_train = get_corpus(rev_train)
```

I’m omitting showing a few steps of additional pre-processing (punctuation, stripping newlines, etc) for brevity in this post.

There are really 2 key items in this code block:

1.  Gensim’s Phrases class allows you to group related phrases into one token for LDA. For example, note that in the list of found topics toward the start of this post that _ice\_cream_ was listed as a single token. Thus the output of this line `bigram = [bigram_mod[review] for review in words]` is a list of lists where each list represents a review and the strings in each list are a mix of unigrams and bigrams. This is the case since the what we’ve done is apply the `bigram_mod` phrase modeling model to every review.
2.  Once you have that list of lists of unigrams and bigrams, you can pass it to gensim’s Dictionary class. This will output the word frequency count of each word _for each review_. I found that I had the best results with LDA when I additionally did some processing to remove the most common and the rarest words in the corpus as shown in line 21 of the above code block. Finally here’s what `doc2bow()`  is doing, from their official [examples](https://radimrehurek.com/gensim/tut1.html) \[3\]:

> “The function `_doc2bow()_` simply counts the number of occurrences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector. The sparse vector `_[(0, 1), (1, 1)]_` therefore reads: in the document “Human computer interaction”, the words computer (id 0) and human (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.”

Line 23 above then gives us the corpus in the representation needed for LDA.

To give an example of the type of text we’re dealing with, here’s a snapshot of a Yelp review:

![Image]({{site.url}}/images/lda_img3.png)


Choosing the Number of Topics for LDA
=====================================

In order to train a LDA model you need to provide a fixed assume number of topics across your corpus. There are a number of ways you could approach this:

1.  Run LDA on your corpus with different numbers of topics and see if word distribution per topic looks sensible.
2.  Examine the coherence scores of your LDA model, and effectively grid search to choose the highest coherence \[4\].
3.  Create a handful of LDA models with different topic values, then see how these perform in the supervised classification model training. This is specific to my goals here, since my ultimate aim is to see if the topic distributions have predictive value.

Of these: I don’t trust #1 as a method at all. Who am I to say what’s sensible or not in this context? I’m relying on LDA to identify latent topic representations of 100,000 documents, and it’s possible that it won’t necessarily be intuitive. For #2: I talked to some former NLP professionals and they dissuaded me from relying on coherence scores based on their experience in industry. Approach #3 would be reasonable for my purpose but there’s a reality that LDA takes a non-trivial of time to train even though I was using a 16GB 8-Core AWS instance.

Thus I came up with an idea that I think is fairly novel — at the very least, I haven’t seen anyone do this online or in papers:

Gensim also provides a Hierarchical Dirichlet Process (HDP) class \[5\]. HDP is similar to LDA, except it seeks to learn the correct number of topics from the data; that is, you don’t need to provide a fixed number of topics. I figured I would run HDP on my 100,000 reviews a few times and see the number of topics it was learning. In my case this was always 20 topics, so I went with that.

To get an intuitive feel for HDP: I found a number of sources online that said it’s most similar to a Chinese Restaurant Process. This is explained brilliantly by Edwin Chen [here](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/) \[6\], and beautifully visualized in \[7\] [here](http://gerin.perso.math.cnrs.fr/ChineseRestaurant.html). Here’s the visualization from \[7\]:

![Image]({{site.url}}/images/lda_img4.jpeg)

In this example, we need to assign 8 to a topic. There’s a 3/8 probability 8 will land in topic C1, a 4/8 probability 8 will land in topic C2, and a 1/8 probability a new topic C3 will be created. In this way a number of topics is discovered. So the bigger a cluster is, the more likely it is for someone to join that cluster. I felt this was just as reasonable as any other method to choose a fixed topic number for LDA. If anyone with a heavier Bayesian Inference background has thoughts on this please weigh in!

Creating a LDA Model
====================

Here’s the code to run LDA with Gensim:

```python
import gensim
import logging # This allows for seeing if the model converges. A log file is created.
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lda_train = gensim.models.ldamulticore.LdaMulticore(
                           corpus=train_corpus,
                           num_topics=20,
                           id2word=train_id2word,
                           chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)
    lda_train.save('lda_train.model')
```

By turning on the `eval_every` flag we’re able to process the corpus in chunks: in my case chunks of 100 documents worked fairly well for convergence. The number of passes is separate passes over the entire corpus.

Once that’s done, you can view the words making up each topic as follows:

```python
lda_train.print_topics(20,num_words=15)[:10]
```

With that code you’ll see 10 of the 20 topics and the 15 top words for each.

Converting Topics to Feature Vectors
====================================

Now comes the interesting bit. We’re going to use the LDA model to grab the distribution of these 20 topics for every review. This 20-vector will be our feature vector for supervised classification, with the supervised learning goal being to determine positive or negative sentiment.

Note that I think this approach for supervised classification using topic model vectors is not very common. When I did it I wasn’t aware of any example online of people trying this, though later on when I was done I discovered this [paper](http://gibbslda.sourceforge.net/fp224-phan.pdf) where it was done in 2008 \[8\]. Please let me know if there are other examples out there!

> The ultimate goal is not only to see how this performs in a train/test CV split of the current data, but whether the topics have hit on something fundamental that translates to unseen test data in the future (in my case, data from a year later).

Here’s what I did to grab the feature vectors for every review:

```python
train_vecs = []
for i in range(len(rev_train)):
    top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    topic_vec.extend([rev_train.iloc[i].real_counts]) # counts of reviews for restaurant
    topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
    train_vecs.append(topic_vec)
```

The key bit is using `minimum_probability=0.0` in line 3. This ensures that we’ll capture the instances where a review is presented with 0% in some topics, and the representation for each review will add up to 100%.

Lines 5 and 6 are two hand-engineered features I added.

Thus a single observation for a review for supervised classification now looks like this:

![Image]({{site.url}}/images/lda_img5.png)

The first 20 items represent the distribution for the 20 found topics for each review.

Training a Supervised Classifier
================================

We’re now ready to train! Here I’m using 100,000 2016 restaurant reviews and their topic-model distribution feature vector + two hand-engineered features:

```python
X = np.array(train_vecs)
y = np.array(rev_train.target)

kf = KFold(5, shuffle=True, random_state=42)
cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1,  = [], [], []

for train_ind, val_ind in kf.split(X, y):
    # Assign CV IDX
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind]
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_val_scale = scaler.transform(X_val)

    # Logisitic Regression
    lr = LogisticRegression(
        class_weight= 'balanced',
        solver='newton-cg',
        fit_intercept=True
    ).fit(X_train_scale, y_train)

    y_pred = lr.predict(X_val_scale)
    cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))
    
    # Logistic Regression SGD
    sgd = linear_model.SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        loss='log',
        class_weight='balanced'
    ).fit(X_train_scale, y_train)
    
    y_pred = sgd.predict(X_val_scale)
    cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))
    
    # SGD Modified Huber
    sgd_huber = linear_model.SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        alpha=20,
        loss='modified_huber',
        class_weight='balanced'
    ).fit(X_train_scale, y_train)
    
    y_pred = sgd_huber.predict(X_val_scale)
    cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')
```

A couple notes on this:

1.  I landed on a comparison between standard Logistic Regression, Stochastic Gradient Descent with Log Loss, and Stochastic Gradient Descent with Modified Huber loss.
2.  I’m running a 5-fold CV, so that in each run 1/5 of the reviews are held-out as validation data, and the other 4/5 are training data. This is repeated for every fold, and the f1 score results are averaged at the end.
3.  My classes are imbalanced. In particular, there are a disproportionate amount of 4 and 5 star reviews in the Yelp review dataset. The `class_weight='balanced'` line in the models approximates undersampling to correct for this. See \[9\] for a justification of this choice.
4.  I’ve also restricted the analysis to just restaurants that had above the 25th percentile in total reviews in the dataset. This was more of a restriction to speed things up since the initial dataset was something like 4 million+ reviews.

2016 Training Results
=====================

Here are the f1-score results:

![Image]({{site.url}}/images/lda_img6.png)

I’ll first walk-through the lower .53 and .62 f1-scores using Logisitic Regression. When I initially started training I tried to predict the individual review ratings: 1,2,3,4 or 5 stars. As you can see this was not successful. I was a bit discouraged with the initial .53 score, so went back to examine my initial EDA charts to see if I could notice anything in the data. I had run this chart earlier on:

![Image]({{site.url}}/images/lda_img7.png)

This shows the word count IQR range by rating. Since the main IQR range was pretty compact, I decided to try re-running the LDA pre-processing and model restricted to just (roughly) the IQR range. Making this change increased my Logistic Regression score on f1 to .62 for 1,2,3,4,5 star classification. Still not great.

At this point I decided to see what would happen if I got rid of the 3-stars, and grouped the 1,2 stars as ‘bad’ and 4,5 stars as ‘good’ sentiment scores. As you’ll see in the graph above, this worked wonder! Now Logistic Regression yieled a .869 f1-score.

Modified Huber Loss
===================

When I was running these, I noticed a ‘modified huber’ loss option in SKLearn’s Stochastic Gradient Descent implementation \[10\]. In this case, the penalty for being wrong is much worse than either Hinge (SVC) or Log Loss:

![Image]({{site.url}}/images/lda_img8.png)

I’m still grappling with why this worked so well, but my initial thought is that these punishing penalties caused SGD (remember, 1 by 1 weight updates) to learn quickly. Regularization helped immensely here as well. The alpha in line 42 in the code above is a regularization parameter (think like in Ridge or Lasso regularization), and this helped in getting my f1-score up to .936.

Applying the Model on Unseen Data
=================================

At this point I was pretty thrilled by these results, but wanted to further see what would happen on completely unseen data.

Specifically:

1.  Take the LDA Model from the 2016 reviews, and grab feature vectors on test data. It’s important to note that the same 2016 model can be used to do this!
2.  Re-run the models on the test-vectors.

All that’s required is making bigrams for the test corpus, then throwing this into the test-vector extraction approach as before:
```python
def get_bigram(df):
    """
    For the test data we only need the bigram data built on 2017 reviews,
    as we'll use the 2016 id2word mappings. This is a requirement due to 
    the shapes Gensim functions expect in the test-vector transformation below.
    With both these in hand, we can make the test corpus.
    """
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram = bigrams(words)
    bigram = [bigram[review] for review in words]
    return bigram
  
bigram_test = get_bigram(rev_test)

test_corpus = [train_id2word.doc2bow(text) for text in bigram_test]

test_vecs = []
for i in range(len(rev_test)):
    top_topics = lda_train.get_document_topics(test_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    topic_vec.extend([rev_test.iloc[i].real_counts]) # counts of reviews for restaurant
    topic_vec.extend([len(rev_test.iloc[i].text)]) # length review
    test_vecs.append(topic_vec)
```


Finally, the results:

![Image]({{site.url}}/images/lda_img9.png)

Somewhat shockingly to me, this generalizes!

I was thrilled with this result, as I believe this approach could work generally for any company trying to train a classifier in this way. I also did a hypothesis test at the end using mlxtend \[11\] and the final result is indeed statistically significant.

Future Work
===========

I intend on extending this a bit in the future, and I’ll leave you with that:

![Image]({{site.url}}/images/lda_img10.png)

I’ve also hosted all the code for this and the trained LDA model on my GitHub [here](https://github.com/marcmuon/nlp_yelp_review_unsupervised).

Thanks for reading!

Sources
=======

\[1\] [https://en.wikipedia.org/wiki/Latent\_Dirichlet\_allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)  
\[2\] [https://www.yelp.com/dataset](https://www.yelp.com/dataset)  
\[3\] [https://radimrehurek.com/gensim/tut1.html](https://radimrehurek.com/gensim/tut1.html)  
\[4\] [https://radimrehurek.com/gensim/models/coherencemodel.html](https://radimrehurek.com/gensim/models/coherencemodel.html)  
\[5\] [https://radimrehurek.com/gensim/models/hdpmodel.html](https://radimrehurek.com/gensim/models/hdpmodel.html)  
\[6\] [http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/)  
\[7\] [http://gerin.perso.math.cnrs.fr/ChineseRestaurant.html](http://gerin.perso.math.cnrs.fr/ChineseRestaurant.html)  
\[8\] [http://gibbslda.sourceforge.net/fp224-phan.pdf](http://gibbslda.sourceforge.net/fp224-phan.pdf)  
\[9\] [http://blog.madhukaraphatak.com/class-imbalance-part-2/](http://blog.madhukaraphatak.com/class-imbalance-part-2/)  
\[10\] [https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.SGDClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
\[11\] [http://rasbt.github.io/mlxtend/user\_guide/evaluate/mcnemar/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/)
