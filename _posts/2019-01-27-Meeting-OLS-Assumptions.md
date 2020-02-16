---
layout: post
title: Meeting the Assumptions of OLS Regression
---

*This article was originally published on Medium [here](https://towardsdatascience.com/what-to-do-when-your-data-fails-ols-regression-assumptions-916272367f66).*


Regression analysis falls in the realm of inferential statistics. Consider the following equation:

**y** ≈ β0 + β1**x** + e

The approximate equals sign indicates that there is an approximate linear relationship between **x** and **y**. The error *e* term indicates that this model isn't going to fully reflect reality via a simple linear relation. The machine learning task is to estimate the beta parameters as follows:

**ŷ** = β̂0 + β̂1**x**

Note with there approximations of the betas it's now possible to compute predictions for previously unseen values of the dependent variable.

Unpacking this a little bit in ML terms, you would: 

1. Take some data set with a feature vector **x** and a target vector **y** 
2. Split the data set into train/test sections randomly
3. Train the model and find estimates (β̂0, β̂1) of the true beta intercept and slope
4. See how your model generalizes by using your trained beta parameters to predict values of ŷ on the held-out test data
5. Compute residual errors between **y** and **ŷ** and quantify how good/bad you did with something like Mean Absolute Error or Root Mean Squared Error.

## Ordinary Least Squares

This above model is a very simple example, so instead consider the more realistic multiple linear regression case where the goal is to find beta parameters as follows:

**ŷ** = β̂0 + β̂1**x1** + β̂2**x2** + ... + β̂p**xp**

How does the model figure out what β̂ parameters to use as estimates? Ordinary Least Squares is a method where the solution finds all the β̂ coefficients which minimize the sum of squares of the residuals, i.e. minimizing the sum of these differences: (**y** - **ŷ**)^2, for all values of **y** and **ŷ** in the training observations. Think of **y** and **ŷ** as column vectors with entries equal to the number of your total observations.

The fascinating bit is that OLS provides the **best linear unbiased estimator (BLUE)** of **y** under a set of classical assumptions. That's a bit of a mouthful, but note that:

* “best” = minimal variance of the OLS estimation of the true betas (i.e. *no other linear estimator* has less variance!)
* “unbiased” = expected value of the estimated beta-hats equal the true beta values

The proof of this is due to the heavyweight Gauss-Markov theorem, which is far beyond the scope of this post. However, it's clearly beneficial to meet these assumptions and obtain a 'BLUE' estimator, so without further ado here are the assumptions:

1. Regression is linear in the β parameters 
2. Residuals should be normally distributed with 0 mean
3. Residuals must have constant variance
4. Errors are uncorrelated across observations
5. No independent variable is a perfect linear function of any other independent variable

## When things go awry

In real life your data is unlikely to perfectly meet these assumptions. In this section I'll show an example where my base data set blatantly violates assumption #2 and #3 above, and explicitly what I did to fix it.

The example data came from funding levels of online-crowdsourced projects, and a variety of features such as campaign length, description text sentiment, number of photos and many more. By simply running OLS on the features and target here's what the residuals looked like:
![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 12.31.43 PM.png)The red line indicates perfect normality, and clearly the residuals are not normally distributed in violation of assumption #2. 

Next, here's a plot to check if the residuals are spread evenly across the range of predictors (assumption #3 for equal variance):

![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 12.33.01 PM.png)
Clearly the residual errors are not spread evenly across the range of predictors, so we have issues here as well.

## Data Transformation

Here's a pair plot of my untransformed data set with a few select problem features:
![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.35.13 PM.png)In this case pledged was my dependent **y**, and the number of gift options and photo count were two selected features. While not a guarantee, it's sometime the case that transforming features or the target to a more normal distribution can help with the problematic OLS assumptions mentioned above.

In this case, the pledged amount **y** is a classic example begging for a transformation to log space. Its individual y values take on anything from $2 to $80000. In my case 'pledged' was in a Pandas dataframe, so I quickly converted the entire column via numpy's log function:

```python
df['y_log'] = np.log(df['pledged'])
plt.ylabel('Count')
plt.xlabel('LOG of Pledged Amount')
plt.title('Dist. of Transformed Pledged Amount - Dependent Target')
plt.hist(df['y_log'])
```

This resulted in the following transformation:

![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.24.19 PM.png)One quick aside is that when you transform **y** to log space, you'll implicitly end up interpreting unit changes in X as having a percentage change interpretation in the original non-log y at the end. The answer in [this Stack Overflow question](https://stats.stackexchange.com/questions/16747/interpreting-percentage-units-regression) has a very clear explanation of why this is the case by using the property of the natural log's derivative.

Now on to the features. I've found the [Box-Cox transformation](https://en.wikipedia.org/wiki/Power_transform) to help immensely with regard to fixing residual normality for feature distributions with strange shapes. If you look at the center box in the pair plot above, you'll see the un-transformed distribution of the number of gift options. It's hard to even draw a parallel with a standard probability distribution, so here's how to run a Box-Cox transformation using scipy.stats:

```python
lamb=stats.boxcox_normmax(df.num_gift_options, brack=(-1.9, 1.9))
print("Lambda:", lamb)
num_gift_options_t =(np.power(df.num_gift_options,lamb)-1)/lamb
df['num_gift_options_t'] = (np.power(df.num_gift_options,lamb)-1)/lamb
```

Note here that the [stats.boxcox_normax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) function from scipy.stats will find the best lambda to use in the power transformation.

Here's how it looks post-transformation:

![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.31.49 PM.png)If the feature in question has zero or negative values, neither the log transform or the box-cox will work. Thankfully, the [Yeo-Johnson power transformation](https://www.stat.umn.edu/arc/yjpower.pdf) solves for this case explicitly. Conveniently, the Yeo-Johnson method is the default case in [sklearn.preprocessing's](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) PowerTransformer:

```python
pt = PowerTransformer()
pt.fit(df['photo_cnt'].values.reshape(-1,1))
df['photo_cnt_t'] = pt.transform(df['photo_cnt'].values.reshape(-1,1))

plt.ylabel('Count')
plt.xlabel('Box Cox Transformed Negative Sentiment')
plt.title('Dist. of Transformed Negative Sentiment - Feature')
plt.hist(df['all_sentiment_neg_t'])
```

Here's what that looks like post-transformation:

![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.37.04 PM.png)While the transformed features are by no means normally distributed themselves, look at what we get for our residual distribution and variance plots post-transformation:

![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.39.00 PM.png)![Image]({{ site.url }}/images/Screen Shot 2019-01-27 at 1.39.34 PM.png)This is night and day from where we started, and we can now say that we have essentially normally distributed residuals and constant variance among the residuals! Hence the OLS assumptions are met and we can proceed with modeling.

## Model Testing and Interpretation

This is by no means the end point of the analysis. In this specific case, I ended up running a 3-fold Cross-Validation testing out Linear Regression, Ridge Regression, and Huber Regression on a validation split of my training data, and then finally testing the winner on the held-out test data to see if the model generalized. The overall point is that it's best to make sure you have met the OLS assumptions before going into a full train/validation/test loop on a number of models for the regression case.

One note is that when you transform a feature, you lose the ability to interpret the coefficients effect on y at the end. For example, I did not transform the project length feature in this analysis, and at the end I was able to say that a unit increase (+1 day) in project length led to an 11% decrease in funding amount.

Since I used these transformations on the photo count and number of gift options features, I can't make the same assertion given a unit increase in X, as the coefficient predictions are relative to the transformation. Thus transformations do have a downside, but it's worth it to know you're get a BLUE estimator via OLS.
