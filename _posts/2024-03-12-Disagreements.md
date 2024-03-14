---
title: 'Are Explainers Just Another Black-box?'
date: 2023-03-12
permalink: /posts/2023/03/Another-Black-Box/
tags:
  - Post-hoc Explainers
  - Black Boxes
  - Disagreement Problem
---

In this first blog post, we will discuss post-hoc explanation methods and whether
they are just another black-box on top of the Machine Learning model.

## Machine Learning Models

Machine Learning aims at *teaching* computers how to solve complex problems. This methodology is useful
for solving tasks where it is hard to write a conventional program. For example, lets say you are working
for a bike rental company and you want to predict the number of bike rentals at a certain hour given information about
the weather and the day of the week. A classical program might look like this

```python
if temperature=cold or time=late or time=early:
  return few-bike-rentals
else if temperature=hot_but_not_too_hot:
    return many-bike-rentals
else:
  return medium-bike-rentals
```

This program has several issues. First, it is not clear what ``hot`` and ``cold`` temperature actually mean numerically.
What thresholds should be used? Same thing with `time=late` and `time=early`. The solution proposed with Machine Learning is to
*learn* those thresholds (and the program itself) automatically based on historical data of bike rentals given various hours
and temperatures. In this new paradigm, you would need access to data comprised of

1. A $(N, d)$ feature matrix **X** representing the $d$ characteristics (time, temperature, day of the week etc.)
of $N$ instances.
2. A $(N,)$ vector **y** containing the actual number of bike rentals for the given instance.

Here is an example of such a dataset available in the [PyFD](https://github.com/gablabc/PyFD) Python library.

```python
import numpy as np
import matplotlib.pyplot as plt
from pyfd.data import get_data_bike

# load the data
X, y, features = get_data_bike()
print(X.shape)
>>> (17379, 10)
print(y.shape)
>>> (17379,)
print(features.print_names())
>>> ['yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
```

We have 17K data instances each comprised of $d=10$ time/weather features.
Given this collection of data, one then fits a Machine Learning model $h$ in order to get accurate
predictions of $y$ given $x$. In this blog, we will investigate three regressors:

1. Random Forests
2. Gradient Boosted Trees
3. Neural Networks.

To assert if these models are good or not, we have to keep aside some portion of the data (here 20%)
and used it afterward to compute an unbiased measure of performance. The $R^2$ measure of performance is
used

$$R^2(h) = 1-\frac{\sum_{i\in \text{Test}}(\,h(x^{(i)})-y^{(i)}\,)^2}{\sum_{i\in \text{Test}}(\,\bar{y}-y^{(i)}\,)^2},$$

where $\bar{y}$ is the average value of the target on the test set. Simply put, the $R^2$ represents how better the
model $h$ is at predicting $y$ than the *dummy* predictor $h_{\text{dummy}}(x) = \bar{y}$.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the models
model_1 = Pipeline([('preprocessor', StandardScaler()), 
                    ('predictor', MLPRegressor(random_state=0, 
                                               max_iter=1000,
                                               learning_rate_init=0.01))]).fit(X_train, y_train)
model_2 = RandomForestRegressor(random_state=0, max_depth=10).fit(X_train, y_train)
model_3 = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)

# Evaluate the models
print(f"R^2 of model_1 : {model_1.score(X_test, y_test):.3f}")
print(f"R^2 of model_2 : {model_2.score(X_test, y_test):.3f}")
print(f"R^2 of model_3 : {model_3.score(X_test, y_test):.3f}")

>>>
R^2 of model_1 : 0.908
R^2 of model_2 : 0.919
R^2 of model_3 : 0.943
```

Note that the three models have acceptable performance on the test set and hence these programs are
able to accurately predict the number of bike rentals.

## Post-hoc explanations

Let's be frank, do you think that the three models presented earlier are **trustworthy**?
Would you deploy them if you were working at a bike rental company? Personally,
the current results are not enough to convince me. Indeed, we have reported some encouraging indices
of performance, yet we do not know how the program works internally. What if there is a bug in the model?
What if the model does not work well in deployment? We cannot answer those questions with certainty because
our models are **black-boxes**: they take an input $x$ and return an output $h(x)$, but we do not understand the
mechanisms involved.

![bb](/images/blog-bb/bb.png)

Our lack of understanding of $h$ is the main motivation behind eXplainable Artificial Intelligence (XAI), a research
initiative to shed light on the decision-making of ML models. This research field has recently introduced many techniques

1. Local Feature Attributions $\phi_i(h, x)\,\forall i=1,2,\ldots,d$ are vectors that attribute to each feature an
importance toward the prediction $h(x)$.
2. Global Feature Importance $\Phi_i(h)\,\forall i=1,2,\ldots,d$ are positive vectors that illustrates how much the
feature is used overall by the model.

We will focus on Global Feature Importance as they are the simplest to understand. We will employ three techniques
[Partial Dependence Plots (PDP)](https://scikit-learn.org/stable/modules/partial_dependence.html),
[Permutation Feature Importance (PFI)](https://scikit-learn.org/stable/modules/permutation_importance.html),
[SHAP](https://github.com/shap/shap). Here is how you would compute the global feature importance of the Gradient Boosted Trees
using SHAP

```python
from pyfd.shapley import interventional_treeshap
from pyfd.plots import bar

shapley_values = interventional_treeshap(model_3, X_test, X_test, algorithm="leaf")
global_shapley_importance = np.mean(np.abs(shapley_values), axis=0)
bar(global_shapley_importance, features.print_names())
plt.xlabel("Global Feature Importance")
plt.show()
```

![bb](/images/blog-bb/importance.png)

This bar chart shows the global importance of the ten features in our dataset. We note that the Gradient Boosted Trees
relies more on the feature `hr` than any other features. This is an example of insight that post-hoc explanations can give you
on your model.

## Another Black-box?

Ok, now do you trust that the Gradient Boosted Trees is a good program? **I still don't!** The reason is that the explanation methods are
themselves very complicated and hard to understand. I would even argue that they are another black box on top of the old
one.

![bb](/images/blog-bb/bb_2.png)

Even worst, it is very hard to judge the quality of an explanation. This makes explanations evaluation even harder than
model evaluation. Indeed, although our models are black-boxes, we can still evaluate their test set performance to get some
idea of which model is better/worst that the others. We do not have access to such metrics in explainability : if I compute a
PDP, SHAP, and PFI feature importance, it is not clear which one is best.

![bb](/images/blog-bb/importance_3.png)

This figure presents the feature importance yielded by three different explainers (PFI, SHAP, PDP) in different opacities
(opaque, semi-transparent, transparent). We see that explainers do not agree on the importance of `workingday` : PFI says its
important while PDP says it is not. Which one is right? This is an important question to address because if we cannot trust
the post-hoc explainers, we cannot trust the model in the first place and we are back to square one.

## Ouverture

The simple example of explanation disagreements I presented occurs in a variety of ML use-cases and models
[(Krishna et al., 2022)](https://arxiv.org/abs/2202.01602). This issue is called the Disagreement Problem and it
highlights the need to **quantify** trust in explainability methods. Otherwise, when explanations provide contradictory interpretations, practitioners cannot be expected to pick the right one. This is the main motivation behind my PhD research :
*how do we quantify trust in post-hoc explanation methods?* I will not have the time to present my ideas/solutions in this
blog post, but future posts will discuss them. Stay tuned!
