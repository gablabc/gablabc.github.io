---
permalink: /
title: "Research Interests"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

### Research Questions

I am Machine Learning enthusiast who is looking to merge complex mathematical theory with real-world applications.
My current research interests are in the field of eXplainable Artificial Intelligence (XAI), where the goal is
to make complex black-box models more interpretable. Indeed, although complex models such as Random Forests or
Neural Network attain high performance, they are not easily understood by humans. Depending on the task, not being
able to explain predictions can be a considerable roadblock toward model acceptance and deployment. For instance, think of
safety-critical domains such as Aerospace, or applications where human beings are impacted by decisions:
Medicine, Banking, and Insurance.

Various post-hoc techniques have been proposed to get insight into model behavior, notably
[Partial Dependence Plots (PDP)](https://scikit-learn.org/stable/modules/partial_dependence.html),
[Permutation Feature Importance (PFI)](https://scikit-learn.org/stable/modules/permutation_importance.html),
[SHAP](https://github.com/shap/shap) and [Expected Gradients](https://www.nature.com/articles/s42256-021-00343-w).
Although these techniques hold the promise of ''explaining'' the behavior of any black-box model, fundamental questions
still remain to be investigated.

1. What are the theoretical relationships between existing methods. What do they characterize about models?
In what scenarios can we expect them to agree (or be contradictory)?
2. When are practitioners allowed (disallowed) to **trust** the explanations provided? Without the existence of
ground-truths in explainability, it is hard to define accurate trustworthiness metrics.

### Contributions

My Doctoral degree is part of the [DEEL](https://deel.quebec/) research initiative and aims at answered these two research questions.
The first question is tackled in our [FDTrees](https://gablabc.github.io/publication/FDTrees.html) paper
where various post-hoc explanations are unified through the lens of **Functional Decompositions** and it is demonstrate that
disagreements are caused by so-called **Feature Interactions**. This discovery clarifies the relationship between the various explainers.
For the second question, we propose to use **Uncertainty** as a proxy of trustworthiness of post-hoc explanations. The higher the
uncertainty, the lower the trust. We define three critical types of uncertainties that we recommend computing in a XAI pipeline :

- **Oversimplification uncertainty**: the amount by which the explanation oversimplifies the model because of feature
interactions. This uncertainty is reduced by using [FDTrees](https://gablabc.github.io/publication/FDTrees.html).
- **Sub-sampling uncertainty** : the stochasticity induced by the necessity to provide subsamples of data to the explainers
instead of the whole dataset. The importance of considering this uncertainty is demonstrated by our
[FoolSHAP](https://gablabc.github.io/publication/fool_SHAP.html) attack that can make an unfair model look acceptable.
- **Under-specification uncertainty** : the uncertainty caused by the existence of an equivalent class of models with good
empirical performance *i.e* a Rashomon Set. This methodology is introduced in our
[JMLR](https://gablabc.github.io/publication/partial_order.html) paper.

Each of these is elaborated on in a published article from my PhD thesis.
