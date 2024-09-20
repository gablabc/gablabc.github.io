---
title: "Tackling the XAI Disagreement Problem with Regional Explanations"
collection: publications
category: conferences
permalink: /publication/FDTrees
excerpt: 'Compute explanations on regions defined by a decision tree'
date: 2024-03-08
venue: 'AISTATS'
citation: 'Laberge, G., Pequignot, Y., Marchand, M., & Khomh, F. (2024, May). Tackling the XAI Disagreement Problem with Regional Explanations. In International Conference on Artificial Intelligence and Statistics (AISTATS) (Vol. 238).'
---

The XAI Disagreement Problem concerns the fact that various explainability methods yield different local/global insights on model
behavior. Thus, given the lack of ground truth in explainability, practitioners are left wondering “Which explanation should I believe?”.
In this work, we approach the Disagreement Problem from the point of view of Functional Decomposition (FD). First, we
demonstrate that many XAI techniques disagree because they handle feature interactions differently. Secondly, we reduce interactions
locally by fitting a so-called FD-Tree, which partitions the input space into regions where the model is approximately additive.
Thus instead of providing global explanations aggregated over the whole dataset, we advocate reporting the FD-Tree structure as well
as the regional explanations extracted from its leaves. The beneficial effects of FD-Trees on the Disagreement Problem are demonstrated
on toy and real datasets

![california](/images/papers/results.png)

[Paper](https://hal.science/hal-04480870/document)

[Github](https://github.com/gablabc/UXAI_ANOVA)
