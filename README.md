# A Short Introduction to Bossting implementation -- AdaBoost
This is an implementation of the research paper ["A Short Introduction to Boosting"](http://www.cs.columbia.edu/~jebara/6772/papers/IntroToBoosting.pdf) written by Yoav Freund and Robert E. Schapire.

## Inspiration
Machine Learning algorithms specially those concerning classification and regression can perform weakly while encountering huge datasets. In order to overcome such inconveniences, a number of optimisation algorithms were developed that could improve a model's performance significantly. Adaboost is one such boosting technique which we have implemented here to analyse the improvement in the performance of our classification model.
<hr>

<img src = "https://www.edureka.co/blog/wp-content/uploads/2019/06/How-Does-Boosting-Algorithm-Work-Boosting-Machine-Learning-Edureka-min-528x254.png">

## Introduction
Boosting refers to a general and provably effective method of producing a very accurate prediction rule by combining rough and moderately inaccurate rules of thumb. The AdaBoost algorithm was introduced in 1995 by Freund and Schapire which solved many of the practical difficulties of the earlier boosting algorithms. The algorithm takes as input a training set (x<sub>1</sub>,y<sub>1</sub>),...., (x<sub>m</sub>,y<sub>m</sub>) where each x<sub>i</sub> belongs to some domain or instance space, and each label y<sub>i</sub> is in some label set assuming Y = {-1, 1}. AdaBoost calls a given weak or base learning algorithm repeatedly in a series of rounds t = 1,..., T whose job is to find a weak hypothesis h<sub>t</sub> and outputs a final hypothesis H which is a weighted majority vote of the T weak hypotheses.
<hr>

## Model Components
Our model architecture consists of the following components :-
- The weak learner was decided to be a Decision Tree Classifier with two leaf nodes.
- The output obtained from the weak learners were combined into a weighted sum that represented the final boosted output.
<hr>

## Requirements

scikit-learn==0.24.1<br>
numpy==1.19.2<br>
matplotlib==3.3.4<br>
typing==3.7.4.3
<hr>
