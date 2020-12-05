# CSE 258 Assignment 2 Report

## Outline
1. Identify a dataset to study, and perform an exploratory analysis of the data. Describe the dataset, including its basic statistics and properties, and report any interesting findings. This exploratory analysis should motivate the design of your model in the following sections. Datasets should be reasonably large (e.g. more than 50,000 samples).

The dataset we choose is renttherunray cloth rental data. 
- plots ..
 

2. Identify a predictive task that can be studied on this dataset. Describe how you will evaluate your model at this predictive task, what relevant baselines can be used for comparison, and how you will assess the validity of your model’s predictions. It’s fine to use models that were described in class here (i.e., you don’t have to invent anything new (though you may!)), though you should explain and justify which model was appropriate for the task. It’s also important in this section to carefully describe what features you will use and how you had to process the data to obtain them.

The task is to use user reviews to predict if user feels the rental dress fits perfectly, is smaller, or is larger. We cast the problem as a classification prblem. Given text features, classfify as class 0 (fit), class 1 (small) or class 2 (large).

- performance matrix
We decide to use balanced error rate $\frac{1}{3} \sum_{\text{all classes}} \text{1 - class accuracy}$

- base line: Logistics + unigram

- model tesing
train, validation, test piple line. 20% of data are used for validation and parameter toneing. We select the best model to perform on the test set.

- Features 
1. remove stopwords exluding stopwords that negate meaning such as "not","wouldn't",etc to preseve sentiment of the sentance. For instance, the meaning of "this dress does not fit" would change dramatically if we remove the word "not."

2. only include ngrams that contain at least one adjetive in the bag of words
3. use multiple ngram strategy
- unigram
- unigram + bigram
- unigram + bigram + trigram
- unigram + bigram + trigram + 4-gram

4. If feature size is too large, we use tuncated svd to preform dementionality reduction. 

- Model selection
We choose models that are suitable for classification, such as KNN, Naive Base,  Logistics, SVM. These kind of models produce distrect value ouptut and each output label belongs to a class. Regression model will not be suitable for this task because the notion of small, fit, large is on an arbitrary scale and labels cannot be represented as continuous values. If the maping small->0, fit->1, large->2, regression model (MSE loss) would make the assumption that predicting a fit review small is the same as redicting large, but these two are different classes. We have considered using Ordinal Regression, which preseve ordering information of different categories. However, ...


3. Describe your model. Explain and justify your decision to use the model you proposed. How will you optimize it? Did you run into any issues due to scalability, overfitting, etc.? What other models did you consider for comparison? What were your unsuccessful attempts along the way? What are the strengths and weaknesses of the different models being compared?

    When comparing numerous models, we ran into the issue of scalability. With a combination of unigram, bigram, trigram, and 4-grams, the number of features is around 8000. My unsuccessful attempt was directly passing the training data to the model. To deal with this issue, we used truncate SVD to reduce number of features down to 500. 

4. Describe literature related to the problem you are studying. If you are using an existing dataset, where did it come from and how was it used? What other similar datasets have been studied in the past and how? What are the state-of-the-art methods currently employed to study this type of data? Are the conclusions from existing work similar to or different from your own findings?

5. Describe your results and conclusions. How well does your model perform compared to alternatives, and what is the significance of the results? Which feature representations worked well and which do not? What is the interpretation of your model’s parameters? Why did the proposed model succeed why others failed (or if it failed, why did it fail)?



## Dataset

## Models
- SVM
- Logistics
- KNN
- Naive Bayse

## Features
### TF based 
- unigram
- unigram + bigram
- unigram + bigram + trigram
- unigram + bigram + trigram + 4-gram


### TF-IDF based
- unigram
- unigram + bigram
- unigram + bigram + trigram
- unigram + bigram + trigram + 4-gram
- unigram + bigram + trigram + 4-gram + 5-gram


## Literature Review