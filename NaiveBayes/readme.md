# Naive Bayes Classier for Text Classication
Annie Steenson

COM S 474 - Machine Learning

Lab 1

### How to run the program
NaiveBayes is the class with the main function

##### Arguments:
1. vocabulary.txt
2. map.csv
3. train_label.csv
4. train_data.csv
5. test_label.csv
6. test_data.csv 

Example run command:

`
java NaiveBayes vocabulary.txt map.csv training label.csv training data.csv
testing label.csv testing data.csv
`

**Note**

The confusion matrix starts at indices (1,1). The row 0 and column 0 is not used. 
This is because the training and test data id's begin at 1.

### Lab Report

**Question 1:** 

Now you will evaluate your classiers on the testing data set. First, repeat the experiments described in 
Section 2.2.1 on testing dataset. Compare the results obtained with the results you have
obtained in Section 2.2.1. What do you observe? Discuss.

**Answer 1:**

The accuracy on the training data is much higher (~ 94%) than it is on the test data (~ 78%). However, it is 
peculiar that the accuracy on the training data is not 100% percent since that is the same data set we created the 
model off of. The reason for the training data not showing 100% accuracy could be that some newsgroups' documents are 
too similar and, therefore, hard to differentiate. Another reason could be that the training dataset is not plentiful 
enough to achieve a near 100% accurate representation of the newsgroups.

**Question 2:** 

Next repeat the experiments (on testing data) using your Maximum Likelihood estimator PMLE(wkj!j)
instead of the Bayesian estimators. Compare your results to the results obtained using your
Bayesian estimators. Can you observe the difference? Which one is better?

**Answer 2:**

The experience using the Maximum Likelihood Estimator returns very, very low accuracy (~ 4%). The Bayesian estimator is
by far much better on this dataset than the Maximum Likelihood Estimator (MLE). 
To analyze why this might be, we should look at how each estimate is defined.

Here is how I calculate each estimate in my code:

 `
 max_likelihood_estimate[label][word] = (double) ((total_occurences) / total_words);
 `
 
 `
 baysian_estimate[label][word] = (double)  (total_occurences + 1) / (total_words + num_words);
 `
 
Assume that label is synonymous to the category/newsgroup of a document.

As apparent above, MLE neglects to consider the total_words that are in that document group. Therefore, MLE applied a 
more general estimate rather than one that references the document group's context. 

Looking at the class accuracy for MLE on the test data, it looks like all of my documents are converging to the first 
newsgroup/label.