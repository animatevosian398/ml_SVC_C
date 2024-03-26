#!/usr/bin/env python
# coding: utf-8

# # Toxic comments
# 
# This notebook takes you though a complete iteration of Machine Learning Assignment 1 - Toxic comments. The assignment details (including links to download the data) can be found [here](https://docs.google.com/document/d/1WGYw99e5q6j5V0Zrf2HveagU6URt_kVvdR8B9HYQ99E/edit?usp=sharing). 

# In[1]:


# all imports and magic commands
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_measures import BinaryClassificationPerformance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### IMPORTANT!!! Make sure you are using `BinaryClassificationPerformance` v1.02

# In[2]:


help(BinaryClassificationPerformance)


# # Function for feature building and extraction on natural language data

# In[3]:


# function that takes raw data and completes all preprocessing required before model fits
def process_raw_data(fn, my_random_seed, test=False):
    # read and summarize data
    toxic_data = pd.read_csv(fn)
    if (not test):
        # add an indicator for any toxic, severe toxic, obscene, threat, insult, or indentity hate
        toxic_data['any_toxic'] = (toxic_data['toxic'] + toxic_data['severe_toxic'] + toxic_data['obscene'] + toxic_data['threat'] + toxic_data['insult'] + toxic_data['identity_hate'] > 0)
    print("toxic_data is:", type(toxic_data))
    print("toxic_data has", toxic_data.shape[0], "rows and", toxic_data.shape[1], "columns", "\n")
    print("the data types for each of the columns in toxic_data:")
    print(toxic_data.dtypes, "\n")
    print("the first 10 rows in toxic_data:")
    print(toxic_data.head(5))
    if (not test):
        print("The rate of 'toxic' Wikipedia comments in the dataset: ")
        print(toxic_data['any_toxic'].mean())

    # vectorize Bag of Words from review text; as sparse matrix
    if (not test): # fit_transform()
        #######changes below#########
        #added binary=true 3/4/23 - to show occurrence
        #countvectorizer instead of hashingVectorizer so removed (n_features=2 ** 17, alternate_sign=False)
        #now added ngram (bigram) 2,2 - 3/5/23
#      took out   ngram_range = (2, 2)
        hv = HashingVectorizer(n_features=2 ** 17, alternate_sign=False,binary=True)
        X_hv = hv.fit_transform(toxic_data.comment_text)
        fitted_transformations.append(hv)
        print("Shape of HashingVectorizer X:")
        print(X_hv.shape)
    else: # transform() 
        X_hv = fitted_transformations[0].transform(toxic_data.comment_text)
        print("Shape of HashingVectorizer X:")
        print(X_cv.shape)
        
#     if binary=True, then we have a boolean, and don't need this transformation below
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
#     if (not test):
#         transformer = TfidfTransformer()
#         X_tfidf = transformer.fit_transform(X_hv)
#         fitted_transformations.append(transformer)
#     else:
#         X_tfidf = fitted_transformations[1].transform(X_hv)
    
    # create additional quantitative features
    # features from Amazon.csv to add to feature set
    toxic_data['word_count'] = toxic_data['comment_text'].str.split(' ').str.len()
    toxic_data['punc_count'] = toxic_data['comment_text'].str.count("\.")

    X_quant_features = toxic_data[["word_count", "punc_count"]]
    print("Look at a few rows of the new quantitative features: ")
    print(X_quant_features.head(10))
    
    # Combine all quantitative features into a single sparse matrix
    X_quant_features_csr = csr_matrix(X_quant_features)
    
    X_combined = hstack([X_hv, X_quant_features_csr])
#     X_combined = hstack([X_tfidf, X_quant_features_csr]) #if binary=true, no tfidf
    
    X_matrix = csr_matrix(X_combined) # convert to sparse matrix
    print("Size of combined bag of words and new quantitative variables matrix:")
    print(X_matrix.shape)
    
    # Create `X`, scaled matrix of features
    # feature scaling
    if (not test):
        sc = StandardScaler(with_mean=False)
        X = sc.fit_transform(X_matrix)
        fitted_transformations.append(sc)
        print(X.shape)
        y = toxic_data['any_toxic']
    else:
        X = fitted_transformations[1].transform(X_matrix) #changed to access [1] index, since with countvectorizer we have less transformations, less columns?
        print(X.shape)
    
    # Create Training and Test Sets
    # enter an integer for the random_state parameter; any integer will work
    if (test):
        X_submission_test = X
        print("Shape of X_test for submission:")
        print(X_submission_test.shape)
        print('SUCCESS!')
        return(toxic_data, X_submission_test)
    else: 
        X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(X, y, toxic_data, test_size=0.2, random_state=my_random_seed)
        print("Shape of X_train and X_test:")
        print(X_train.shape)
        print(X_test.shape)
        print("Shape of y_train and y_test:")
        print(y_train.shape)
        print(y_test.shape)
        print("Shape of X_raw_train and X_raw_test:")
        print(X_raw_train.shape)
        print(X_raw_test.shape)
        print('SUCCESS!')
        return(X_train, X_test, y_train, y_test, X_raw_train, X_raw_test)


# # Create training and test sets from function

# In[4]:


# create an empty list to store any use of fit_transform() to transform() later
# it is a global list to store model and feature extraction fits
fitted_transformations = []

# CHANGE FILE PATH and my_random_seed number (any integer other than 74 will do): 
X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = process_raw_data(fn='/Users/animatevosian/Desktop/ml/final_assignment_1/toxiccomments_train.csv', my_random_seed=1626)

# print("Number of fits stored in `fitted_transformations` list: ")
# print(len(fitted_transformations))


# # Fit (and tune) Various Models

# ### MODEL: ordinary least squares

# In[5]:


from sklearn import linear_model
ols = linear_model.SGDClassifier(loss="squared_error") #changed from squared_loss to squared_error based on documentation
# ols = linear_model.SGDClassifier(loss="squared_loss")
ols.fit(X_train, y_train)

ols_performance_train = BinaryClassificationPerformance(ols.predict(X_train), y_train, 'ols_train')
ols_performance_train.compute_measures()
print(ols_performance_train.performance_measures)


# ### MODEL: SVM, linear

# In[6]:


from sklearn import linear_model
svm = linear_model.SGDClassifier()
svm.fit(X_train, y_train)

svm_performance_train = BinaryClassificationPerformance(svm.predict(X_train), y_train, 'svm_train')
svm_performance_train.compute_measures()
print(svm_performance_train.performance_measures)


# In[7]:


from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#DEFAULT C = 1.0
svc_1 = LinearSVC(random_state=0, tol=1e-5)
svc_1.fit(X_train, y_train)

svc_1_performance_train = BinaryClassificationPerformance(svc_1.predict(X_train), y_train, 'svc1_train')
svc_1_performance_train.compute_measures()
print(svc_1_performance_train.performance_measures)


# In[8]:


from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
 #C = 0.01
svc_0_01 = LinearSVC(random_state=0, tol=1e-5, C=0.01 )
svc_0_01.fit(X_train, y_train)

svc_0_01_performance_train = BinaryClassificationPerformance(svc_0_01.predict(X_train), y_train, 'svc0_01_train')
svc_0_01_performance_train.compute_measures()
print(svc_0_01_performance_train.performance_measures)


# In[9]:


from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
 #C = 0.5
svc_0_5 = LinearSVC(random_state=0, tol=1e-5, C=0.5 )
svc_0_5.fit(X_train, y_train)

svc_0_5_performance_train = BinaryClassificationPerformance(svc_0_5.predict(X_train), y_train, 'svc0_5_train')
svc_0_5_performance_train.compute_measures()
print(svc_0_5_performance_train.performance_measures)


# In[10]:


from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#c=5
svc_5 = LinearSVC(random_state=0, tol=1e-5, C=5.0)
svc_5.fit(X_train, y_train)

svc_5_performance_train = BinaryClassificationPerformance(svc_5.predict(X_train), y_train, 'svc5_train')
svc_5_performance_train.compute_measures()
print(svc_5_performance_train.performance_measures)


# In[11]:


from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#C=10
svc_10 = LinearSVC(random_state=0, tol=1e-5, C=10.0)
svc_10.fit(X_train, y_train)

svc_10_performance_train = BinaryClassificationPerformance(svc_10.predict(X_train), y_train, 'svc10_train')
svc_10_performance_train.compute_measures()
print(svc_10_performance_train.performance_measures)


# In[12]:


# from sklearn.svm import LinearSVC
# # from sklearn.pipeline import make_pipeline
# # from sklearn.preprocessing import StandardScaler
# #C=100
# svc_100 = LinearSVC(random_state=0, tol=1e-5, C=100.0)
# svc_100.fit(X_train, y_train)

# svc_100_performance_train = BinaryClassificationPerformance(svc_100.predict(X_train), y_train, 'svc100_train')
# svc_100_performance_train.compute_measures()
# print(svc_100_performance_train.performance_measures)


# ### MODEL: logistic regression

# In[13]:


# from sklearn import linear_model
# lgs = linear_model.SGDClassifier(loss='log_loss') #changed to log_loss based on documentation
# # lgs = linear_model.SGDClassifier(loss='log')
# lgs.fit(X_train, y_train)

# lgs_performance_train = BinaryClassificationPerformance(lgs.predict(X_train), y_train, 'lgs_train')
# lgs_performance_train.compute_measures()
# print(lgs_performance_train.performance_measures)


# ### MODEL: Naive Bayes

# In[14]:


# from sklearn.naive_bayes import MultinomialNB
# nbs = MultinomialNB()
# nbs.fit(X_train, y_train)

# nbs_performance_train = BinaryClassificationPerformance(nbs.predict(X_train), y_train, 'nbs_train')
# nbs_performance_train.compute_measures()
# print(nbs_performance_train.performance_measures)


# ### MODEL: Perceptron

# In[15]:


# from sklearn import linear_model
# prc = linear_model.SGDClassifier(loss='perceptron')
# prc.fit(X_train, y_train)

# prc_performance_train = BinaryClassificationPerformance(prc.predict(X_train), y_train, 'prc_train')
# prc_performance_train.compute_measures()
# print(prc_performance_train.performance_measures)


# ### MODEL: Ridge Regression Classifier

# In[16]:


# from sklearn import linear_model
# rdg = linear_model.RidgeClassifier(alpha=10000)
# rdg.fit(X_train, y_train)

# rdg_performance_train = BinaryClassificationPerformance(rdg.predict(X_train), y_train, 'rdg_train')
# rdg_performance_train.compute_measures()
# print(rdg_performance_train.performance_measures)


# ### MODEL: Random Forest Classifier

# In[17]:


# from sklearn.ensemble import RandomForestClassifier
# rdf = RandomForestClassifier(max_depth=2, random_state=0)
# rdf.fit(X_train, y_train)

# rdf_performance_train = BinaryClassificationPerformance(rdf.predict(X_train), y_train, 'rdf_train')
# rdf_performance_train.compute_measures()
# print(rdf_performance_train.performance_measures)


# ### ROC plot to compare performance of various models and fits

# In[18]:


# fits = [ols_performance_train, svm_performance_train, lgs_performance_train, nbs_performance_train, prc_performance_train, rdg_performance_train, rdf_performance_train]
fits = [
        svc_0_01_performance_train,
        svc_0_5_performance_train,
        svc_1_performance_train, 
        svc_5_performance_train, 
        svc_10_performance_train,
        ] #testing svc, C - march 25
for fit in fits:
    plt.plot(fit.performance_measures['FP'] / fit.performance_measures['Neg'], 
             fit.performance_measures['TP'] / fit.performance_measures['Pos'], 'bo')
    plt.text(fit.performance_measures['FP'] / fit.performance_measures['Neg'], 
             fit.performance_measures['TP'] / fit.performance_measures['Pos'], fit.desc)
plt.axis([0, 1, 0, 1])
plt.title('ROC plot: train set')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# ### looking at reviews based on their classification
# 
# Let's say we decide that Ordinary Least Squares (OLS) Regression is the best model for generalization. Let's take a look at some of the reviews and try to make a (subjective) determination of whether it's generalizing well. 

# ### let's look at some false positives:

# In[19]:


# ols_predictions = ols.predict(X_train)


# In[20]:


# # false positives

# print("Examples of false positives:")

# import random, time

# for i in range(0, len(ols_predictions)):
#     if (ols_predictions[i] == 1):
#         if (X_raw_train.iloc[i]['any_toxic'] == 0):
#             if (random.uniform(0, 1) < 0.05): # to print only 5% of the false positives
#                 print(i)
# #                 print(X_raw_train.iloc[i]['comment_text'])
#                 print('* * * * * * * * * ')


# ---
# 
# # <span style="color:red">WARNING: Don't look at test set performance too much!</span>
# 
# ---
# 
# The following cells show performance on your test set. Do not look at this too often! 

# # Look at performance on the test set

# ### MODEL: ordinary least squares

# In[21]:


# ols_performance_test = BinaryClassificationPerformance(ols.predict(X_test), y_test, 'ols_test')
# ols_performance_test.compute_measures()
# print(ols_performance_test.performance_measures)


# ### MODEL: SVM, linear

# In[22]:


svm_performance_test = BinaryClassificationPerformance(svm.predict(X_test), y_test, 'svm_test')
svm_performance_test.compute_measures()
print(svm_performance_test.performance_measures)


# ### MODEL: SVC, adjusting c parameter
# 

# In[23]:


#c=0.01
svc_0_01_performance_test = BinaryClassificationPerformance(svc_0_01.predict(X_test), y_test, 'svc0_01_test')
svc_0_01_performance_test.compute_measures()
print(svc_0_01_performance_test.performance_measures)


# In[24]:


svc_0_5_performance_test = BinaryClassificationPerformance(svc_0_5.predict(X_test), y_test, 'svc0_5_test')
svc_0_5_performance_test.compute_measures()
print(svc_0_5_performance_test.performance_measures)


# In[25]:


svc_1_performance_test = BinaryClassificationPerformance(svc_1.predict(X_test), y_test, 'svc1_test')
svc_1_performance_test.compute_measures()
print(svc_1_performance_test.performance_measures)


# In[26]:


svc_5_performance_test = BinaryClassificationPerformance(svc_5.predict(X_test), y_test, 'svc5_test')
svc_5_performance_test.compute_measures()
print(svc_5_performance_test.performance_measures)


# In[27]:


svc_10_performance_test = BinaryClassificationPerformance(svc_10.predict(X_test), y_test, 'svc10_test')
svc_10_performance_test.compute_measures()
print(svc_10_performance_test.performance_measures)


# In[28]:


# svc_100_performance_test = BinaryClassificationPerformance(svc_100.predict(X_test), y_test, 'svc100_test')
# svc_100_performance_test.compute_measures()
# print(svc_100_performance_test.performance_measures)


# ### MODEL: logistic regression

# In[29]:


# lgs_performance_test = BinaryClassificationPerformance(lgs.predict(X_test), y_test, 'lgs_test')
# lgs_performance_test.compute_measures()
# print(lgs_performance_test.performance_measures)


# ### MODEL: Naive Bayes

# In[30]:


# nbs_performance_test = BinaryClassificationPerformance(nbs.predict(X_test), y_test, 'nbs_test')
# nbs_performance_test.compute_measures()
# print(nbs_performance_test.performance_measures)


# ### MODEL: Perceptron

# In[31]:


# prc_performance_test = BinaryClassificationPerformance(prc.predict(X_test), y_test, 'prc_test')
# prc_performance_test.compute_measures()
# print(prc_performance_test.performance_measures)


# ### MODEL: Ridge Regression Classifier

# In[32]:


# rdg_performance_test = BinaryClassificationPerformance(rdg.predict(X_test), y_test, 'rdg_test')
# rdg_performance_test.compute_measures()
# print(rdg_performance_test.performance_measures)


# ### MODEL: Random Forest Classifier

# In[33]:


# rdf_performance_test = BinaryClassificationPerformance(rdf.predict(X_test), y_test, 'rdf_test')
# rdf_performance_test.compute_measures()
# print(rdf_performance_test.performance_measures)


# ### ROC plot to compare performance of various models and fits

# In[34]:


# fits = [ols_performance_test, svm_performance_test, lgs_performance_test, nbs_performance_test, prc_performance_test,rdg_performance_test, rdf_performance_test]
fits = [
        svc_0_01_performance_test,
        svc_0_5_performance_test,
        svc_1_performance_test, 
        svc_5_performance_test, 
        svc_10_performance_test,
       ]
for fit in fits:
    plt.plot(fit.performance_measures['FP'] / fit.performance_measures['Neg'], 
             fit.performance_measures['TP'] / fit.performance_measures['Pos'], 'bo')
    plt.text(fit.performance_measures['FP'] / fit.performance_measures['Neg'], 
             fit.performance_measures['TP'] / fit.performance_measures['Pos'], fit.desc)
plt.axis([0, 1, 0, 1])
plt.title('SVC ROC plot: test set')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# ---
# 
# # <span style="color:red">SUBMISSION</span>
# 
# ---

# In[35]:


# # read in test data for submission
# # CHANGE FILE PATH and my_random_seed number (any integer other than 74 will do): 
# raw_data, X_test_submission = process_raw_data(fn='/Users/animatevosian/Desktop/ml/final_assignment_1/data/toxiccomments_test.csv', my_random_seed=1626, test=True)
# print("Number of rows in the submission test set (should be 153,164): ")


# ---
# 
# Choose a <span style="color:red">*single*</span> model for your submission. In this code, I am choosing the Ordinary Least Squares model fit, which is in the `ols` object. But you should choose the model that is performing the best for you! 

# In[36]:


# # store the id from the raw data
# my_submission = pd.DataFrame(raw_data["id"])
# # concatenate predictions to the id
# my_submission["prediction"] = ols.predict(X_test_submission)
# # look at the proportion of positive predictions
# print(my_submission['prediction'].mean())


# In[37]:


# raw_data.head()


# In[38]:


# my_submission.head()


# In[39]:


# my_submission.shape


# In[40]:


# # export submission file as pdf
# # CHANGE FILE PATH: 
# my_submission.to_csv('/Users/animatevosian/Desktop/ml/final_assignment_1/submissionPractice/toxiccomments_submission.csv', index=False)


# # Submit to Canvas: 1) the CSV file that was written in the previous cell and 2) the url to the repository (GitHub or other) that contains your code and documentation
