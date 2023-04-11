# COLLEGE-PROJECT

1.Importing Libraries: The first few lines of code imports necessary libraries such as numpy, pandas, seaborn, matplotlib, warnings, string, nltk, and various machine learning models from sklearn.

2.Loading Data: The code loads the preprocessed dataset in csv format and stores it in the 'df' variable. The first 5 rows of the dataset are then displayed using the 'head()' function.

3.Data Cleaning: The 'Unnamed: 0' column is dropped from the dataset as it contains redundant information. The 'dropna()' function is used to drop any rows that contain null or missing values. A new column named 'length' is created that contains the length of each review in the dataset.

4.Data Visualization: A histogram is plotted to visualize the distribution of review lengths in the dataset. The 'groupby()' function is used to group reviews by their labels and descriptive statistics (such as count, mean, standard deviation, etc.) are computed for each label. A histogram is then plotted to visualize the distribution of review lengths for each label.

5.Bag of Words (BoW) Model: A function named 'text_process' is defined that removes any punctuation and stop words from each review. A 'CountVectorizer()' object is created with the 'analyzer' parameter set to the 'text_process' function. The 'fit()' function is called on the 'CountVectorizer()' object to create a bag of words model of the entire dataset. The 'vocabulary_' attribute of the bag of words model is used to display the total vocabulary size.

6.Transforming a single review: A single review is selected from the dataset and its BoW representation is computed using the 'transform()' function. The 'shape' attribute is used to display the dimensions of the BoW representation. The 'get_feature_names()' function is used to display the words that are present in the vocabulary at specific indices.

7.Transforming the entire dataset: The 'transform()' function is called on the entire dataset to obtain the BoW representation of each review. The 'shape' attribute is used to display the dimensions of the BoW model. The 'nnz' attribute is used to display the number of non-zero values in the sparse matrix representation of the BoW model. The 'sparsity' of the BoW model is computed and displayed.

8.TF-IDF Model: A 'TfidfTransformer()' object is created and fit on the BoW model to obtain the TF-IDF representation of each review. The 'transform()' function is used to compute the TF-IDF representation of a single review. The 'idf_' attribute of the 'TfidfTransformer()' object is used to display the IDF value of specific words in the vocabulary.

9.Train-Test Split: The dataset is split into training and testing sets using the 'train_test_split()' function.

10.Multinomial Naive Bayes Model: A 'Pipeline()' object is created that consists of a BoW model, a TF-IDF model, and a Multinomial Naive Bayes classifier. The 'fit()' function is called on the 'Pipeline()' object to train the model on the training data. The 'predict()' function is used to obtain predictions on the testing data. The classification report, confusion matrix, and accuracy score are displayed to evaluate the performance of the model.

11.Random Forest Classifier: A 'Pipeline()' object is created that consists of a BoW model, a TF-IDF model, and a Random Forest classifier. The 'fit()' function is called on the 'Pipeline()' object to train the model on the training data. The 'predict()' function is used to obtain predictions on the testing data. The classification report,
