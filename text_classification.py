__author__ = 'Saeid SOHILY-KHAH'
"""
Natural Language Processing algorithms: Text Classification using Logistic Regression [Scikit-Learn]
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# Load sample text documents
def load_text():
    '''
    Load large movie review dataset (binary sentiment) [set of 50,000 highly polar movie reviews]
    [ref.: http://www.aclweb.org/anthology/P11-1015]
    :return: pandas dataframe (label: binary sentiment [0:negative,1:positive], text: movie review)
    '''
    try:
        df = pd.read_csv('movie_review_dataset.csv')
    except:
        # File not exist
        print("unzip the 'movie_review_dataset.csv.zip' and put .csv file in the same directory as .py file")
        exit()
    return df


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load sample text documents (text documents for classification)
    df = load_text()

    # Define a vectorizer using TfidfVectorizer SkLearn by considering stopwords,..., 1-gram and 2-grams
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.4, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df.text) # transform the texts to the vectorizer
    y = df.label.values # target values (i.e. labels)

    # Split data into training and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)

    # Define Logistic Regression classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train) # fit classifier according to the training data

    # Prediction
    y_pred = clf.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred) # creating confusion matrix

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot confusion matrix
    sns.heatmap(cm.T,
                square=True,
                annot=True,
                fmt='d',
                cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title('CONFUSION MATRIX (ACCURACY = {}%)'.format(round(accuracy_score(y_test, y_pred),4)*100),
              fontsize=12)

    # To save the plot locally
    plt.savefig('text_classification.png', bbox_inches='tight')
    plt.show()
