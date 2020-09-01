__author__ = 'Saeid SOHILY-KHAH'
"""
Natural Language Processing algorithms: LSA (Latent Semantic Analysis) using Bag-of-Words and TF-IDF [Scikit-Learn]
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Load sample corpus
def load_corpus():
    '''
    Load sample corpus with the assigned titles
    :return:
    '''
    title = [
        'Data Science',
        'Data Science',
        'Data Science',
        'Data Science',
        'Data Science',
        'Data Science',
        'Data Science',
        'Art',
        'Art',
        'Art',
        'Art',
        'Art',
        'Art',
        'Art'
    ]
    corpus = [
        'Artificial intelligence is a technique, which enables machines to mimic human behaviour.',
        'Data Science is various tools, algorithms, and machine learning principles.',
        'Machine learning is a subset of artificial intelligence.',
        'Supervised learning is a machine learning technique, where the algorithm learns to predict using label data.',
        'Unsupervised learning is a machine learning technique, where the machine is trained using unlabelled data.',
        'Classification predicts discrete values, and Regression predicts continues values.',
        'Gradient boosting is a Machine Learning technique for regression and classification problems.',
        'Art is a diverse range of human activities in creating visual, auditory or performing artifacts.',
        'The three classical branches of visual art are painting, sculpture and architecture.',
        'Music, theatre, film, dance, literature are included in a broader definition of the arts.',
        'Art can be defined as an act of expressing feelings, thoughts, and observations.',
        'Art may seek to bring about a particular emotion, for the purpose of relaxing or entertaining the viewer.',
        'Painting is the practice of applying paint, pigment, color or other medium to a solid surface',
        'Color, made up of hue, saturation, and value, dispersed over a surface is the essence of painting.'
    ]
    return title, corpus


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load list of sample documents with the assigned titles (text documents to analyse)
    title, corpus = load_corpus()

    # Convert corpus to pandas dataframe
    df_documents = pd.DataFrame(title, columns=['title'])
    df_documents['corpus'] = corpus

    # Method 1: using countVectorizer SkLearn
    # Define a vectorizer using countVectorizer SkLearn
    vectorizer1 = CountVectorizer(min_df=1, stop_words='english')
    bag_of_words = vectorizer1.fit_transform(df_documents.corpus) # transform the documents to the vectorizer

    # Create a model capable of returning a dataset with fewer features
    svd1 = TruncatedSVD(n_components=2)
    lsa1 = svd1.fit_transform(bag_of_words) # transform the bag of words into the LSA

    # Convert the count vectorizer encoded documents into the pandas dataframe
    df_documents_cv_encoded = pd.DataFrame(lsa1, columns=['topic1', 'topic2'])
    df_documents_cv_encoded['corpus'] = corpus
    df_documents_cv_encoded['title'] = title
    df_documents_cv_encoded = df_documents_cv_encoded[['corpus', 'title', 'topic1', 'topic2']]

    # Method 2: using TfidfVectorizer SkLearn
    # Define a vectorizer using countVectorizer SkLearn
    vectorizer2 = TfidfVectorizer(min_df=1, stop_words='english')
    tfidfs = vectorizer2.fit_transform(df_documents.corpus) # transform the documents to the vectorizer

    # Create a model capable of returning a dataset with fewer features
    svd2 = TruncatedSVD(n_components=2)
    lsa2 = svd2.fit_transform(tfidfs) # transform the tf-idfs into the LSA

    # Convert the tfidf vectorizer encoded documents into the pandas dataframe
    df_documents_tc_encoded = pd.DataFrame(lsa2, columns=['topic1', 'topic2'])
    df_documents_tc_encoded['corpus'] = corpus
    df_documents_tc_encoded['title'] = title
    df_documents_tc_encoded = df_documents_tc_encoded[['corpus', 'title', 'topic1', 'topic2']]

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot results
    # Plot method 1: using countVectorizer SkLearn
    for ttl in df_documents_cv_encoded.title.unique():
        x1 = df_documents_cv_encoded[df_documents_cv_encoded.title == ttl]['topic1'].values
        x2 = df_documents_cv_encoded[df_documents_cv_encoded.title == ttl]['topic2'].values
        color = 'blue' if ttl == 'Data Science' else 'red'
        axes[0].scatter(x1, x2, c=color, alpha=0.4, label=ttl)

    for document, x1, x2 in zip(df_documents_cv_encoded['corpus'],
                                df_documents_cv_encoded['topic1'],
                                df_documents_cv_encoded['topic2']):
        axes[0].annotate(document[:20] + '...', (x1, x2), alpha=0.5)

    axes[0].set_xlabel('Topic 1')
    axes[0].set_ylabel('Topic 2')
    axes[0].axvline(linewidth=0.5)
    axes[0].axhline(linewidth=0.5)
    axes[0].set_title('LSA using CountVectorizer', fontsize=12)
    axes[0].legend()

    # Plot method 2: using TfidfVectorizer SkLearn
    for ttl in df_documents_tc_encoded.title.unique():
        x1 = df_documents_tc_encoded[df_documents_tc_encoded.title == ttl]['topic1'].values
        x2 = df_documents_tc_encoded[df_documents_tc_encoded.title == ttl]['topic2'].values
        color = 'blue' if ttl == 'Data Science' else 'red'
        axes[1].scatter(x1, x2, c=color, alpha=0.4, label=ttl)

    for document, x1, x2 in zip(df_documents_tc_encoded['corpus'],
                                df_documents_tc_encoded['topic1'],
                                df_documents_tc_encoded['topic2']):
        axes[1].annotate(document[:20] + '...', (x1, x2), alpha=0.5)

    axes[1].set_xlabel('Topic 1')
    axes[1].set_ylabel('Topic 2')
    axes[1].axvline(linewidth=0.5)
    axes[1].axhline(linewidth=0.5)
    axes[1].set_title('LSA using TF-IDF Vectorizer', fontsize=12)
    axes[1].legend()

    # To save the plot locally
    plt.savefig('lsa.png', bbox_inches='tight')
    plt.show()