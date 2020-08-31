__author__ = 'Saeid SOHILY-KHAH'
"""
Natural Language Processing algorithms: TF-IDF
"""
import math
import operator
from nltk.corpus import stopwords

# Load sample documents
def load_documents():
    '''
    Load a sample documents
    :return: list of documents
    '''
    documents = [
        ['Artificial intelligence is a technique which enables a machine in artificial way to mimic human behaviour'],
        ['Machine learning is a subset of artificial intelligence divided into supervised and unsupervised learning'],
        ['Supervised learning is a machine learning technique where the algorithm learns to predict using label data'],
        ['Unsupervised learning is a machine learning technique where the machine is trained using unlabelled data'],
        ['A classification algorithm predicts a discrete value, and a regression algorithm predicts a continues value'],
        ['Gradient boosting is a Machine Learning technique for regression and classification problems with boosting']]
    return documents


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load list of sample documents (text documents to analyse)
    documents = load_documents()

    # Generate a bag of words for each document
    bag_of_words = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(documents)):
        bag_of_words.append([x.lower() for x in documents[i][-1].split()])
        # Remove the stop words
        for stop_word in stop_words:
            while stop_word in bag_of_words[-1]:
                bag_of_words[-1].remove(stop_word)

    # Create a set of unique words in all documents
    unique_words = set()
    for i in range(len(bag_of_words)):
        unique_words = unique_words.union(set(bag_of_words[i]))

    # Compute Tfs; occurence rate for each word in a document
    TFs = []
    for i in range(len(bag_of_words)):
        TFs.append(dict.fromkeys(unique_words, 0))

        doc_len = len(bag_of_words[i]) # number of words in the document
        for word in bag_of_words[i]:
            TFs[i][word] += 1/doc_len

    # Compute IDFs; Inverse Data Frequency (IDF); log(number of documents)/(number of documents that contain the word)
    N = len(documents)
    IDF = dict.fromkeys(unique_words, 0)
    for item in TFs:
        for word, tfs in item.items():
            if tfs > 0:
                IDF[word] += 1

    for word, value in IDF.items():
        IDF[word] = math.log(N / float(value)+1e-6)

    # Compute TF-IDFs; TFs multiplied by IDF
    TF_IDFs = []
    for item in TFs:
        TF_IDFs.append({})
        for word, value in item.items():
            TF_IDFs[-1][word] = round(value * IDF[word],3)

    # Summarize result
    for i, item in enumerate(TF_IDFs):
        sorted_item = sorted(item.items(), key=operator.itemgetter(1))
        print('document {} is mostly about {} and {}.'.format(i+1, sorted_item[-1][0], sorted_item[-2][0]))