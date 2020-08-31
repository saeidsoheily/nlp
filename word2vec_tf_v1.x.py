__author__ = 'Saeid SOHILY-KHAH'
"""
Natural Language Processing algorithms: Word2Vec using TensorFlow 1.x
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


# Load sample corpus
def load_corpus():
    '''
    Load a sample text
    :return: corpus as a sample text
    '''
    corpus = ['Artificial intelligence is a technique, which enables machines to mimic human behaviour.',
              'Data Science is various tools, algorithms, and machine learning principles.',
              'Machine learning is a subset of artificial intelligence.',
              'Supervised learning is a machine learning technique, where the algorithm learns to predict using label data.',
              'Unsupervised learning is a machine learning technique, where the machine is trained using unlabelled data.',
              'Classification predicts discrete values, and Regression predicts continues values.',
              'Gradient boosting is a Machine Learning technique for regression and classification problems.']
    return corpus


# Remove the stop words from corpus
def remove_stop_words(corpus, stop_words):
    '''
    Remove stopwords from text body
    :param corpus: text body
    :param stop_words:
    :return:
    '''
    results = []
    for item in corpus:
        words_lst = item.split(' ')
        for stop_word in stop_words:
            while stop_word in words_lst:
                words_lst.remove(stop_word)
        results.append(" ".join(words_lst))
    return results


# Create a set of words exist in corpus
def create_set_of_words(corpus):
    '''
    Create a set of words
    :param corpus: text body
    :return:
    '''
    words = []
    for text in corpus:
        for word in text.split(' '):
            words.append(word.lower().strip().strip(".").strip(","))
    return set(words)


# Generate training data (pandas dataframe)
def generate_train_date(corpus, step_size):
    '''
    Generate training data in pandas dataframe format
    :param corpus: text body
    :param step_size:
    :return:
    '''
    train_data = []
    for item in corpus:
        sentence = item.split()
        for idx, word in enumerate(sentence):
            word = word.lower().strip().strip(".").strip(",")
            for neighbor in sentence[max(idx - step_size, 0): min(idx + step_size, len(sentence)) + 1]:
                neighbor = neighbor.lower().strip(".").strip(",")
                if neighbor != word:
                    train_data.append([word, neighbor])

    return pd.DataFrame(train_data, columns=['input', 'label'])


# Convert numbers to one hot vectors
def one_hot_encoder(index, num_dimensions):
    '''
    One Hot Encoder
    :param index: word index
    :param num_dimensions: number of dimensions (len of words)
    :return:
    '''
    encoded = np.zeros(num_dimensions)
    encoded[index] = 1
    return encoded


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load sample corpus (text body to analyse)
    corpus = load_corpus()

    # Remove the stop words from corpus
    stop_words = set(stopwords.words('english'))
    corpus = remove_stop_words(corpus, stop_words)

    # Create a set of words exist in corpus
    words = create_set_of_words(corpus)

    # Generate training data pandas data frame
    df = generate_train_date(corpus, step_size=4)

    # Convert words to int numbers (dictionary)
    word2int = {}
    for i, word in enumerate(words):
        word2int[word] = i

    # Create training input/label
    X = []  # input word
    Y = []  # label word
    num_dimensions = len(words)
    for x, y in zip(df['input'], df['label']):
        X.append(one_hot_encoder(word2int[x], num_dimensions))
        Y.append(one_hot_encoder(word2int[y], num_dimensions))

    X_train = np.asarray(X)  # convert X_train to numpy arrays
    Y_train = np.asarray(Y)  # convert Y_train to numpy arrays

    # TensorFlow: Model initialization
    embedding_dim = 2 # word embedding will be 2 dimensions for simply 2D visualization
    learning_rate = 0.05 # learning rate for GradientDescentOptimizer
    training_epochs = 15000  # number of epochs in training stage
    model_path = os.getcwd() + '/saved_model'  # to save trained model in the current directory

    # TensorFlow: Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, num_dimensions))
    y_ = tf.placeholder(tf.float32, shape=(None, num_dimensions))

    # TensorFlow: Create model
    # hidden layer: represents word vector
    W1 = tf.Variable(tf.random_normal([num_dimensions, embedding_dim]))
    b1 = tf.Variable(tf.random_normal([embedding_dim]))  # bias
    hidden_layer = tf.add(tf.matmul(x, W1), b1)

    # output layer
    W2 = tf.Variable(tf.random_normal([embedding_dim, num_dimensions]))
    b2 = tf.Variable(tf.random_normal([num_dimensions]))
    y_pred = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))

    # TensorFlow: Create the cost function and optimizer
    cost_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred), axis=[1])) # loss function: cross entropy
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function) # training operation

    # TensorFlow: Model evaluation
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TensorFlow: Create a session to run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    # TensorFlow: Create and instance of train.Saver
    saver = tf.train.Saver()

    # TensorFlow: Create an object class for writting summaries
    file_writer = tf.summary.FileWriter(model_path, sess.graph)

    sess.run(init) # execute the initializer

    # TensorFlow: Training...
    for epoch in range(training_epochs):
        sess.run(training_step, feed_dict={x: X_train, y_: Y_train})
        if epoch % 2000 == 0:
            print('Epoch:{:<5} ->  Training_Cost={:.5f}'
                .format(epoch, sess.run(cost_function, feed_dict={x: X_train, y_: Y_train})))

    # TensorFlow: Save the model
    saver.save(sess, model_path)

    # TensorFlow: Restore the saved model
    #saver.restore(sess, model_path')

    # TensorFlow: Evaluation the model
    # Summarize result
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TensorFlow: Run the hidden layer as words' vector
    vectors = sess.run(W1 + b1)
    df_word2vec = pd.DataFrame(vectors, columns=['dim1', 'dim2'])
    df_word2vec['word'] = words
    df_word2vec = df_word2vec[['word', 'dim1', 'dim2']]

    # Plot settings
    fig, axes = plt.subplots(figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    for word, dim1, dim2 in zip(df_word2vec['word'], df_word2vec['dim1'], df_word2vec['dim2']):
        axes.annotate(word, (dim1, dim2), color='red')

    x_axis_min = np.amin(vectors, axis=0)[0] - .5
    x_axis_max = np.amax(vectors, axis=0)[0] + .5
    y_axis_min = np.amin(vectors, axis=0)[1] - .5
    y_axis_max = np.amax(vectors, axis=0)[1] + .5

    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_nlp_word2vec.png', bbox_inches='tight')
    plt.show()
