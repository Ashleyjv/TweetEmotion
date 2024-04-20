#comp
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import ssl
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download('stopwords')

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import matplotlib.pyplot as plt
nltk.download('stopwords')



#joy, love, sadness, anger, fear, surprise
class LRModel:
    def __init__(self, num_features, num_classes):
        # Initialize weights and biases
        self.weights = np.zeros((num_features, num_classes))
        self.biases = np.zeros(num_classes)
    
    def softmax(self, z):
        # Apply the softmax function
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def train(self, X, y, learning_rate=0.9, num_iterations=4000):
        """
        Train the logistic regression model using gradient descent.
        X: Numpy array of training data
        y: Numpy array of labels
        learning_rate: Step size for gradient descent
        num_iterations: Number of iterations for the training loop
        """
        for i in range(num_iterations):
            # Compute the predictions
            z = np.dot(X, self.weights) + self.biases
            predictions = self.softmax(z)
            
            # Compute the gradient of the loss function w.r.t. weights and biases
            diff = predictions - y  # Error in predictions
            grad_weights = np.dot(X.T, diff) / X.shape[0]
            grad_biases = np.sum(diff, axis=0) / X.shape[0]
            
            # Update the weights and biases
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases

            # Optionally, you can print the loss here or use any convergence criteria
    def train2(self, X, y, class_weights, learning_rate=0.9, num_iterations=4000):
        """
    Train the logistic regression model using gradient descent, with class weights.
    
    Parameters:
    - X: Numpy array of training data.
    - y: Numpy array of labels, one-hot encoded.
    - class_weights: Dictionary mapping class indices to weights.
    - learning_rate: Step size for gradient descent.
    - num_iterations: Number of iterations for the training loop.
    """
        for i in range(num_iterations):
            # Compute the predictions
            z = np.dot(X, self.weights) + self.biases
            predictions = self.softmax(z)
            
            # Apply class weights
            for j in range(6):
                predictions[:, j] *= class_weights[j]
            
            # Compute the error and gradients with weighted predictions
            diff = predictions - y
            grad_weights = np.dot(X.T, diff) / X.shape[0]
            grad_biases = np.sum(diff, axis=0) / X.shape[0]
            
            # Update weights and biases
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases
    
    
    def predict(self, X):
        """
        Make predictions using the trained weights
        X: Numpy array of data
        """
        z = np.dot(X, self.weights) + self.biases
        predictions = self.softmax(z)
        return np.argmax(predictions, axis=1)  # Return the class with the highest probability
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        X: Numpy array of data
        y: Numpy array of labels
        """
        predictions = self.predict(X)
        accuracy = (predictions == np.argmax(y, axis=1)).mean()  # Assuming y is one-hot encoded
        return accuracy
    
    # def predict(self, X):
    #     # Compute the scores using the learned weights and bias
    #     z = np.dot(X, self.weights) + self.biases
    #     # Use softmax to get probabilities
    #     probabilities = self.softmax(z)
    #     # Pick the class with the highest probability as the prediction
    #     predictions = np.argmax(probabilities, axis=1)
    #     return predictions
    
    def predict_emotion(self, X, index_to_emotion):
        # Make predictions using the trained weights
        predictions_indices = self.predict(X)
        # Convert numerical predictions back to emotion strings
        predictions_emotions = [index_to_emotion[index] for index in predictions_indices]
        return predictions_emotions
 
 
 
import numpy as np
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            "W1": np.random.randn(hidden_size, input_size) * 0.01,
            "b1": np.zeros((hidden_size, 1)),
            "W2": np.random.randn(output_size, hidden_size) * 0.01,
            "b2": np.zeros((output_size, 1))
        }

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return e_Z / np.sum(e_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        
        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[1]
        logprobs = -np.log(Y_hat[Y.astype(bool)]).mean()
        return logprobs
    
    def compute_weighted_loss(self, Y_hat, Y, class_weights):
        m = Y.shape[1]
        weighted_losses = []
        for i in range(Y.shape[0]):  # Iterate over each class
            # Select the predictions and true labels for the current class
            Y_hat_class = Y_hat[i, :]
            Y_class = Y[i, :]

            # Compute the weighted loss for the current class
            class_weight = class_weights[i]
            weighted_loss = -np.sum(class_weight * (Y_class * np.log(Y_hat_class + 1e-9)))
            weighted_losses.append(weighted_loss)

        # Compute the mean of weighted losses
        total_weighted_loss = np.sum(weighted_losses) / m
        return total_weighted_loss
    
    def compute_weighted_loss2(self, Y_hat, Y, class_weights):
        m = Y.shape[1]
        # Extract individual class weights
        weights = np.asarray([class_weights[i] for i in np.argmax(Y, axis=0)])
        # Compute cross-entropy loss
        loss = -np.sum(weights * (Y * np.log(Y_hat + 1e-9))) / m
        return loss
    

    def backward_propagation(self, cache, X, Y):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * self.relu_derivative(Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def update_parameters(self, grads, learning_rate):
        self.params['W1'] -= learning_rate * grads['dW1']
        self.params['b1'] -= learning_rate * grads['db1']
        self.params['W2'] -= learning_rate * grads['dW2']
        self.params['b2'] -= learning_rate * grads['db2']

    def train(self, X_train, Y_train, num_iterations=1000, learning_rate=0.01, print_cost=True):
        for i in range(num_iterations):
            A2, cache = self.forward_propagation(X_train)
            cost = self.compute_loss(A2, Y_train)
            grads = self.backward_propagation(cache, X_train, Y_train)
            self.update_parameters(grads, learning_rate)
            
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
            
    def train2(self, X_train, Y_train, class_weights, num_iterations=1000, learning_rate=0.01, print_cost=True):
        for i in range(num_iterations):
            A2, cache = self.forward_propagation(X_train)
            
            # Use the modified loss function that includes class weights
            cost = self.compute_weighted_loss2(A2, Y_train, class_weights)
            
            grads = self.backward_propagation(cache, X_train, Y_train)
            self.update_parameters(grads, learning_rate)
            
            # if print_cost and i % 100 == 0:
                # print(f"Cost after iteration {i}: {cost}")


    def predict(self, X):
        A2, cache = self.forward_propagation(X)
        predictions = np.argmax(A2, axis=0)
        return predictions


def preprocess_and_stem(sentence):
    # Initialize the PorterStemmer
    ps = PorterStemmer()
    # Initialize stop words
    stop_words = set(stopwords.words('english'))
    # Clean the sentence
    # Remove URLs, hashtags, mentions, and special characters
    sentence_clean = re.sub(r"http\S+|www\S+|https\S+", '', sentence, flags=re.MULTILINE)
    sentence_clean = re.sub(r'\@\w+|\#','', sentence_clean)
    sentence_clean = re.sub(r"[^a-zA-Z\s]", '', sentence_clean, re.I|re.A)
    # Strip whitespace, convert to lowercase
    sentence_clean = sentence_clean.strip().lower()
    # Tokenize the cleaned sentence
    tokens = word_tokenize(sentence_clean)
    # Remove stopwords and stem each token
    stemmed_tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    # Optionally: return both the cleaned and stemmed sentence and the token list
    return ' '.join(stemmed_tokens), stemmed_tokens


def preprocess_train_df(some_df):
    train_map = {}
    for row in some_df.iterrows():
        train_map[row[1][0]] = [row[1][1], row[1][2]]
        
    # print (train_map)
    
    #stemming the train_map
    for key, value in train_map.items():
        sentence, emotion = value
        st_token_sent, st_token_arr = preprocess_and_stem(sentence)
        train_map[key] = [st_token_sent, emotion]
    
    return train_map

def Getlabels(data_map):
    # Assuming `data_map` is your map: {id: [stemmed_sentence, emotion]}
    # Extract labels
    raw_labels = [value[1] for value in data_map.values()]  # List of all emotions

    # You need to convert these labels to integers or one-hot encode them for multi-class logistic regression
    # For example, let's say you have 6 emotions: joy, sadness, anger, fear, love, surprise
    emotion_to_index = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}

    # Convert raw labels to indices
    labels_indices = [emotion_to_index[label] for label in raw_labels]

    # Now you would one-hot encode these indices
    def one_hot_encode(indices, num_classes):
        one_hot = np.zeros((len(indices), num_classes))
        one_hot[np.arange(len(indices)), indices] = 1
        return one_hot

    # Assuming you have 6 emotions
    num_classes = 6
    # print (labels)
    labels = one_hot_encode(labels_indices, num_classes)
    # print (labels)
    return labels

import math

# Suppose 'data' is your map: {id: [stemmed_sentence, emotion]}
# documents = [value[0] for value in data.values()]  # List of all stemmed sentences

# Step 1: Calculate TF (Term Frequency)
def compute_tf(document):
    # Split the document into terms
    words = document.split()
    # Count the number of times each term occurs in the document
    tf_dict = {}
    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    # Normalize by the total number of terms in the document
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(words)
    return tf_dict


# Step 2: Calculate IDF (Inverse Document Frequency)
def compute_idf(documents):
    # Count the number of documents that contain each term
    idf_dict = {}
    N = len(documents)
    documents = [doc.split() for doc in documents]
    for doc in documents:
        for word in set(doc):
            idf_dict[word] = idf_dict.get(word, 0) + 1
    # Compute IDF, taking the log of the ratio of the total number of documents to
    # the number of documents containing each term
    for word in idf_dict:
        idf_dict[word] = math.log(N / float(idf_dict[word]))
    return idf_dict

# Step 3: Calculate TF-IDF Scores
def compute_tfidf(tf, idfs):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idfs.get(word, 0)
    return tfidf

#for training
def ConvertToDenseMatrix(tfidf_documents):
    # Assuming `tfidf_documents` is your list of dictionaries with TF-IDF scores
    # First, get the complete vocabulary
    vocabulary = set(term for doc in tfidf_documents for term in doc.keys())

    # Create a mapping of vocabulary terms to column indices
    vocab_to_index = {term: i for i, term in enumerate(vocabulary)}

    # Initialize an empty matrix of zeros with the appropriate shape
    num_documents = len(tfidf_documents)
    num_terms = len(vocabulary)
    dense_matrix = np.zeros((num_documents, num_terms))

    # Fill in the matrix with TF-IDF scores
    for i, doc in enumerate(tfidf_documents):
        for term, score in doc.items():
            dense_matrix[i, vocab_to_index[term]] = score
    return dense_matrix

#for testing
def ConvertToDenseMatrix2(tfidf_documents, vocab_to_index):
    # Use the provided vocab_to_index to ensure alignment with the model's expected features
    num_documents = len(tfidf_documents)
    num_terms = len(vocab_to_index)  # This should match the model's weights
    dense_matrix = np.zeros((num_documents, num_terms))
    
    for i, doc in enumerate(tfidf_documents):
        for term, score in doc.items():
            if term in vocab_to_index:  # Only use terms that were seen during training
                dense_matrix[i, vocab_to_index[term]] = score
    print ("in CTDM2 ", dense_matrix.shape)
    return dense_matrix
    

def perform_cross_validation(X, y, num_features, num_classes, k, class_weights):
    fold_size = len(X) // k
    lrs = [0.1, 0.2, 0.5, 0.7, 0.9]  # Ensure this matches k if k is variable
    scores_per_lr = {lr: [] for lr in lrs}  # Dictionary to hold scores for each learning rate

    for lr in lrs:
        for i in range(k):
            valid_data = X[i * fold_size:(i + 1) * fold_size]
            valid_labels = y[i * fold_size:(i + 1) * fold_size]
            train_data = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]], axis=0)
            train_labels = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]], axis=0)

            model = LRModel(num_features, num_classes)
            model.train(train_data, train_labels, learning_rate=lr, num_iterations=4000)  # Assume train handles class_weights internally if needed
            accuracy = model.evaluate(valid_data, valid_labels)
            scores_per_lr[lr].append(accuracy)  # Append accuracy for this fold

        print(f"Average accuracy for LR={lr}: {np.mean(scores_per_lr[lr])}")

    # After collecting accuracies, plot them
    avg_scores = [np.mean(scores) for scores in scores_per_lr.values()]  # Calculate average score for each lr
    plt.plot(list(scores_per_lr.keys()), avg_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Learning Rate vs. Validation Accuracy')
    plt.xscale('log')
    plt.show()

    return avg_scores  # Return average scores for further analysis or decision-making

def preprocess_test_df2(test_df):
    test_map = {}
    for i, row in test_df.iterrows():
        processed_sentence, _ = preprocess_and_stem(row['text'])
        # Ensure every document is included, even if preprocessing results in an empty string
        if not processed_sentence:  # Check if the result is empty
            processed_sentence = "emptydoc"  # Use a placeholder
        test_map[row['id']] = [processed_sentence]
    return test_map


def  test_and_write_LR(model, vocab_to_index, index_to_emotion, idf):  # Pass 'idf' from training phase
    # print (len(idf))
    test_df = pd.read_csv("test.csv")
    test_df = test_df.drop_duplicates(subset=['id'], keep='first')

    # print (test_df.shape)
    # Check for empty or NaN entries in the test dataframe
    # empty_docs_indices = test_df[test_df['text'].isnull() | (test_df['text'] == '')].index.tolist()
    # print("Indices of empty or NaN documents:", empty_docs_indices)

    test_map = preprocess_test_df2(test_df)
    # print (test_map)
    test_documents = [value[0] for value in test_map.values()]
    # print ("test_documents", len(test_documents))
    # emptydoc_count = sum(1 for doc in test_documents if doc == "emptydoc")
    # print("Number of 'emptydoc' placeholders:", emptydoc_count) 

    test_tfs = [compute_tf(doc) for doc in test_documents]
    test_tfidf_documents = [compute_tfidf(tf, idf) for tf in test_tfs]  # Use 'idf' from training
    test_dense_tfidf = ConvertToDenseMatrix2(test_tfidf_documents, vocab_to_index)  # Ensure this uses training 'vocab_to_index'
    print (test_dense_tfidf.shape)
    predictions = model.predict(test_dense_tfidf)  # Correct shape for prediction
    
    # Convert indices to emotions and add to DataFrame, then save
    predicted_emotions = [index_to_emotion[pred] for pred in predictions]
    print (set(predicted_emotions))
    emo_map = {}
    for i in predicted_emotions:
        if i in emo_map:
            emo_map[i] += 1
        else:
            emo_map[i] = 1
            
    print ("FINAL EMOTION MAP ", emo_map)
    test_df['emotion'] = predicted_emotions
    test_df.to_csv('test_lr.csv', index=False)


def LR():
    # Load data
    cross_validate = False
    train_df = pd.read_csv("train.csv")

    # Preprocess data
    train_map = preprocess_train_df(train_df)

    # Encoding labels
    unique_emotions = train_df['emotions'].unique()
    emotion_to_index = {emotion: index for index, emotion in enumerate(unique_emotions)}
    index_to_emotion = {index: emotion for emotion, index in emotion_to_index.items()}
    y_train_encoded = train_df['emotions'].map(emotion_to_index).values

    # Compute class weights
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    class_weights_normalized = {cls: (sum(counts) / (len(unique) * count)) for cls, count in zip(unique, counts)}

    # TF-IDF Conversion
    documents = [value[0] for value in train_map.values()]
    tfs = [compute_tf(doc) for doc in documents]
    idf = compute_idf(documents)
    tfidf_documents = [compute_tfidf(tf, idf) for tf in tfs]

    # Prepare data
    vocabulary = set(term for doc in tfidf_documents for term in doc.keys())
    vocab_to_index = {term: i for i, term in enumerate(sorted(vocabulary))}
    # dense_tfidf = ConvertToDenseMatrix(tfidf_documents)
    dense_tfidf = ConvertToDenseMatrix2(tfidf_documents, vocab_to_index)
    np_labels = np.array(Getlabels(train_map))  # Assuming Getlabels() function converts emotions to one-hot encoded labels

    # Cross-validation
    num_features = dense_tfidf.shape[1]
    num_classes = np_labels.shape[1]
    k = 5
    if (cross_validate):
        scores = perform_cross_validation(dense_tfidf, np_labels, num_features, num_classes, k, class_weights_normalized)
        print("Cross-validated accuracy:", np.mean(scores))
        return None
    else:
        model = LRModel(num_features, num_classes)
        # model.train2(dense_tfidf, np_labels, class_weights_normalized, learning_rate=0.8, num_iterations=4000)
        model.train(dense_tfidf, np_labels, learning_rate=0.8, num_iterations=4000)

        print ('in else ready for testing')
        return model, vocab_to_index, idf

def test_and_write_NN(model, vocab_to_index, index_to_emotion, idf):
    test_df = pd.read_csv("test.csv")
    test_df = test_df.drop_duplicates(subset=['id'], keep='first')

    empty_docs_indices = test_df[test_df['text'].isnull() | (test_df['text'] == '')].index.tolist()
    if empty_docs_indices:
        print("Indices of empty or NaN documents:", empty_docs_indices)

    test_map = preprocess_test_df2(test_df)
    test_documents = [value[0] for value in test_map.values()]

    test_tfs = [compute_tf(doc) for doc in test_documents]
    # Directly use the idf from training phase, do not recompute
    test_tfidf_documents = [compute_tfidf(tf, idf) for tf in test_tfs]
    test_dense_tfidf = ConvertToDenseMatrix2(test_tfidf_documents, vocab_to_index)

    predictions = model.predict(test_dense_tfidf.T)  # Ensure correct shape for prediction
    predicted_emotions = [index_to_emotion[pred] for pred in predictions]

    # Counting the occurrences of each predicted emotion
    emo_map = {emotion: predicted_emotions.count(emotion) for emotion in set(predicted_emotions)}
    
    print(emo_map)
    test_df['emotion'] = predicted_emotions
    test_df.to_csv('test_nn.csv', index=False)

def NN():
    print ("NN")

    cross_validate = False
    # your Multi-layer Neural Network
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    train_map = {}
    test_map = {}
    st_train_df = pd.DataFrame()
    st_test_df = pd.DataFrame()
    ctr = 0
    for row in train_df.iterrows():
        train_map[row[1][0]] = [row[1][1], row[1][2]]
        
    # print (train_map)
    
    #stemming the train_map
    for key, value in train_map.items():
        sentence, emotion = value
        st_token_sent, st_token_arr = preprocess_and_stem(sentence)
        train_map[key] = [st_token_sent, emotion]
    
    # print (train_map)
        # Encoding labels
    unique_emotions = train_df['emotions'].unique()
    emotion_to_index = {emotion: index for index, emotion in enumerate(unique_emotions)}
    index_to_emotion = {index: emotion for emotion, index in emotion_to_index.items()}
    y_train_encoded = train_df['emotions'].map(emotion_to_index).values

    # Compute class weights
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    class_weights_normalized = {cls: (sum(counts) / (len(unique) * count)) for cls, count in zip(unique, counts)}


    # Suppose 'data' is your map: {id: [stemmed_sentence, emotion]}
    documents = [value[0] for value in train_map.values()]  # List of all stemmed sentences
    tfs = [compute_tf(doc) for doc in documents]
    # print (tfs)
    idf = compute_idf(documents)
    # print (idf)
    tfidf_documents = [compute_tfidf(tf, idf) for tf in tfs]
    # print (tfidf_documents)
    
    #cross validation
    labels = Getlabels(train_map)  # This is correct
    dense_tfidf = ConvertToDenseMatrix(tfidf_documents)  # Correct conversion to dense matrix
    np_labels = np.array(labels)  # Correct conversion to NumPy array for labels
    num_features = dense_tfidf.shape[1]  # Correctly obtaining the number of features
    num_classes = np_labels.shape[1]  # Correctly obtaining the number of classes

    if (cross_validate):
        k = 5  # Number of folds
        fold_size = dense_tfidf.shape[0] // k
        neuron_counts = [10, 50, 100, 150, 200]  # Specify the different numbers of neurons you want to test
        neuron_scores = {neuron_count: [] for neuron_count in neuron_counts}  # Dictionary to keep track of scores for each neuron count

        for neuron_count in neuron_counts:
            for fold in range(k):
                print(f"Training on fold {fold+1}/{k} with {neuron_count} neurons...")
                # Create validation and training sets
                start, end = fold * fold_size, (fold + 1) * fold_size
                valid_data = dense_tfidf[start:end]
                valid_labels = np_labels[start:end]
                train_data = np.concatenate([dense_tfidf[:start], dense_tfidf[end:]])
                train_labels = np.concatenate([np_labels[:start], np_labels[end:]])

                # Transpose data for the neural network
                X_train, Y_train = train_data.T, train_labels.T
                X_valid, Y_valid = valid_data.T, valid_labels.T

                # Initialize and train the neural network
                nn_model = SimpleNeuralNetwork(num_features, neuron_count, num_classes)
                # Assuming you have a constant learning rate that you have decided to use
                learning_rate = 0.7
                nn_model.train2(X_train, Y_train, learning_rate=learning_rate, class_weights = class_weights_normalized,num_iterations=2000)

                # Predict and evaluate
                predictions = nn_model.predict(X_valid)
                accuracy = np.mean(predictions == np.argmax(Y_valid, axis=0))
                neuron_scores[neuron_count].append(accuracy)

        # Plotting
        avg_scores = {neuron_count: np.mean(scores) for neuron_count, scores in neuron_scores.items()}
        plt.plot(list(avg_scores.keys()), list(avg_scores.values()), marker='o')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Average Validation Accuracy')
        plt.title('Number of Neurons vs Validation Accuracy')
        plt.show()


    else :
        X_train_full, Y_train_full = dense_tfidf.T, np_labels.T  # Assuming Getlabels() provides correct format

        nn_model_final = SimpleNeuralNetwork(input_size=num_features, hidden_size=100, output_size=num_classes)
        nn_model_final.train2(X_train_full, Y_train_full, class_weights=class_weights_normalized, num_iterations=2000, learning_rate=0.7)
        return nn_model_final, idf

    print("End of Neural Network with Cross-Validation.")
    # return nn_model




if __name__ == '__main__':
    print ("..................Beginning of Logistic Regression................")
    train_df2 = pd.read_csv("train.csv")
    unique_emotions = train_df2['emotions'].unique()  # Assuming 'emotion' is the label column in your training set
    emotion_to_index = {emotion: index for index, emotion in enumerate(unique_emotions)}
    index_to_emotion = {index: emotion for emotion, index in emotion_to_index.items()}
    # index_to_emotion
    # print (index_to_emotion)
    lr_model, vocab_to_index, idf = LR()
    test_and_write_LR(lr_model, vocab_to_index, index_to_emotion, idf)
    # test_and_write(model)
    print ("..................End of Logistic Regression................")
    # index_to_emotion = {index: emotion for emotion, index in emotion_to_index.items()}

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    nn_model, idf2 = NN()
    test_and_write_NN(nn_model, vocab_to_index, index_to_emotion, idf)

    print ("..................End of Neural Network................")
