import os
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import joblib

class Utilities:
    """
    Class with static methods to perform helper utility functions.
    """
    @staticmethod
    def load_data(loc='./dataset/', _train=False, _test=False):
        """
        Function to load the data from disk.

        Arguments:
            loc: Location of the dataset
            _train: Flag to load the training data
            _test: Flag to load the testing data

        Returns:
            Returns the dataset as a dataframe.
        """
        trainloc = loc + 'msr_paraphrase_train.txt'
        testloc = loc + 'msr_paraphrase_test.txt'

        if _train:
            return pd.read_csv(trainloc, delimiter='\t', usecols=[0,1,2,3,4])
        else:
            return pd.read_csv(testloc, delimiter='\t', usecols=[0,1,2,3,4])

    @staticmethod
    def update_data_columns(df):
        """
        Function to update the data columns. Drops Sentence ID parameters.
        Renames columns to easily recognizable names. Drops NA values.

        Arguments:
            df: Dataframe to update

        Returns: Updated dataframe
        """
        df.drop(columns=['#1 ID', '#2 ID'], inplace=True)
        df.rename(columns={'Quality':'Label','#1 String':'Sentence1','#2 String':'Sentence2'}, inplace=True)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def display_metrics(y_true, y_pred):
        """
        Function to display the metrics of the model.
        Metrics displayed: Accuracy, Precision, Recall, F1 Score.

        Arguments:
            y_true: Ground truth labels
            y_pred: Predicted labels
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')

    @staticmethod
    def display_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
        """
        Function to display the confusion matrix.

        Arguments:
            y_true: Ground truth labels
            y_pred: Predicted labels
            title: Title of the confusion matrix
        """
        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Non Paraphrase', 'Predicted Paraphrase'],
                    yticklabels=['Actual Non Paraphrase', 'Actual Paraphrase'])

        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.title(title)
        plt.show()


class ManualFeatures:
    """
    Class that handles functionalities related to manual engineered features.
    """
    def __init__(self):
        # Loading en_core_web_sm from spacy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        self.word2vec_model = None

    def tokenize_text(self, text):
        """
        Function to tokenize the text using SpaCy library. Converts text to lowercase.
        Removes stopwords, punctuations and empty tokens.

        Arguments:
            text: Text to be tokenized

        Returns: Tokenized text.
        """
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != '']
        return " ".join(tokens)

    def preprocess_dataframe(self, df):
        """
        Function to preprocess the dataframe. Applies tokenize_text() function to both the sentences.
        Stores preprocessed sentences in a new column.

        Arguments:
            df: Dataframe to be preprocessed

        Returns: Preprocessed dataframe
        """
        df['Preproc_Sentence1'] = df['Sentence1'].apply(self.tokenize_text)
        df['Preproc_Sentence2'] = df['Sentence2'].apply(self.tokenize_text)
        return df

    def get_num_tokens(self, preproc_text):
        """
        Function to get the number of tokens in the preprocessed sentence.

        Arguments:
            preproc_text: Preprocessed sentence

        Returns: Number of tokens in the preprocessed sentence
        """
        return len(preproc_text.split())

    def get_common_lemmas(self, preproc_text1, preproc_text2):
        """
        Function to get the common lemmas between two sentences.

        Arguments:
            preproc_text1: Preprocessed sentence1
            preproc_text2: Preprocessed sentence2

        Returns: Length of list of common lemmas
        """
        lemmas1 = set(preproc_text1.split())
        lemmas2 = set(preproc_text2.split())
        return len(lemmas1.intersection(lemmas2))

    def get_avg_embedding(self, tokens, embedding_model):
        """
        Function to get the average embedding of tokens using the embedding_model(here Word2Vec) provided.

        Arguments:
            tokens: List of tokens
            embedding_model: Embedding model used

        Returns: Average embedding of tokens
        """
        valid_embeddings = [embedding_model.wv[token] for token in tokens if token in embedding_model.wv]
        return np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(embedding_model.vector_size)

    def compute_cosine_similarity(self, vec1, vec2):
        """
        Function to compute the cosine similarity between two vectors.

        Arguments:
            vec1: First vector
            vec2: Second vector

        Returns: Cosine similarity between two vectors
        """
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    def compute_features(self, df, embedding_model):
        """
        Function to compute the features in the given dataframe. Computes the length of both the sentences,
        difference in length between both the sentences, number of common words in both the sentences,
        cosine similarity between both the sentences.

        Arguments:
            df: Dataframe to be processed
            embedding_model: Embedding model used

        Returns: dataframe with computed features in new columns.
        """
        df['text1_len'] = df['Preproc_Sentence1'].apply(self.get_num_tokens)
        df['text2_len'] = df['Preproc_Sentence2'].apply(self.get_num_tokens)
        df['diff_len'] = (df['text1_len'] - df['text2_len']).abs()

        df['common_lemma'] = df.apply(
            lambda row: self.get_common_lemmas(
                row['Preproc_Sentence1'],
                row['Preproc_Sentence2']), axis=1)

        df['pair_sim'] = df.apply(
            lambda row: self.compute_cosine_similarity(
                self.get_avg_embedding(row['Preproc_Sentence1'].split(), embedding_model),
                self.get_avg_embedding(row['Preproc_Sentence2'].split(), embedding_model)
            ),
            axis=1
        )

        return df

    def run(self):
        """
        Function to run the entire pipeline on the manually engineered features.
        """
        train_data = Utilities.load_data(_train=True)
        test_data = Utilities.load_data(_train=False)

        train_data = Utilities.update_data_columns(train_data)
        test_data = Utilities.update_data_columns(test_data)

        train_data = self.preprocess_dataframe(train_data)
        test_data = self.preprocess_dataframe(test_data)

        # Check if Word2Vec model exists
        word2vec_model_path = './models/word2vec_model.bin'
        if os.path.exists(word2vec_model_path):
            self.word2vec_model = Word2Vec.load(word2vec_model_path)
            print("Loaded existing Word2Vec model.")
        else:
            # Else create a word2vec model trained on the dataset loaded.
            sentences = train_data['Preproc_Sentence1'].tolist() + train_data['Preproc_Sentence2'].tolist()
            self.word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
            self.word2vec_model.save(word2vec_model_path)
            print("Trained and saved new Word2Vec model.")

        train_data = self.compute_features(train_data, self.word2vec_model)
        test_data = self.compute_features(test_data, self.word2vec_model)

        # Features to utilize for prediction.
        features = ['text1_len', 'text2_len', 'diff_len', 'common_lemma', 'pair_sim']

        X_train = train_data[features]
        y_train = train_data['Label']
        X_test = test_data[features]
        y_test = test_data['Label']

        # Check if GradientBoosting model exists
        gb_model_path = './models/gb_model_manual_features.pkl'
        if os.path.exists(gb_model_path):
            gb_model = joblib.load(gb_model_path)
            print("Loaded existing GradientBoosting model trained on manual features.")
        else:
            # Else fit and train a gradient boosting classifier and save it
            gb_model = GradientBoostingClassifier()
            gb_model.fit(X_train, y_train)
            joblib.dump(gb_model, gb_model_path)
            print("Trained and saved new GradientBoosting model on manual features.")

        y_pred_gb = gb_model.predict(X_test)

        Utilities.display_metrics(y_test, y_pred_gb)
        Utilities.display_confusion_matrix(y_test, y_pred_gb, title='Gradient Boosting using manually computed features')


class LSTMFeatures:
    """
    Class that handles the functionalities related to building the LSTM model for Paraphrase Identification.
    """
    def __init__(self):
        # Setting hyperparameters
        self.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
        self.VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "14000"))
        self.EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "200"))
        self.HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
        self.NEPOCH = int(os.environ.get("NEPOCH", "20"))
        self.tokenizer = Tokenizer(num_words=self.VOCABULARY_SIZE, oov_token="<OOV>")

    def tokenize_and_pad(self, df, max_len=50):
        """
        Function to tokenize the dataframe and pad it to max_len.
        Arguments:
            df: Dataframe to be tokenized
            max_len: Max length of sentence

        Returns: Tokenized dataframe
        """
        seq1 = self.tokenizer.texts_to_sequences(df['Sentence1'].tolist())
        seq2 = self.tokenizer.texts_to_sequences(df['Sentence2'].tolist())
        seq1_padded = pad_sequences(seq1, maxlen=max_len, padding='post', truncating='post')
        seq2_padded = pad_sequences(seq2, maxlen=max_len, padding='post', truncating='post')
        return seq1_padded, seq2_padded

    def build_lstm_model(self):
        """
        Function to build the LSTM model with the parameters.
        Model uses binary_crossentropy loss function and Adam optimizer.
        Returns: LSTM model
        """
        model = Sequential([
            Embedding(input_dim=self.VOCABULARY_SIZE, output_dim=self.EMBEDDING_DIM, input_length=50),
            Bidirectional(LSTM(self.HIDDEN_DIM, return_sequences=False)),
            Dense(self.HIDDEN_DIM, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE), metrics=['accuracy'])
        return model

    def run(self):
        """
        Function to run the entire pipeline to predict using LSTM model.
        """
        train_data = Utilities.load_data(_train=True)
        test_data = Utilities.load_data(_test=True)

        train_data = Utilities.update_data_columns(train_data)
        test_data = Utilities.update_data_columns(test_data)

        self.tokenizer.fit_on_texts(train_data['Sentence1'].tolist() + train_data['Sentence2'].tolist())

        seq1_train, seq2_train = self.tokenize_and_pad(train_data)
        seq1_test, seq2_test = self.tokenize_and_pad(test_data)

        lstm_model_path = './models/lstm_model.h5'
        # Loads the LSTM model if it is already present.
        if os.path.exists(lstm_model_path):
            lstm_model = load_model(lstm_model_path)
            print("Loaded existing LSTM model.")
        else:
            # Else creates an LSTM model, trains it and saves it.
            lstm_model = self.build_lstm_model()
            y_train_binary = np.array(train_data['Label']).reshape(-1, 1)
            lstm_model.fit(seq1_train, y_train_binary, epochs=self.NEPOCH, batch_size=32, validation_split=0.2)
            lstm_model.save(lstm_model_path)
            print("Trained and saved new LSTM model.")

        # Get the feature embeddings for Sentence 1 and Sentence 2 from the LSTM model
        train_embeddings1, train_embeddings2 = lstm_model.predict(seq1_train), lstm_model.predict(seq2_train)
        test_embeddings1, test_embeddings2 = lstm_model.predict(seq1_test), lstm_model.predict(seq2_test)

        # Concatenate each emebeddings.
        train_features = np.concatenate([train_embeddings1, train_embeddings2], axis=1)
        test_features = np.concatenate([test_embeddings1, test_embeddings2], axis=1)

        gb_model_lstm_path = './models/gb_model_lstm_features.pkl'
        # Loads the Gradient Boosting classifer for LSTM if already present
        if os.path.exists(gb_model_lstm_path):
            gb_model_lstm = joblib.load(gb_model_lstm_path)
            print("Loaded existing GradientBoosting model trained on LSTM features.")
        else:
            # Else trains a Gradient Boosting Classifier model and saves it.
            gb_model_lstm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
            gb_model_lstm.fit(train_features, np.ravel(y_train_binary))
            joblib.dump(gb_model_lstm, gb_model_lstm_path)
            print("Trained and saved new GradientBoosting model on LSTM features.")

        y_pred_gb = gb_model_lstm.predict(test_features)

        Utilities.display_metrics(test_data['Label'], y_pred_gb)
        Utilities.display_confusion_matrix(test_data['Label'], y_pred_gb, title='Gradient Boosting using LSTM extracted features')


if __name__ == "__main__":

    # Load the dataset
    train_data = Utilities.load_data(_train=True)
    test_data = Utilities.load_data(_test=True)

    # Train and predict using Manually Engineered Features
    manual_features = ManualFeatures()
    manual_features.run()

    # Train and predict using LSTM Features
    lstm_features = LSTMFeatures()
    lstm_features.run()
