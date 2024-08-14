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
    @staticmethod
    def load_data(loc='./dataset/', _train=False, _test=False):
        trainloc = loc + 'msr_paraphrase_train.txt'
        testloc = loc + 'msr_paraphrase_test.txt'

        if _train:
            return pd.read_csv(trainloc, delimiter='\t', usecols=[0,1,2,3,4])
        else:
            return pd.read_csv(testloc, delimiter='\t', usecols=[0,1,2,3,4])

    @staticmethod
    def update_data_columns(df):
        df.drop(columns=['#1 ID', '#2 ID'], inplace=True)
        df.rename(columns={'Quality':'Label','#1 String':'Sentence1','#2 String':'Sentence2'}, inplace=True)
        return df

    @staticmethod
    def display_metrics(y_true, y_pred):
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
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        self.word2vec_model = None

    def tokenize_text(self, text):
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != '']
        return " ".join(tokens)

    def preprocess_dataframe(self, df):
        df.dropna(inplace=True)
        df['Preproc_Sentence1'] = df['Sentence1'].apply(self.tokenize_text)
        df['Preproc_Sentence2'] = df['Sentence2'].apply(self.tokenize_text)
        return df

    def get_num_tokens(self, preproc_text):
        return len(preproc_text.split())

    def get_common_lemmas(self, preproc_text1, preproc_text2):
        lemmas1 = set(preproc_text1.split())
        lemmas2 = set(preproc_text2.split())
        return len(lemmas1.intersection(lemmas2))

    def get_avg_embedding(self, tokens):
        valid_embeddings = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
        return np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(self.word2vec_model.vector_size)

    def compute_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    def compute_features(self, df):
        df['text1_len'] = df['Preproc_Sentence1'].apply(self.get_num_tokens)
        df['text2_len'] = df['Preproc_Sentence2'].apply(self.get_num_tokens)
        df['diff_len'] = (df['text1_len'] - df['text2_len']).abs()

        df['common_lemma'] = df.apply(
            lambda row: self.get_common_lemmas(
                row['Preproc_Sentence1'],
                row['Preproc_Sentence2']), axis=1)

        df['pair_sim'] = df.apply(
            lambda row: self.compute_cosine_similarity(
                self.get_avg_embedding(row['Preproc_Sentence1'].split()),
                self.get_avg_embedding(row['Preproc_Sentence2'].split())
            ),
            axis=1
        )

        return df

    def run(self):

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
            sentences = train_data['Preproc_Sentence1'].tolist() + train_data['Preproc_Sentence2'].tolist()
            self.word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
            self.word2vec_model.save(word2vec_model_path)
            print("Trained and saved new Word2Vec model.")

        train_data = self.compute_features(train_data)
        test_data = self.compute_features(test_data)

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
            gb_model = GradientBoostingClassifier()
            gb_model.fit(X_train, y_train)
            joblib.dump(gb_model, gb_model_path)
            print("Trained and saved new GradientBoosting model on manual features.")

        y_pred_gb = gb_model.predict(X_test)

        Utilities.display_metrics(y_test, y_pred_gb)
        Utilities.display_confusion_matrix(y_test, y_pred_gb, title='Gradient Boosting using manually computed features')


class LSTMFeatures:
    def __init__(self):
        self.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
        self.VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "14000"))
        self.EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "200"))
        self.HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
        self.NEPOCH = int(os.environ.get("NEPOCH", "20"))
        self.tokenizer = Tokenizer(num_words=self.VOCABULARY_SIZE, oov_token="<OOV>")

    def tokenize_and_pad(self, df, max_len=50):
        seq1 = self.tokenizer.texts_to_sequences(df['Preproc_Sentence1'].tolist())
        seq2 = self.tokenizer.texts_to_sequences(df['Preproc_Sentence2'].tolist())
        seq1_padded = pad_sequences(seq1, maxlen=max_len, padding='post', truncating='post')
        seq2_padded = pad_sequences(seq2, maxlen=max_len, padding='post', truncating='post')
        return seq1_padded, seq2_padded

    def build_lstm_model(self):
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
        train_data = Utilities.load_data(_train=True)
        test_data = Utilities.load_data(_test=True)

        train_data = Utilities.update_data_columns(train_data)
        test_data = Utilities.update_data_columns(test_data)

        manual_features = ManualFeatures()
        train_data = manual_features.preprocess_dataframe(train_data)
        test_data = manual_features.preprocess_dataframe(test_data)

        self.tokenizer.fit_on_texts(train_data['Preproc_Sentence1'].tolist() + train_data['Preproc_Sentence2'].tolist())

        seq1_train, seq2_train = self.tokenize_and_pad(train_data)
        seq1_test, seq2_test = self.tokenize_and_pad(test_data)

        lstm_model_path = './models/lstm_model.h5'
        if os.path.exists(lstm_model_path):
            lstm_model = load_model(lstm_model_path)
            print("Loaded existing LSTM model.")
        else:
            lstm_model = self.build_lstm_model()
            y_train_binary = np.array(train_data['Label']).reshape(-1, 1)
            lstm_model.fit(seq1_train, y_train_binary, epochs=self.NEPOCH, batch_size=32, validation_split=0.2)
            lstm_model.save(lstm_model_path)
            print("Trained and saved new LSTM model.")

        train_embeddings1, train_embeddings2 = lstm_model.predict(seq1_train), lstm_model.predict(seq2_train)
        test_embeddings1, test_embeddings2 = lstm_model.predict(seq1_test), lstm_model.predict(seq2_test)

        train_features = np.concatenate([train_embeddings1, train_embeddings2], axis=1)
        test_features = np.concatenate([test_embeddings1, test_embeddings2], axis=1)

        gb_model_lstm_path = './models/gb_model_lstm_features.pkl'
        if os.path.exists(gb_model_lstm_path):
            gb_model_lstm = joblib.load(gb_model_lstm_path)
            print("Loaded existing GradientBoosting model trained on LSTM features.")
        else:
            gb_model_lstm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
            gb_model_lstm.fit(train_features, np.ravel(y_train_binary))
            joblib.dump(gb_model_lstm, gb_model_lstm_path)
            print("Trained and saved new GradientBoosting model on LSTM features.")

        y_pred_gb = gb_model_lstm.predict(test_features)

        Utilities.display_metrics(test_data['Label'], y_pred_gb)
        Utilities.display_confusion_matrix(test_data['Label'], y_pred_gb, title='Gradient Boosting using LSTM extracted features')


if __name__ == "__main__":
    train_data = Utilities.load_data(_train=True)
    test_data = Utilities.load_data(_test=True)

    manual_features = ManualFeatures()
    manual_features.run()

    lstm_features = LSTMFeatures()
    lstm_features.run()
