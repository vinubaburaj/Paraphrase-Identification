import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from gensim.models import Word2Vec

from ParaphraseIdentification import ManualFeatures, LSTMFeatures, Utilities


class Demo:
    def __init__(self):
        self.model_type = "LSTM"
        self.manual_gb_model = None
        self.lstm_model = None
        self.lstm_gb_model = None

    # def choose_model(self):
    #     print("Choose a model to run:")
    #     print("1. Manual Features")
    #     print("2. LSTM Features")
    #     choice = input("Enter 1 or 2: ")
    #
    #     if choice == "1":
    #         self.model_type = "Manual"
    #     elif choice == "2":
    #         self.model_type = "LSTM"
    #     else:
    #         print("Invalid choice. Please enter 1 or 2.")
    #         self.choose_model()

    def load_models(self):
        if self.model_type == "Manual":
            self.manual_gb_model = joblib.load('models/gb_model_manual_features.pkl')
        elif self.model_type == "LSTM":
            self.lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
            self.lstm_gb_model = joblib.load('models/gb_model_lstm_features.pkl')

    def run_demo(self):
        # self.choose_model()
        self.load_models()

        print("Welcome to this demo of Paraphrase Identification using LSTM! Modify the 'sample_sentences' to test for different inputs")

        sample_sentences = [
            ("The cat is sitting on the mat.", "A cat is resting on the mat."),
            ("She is writing a letter.", "She is composing a letter."),
            ("The weather is quite cold today.", "It's pretty chilly outside today."),
            ("He quickly finished his homework.", "He completed his homework in no time."),
            ("They went to the store to buy groceries.", "They visited the shop to purchase food items."),
            ("The dog is barking loudly.", "She is reading a book in the library."),
            ("He is playing basketball at the park.", "The car broke down on the highway."),
            ("The sun is shining brightly in the sky.", "She enjoys painting abstract art in her free time."),
            ("They are discussing the new project at work.", "A bird is building a nest in the tree."),
            ("He went to school", "They are at the playground."),
        ]

        # if self.model_type == "Manual":
        #     manual_features = ManualFeatures()
        #
        #     word2vec_model_path = './models/word2vec_model.bin'
        #
        #     word2vec_model = Word2Vec.load(word2vec_model_path)
        #
        #     sample_df = pd.DataFrame(sample_sentences, columns=["Sentence1", "Sentence2"])
        #     sample_df = manual_features.preprocess_dataframe(sample_df)
        #     sample_df = manual_features.compute_features(sample_df, word2vec_model)
        #
        #     features = ['text1_len', 'text2_len', 'diff_len', 'common_lemma', 'pair_sim']
        #     # features = ['common_lemma', 'pair_sim']
        #     X_sample = sample_df[features]
        #
        #     predictions = self.manual_gb_model.predict(X_sample)
        #     print("Predictions using Manual Features:")
        #     print(predictions)

        if self.model_type == "LSTM":
            lstm_features = LSTMFeatures()
            lstm_features.lstm_model = self.lstm_model

            train_data = Utilities.load_data(_train=True)
            train_data = Utilities.update_data_columns(train_data)


            sample_df = pd.DataFrame(sample_sentences, columns=["Sentence1", "Sentence2"])


            tokenizer = lstm_features.tokenizer
            tokenizer.fit_on_texts(train_data["Sentence1"].to_list() + train_data["Sentence2"].to_list())

            seq1_sample, seq2_sample = lstm_features.tokenize_and_pad(sample_df)

            sample_embeddings1, sample_embeddings2 = lstm_features.lstm_model.predict(seq1_sample), lstm_features.lstm_model.predict(seq2_sample)

            sample_features = np.concatenate([sample_embeddings1, sample_embeddings2], axis=1)
            predictions = self.lstm_gb_model.predict(sample_features)

            print("Predictions using LSTM Features:")
            print("=================================================")
            for i in range(predictions.shape[0]):
                print(sample_sentences[i][0])
                print(sample_sentences[i][1])
                paraphraseYesNo: str = "Yes" if predictions[i] == 1 else "No"
                print("Paraphrase: ",paraphraseYesNo)
                print("=================================================")

if __name__ == "__main__":
    demo = Demo()
    demo.run_demo()
