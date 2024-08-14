import unittest
import pandas as pd
import numpy as np
from ParaphraseIdentification import Utilities, ManualFeatures, LSTMFeatures
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

class TestUtilities(unittest.TestCase):

    def test_load_data(self):
        df_train = Utilities.load_data(_train=True)
        df_test = Utilities.load_data(_test=True)
        self.assertIsInstance(df_train, pd.DataFrame)
        self.assertIsInstance(df_test, pd.DataFrame)

    def test_update_data_columns(self):
        df = pd.DataFrame({
            '#1 ID': [1, 2],
            '#2 ID': [3, 4],
            '#1 String': ["The car is red.", "He reads a book."],
            '#2 String': ["The vehicle is red.", "He is reading a book."],
            'Quality': [1, 0]
        })
        updated_df = Utilities.update_data_columns(df)
        self.assertNotIn('#1 ID', updated_df.columns)
        self.assertNotIn('#2 ID', updated_df.columns)
        self.assertIn('Sentence1', updated_df.columns)
        self.assertIn('Sentence2', updated_df.columns)
        self.assertIn('Label', updated_df.columns)
        self.assertEqual(updated_df.isna().sum().sum(), 0)


class TestManualFeatures(unittest.TestCase):

    def setUp(self):
        self.manual_features = ManualFeatures()
        self.test_df = pd.DataFrame({
            'Sentence1': ["The car is red.", "He reads a book."],
            'Sentence2': ["The vehicle is red.", "He is reading a book."]
        })
        self.test_df = self.manual_features.preprocess_dataframe(self.test_df)

        # Mock Word2Vec model for testing purposes
        sentences = [["car", "red"], ["vehicle", "red"], ["reads", "book"], ["reading", "book"]]
        self.word2vec_model = Word2Vec(sentences, vector_size=10, min_count=1)

    def test_tokenize_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.manual_features.tokenize_text(text)
        self.assertIsInstance(tokens, str)
        self.assertTrue(len(tokens.split()) > 0)

    def test_preprocess_dataframe(self):
        df = pd.DataFrame({
            'Sentence1': ["The car is red.", "He reads a book."],
            'Sentence2': ["The vehicle is red.", "He is reading a book."]
        })
        preprocessed_df = self.manual_features.preprocess_dataframe(df)
        self.assertIn('Preproc_Sentence1', preprocessed_df.columns)
        self.assertIn('Preproc_Sentence2', preprocessed_df.columns)

    def test_get_num_tokens(self):
        num_tokens = self.manual_features.get_num_tokens("This is a sentence.")
        self.assertEqual(num_tokens, 4)

    def test_get_common_lemmas(self):
        common_lemmas = self.manual_features.get_common_lemmas("car red", "vehicle red")
        self.assertEqual(common_lemmas, 1)

    def test_compute_cosine_similarity(self):
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])
        similarity = self.manual_features.compute_cosine_similarity(vec1, vec2)
        expected_similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        self.assertEqual(similarity, expected_similarity)

    def test_compute_features(self):
        computed_df = self.manual_features.compute_features(self.test_df, self.word2vec_model)
        self.assertIn('pair_sim', computed_df.columns)


class TestLSTMFeatures(unittest.TestCase):

    def setUp(self):
        self.lstm_features = LSTMFeatures()
        self.test_df = pd.DataFrame({
            'Sentence1': ["The car is red.", "He reads a book."],
            'Sentence2': ["The vehicle is red.", "He is reading a book."]
        })

    def test_tokenize_and_pad(self):
        self.lstm_features.tokenizer.fit_on_texts(self.test_df['Sentence1'].tolist() + self.test_df['Sentence2'].tolist())
        seq1_padded, seq2_padded = self.lstm_features.tokenize_and_pad(self.test_df)
        self.assertEqual(seq1_padded.shape[1], 50)
        self.assertEqual(seq2_padded.shape[1], 50)



if __name__ == '__main__':
    unittest.main()
