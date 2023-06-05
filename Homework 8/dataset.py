# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# other stuff
import os
import sys
import gzip
import pickle
import random
import gensim.downloader as genapi
from gensim.models import KeyedVectors 
import numpy as np 

class SentimentDataset(Dataset):

    """
    MOST OF THE PART OF THIS CLASS WAS TAKEN FROM PROF. KAK'S SentimentAnalysisDataset CLASS! 
    I REMOVE THE UNNECESSARY COMPONENTS AND MAKE SOME CHANGES FOR THE HOMEWORK
    """

    def __init__(self, mode="train", path="data"):
        super(SentimentDataset, self).__init__()

        self.path = path
        self.mode = mode
        # get the word2vec pre-trained embeddings 
        if os.path.exists('vectors.kv'):
            self.word_vectors = KeyedVectors.load('vectors.kv')
        else: 
            self.word_vectors = genapi.load("word2vec-google-news-300")   
            self.word_vectors.save('vectors.kv') 
        # dataset file name
        file_name = f"sentiment_dataset_{mode}_400.tar.gz"
        f = gzip.open(os.path.join(path, file_name), "rb")
        dataset = f.read()
        # get positive & negative reviews and vocabulary
        if sys.version_info[0] == 3:
            self.pos_reviews, self.neg_reviews, self.vocab = pickle.loads(dataset, encoding='latin1')
        else:
            self.pos_reviews, self.neg_reviews, self.vocab = pickle.loads(dataset)
        # get categories
        self.categories = sorted(list(self.pos_reviews.keys()))
        # positive negative review category sizes
        self.cat_sizes_pos = {category: len(self.pos_reviews[category]) for category in self.categories}
        self.cat_sizes_neg = {category: len(self.neg_reviews[category]) for category in self.categories}
        # dataset consisting of pos./neg. reviews
        self.indexed_dataset = []
        for category in self.pos_reviews:
            for review in self.pos_reviews[category]:
                self.indexed_dataset.append([review, category, 1])
        for category in self.neg_reviews:
            for review in self.neg_reviews[category]:
                self.indexed_dataset.append([review, category, 0])
        # shuffle the dataset
        random.shuffle(self.indexed_dataset)

    def get_vocab_size(self):
        return len(self.vocab)

    # gets one-hot vector for the given word
    def one_hotvec_for_word(self, word):
        word_index = self.vocab.index(word)
        hotvec = torch.zeros(1, len(self.vocab))
        hotvec[0, word_index] = 1
        return hotvec

    def sentiment_to_tensor(self, sentiment):
        """
        Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
        sentiment and 1 for positive sentiment.  We need to pack this value in a
        two-element tensor.
        """
        sentiment_tensor = torch.zeros(2)
        if sentiment == 1:
            sentiment_tensor[1] = 1
        elif sentiment == 0:
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def review_to_tensor(self, review):
        list_of_embeddings = []
        for i, word in enumerate(review):
            if word in self.word_vectors.key_to_index:
                embedding = self.word_vectors[word]
                list_of_embeddings.append(np.array(embedding))
            else:
                next
        review_tensor = torch.FloatTensor(np.array(list_of_embeddings))
        return review_tensor

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, idx):
        sample = self.indexed_dataset[idx]
        review = sample[0]
        review_category = sample[1]
        review_sentiment = sample[2]
        review_sentiment = self.sentiment_to_tensor(review_sentiment)
        review_tensor = self.review_to_tensor(review)
        category_index = self.categories.index(review_category)
        sample = {'review': review_tensor,
                  'category': category_index,  # should be converted to tensor, but not yet used
                  'sentiment': review_sentiment}
        return sample

# test code
if __name__ == "__main__":

    train_data = SentimentDataset()
    train_loader = DataLoader(train_data, batch_size=1)
    train_loader = iter(train_loader)
    print("Data size:", len(train_data))
