#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
Algorithm design:

CLEANING
In the method => tokenise(), all the punctuations are removed by importing string.punctuations
i.e !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
After removing the punctuations, digits were also removed but eventually that had an 
negative effect on the weighted average hence commented that part.
Common stopwords inspired from nltk english library are added in the stopWords dictionary.
In post processing, infrequent words were removed but did not have any positive effect on the 
weighted average and also slowed the whole process. Hence, that part is commented.


RANDOMIZATION
torch.manual_seed() is used for better reproducibility as this controls the sources of randomness.
Also, torch.backends.cudnn.deterministic is set as True to avoid nondeterministic algorithms.
wordVectors' dimension for GloVe is finally chosen as 200, tried with all the different possible 
values such as 50,100,200,300 but best results are obtained with dim=200


OPTIMIZATION
Default value for optimizer is not chosen as Adam’s method is considered as a method of Stochastic 
Optimization. It is a technique implementing adaptive learning rate. Whereas in normal SGD (default) 
the learning rate has an equivalent type of effect for all the weights/parameters of the model.
Number of epochs is reduced to 6 from default value of 10, as it reduces the chance of overfitting 
and produces better accuracy for this particular dataset in the assignment.


ARCHITECTURE
The main architecture has regularization dropout, linear model and GRU model to predict ratings and category.
For this problem, we needed to form a neural network with a memory and not just a regular RNN or CNN model.
Firstly LSTM was selected due to its popularity in text classification as it has feedback connections.
The accuracy with both GRU and LSTM was similar. But after doing more research GRU was selected over LSTM 
as it couples and forgets as well as input gates. GRU uses less training parameters and therefore uses less
memory, executes faster and trains faster than LSTM. 


PROCESS
Firstly, the mean of the word embeddings is calculated and fed into the fully connected linear model.
Then dropout is fed into the GRU model, the output of the GRU models helps in predicting ratings when fed 
into a fully connected linear layer model.
Now the output of the GRU model and word embeddings are concatenated. Now, the result is fed into two fully 
connected linear layers to finally predict the category.
The predicted labels with highest probability were selected and converted back to long integers.
Now, as we have the multilabel problem in hand, the loss is calculated by cross entropy. The total loss is 
calculated by summation of the individual loss of ratings and category.


"""

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################




import string
import numpy as np
from torchtext.vocab import GloVe
import torch.optim as toptim
import torch.nn as tnn
import torch
import random
def setup_seed(seed):
    """
    reduces randomness and increase reproducibilty
    """
    # fix random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # avoiding nondeterministic algorithms
    torch.backends.cudnn.deterministic = True


# random number -- 1000
setup_seed(1000)


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    # removing punctuations -- string.punctuation
    translator = str.maketrans(string.punctuation, " "*len(string.punctuation))
    sample = sample.translate(translator)
    # tokenizing the string
    processed = sample.split()

    return processed


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # removing digits
    # translator = str.maketrans(string.digits, " "*len(string.digits))
    # sample = " ".join(sample).translate(translator)
    # processed = sample.split()

    return sample
    # return processed


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    # Remove infrequent words from batch
    # vocabCount = vocab.freqs
    # vocabITOS = vocab.itos
    # for sentence in batch:
    #     for j, word in enumerate(sentence):
    #         if vocabCount[vocabITOS[word]] < 3:
    #             sentence[j] = -1
    return batch


# stopWords = {}
# common stopwords -- inspired by nltk english stopwords
stopWords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
             "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
             "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
             "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
             "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
             "such", "only", "own", "same", "so", "than", "too", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll", "m"}

# best results with dim=200
wordVectors = GloVe(name='6B', dim=200)


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # return the position of the highest predicted value
    ratingOutput = torch.argmax(ratingOutput, 1, keepdim=True).long()
    categoryOutput = torch.argmax(categoryOutput, 1, keepdim=True).long()

    return ratingOutput, categoryOutput


################################################################################
###################### The following determines the model ######################
################################################################################


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        # dropout
        self.dropout = tnn.Dropout(0.2)
        # GRU model
        self.gru_model = tnn.GRU(input_size=200, hidden_size=128,
                                 num_layers=1, batch_first=True, bidirectional=True)
        # linear model i.e fully connected layer for vocabulary embedding
        self.linear_model_word_embedding = tnn.Linear(
            in_features=200, out_features=200, bias=True)
        # linear model i.e fully connected layer for rating prediction
        self.linear_model_rating = tnn.Linear(
            in_features=128 * 2, out_features=2, bias=True)
        # linear model i.e fully connected layer for category prediction
        self.linear_model_category_hidden = tnn.Linear(
            in_features=128 * 2 + 200, out_features=128, bias=True)
        self.linear_model_category = tnn.Linear(
            in_features=128, out_features=5, bias=True)

    def forward(self, input, length):
        # calculate mean of vocab embeddings
        mean_word_embedding = torch.mean(input, dim=1)
        # feeding mean_word_embedding into a linear model i.e fully connected layer
        lm_word_embedding = self.linear_model_word_embedding(
            mean_word_embedding)
        word_embedding_out = torch.relu(lm_word_embedding)
        # dropout
        dropout = self.dropout(input)

        # feeding dropout into a GRU model
        gru_out = self.gru_model(dropout)[0][:, -1, :]
        # predicting rating by linear model
        ratingOutput = self.linear_model_rating(gru_out)
        # predicting category by concatenating word embedding and gru_model output
        input_category = torch.cat([gru_out, word_embedding_out], dim=1)
        # feeding input_category into 2 fully connected layers
        lm_category_hidden = self.linear_model_category_hidden(
            input_category)
        category_out = torch.relu(lm_category_hidden)
        categoryOutput = self.linear_model_category(category_out)

        return ratingOutput, categoryOutput


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # cross entropy
        self.loss_func = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # adding both the losses calculated by cross entropy
        return self.loss_func(ratingOutput, ratingTarget) + self.loss_func(categoryOutput, categoryTarget)


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 6
optimiser = toptim.Adam(net.parameters(), lr=0.001)
