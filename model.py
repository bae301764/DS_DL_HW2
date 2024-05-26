import torch.nn as nn
import torch.nn.functional as F
import torch

class CharRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_class, n_layers):
        super(CharRNN, self).__init__()
        # write your codes here
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_class = n_class

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, int(self.hidden_size/4))
        self.fc2 = nn.Linear(int(self.hidden_size/4), int(self.hidden_size/16))
        self.fc3 = nn.Linear(int(self.hidden_size/16), self.n_class)
        self.activation = nn.GELU()

    def forward(self, input, hidden):
        x = self.embedding(input)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.activation(self.fc(out))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return initial_hidden
    

class CharLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_class, n_layers):
        super(CharLSTM, self).__init__()
        # write your codes here
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_class = n_class
        self.activation = nn.GELU()

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, int(self.hidden_size/4))
        self.fc2 = nn.Linear(int(self.hidden_size/4), int(self.hidden_size/16))
        self.fc3 = nn.Linear(int(self.hidden_size/16), self.n_class)

    def forward(self, input, hidden):
        x = self.embedding(input)
        out, hidden = self.LSTM(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.activation(self.fc(out))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out, hidden

    def init_hidden(self, batch_size):
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size))
        
        return initial_hidden
    