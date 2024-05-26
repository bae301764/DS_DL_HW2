# import some packages you need here
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        os.chdir(r'/home/iai3/Desktop/jeongwon/2024 딥러닝 과제/과제2')
        with open(input_file, 'r', encoding='utf-8') as file:
            self.data = file.read()
        self.sequnce_length = 30
        self.char_vocab = sorted(list(set(self.data)))
        self.index_char = dict((index, char) for index, char in enumerate(self.char_vocab))
        self.char_index = dict((char, index) for index, char in enumerate(self.char_vocab))
        self.text_index = [self.char_index[char] for char in self.data]

    def __len__(self):
        return len(self.data)-self.sequnce_length
    
    def __getitem__(self, idx):
        input = torch.tensor(self.text_index[idx : idx+self.sequnce_length])
        target = torch.tensor(self.text_index[idx+1 : idx+self.sequnce_length+1])
        return input, target

if __name__ == '__main__':
    dataset = Shakespeare('./shakespeare_train.txt')
    input, target = dataset[0]
    # 첫 번째 데이터 샘플 출력
    print(dataset.data[:33])
    print(dataset.index_char)
    print(input)  
    print(target)