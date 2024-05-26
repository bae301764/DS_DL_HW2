import dataset
import numpy as np
from model import CharRNN, CharLSTM
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from tqdm import tqdm
from generate import generate
import os
import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    trn_loss = 0
    train_count = 0
    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))

        if isinstance(hidden, tuple):
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)

        output, hidden = model(inputs, hidden)
        hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
        loss = criterion(output.view(-1, model.n_class), targets.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += (loss.item()*inputs.size(0))
        train_count += inputs.size(0)

    trn_loss = trn_loss/train_count

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """
    model.eval()

    with torch.no_grad():
        val_loss = 0
        val_total = 0

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))

            if isinstance(hidden, tuple):
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)
    

            outputs, hidden = model(inputs, hidden)
            hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

            loss = criterion(outputs.view(-1, model.n_class), targets.view(-1))
            val_loss += loss.item() * inputs.size(0)
            val_total += targets.size(0)

    val_loss = val_loss / val_total

    return val_loss

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    os.chdir('/home/iai3/Desktop/jeongwon/2024 딥러닝 과제/과제2')
    device = torch.device('cuda')

    seed = 5
    gpuid = 0
    np.random.seed(seed)
    torch.cuda.set_device(gpuid)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    data = dataset.Shakespeare('shakespeare_train.txt')
    dataset_idx = list(range(len(data)))
    np.random.shuffle(dataset_idx)
    valid_idx = int(0.2*len(data))
    train_idx, val_idx = dataset_idx[valid_idx:], dataset_idx[:valid_idx]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)    

    batch_size = 256
    train_loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=8, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=8, sampler=val_sampler, drop_last=True)
    
    input_size = len(data.char_vocab)
    embedding_size = 128
    hidden_size = 256
    output_size = len(data.char_vocab)
    n_layers = 2

    model_RNN = CharRNN(input_size, embedding_size, hidden_size, output_size, n_layers).to(device)
    model_LSTM = CharLSTM(input_size, embedding_size, hidden_size, output_size, n_layers).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_RNN = torch.optim.AdamW(model_RNN.parameters(), lr=0.001, weight_decay=0.001)
    optimizer_LSTM = torch.optim.AdamW(model_LSTM.parameters(), lr=0.001, weight_decay=0.001)

    epochs = 250
    rnn_valid_best_loss, lstm_valid_best_loss = float('inf'), float('inf')
    rnn_trainloss, rnn_validloss, lstm_trainloss, lstm_validloss = [], [], [], []

    dir = './model_weight/RNN'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = './model_weight/LSTM'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    
    for epoch in tqdm(range(epochs)):
        train_loss = train(model_RNN, train_loader, device, criterion, optimizer_RNN)
        valid_loss = validate(model_RNN, val_loader, device, criterion)
        rnn_trainloss.append(train_loss)
        rnn_validloss.append(valid_loss)

        if valid_loss < rnn_valid_best_loss:
            rnn_valid_best_loss = valid_loss
            print(f'best rnn loss : {round(rnn_valid_best_loss, 4)}')
            best_RNN_weight = model_RNN.state_dict()
            torch.save(best_RNN_weight, f'./model_weight/RNN/best_model_epoch2.pth')


        train_loss = train(model_LSTM, train_loader, device, criterion, optimizer_LSTM)
        valid_loss = validate(model_LSTM, val_loader, device, criterion)
        lstm_trainloss.append(train_loss)
        lstm_validloss.append(valid_loss)

        if valid_loss < lstm_valid_best_loss:
            lstm_valid_best_loss = valid_loss
            print(f'best lstm loss : {round(lstm_valid_best_loss, 4)}')
            best_LSTM_weight = model_LSTM.state_dict()
            torch.save(best_LSTM_weight, f'./model_weight/LSTM/best_model_epoch2.pth')

    # validation을 이용한 성능 비교
    plt.plot(rnn_validloss, color='red', label='RNN')
    plt.plot(lstm_validloss, color='blue', label='LSTM')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.savefig("Compare Language Model's Loss2.png")
    plt.cla()

    plt.plot(rnn_validloss, color='red', label='RNN')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.savefig("RNN's Loss2.png")
    plt.cla()

    plt.plot(lstm_validloss, color='blue', label='LSTM')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.savefig("LSTM's Loss2.png")
    plt.cla()

    seed_sentence = ["First Citizen: O, fleeting time",
                    "All: In twilight's grasp, I fi",
                    "Second Citizen: Whispered drea",
                    "MENENIUS: Upon yon hills, our ",
                    "MARCIUS: Stars above, guide me"]
    temperature = [0.5, 0.7, 1, 1.5, 2]

    with open("RNN generate.txt", "w") as file:
        pass
    with open("LSTM generate.txt", "w") as file:
        pass

    for temp in temperature:
        with open("RNN generate.txt", "a") as file:
            file.write('\n\n\nGenerate at temp : ' + str(temp) + "\n")
        with open("LSTM generate.txt", "a") as file:
            file.write('\n\n\nGenerate at temp : ' + str(temp) + "\n")
        for sentence in seed_sentence:
            # RNN 모델 불러와서 생성
            weight_dir = './model_weight/RNN/best_model_epoch2.pth'
            model_RNN.load_state_dict(torch.load(weight_dir))
            sentence_RNN = generate(model_RNN, sentence, temp, data, 100, device)
            with open("RNN generate.txt", "a") as file:
                file.write(sentence_RNN + "\n\n")
            # LSTM 모델 불러와서 생성
            weight_dir = './model_weight/LSTM/best_model_epoch2.pth'
            model_LSTM.load_state_dict(torch.load(weight_dir))
            sentence_LSTM = generate(model_LSTM, sentence, temp, data, 100, device)
            with open("LSTM generate.txt", "a") as file:
                file.write(sentence_LSTM + "\n\n")

    # write your codes here
if __name__ == '__main__':
    main()