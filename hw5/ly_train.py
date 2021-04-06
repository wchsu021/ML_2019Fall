from gensim.models import word2vec
import numpy as np
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ly_util import collate_fn, TextDataset, TextNet

def word2index(text_list, word_dict):
    index_tensor = torch.zeros(len(text_list), len(text_list[0]), dtype=torch.long)
    for i, text in enumerate(text_list):
        index = [word_dict[x] if x in word_dict else word_dict['<UNK>'] for x in text]
        index_tensor[i, :] = torch.tensor(index)
    return index_tensor

if __name__ == '__main__':
    print("Load data!")
    data = pd.read_csv(sys.argv[1])
    data = data.values[:,1]
    label = pd.read_csv(sys.argv[2])
    label = label.values[:,1]
    
    train_data = data[:10000]
    train_label = label[:10000]
    valid_data = data[10000:]
    valid_label = label[10000:]
    
    train_data = [text.split() for text in train_data]
    #train_data = [list(filter(str.strip, text_list)) for text_list in train_data] # Remove " " element
    valid_data = [text.split() for text in valid_data]
    #valid_data = [list(filter(str.strip, text_list)) for text_list in valid_data] # Remove " " element
    
    #print(type(train_data)) #(13240,)
    #print(type(train_data)) #(13240,)
    
    ### Word Embedding
    print("Word Embedding!")
    train_num = label.shape[0]
    word_dim = 256
    max_length = 100
    Word_Emb_train_data = []
    for i in range(train_num): 
        Word_Emb_train_data.append(data[i].split())
    Word_Emb_train_data = Word_Emb_train_data + [['<UNK>'] * 5] + [['<PAD>'] * 5]
    w2v_train_model = word2vec.Word2Vec(Word_Emb_train_data, size=word_dim, window=5, min_count=5, sg=0)
    #words = list(model.wv.vocab)
    w2v_train_model.save('Word_Emb_model_NEW.bin')
    print("w2v_model saved!")
    
    ### RNN Data Processing ###
    print("w2v_model loading!")
    # Load Word Embedding Model
    w2v_model = word2vec.Word2Vec.load('Word_Emb_model_NEW.bin')
    weights = torch.FloatTensor(w2v_model.wv.vectors)
    word_dict = {word : w2v_model.wv.vocab[word].index for word in w2v_model.wv.vocab}
    
    
    train_set = TextDataset(train_data, train_label, max_length)
    valid_set = TextDataset(valid_data, valid_label, max_length)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    model = TextNet(weights, max_length).cuda()

    best_acc = 0.0
    num_epoch = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    
    ### RNN Data Processing End ###
    print("Start RNN training")
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, batch in enumerate(train_loader):
            text, label, length, number = batch
            text = word2index(text, word_dict).cuda()
            length = length.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(text, length)
            batch_loss = loss(output, label)

            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.cpu().numpy().reshape(-1, 1))
            train_loss += batch_loss.item()

            torch.cuda.empty_cache()

            progress = ('=' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            (time.time() - epoch_start_time), progress), end='\r', flush=True)
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_set)

        model.eval()
        for i, batch in enumerate(valid_loader):
            text, label, length, number = batch

            text = word2index(text, word_dict).cuda()
            length = length.cuda()
            label = label.cuda()

            output = model(text, length)
            batch_loss = loss(output, label)
            val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.cpu().numpy().reshape(-1, 1))
            val_loss += batch_loss.item()

            torch.cuda.empty_cache()

        val_loss = val_loss / len(valid_loader)
        val_acc = val_acc / len(valid_set)
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time() - epoch_start_time, \
            train_acc, train_loss, val_acc, val_loss))

        if val_acc > best_acc:
            torch.save(model, 'model.pth')
            best_acc = val_acc
            print('Model Saved!')
            
    print("Finish Training")
        