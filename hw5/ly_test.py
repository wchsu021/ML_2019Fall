from gensim.models import word2vec
import numpy as np
import pandas as pd
import sys
import time
import csv
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
    test_data = pd.read_csv(sys.argv[1])
    test_data = test_data.values[:,1]
    #print(test_data)
    test_data = [text.split() for text in test_data]
    
    test_num = len(test_data)
    word_dim = 256
    max_length = 100
    
    w2v_model = word2vec.Word2Vec.load('Word_Emb_model_NEW.bin')
    weights = torch.FloatTensor(w2v_model.wv.vectors)
    word_dict = {word : w2v_model.wv.vocab[word].index for word in w2v_model.wv.vocab}

    test_set = TextDataset(test_data, np.arange(len(test_data)), max_length) # random sequence for label parameter
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = torch.load('model.pth')
    model.eval()
    model.cuda()

    ans = np.zeros((0, 1),dtype=np.int)

    for i, batch in enumerate(test_loader):
        text, _, length, number = batch
        text = word2index(text, word_dict).cuda()
        length = length.cuda()
        output = model(text, length)
        label = np.argmax(output.cpu().data.numpy(), axis=1)
        _, label = map(list, zip(*sorted(zip(number, label.tolist()))))
        label = np.array(label).reshape(-1, 1)
        ans = np.concatenate((ans, label), axis=0)
        torch.cuda.empty_cache()
    #print(ans.shape)
    f = open(sys.argv[2], "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(ans.shape[0]):
        content = [str(i), str(ans[i])[1]]
        w.writerow(content)
    f.close()
    print("Finish testing")
