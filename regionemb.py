import os
import math
import shutil,time
import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
####################
list_max_sequence_length = [1024,256,256,256,1024,256]
list_n_classes = [5,2,5,4,10,14]
list_vocab_size = [124273,394385,356312,42783,361926,227863]
list_task = ['yelp_full','amazon_polarity','amazon_full','ag','yahoo','dbpedia']
emb_size = 128
region_size = 7
region_radius = region_size//2
batch_size = 16
max_epoch = 20
learning_rate = 0.0001
#####################
#####################
base_path = 'data/'
print_step = 200
ctx = mx.gpu(2)
index = 3 # which task
n_classes = list_n_classes[index]
vocab_size = list_vocab_size[index]
max_sequence_length = list_max_sequence_length[index]
task_path = list_task[index]+'/'
####################
class Net(nn.HybridBlock):
    def __init__(self):
        super(Net, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size,region_size*emb_size)
            self.embedding_region = nn.Embedding(vocab_size,emb_size)
            self.max_pool = nn.GlobalMaxPool1D()
            self.dense = nn.Dense(n_classes)
    def hybrid_forward(self, F,aligned_seq,trimed_seq,mask):
        region_aligned_seq = aligned_seq.transpose((1, 0, 2))
        region_aligned_emb = self.embedding_region(region_aligned_seq).reshape((batch_size,-1,region_size,emb_size))
        context_unit = self.embedding(trimed_seq).reshape((batch_size,-1,region_size,emb_size))
        projected_emb = region_aligned_emb * context_unit
        feature = self.max_pool(projected_emb.transpose((0,1,3,2)).reshape((batch_size,-1,region_size))).reshape((batch_size,-1,emb_size))
        feature = feature*mask
        res = F.sum(feature, axis=1).reshape((batch_size,emb_size))
        res = self.dense(res)
        return res
def read_data(path, slot_indexes, slots_lengthes, delim=';', pad=0, type_dict=None):
    n_slots = len(slot_indexes)
    slots = [[] for _ in range(n_slots)]
    if not type_dict:
        type_dict = {}
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)    
            i += 1
            if i % 10000 == 1:
                print('read %d lines' % i)
            raw = []
            for index in slot_indexes:
                slot_value = items[index].split()
                tp = type_dict.get(index, int)
                raw.append([tp(x) for x in slot_value])
            for index in range(len(raw)):
                slots[index].append(pad_and_trunc(raw[index],slots_lengthes[index],pad=pad,sequence=slots_lengthes[index]>1))
    return slots
def pad_and_trunc(data, length, pad=0, sequence=False):
    if pad < 0:
        return data
    if sequence: 
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
    if len(data) > length:
        return data[:length]
    while len(data) < length:
        data.append(pad)
    return data
def load_data(path):
    print('Loading data...')
    indexes = [0,1]
    lengths = [1,max_sequence_length]
    print('Loading train...')
    train_path = base_path+task_path+'train.csv.id'
    labels_train, sequence_train = read_data(train_path, indexes, lengths)
    print('Loading test...')
    test_path = base_path+task_path+'test.csv.id'
    labels_test, sequence_test = read_data(test_path, indexes, lengths)
    return list(zip(sequence_test, labels_test)),list(zip(sequence_train, labels_train))
def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield batch_num * 100.0 / num_batches_per_epoch,shuffled_data[start_index:end_index]
def accuracy(ouput,label,batch_size):
    out = nd.argmax(output,axis=1)
    res = nd.sum(nd.equal(out.reshape((-1,1)),label))/batch_size
    return res
def batch_process(seq,ctx):
    seq = np.array(seq)
    aligned_seq = np.zeros((max_sequence_length - 2*region_radius,batch_size,region_size))
    for i in range(region_radius, max_sequence_length - region_radius):
        aligned_seq[i-region_radius] = seq[:,i-region_radius:i-region_radius+region_size]
    aligned_seq = nd.array(aligned_seq,ctx)
    batch_sequence = nd.array(seq,ctx)
    trimed_seq = batch_sequence[:, region_radius: max_sequence_length - region_radius]
    mask = nd.broadcast_axes(nd.greater(trimed_seq,0).reshape((batch_size,-1,1)),axis=2,size=128)
    return aligned_seq,nd.array(trimed_seq,ctx),mask
def evaluate(data,batch_size):
    print('lalal')
    test_loss = 0.0
    acc_test = 0.0
    cnt = 0
    for epoch_percent, batch_slots in batch_iter(data,batch_size,shuffle=False):
        batch_sequence, batch_label = zip(*batch_slots)
        batch_sequence = nd.array(batch_sequence,ctx)
        batch_label = nd.array(batch_label,ctx)
        output = net(aligned_seq,trimed_seq,mask)
        loss = SCE(output,batch_label)
        acc_test += accuracy(output,batch_label,batch_size)
        test_loss += nd.mean(loss)
        cnt = cnt+1
    return acc_test/cnt,test_loss/cnt
net = Net()
SCE = mx.gluon.loss.SoftmaxCrossEntropyLoss()
net.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': learning_rate})
data_test,data_train = load_data(task_path)
best_acc,global_step,train_loss,train_acc = 0,0,0,0
net.hybridize()
ctime = time.time()
print(ctx,list_task[index])
for epoch in range(max_epoch):
    for epoch_percent, batch_slots in batch_iter(data_train,batch_size,shuffle=True):
        batch_sequence, batch_label = zip(*batch_slots)
        global_step = global_step + 1
        batch_label = nd.array(batch_label,ctx)
        aligned_seq,trimed_seq,mask = batch_process(batch_sequence,ctx)
        with autograd.record():
            output = net(aligned_seq,trimed_seq,mask)
            loss = SCE(output,batch_label)
        loss.backward()
        trainer.step(batch_size)
        train_acc += accuracy(output,batch_label,batch_size)
        train_loss += nd.mean(loss)
        if global_step%print_step==0:
            print('%.4f %%'%epoch_percent,'train_loss:',train_loss.asscalar()/print_step,' train_acc:',train_acc.asscalar()/print_step,'time:',time.time()-ctime)
            train_loss,train_acc = 0,0
            ctime = time.time()
    test_acc,test_loss = evaluate(data_test,batch_size)
    if test_acc>best_acc:
        net.save_parameters('params/regionemb_'+list_task[index])
    print('epoch %d done'%(epoch+1),'acc = %.4f,loss = %.4f'%(test_acc,test_loss))    
