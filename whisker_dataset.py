import torch
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.random.seed(515)
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 23
def generate_spike_train(prob,dt, verify=False):
    '''
    prob shape: (N,12,110,31,3)
    return spike train shape: (N,12,110*dt,31,3)
    '''
    repeated_prob = np.repeat(prob,dt,axis=2)
    spike_train = np.random.binomial(n=1, p=repeated_prob)

    #verify
    if verify==True:
        selected_prob = prob[0,0,:,0,0] # (110,)
        selected_spike_train = spike_train[0,0,:,0,0] # (110*dt,)
        selected_spike_train = selected_spike_train.reshape(prob.shape[2],dt) # (110,dt,)
        sums = selected_spike_train.sum(axis=1)
        fr = sums/dt

        plt.figure(figsize=(7,7))
        plt.plot(selected_prob,c='gray',linewidth=3, label='Sampling rate')
        plt.plot(fr,c='indianred', linewidth=3, label='Actual rate')
        # plt.xlabel('Time step')
        # plt.ylabel('Rate')
        plt.legend(fontsize='medium')
        # plt.title('dt = 5')

        # 设置坐标轴刻度线的粗细和刻度标签的字体大小
        plt.tick_params(axis='both', which='major', width=3, labelsize=28)

        # 获取并设置边框线的粗细
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.8)

        plt.show()
        exit()

    return spike_train
def sigmoid(x, c):
    return 1 / (1 + np.exp(-(x-c)))

class SpikingBased_Whisker_Dataset(Dataset):
    def __init__(self,h5_path,dt=5):
        self.dt = dt
        with h5py.File(h5_path, 'r') as f:
            data_force = f['force'][:]
            data_torque = f['torque'][:]
            label = f['label'][:]
            label[label == 53.] = 0 # 53：helmet, 55:box, 113:plane
            label[label == 55.] = 1
            label[label == 113.] = 2

        self.force_mod, self.torque_mod = self.CalMod(data_force,data_torque)
        self.label = label.astype('int32')
        self.spikes = self.GenerateSpikeTrains() # (218, 2, 12, 550, 31, 3)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,idx):
        sample = torch.tensor(self.spikes[idx], dtype=torch.float32)
        return sample, torch.tensor(self.label[idx], dtype=torch.int64)

    def CalMod(self,force,torque):
        # 计算force与torque的模
        force_mod = np.sqrt(np.sum(force**2,axis=-1,keepdims=True)).squeeze()
        torque_mod = np.sqrt(np.sum(torque ** 2, axis=-1, keepdims=True)).squeeze()
        force_mod[force_mod == 0] = 0.00001
        torque_mod[torque_mod == 0] = 0.00001
        return force_mod, torque_mod

    def GenerateSpikeTrains(self):
        f_mod_log = np.log(self.force_mod)
        prob_f = sigmoid(f_mod_log, c=2)
        spike_f = generate_spike_train(prob_f, dt=self.dt)

        t_mod_log = np.log(self.torque_mod)
        prob_t = sigmoid(t_mod_log, c=4)
        spike_t = generate_spike_train(prob_t, dt=self.dt)

        return np.stack((spike_f,spike_t),axis=1)

    def type_specific_data(self):
        # 返回各类别的所有data
        index_0, index_1, index_2 = np.where(self.label==0), np.where(self.label==1), np.where(self.label==2)
        data_0, data_1, data_2 = self.spikes[index_0], self.spikes[index_1], self.spikes[index_2]
        return data_0, data_1, data_2

class RealValued_Whisker_Dataset(Dataset):
    def __init__(self,h5_path,dt=5):
        self.dt = dt
        with h5py.File(h5_path, 'r') as f:
            data_force = f['force'][:]
            data_torque = f['torque'][:]
            label = f['label'][:]
            label[label == 53.] = 0 # 53：helmet, 55:box, 113:plane
            label[label == 55.] = 1
            label[label == 113.] = 2

        data_force, data_torque = F.normalize(torch.tensor(data_force)), F.normalize(torch.tensor(data_torque))
        self.ft = np.concatenate((data_force, data_torque),axis=-1)
        self.ft = self.ft.reshape(self.ft.shape[0],1,self.ft.shape[1],self.ft.shape[2],self.ft.shape[3],-1)

        self.label = label.astype('int32')
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,idx):
        sample = torch.tensor(self.ft[idx], dtype=torch.float32)
        return sample, torch.tensor(self.label[idx], dtype=torch.int64)

if __name__ == '__main__':
    train_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5')
    test_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    for data, label in train_loader:
        print(f"Data shape: {data.shape}, Label shape: {label.shape}")
        torch.set_printoptions(threshold=20000)
        print(data[0,0,0,:,:,0])
        exit()




