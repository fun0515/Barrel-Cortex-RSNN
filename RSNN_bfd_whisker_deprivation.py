import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils import data
from SparseLinear import SparseLinear2
from torch.utils.data import Dataset
torch.manual_seed(515)
np.random.seed(515)

def deprived_whiskers(N_whisker=30,x=1):
    return np.random.choice(N_whisker,x,replace = False)


lens = 0.5
num_epochs = 1

b_j0 = 0.03
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale


num_encode = 700
num_classes = 3

def readPops():
    popType = ['E','E','I','I','E','I','I','I','I','I','I','I','I']
    popSize = [440,934,94,93,2232,106,4,55,64,64,34,60,38]
    Exc_ThtoAll = np.load(r'./data/Exc_ThtoAll_prob.npy')[0]
    Exc_AlltoAll = np.load(r'./data/Exc_AlltoAll_prob.npy')
    Inh_AlltoAll = np.load(r'./data/Inh_AlltoAll_prob.npy')
    Prob_AlltoAll = Exc_AlltoAll+Inh_AlltoAll
    Type_AlltoAll = np.where(Exc_AlltoAll > 0, 1., 0.)
    Type_AlltoAll = np.where(Inh_AlltoAll > 0, -1., Type_AlltoAll)

    return popSize, popType, Exc_ThtoAll, Prob_AlltoAll, Type_AlltoAll

def generate_spike_train(prob,dt, verify=False):
    '''
    prob shape: (N,12,110,31,3)
    return spike train shape: (N,12,110*dt,31,3)
    '''
    repeated_prob = np.repeat(prob,dt,axis=2)
    spike_train = np.random.binomial(n=1, p=repeated_prob)
    # spike_train = (np.random.rand(repeated_prob.shape[0], repeated_prob.shape[1], repeated_prob.shape[2]) < repeated_prob).astype('float')

    #verify
    if verify==True:
        selected_prob = prob[0,0,:,0,0] # (110,)
        selected_spike_train = spike_train[0,0,:,0,0] # (110*dt,)
        selected_spike_train = selected_spike_train.reshape(prob.shape[2],dt) # (110,dt,)
        sums = selected_spike_train.sum(axis=1)
        fr = sums/dt

        plt.figure()
        plt.plot(selected_prob,label='prob')
        plt.plot(fr,label='fr')
        plt.legend()
        plt.title('dt = 5')
        plt.show()

    return spike_train
def sigmoid(x, c):
    return 1 / (1 + np.exp(-(x-c)))


class Whisker_Dataset_Deprivation(Dataset):
    def __init__(self,h5_path,index,dt=5):
        self.dt = dt
        self.index = index
        with h5py.File(h5_path, 'r') as f:
            data_force = f['force'][:]
            data_torque = f['torque'][:]
            label = f['label'][:]
            label[label == 53.] = 0 # 53：helmet, 55:box, 113:plane
            label[label == 55.] = 1
            label[label == 113.] = 2

        #print(data_force.shape)
        data_force[:,:,:,index,:,:] = 0.
        data_torque[:,:,:,index,:,:] = 0.

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


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        return grad_input * temp.float() * gamma


def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b

act_fun_adp = ActFun_adp.apply
class SRNN_bfd(nn.Module):
    def __init__(self, input_size=31*3, output_size=3, init_b=0.04, init_w=0.06):
        super(SRNN_bfd, self).__init__()
        self.init_b, self.init_w = init_b, init_w
        global b_j0
        b_j0 = init_b
        self.output_dim = output_size

        self.popSize, self.popType, self.Prob_ThtoAll, self.Prob_AlltoAll, self.Type_AlltoAll = readPops()


        self.fc_f = nn.Linear(input_size, 100, bias=True)
        self.fc_t = nn.Linear(input_size, 100, bias=True)
        self.barrelloid = nn.Linear(200, 200, bias=True)
        self.barrelloid_tau_adp = nn.Parameter(torch.Tensor(200), requires_grad=True)
        self.barrelloid_tau_m = nn.Parameter(torch.Tensor(200), requires_grad=True)

        # 生成丘脑到bfd的连接
        self.ThtoPops = nn.ModuleList()  # 丘脑只对其中的11个群落有兴奋型连接
        for i, conn_prob in enumerate(self.Prob_ThtoAll):
            if conn_prob != 0.:
                self.ThtoPops.append(SparseLinear2(200, self.popSize[i], bias=False, connect_prob=conn_prob, ifPositive=True))

        # 生成神经元群落之间连接
        self.toPops = nn.ModuleList()
        for i, size in enumerate(self.popSize):
            inner_list = nn.ModuleList()
            for j, conn_prob in enumerate(self.Prob_AlltoAll[:,i]):
                if conn_prob != 0. :  # 不排除群落内部连接
                    inner_list.append(SparseLinear2(self.popSize[j], size, bias=False, connect_prob=conn_prob, ifPositive=True))
            self.toPops.append(inner_list)

        # 生成动力学参数
        self.tau_adp, self.tau_m = [], []
        for size in self.popSize:
            self.tau_adp.append(nn.Parameter(torch.Tensor(size), requires_grad=True))
        for size in self.popSize:
            self.tau_m.append(nn.Parameter(torch.Tensor(size), requires_grad=True))

        # readout层
        self.fc2_1 = nn.Linear(self.popSize[0] + self.popSize[1] + self.popSize[4], 128, bias=True)
        self.fc2_2 = nn.Linear(128, output_size)
        self.init_weights()
    def init_weights(self):
        for w in self.ThtoPops:
            w.uniform_init(init_w=self.init_w)
        for a in range(len(self.toPops)):
            for b in range(len(self.toPops[a])):
                self.toPops[a][b].uniform_init(init_w=self.init_w)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.xavier_uniform_(self.fc_f.weight)
        nn.init.xavier_uniform_(self.fc_t.weight)

        for tau_a in self.tau_adp:
            nn.init.constant_(tau_a, 5)
        for tau_m in self.tau_m:
            nn.init.constant_(tau_m, 10)
        nn.init.constant_(self.barrelloid_tau_adp, 5)
        nn.init.constant_(self.barrelloid_tau_m, 10)

        self.b_barrelloid = self.init_b
        self.b = [self.init_b for _ in range(len(self.popSize))]

    def get_all_params(self):
        params_base = []
        for module in self.toPops:
            params_base.extend(list(module.parameters()))
        params_base.extend(list(self.fc_f.parameters()) + list(self.fc_t.parameters()) +
                           list(self.barrelloid.parameters()) + list(self.ThtoPops.parameters()) + list(self.fc2_1.parameters())+list(self.fc2_2.parameters()))
        params_tau_adp = self.tau_adp + [self.barrelloid_tau_adp]
        params_tau_m = self.tau_m + [self.barrelloid_tau_m]
        return params_base, params_tau_adp, params_tau_m

    def forward(self, input):
        batch_size, seq_num = input.size(0), input.size(3) # (batch_size, 2, 12, T, 31*3)

        self.b = [self.init_b for _ in range(len(self.popSize))]
        self.b_barrelloid = self.init_b
        torch.manual_seed(515)
        mem_barrelloid = torch.rand(batch_size, 200).cuda()
        spike_barrelloid = torch.rand(batch_size, 200).cuda()

        mem_layer = [torch.rand(batch_size, size).cuda() for size in self.popSize]
        spike_layer = [torch.rand(batch_size, size).cuda() for size in self.popSize]
        mem_output = torch.rand(batch_size, self.output_dim).cuda()
        output = torch.zeros(batch_size, self.output_dim).cuda()

        h_state = []

        for t in range(seq_num):
            sweep_f = input[:, 0, 0, t, :].squeeze()
            sweep_t = input[:, 1, 0, t, :].squeeze()
            sweep = torch.concat((self.fc_f(sweep_f), self.fc_t(sweep_t)), dim=-1)

            mem_barrelloid, spike_barrelloid, _, self.b_barrelloid = (
                mem_update_adp(self.barrelloid(sweep), mem_barrelloid, spike_barrelloid, self.barrelloid_tau_adp, self.b_barrelloid, self.barrelloid_tau_m))
            for i in range(len(self.popSize)):
                input_current = 0.

                # 计算丘脑对第i个群落的输入(若存在)
                Th2pop_id = np.where(self.Prob_ThtoAll != 0.)[0]
                if i in Th2pop_id:
                    input_current = self.ThtoPops[np.where(Th2pop_id == i)[0][0]](spike_barrelloid)

                # 计算群落之间输入
                source_index = np.where(self.Type_AlltoAll[:, i] != 0.)[0]
                source_type = self.Type_AlltoAll[:, i][source_index]

                for j in range(len(self.toPops[i])):
                    input_current = input_current + source_type[j] * self.toPops[i][j](spike_layer[source_index[j]])

                # 更新神经元状态
                mem_layer[i], spike_layer[i], B, self.b[i] = mem_update_adp(input_current, mem_layer[i], spike_layer[i],
                                                                            self.tau_adp[i], self.b[i], self.tau_m[i])
            if t > 0:
                output= output + F.softmax(self.fc2_2(self.fc2_1(torch.cat((spike_layer[0],spike_layer[1],spike_layer[4]),dim=1))),dim=1)
            h_state.append(torch.cat((spike_layer),dim=1))
        return output, h_state, _, _


def test_whisker_deprivation(model, batch_size=128):
    input_dim, output_dim, seq_dim = 31*3, 3, 550
    cut_whiskers = [1,2,3,4,5,6,7,8,9,10]
    model.eval()
    test_all = []
    test_mean = []
    repetitions = np.arange(31)

    for num_deprived_whiskers in cut_whiskers:
        print("Number of whiskers deprived: ",num_deprived_whiskers)
        test_accs = []
        epochs = 31 if num_deprived_whiskers == 1 else 50
        for epoch in range(epochs):
            if num_deprived_whiskers == 1:
                index = repetitions[epoch]
            else:
                index = deprived_whiskers(30, num_deprived_whiskers)
            print("Index of deprived whiskers: ", index)

            test_dataset = Whisker_Dataset_Deprivation('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5',index,dt=5)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            with torch.no_grad():
                test_correct = 0
                test_total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    input = images.view(images.size(0), images.size(1), images.size(2), seq_dim, -1).cuda()
                    outputs, _, _, _ = model(input)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted.cpu() == labels.long().cpu()).sum()

            test_acc = 100. * test_correct.numpy() / test_total
            test_accs.append(test_acc)
                        
            print('Trial No.', epoch, ' Deprived Whisker Num:', num_deprived_whiskers,' Test Acc:', test_acc)

        print("Deprived Whisker Num: ",num_deprived_whiskers, "Mean Test Acc: ",np.mean(test_accs))
        test_all.append(test_accs)
        test_mean.append(np.mean(test_accs))

    print(test_mean)

###############################
if __name__ == '__main__':
    trained_model = torch.load('./data/RSNN_bfd_0.03b_0.05w.pth',map_location='cuda')
    test_whisker_deprivation(model=trained_model, batch_size=128)