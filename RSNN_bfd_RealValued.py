import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
import seaborn as sns
from scipy.stats import f_oneway
import torch.nn.functional as F
from SparseLinear import SparseLinear2
from whisker_dataset import RealValued_Whisker_Dataset
from utils import single_neuron_dy, plot_forwardRaster, plot_5fr
torch.manual_seed(515)
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 22

lens = 0.5  # hyper-parameters of approximate function
num_epochs = 100 # 150  # n_iters / (len(train_dataset) / batch_size)

b_j0 = 0.03
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale

def readPops():
    popType = ['E','E','I','I','E','I','I','I','I','I','I','I','I']
    popSize = [440,934,94,93,2232,106,4,55,64,64,34,60,38]
    Exc_ThtoAll = np.load(r'/data/mosttfzhu/RSNN_bfd/data/ProbConn/Exc_ThtoAll_prob.npy')[0]
    Exc_AlltoAll = np.load(r'/data/mosttfzhu/RSNN_bfd/data/ProbConn/Exc_AlltoAll_prob.npy')
    Inh_AlltoAll = np.load(r'/data/mosttfzhu/RSNN_bfd/data/ProbConn/Inh_AlltoAll_prob.npy')
    Prob_AlltoAll = Exc_AlltoAll+Inh_AlltoAll
    Type_AlltoAll = np.where(Exc_AlltoAll > 0, 1., 0.)
    Type_AlltoAll = np.where(Inh_AlltoAll > 0, -1., Type_AlltoAll)

    return popSize, popType, Exc_ThtoAll, Prob_AlltoAll, Type_AlltoAll

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
def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m)#.cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

class SRNN_bfd(nn.Module):
    def __init__(self, input_size=31*18, output_size=3, init_b=0.04, init_w=0.06):
        super(SRNN_bfd, self).__init__()
        self.init_b, self.init_w = init_b, init_w
        global b_j0
        b_j0 = init_b
        self.output_dim = output_size

        self.popSize, self.popType, self.Prob_ThtoAll, self.Prob_AlltoAll, self.Type_AlltoAll = readPops()


        self.fc = nn.Linear(input_size, 200, bias=True)
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
        nn.init.xavier_uniform_(self.fc.weight)

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
        params_base.extend(list(self.barrelloid.parameters()) + list(self.ThtoPops.parameters()) + list(self.fc2_1.parameters())+list(self.fc2_2.parameters()))
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
            sweep = self.fc(input[:, 0, 0, t, :])

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


def train(init_b=0.03, init_w=0.05, batch_size=128):
    input_dim, output_dim, seq_dim = 31*18, 3, 110
    train_dataset = RealValued_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5',dt=5)
    test_dataset = RealValued_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5',dt=5)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = SRNN_bfd(input_dim, output_dim, init_b = init_b, init_w = init_w)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.008

    params_base, params_tau_adp, params_tau_m = model.get_all_params()
    optimizer = torch.optim.Adam([
        {'params': params_base, 'lr': learning_rate},
        {'params': params_tau_adp, 'lr': learning_rate * 5},
        {'params': params_tau_m, 'lr': learning_rate * 2}])

    scheduler = StepLR(optimizer, step_size=10, gamma=.5)

    best_accuracy = 0
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        for data, labels in tqdm(train_loader):
            input = data.view(data.size(0), data.size(1), data.size(2), seq_dim, -1).cuda()
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs, h,_,_ = model(input)
            loss = criterion(outputs/seq_dim, labels)
            loss.backward()


            # Updating parameters
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted.cpu() == labels.long().cpu()).sum()

        train_acc = 100. * train_correct.numpy() / train_total
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Train Acc: ', train_acc)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            # Iterate through test dataset
            for data, labels in tqdm(test_loader):
                input = data.view(data.size(0), data.size(1), data.size(2), seq_dim, -1).cuda()
                outputs, _, _, _ = model(input)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted.cpu() == labels.long().cpu()).sum()

            test_acc = 100. * test_correct.numpy() / test_total

        if test_acc >= best_accuracy:
            # torch.save(model, './RSNN_bfd.pth')
            best_accuracy = test_acc
        print('epoch: ', epoch, '. Test Acc: ', test_acc, '. Best Acc: ', best_accuracy)

if __name__ == '__main__':
    train(init_b=0.03, init_w=0.05, batch_size=128)


