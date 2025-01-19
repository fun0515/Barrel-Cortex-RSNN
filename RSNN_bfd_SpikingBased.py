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
from whisker_dataset import SpikingBased_Whisker_Dataset
torch.manual_seed(515)

plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 22

lens = 0.5  # hyper-parameters of approximate function
num_epochs = 100 # 150  # n_iters / (len(train_dataset) / batch_size)

b_j0 = 0.04
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


def train(init_b=0.04, init_w=0.06, batch_size=128):
    input_dim, output_dim, seq_dim = 31*3, 3, 550
    train_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5',dt=5)
    test_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5',dt=5)
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

def ANOVA(model):
    # 分析三类神经元活动是否存在显著特异性，找出特异性神经元并可视化
    T = 110 * 5
    h_state = []
    train_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5')
    test_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5')
    train_data0, train_data1, train_data2 = train_dataset.type_specific_data()
    test_data0, test_data1, test_data2 = test_dataset.type_specific_data()
    data0, data1, data2 = np.concatenate((train_data0, test_data0), axis=0), np.concatenate((train_data1, test_data1),
                                        axis=0), np.concatenate((train_data2, test_data2), axis=0)
    print(data0.shape, data1.shape, data2.shape)

    with torch.no_grad():
        for data in [data0, data1, data2]:
            data = torch.tensor(data, dtype=torch.float32)
            input = data.view(data.size(0), data.size(1), data.size(2), T, -1).cuda()
            _, h, _, _ = model(input)
            h = [r.detach().cpu().numpy() for r in h]
            h = np.stack(h, axis=0).transpose(1, 2, 0)  # (batch_size, N, T)
            h_state.append(h)

    # 按类别计算每个神经元的总共发放次数
    spikes_by_class = [np.sum(h, axis=0) for h in h_state] # return: (3, N, T)
    # 逐个神经元执行ANOVA
    N = spikes_by_class[0].shape[0] # 网络中神经元数量4218
    p_values = np.ones((N,))
    for i in range(N):
        if np.all(spikes_by_class[0][i] == spikes_by_class[1][i]) & np.all(spikes_by_class[0][i] == spikes_by_class[2][i]):
            continue
        else:
            f, p = f_oneway(spikes_by_class[0][i], spikes_by_class[1][i], spikes_by_class[2][i])
        if math.isnan(p) or math.isnan(f):
            continue
        p_values[i] = p
    # 筛选有显著差异的神经元
    significant_index = np.where(p_values < 0.001)[0] # (N_signif, )
    print('find %f neurons that p-value < 0.001' % (significant_index.shape[0]))

    # 统计每个神经元的各类别放电率比率：各类别脉冲数 /（脉冲总数）(N,3)
    firing_proportion = np.zeros((N,3))
    for i in range(N):
        a, b, c = np.sum(spikes_by_class[0][i])+182, np.sum(spikes_by_class[1][i])+182, np.sum(spikes_by_class[2][i])+182 # 防止有些神经元脉冲总数很少
        firing_sum = a + b + c
        firing_proportion[i] = np.array([a/firing_sum, b/firing_sum, c/firing_sum])
    # 按照比率将神经元坐标映射到三条直线上：y = 0.57735x, x = 0, y = -0.57735x, 然后绘制散点图
    neuron_coords = np.zeros((N,2))
    for i in range(N):
        x0, y0 = -(firing_proportion[i][0]/2)*math.sqrt(3), -firing_proportion[i][0]/2
        x1, y1 = 0, firing_proportion[i][1]
        x2, y2 = (firing_proportion[i][2]/2)*math.sqrt(3), -firing_proportion[i][2]/2
        neuron_coords[i] = np.array([x0+x1+x2, y0+y1+y2])

    # 划分出显著神经元与不显著神经元
    index = np.zeros((N,))
    index[significant_index] = 1.
    significant_cells = neuron_coords[np.where(index==1.)[0]]
    unsignificant_cells = neuron_coords[np.where(index==0.)[0]]
    print(firing_proportion[np.where(index==0.)[0]])
    print(np.max(firing_proportion[np.where(index==0.)[0]]))

    # 绘制散点图, 三条线段分别指代三个类别
    plt.figure(figsize=(8,8))
    y_values = np.linspace(start=0, stop=0.265, num=100)
    x_values = y_values * math.sqrt(3)

    plt.plot(-x_values, -y_values, c='black',linestyle = '-', linewidth=2) # 第三象限y=0.57735x
    plt.arrow(-0.433, -0.25, -0.00866, -0.005, head_length=0.05, head_width=0.05, fc='black', ec='black')
    plt.text(-0.4, -0.38, 'Helmet', fontsize=30, color='black', ha='center', va='bottom')

    plt.plot([0]*100, np.linspace(start=0, stop=0.53, num=100), c='black',linestyle = '-', linewidth=2)  # x=0
    plt.arrow(0, 0.5, 0, 0.01, head_length=0.05, head_width=0.05, fc='black', ec='black')
    plt.text(-0.1, 0.53, 'Box', fontsize=30, color='black', ha='center', va='bottom')

    plt.plot(x_values, -y_values, c='black',linestyle = '-', linewidth=2) # 第四象限y=-0.57735x
    plt.arrow(0.433, -0.25, 0.00866, -0.005, head_length=0.05, head_width=0.05, fc='black', ec='black')
    plt.text(0.4, -0.38, 'Plane', fontsize=30, color='black', ha='center', va='bottom')

    plt.scatter(significant_cells[:,0], significant_cells[:,1], marker='o', s=15, linewidths=2, edgecolors='sandybrown', facecolors='none', label = 'P<0.001 cells')
    plt.scatter(unsignificant_cells[:,0], unsignificant_cells[:,1], marker='o', s=15, linewidths=2, edgecolors='cornflowerblue', facecolors='none', label = 'Normal cells')
    plt.xlim(-0.55, 0.55)
    plt.ylim(-0.45,0.65)
    plt.xticks([-0.5,-0.25,0,0.25,0.5],[-0.5,-0.25,0,0.25,0.5])
    plt.title('%.0f P-value < 0.001 Cells' % (significant_index.shape[0]), fontsize=33)
    plt.legend(loc='upper right', markerscale=4,handletextpad=0.1)

    # 设置坐标轴刻度线的粗细和刻度标签的字体大小
    plt.tick_params(axis='both', which='major', width=2, labelsize=30)

    # 获取并设置边框线的粗细
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)  # 设置边框线的粗细为3

    plt.show()


###############################
if __name__ == '__main__':
    train(init_b=0.04, init_w=0.06, batch_size=128)

    # plot neural firing selectivity
    # trained_model = torch.load('/data/mosttfzhu/RSNN_bfd/Adp_LIF_RSNN_bfd_seed515_0.04b_0.06w_batchsize128_0.818.pth',
    #                            map_location='cuda')
    # ANOVA(model=trained_model)

