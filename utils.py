'''CV measure, raster plot, and neural dynamics'''
import copy
import random
import itertools
import matplotlib.pyplot as plt
import torch
import gif
import numpy as np
import scipy.stats as stats
from RSNN_bfd_SpikingBased import SRNN_bfd, mem_update_adp, readPops
from whisker_dataset import SpikingBased_Whisker_Dataset
from scipy.stats import f_oneway, pearsonr
import matplotlib.lines as mlines
import seaborn as sns

plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 25


def calculate_fr(h_state, interval):
    N, T = h_state.shape
    intervals = T // interval
    reshaped_activity = h_state.reshape(N, intervals, interval)

    sums = reshaped_activity.sum(axis=2)
    sums = sums.sum(axis=0)
    fr = sums / N
    return fr * 100

def plot_5fr(h_state, interval=10):
    h_state = h_state.transpose(1, 0)
    N, T = h_state.shape

    l4e_fr = calculate_fr(h_state[0:1374], interval=interval)
    l4i_fr = calculate_fr(h_state[1374:1561], interval=interval)
    l23e_fr = calculate_fr(h_state[1561:3793], interval=interval)
    l23i_fr = calculate_fr(h_state[3793:4218], interval=interval)

    plt.figure(figsize=(10,6))
    x = np.linspace(10, 550, 55)

    plt.plot(x, l4e_fr, label='L4 Exc',color='red')
    plt.plot(x, l4i_fr, label='L4 Inh',color='royalblue')
    plt.plot(x, l23e_fr, label='L2/3 Exc',color='red',linestyle='--',linewidth=2)
    plt.plot(x, l23i_fr, label='L2/3 Inh',color='royalblue',linestyle='--',linewidth=2)

    # plt.xlabel('Time (ms)')
    # plt.xticks(np.arange(0, 550, 50))
    # plt.ylabel('Firing rate')
    plt.xlim(0,550)
    ax = plt.gca()  # 获取当前的Axes对象
    # ax.set_yticks([])  # 隐藏y轴刻度
    # plt.ylim(0,50)
    # plt.gca().set_yticks([])
    plt.legend(loc='upper left')

    plt.show()


@gif.frame
def plot_forwardRaster(h, sample, seq_len):
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    '''绘制所有神经元的raster图; h:binary-array, shape(Batch_size, T,N), Batch_size次trial中N个神经元在T个时刻的发放状态'''
    sample = sample # 15
    spikes_T = h[sample].T  # (N,T)
    N = spikes_T.shape[0]
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = (spikes_T == 1)

    plt.figure(figsize=(10,6))
    for i in range(N):
        if i >= 0 and i <= 439:
            c = 'r'
        elif i >= 440 and i <= 1373:
            c = 'r'
        elif i >= 1561 and i <= 3792:
            c = 'r'
        else:
            c = 'b'
        plt.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=c)
    plt.axhline(1560,color='black')  # L23与L4分界线
    plt.xlim(0,seq_len)
    plt.ylim(0,4218)
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron ID')

    E = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=8, label='Exc')
    I = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=8, label='Inh')
    plt.legend(handles=[E, I],loc='upper left',fontsize='medium')

    # 设置坐标轴刻度线的粗细和刻度标签的字体大小
    plt.tick_params(axis='both', which='major', width=2, labelsize=25)

    # 获取并设置边框线的粗细
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.3)  # 设置边框线的粗细为3

    plt.show()

def calculate_cv(h_state):
    """
    计算变异系数(Coefficient of Variation, CV)
    :param h_state: 二维numpy数组，shape为（神经元数量，时间步长），元素为0或1
    :return: 变异系数
    """
    # 计算每个神经元的平均放电次数
    mean_firing_rates = np.mean(h_state, axis=1)
    mean_firing_rates[mean_firing_rates==0] = 1e-20
    # 计算每个神经元的标准差
    std_devs = np.std(h_state, axis=1)

    # 计算每个神经元的CV
    cvs = std_devs / mean_firing_rates
    return np.mean(cvs)

def NAOI(h_state):
    '''
    Neuron Activity Order Index
    h_state: 二维numpy数组，表示神经元活动（NxT）
    '''
    return calculate_cv(h_state)

def NAOI_along_weights():
    w_list = [0.01 * i for i in range(1, 5)]
    b_list = [0.01 * i for i in range(1, 61)]
    naoi_array = np.zeros((4, 60))
    T = 550
    train_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5')
    test_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5')
    train_data0, train_data1, train_data2 = train_dataset.type_specific_data()
    test_data0, test_data1, test_data2 = test_dataset.type_specific_data()
    data0, data1, data2 = np.concatenate((train_data0, test_data0), axis=0), np.concatenate(
        (train_data1, test_data1),axis=0), np.concatenate((train_data2, test_data2), axis=0)
    data = np.concatenate((data0, data1,data1), axis=0)
    print(data.shape)

    with torch.no_grad():   # 不然会占用很多显存
        for i,w in enumerate(w_list):
            for j,b in enumerate(b_list):
                naoi = 0
                model = SRNN_bfd(input_size=31*3, output_size=3, init_b=b, init_w=w).cuda()
                data = torch.tensor(data, dtype=torch.float32)
                input = data.view(data.size(0), data.size(1), data.size(2), T, -1).squeeze().cuda()
                outputs, h, _, _ = model(input)
                h_state = [r.detach().cpu().numpy() for r in h]
                h_state = np.stack(h_state, axis=0).transpose(1, 2, 0)  # (batch_size, N, T)
                for h in h_state:
                    naoi = naoi + NAOI(h)
                naoi_array[i][j] = naoi / h_state.shape[0]
                print('w: ', w, 'b: ', b, 'naoi: ', naoi / h_state.shape[0])
        print(naoi_array)

def cal_conn_matrix(model=None):
    '''
    :param model: 网络模型
    :return: conn_matrix: shape为（4218,4218）的连接矩阵
    '''
    conn_matrix = np.zeros((4218, 4218))
    popSize, _, _, _, Type_AlltoAll = readPops()
    popSize = [0] + list(itertools.accumulate(popSize)) # 累计和

    toPops = model.toPops
    for i in range(Type_AlltoAll.shape[1]):     # 列
        count = 0 # toPops是紧凑的，与Type_AlltoAll的索引完全一致
        for j in range(Type_AlltoAll.shape[0]): # 行
            if Type_AlltoAll[j][i] != 0:
                conn_matrix[popSize[j]:popSize[j+1],popSize[i]:popSize[i+1]] = toPops[i][count].weight.data.numpy().transpose(1,0)
                print(toPops[i][count].weight.data.numpy())
                count += 1
    print('Connections number: ', np.count_nonzero(conn_matrix))
    return conn_matrix
def plot_degree_distribution(initial_matrix, trained_matrix):
    # 统计比较网络中的度分布
    in_degree1, out_degree1 = np.sum(initial_matrix, axis=0), np.sum(initial_matrix, axis=1)
    in_degree2, out_degree2 = np.sum(trained_matrix, axis=0), np.sum(trained_matrix, axis=1)
    fig, axs = plt.subplots(2,1,figsize=(8, 7))
    axs[0].hist(in_degree1, bins=1000,label='Initial in-degree',color = 'indianred',alpha=0.7)
    axs[0].hist(in_degree2, bins=1000, label='Trained in-degree', color='slategray',alpha=0.7)
    # axs[0].set_title('In degree')
    axs[0].set_xlim([0, 80])
    axs[0].set_ylim([0, 80])
    # axs[0].set_yscale('log')
    axs[0].legend(fontsize='large')
    # 设置坐标轴刻度线的粗细和刻度标签的字体大小
    axs[0].tick_params(axis='both', which='major', width=2, labelsize=26)

    # 获取并设置边框线的粗细
    for spine in axs[0].spines.values():
        spine.set_linewidth(1.3)

    axs[1].hist(out_degree1, bins=1000, label='Initial out-degree', color='indianred', alpha=0.7)
    axs[1].hist(out_degree2, bins=1000, label='Trained out-degree', color='slategray', alpha=0.7)
    # axs[1].set_title('Out degree')
    axs[1].set_xlim([0, 100])
    axs[1].set_ylim([0, 80])
    axs[1].legend(fontsize='large')
    # 设置坐标轴刻度线的粗细和刻度标签的字体大小
    axs[1].tick_params(axis='both', which='major', width=2, labelsize=26)

    # 获取并设置边框线的粗细
    for spine in axs[1].spines.values():
        spine.set_linewidth(1.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.show()

def min_max_normalize(values,a=0.1,b=1.):
    # 将列表中的元素归一化到[a,b]区间
    min_val = min(values)
    max_val = max(values)
    return [a+(b-a)*(x - min_val) / (max_val - min_val) for x in values]

def single_neuron_dy(Tau_adp, Tau_m):
    '''
    :param Tau_adp: 数组列表，训练后的每个神经元亚型的tau_adp
    :param Tau_m: 数组列表，训练后的每个神经元亚型的tau_m
    '''
    popSize = [440, 934, 94, 93, 2232, 106, 4, 55, 64, 64, 34, 60, 38]
    N = len(popSize)
    T = 550
    fr = []
    Izhikevich_fr = [0.01636364, 0.01636364, 0.05454545, 0.02181818, 0.03818182, 0.21272727, 0.21272727,
                     0.05454545, 0.16727273, 0.05454545, 0.05454545, 0.04727273, 0.03454545]
    for i in range(N):
        tau_adp = torch.tensor(Tau_adp[i], device='cuda')
        tau_m = torch.tensor(Tau_m[i], device='cuda')
        mem = torch.zeros((T,popSize[i]), device='cuda')
        spike = torch.zeros((T,popSize[i]), device='cuda')
        b = torch.ones((T,popSize[i]), device='cuda') * 0.04

        for t in range(1, T):
            input_current = torch.tensor(1., device='cuda')
            mem[t], spike[t], _, b[t] = mem_update_adp(input_current, mem[t - 1], spike[t - 1], tau_adp, b[t - 1], tau_m)
        fr.append(torch.sum(spike).cpu().numpy()/(T*popSize[i]))

    print(fr)
    normed_fr = min_max_normalize(fr)
    normed_Izhikevich_fr = min_max_normalize(Izhikevich_fr)

    # 绘制放电率柱状图
    plt.figure(figsize=(7,5))
    x = range(1,27,2)
    # 设置柱状图的位置
    r1 = [x - 0.4 for x in x]
    r2 = [x + 0.4 for x in x]
    plt.bar(r1,normed_fr,color='slateblue',label='Trained',edgecolor='black', linewidth=1.5)
    plt.bar(r2,normed_Izhikevich_fr,color='teal',label='NS experimental',edgecolor='black', linewidth=1.5)
    plt.axhline(0.1764,color='darkred', linestyle='--', linewidth=2)
    plt.xticks(range(1,27,2),range(1,14))
    plt.ylim(0, 1.2)

    # 设置坐标轴刻度线的粗细和刻度标签的字体大小
    plt.tick_params(axis='both', which='major', width=2, labelsize=26)

    # 获取并设置边框线的粗细
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)

    plt.legend(ncol=2,frameon=False,handletextpad=0.15)
    plt.show()

    # 计算放电率的皮尔逊相关系数
    corr, p = pearsonr(normed_fr,normed_Izhikevich_fr)
    print('Pearson correlation after training: ', corr, 'P value: ', p)

    random_corr = 0
    for k in range(100):
        random_fr = [0.5]*13 + np.random.normal(0., 0.000001, 13)
        random_c, _ = pearsonr(Izhikevich_fr,random_fr)
        random_corr += random_c
    print('Random pearson correlation: ', random_corr/100.)

def neuro_dymodel(model, show='L4',parameter='m'):
    # 观察训练后的神经元群落动力学特性
    label_list = ['Pyr_SP', 'Pyr_SS', 'Inh_FS', 'Inh_RSNP', 'Pyr', 'Inh_FSBS', 'Inh_FSCH',
                  'Inh_BSPV', 'Inh_Mar', 'Inh_DBC', 'Inh_Bip', 'Inh_SBC', 'Inh_NG']

    tau_adp = [tau.detach().cpu().numpy() for tau in model.tau_adp]
    tau_m = [tau.detach().cpu().numpy() for tau in model.tau_m]

    single_neuron_dy(tau_adp, tau_m)

    # KDE估计并可视化tau的概率密度分布
    if show == 'L4':
        if parameter == 'm':
            fig, axes = plt.subplots(4, 1, figsize=(8, 7))
            for i, ax in enumerate(axes):
                sns.kdeplot(data=tau_m[i], ax=ax, color = 'steelblue', linewidth=2, common_norm=False, fill=True, label=label_list[i])
                ax.axvline(x=np.mean(tau_m[i]), color='darkred', linestyle='--', linewidth=2)
                ax.axvline(x=10., color='darkolivegreen', linestyle='--', linewidth=2)
                ax.set_ylim(0., 3.5)
                ax.set_xlim(8.5, 11.5)
                ax.set_ylabel('')
                ax.legend()

                # 设置坐标轴刻度线的粗细和刻度标签的字体大小
                ax.tick_params(axis='both', which='major', width=2, labelsize=26)

                # 获取并设置边框线的粗细
                for spine in ax.spines.values():
                    spine.set_linewidth(1.3)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.6, wspace=0.4)
            plt.show()
        elif parameter == 'adp':
            fig, axes = plt.subplots(4, 1, figsize=(8, 7))
            for i, ax in enumerate(axes):
                sns.kdeplot(data=tau_adp[i], ax=ax, color = 'steelblue', linewidth=2, common_norm=False, fill=True, label=label_list[i])
                ax.axvline(x=np.mean(tau_adp[i]), color='darkred', linestyle='--', linewidth=2)
                ax.axvline(x=5., color='darkolivegreen', linestyle='--',linewidth=2)
                ax.set_ylim(0., 2.)
                ax.set_xlim(3.8, 7.5)
                ax.set_ylabel('')
                ax.legend()
                # 设置坐标轴刻度线的粗细和刻度标签的字体大小
                ax.tick_params(axis='both', which='major', width=2, labelsize=26)

                # 获取并设置边框线的粗细
                for spine in ax.spines.values():
                    spine.set_linewidth(1.3)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.6, wspace=0.4)
            plt.show()
    elif show == 'ALL':
        fig, axes = plt.subplots(13, 1, figsize=(10, 20))
        for i,ax in enumerate(axes):
            sns.kdeplot(data = tau_m[i],ax=ax,common_norm=False, fill=True, label = label_list[i])
            ax.axvline(x=np.mean(tau_m[i]), color='red', linestyle='--')
            ax.axvline(x=10., color='green', linestyle='--')
            ax.set_xlim(8.5, 11.5)
            ax.legend()
        plt.tight_layout()
        plt.show()

def simulate(model):
    T = 550
    sample = 15
    train_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_train3.h5')
    test_dataset = SpikingBased_Whisker_Dataset('/data/mosttfzhu/RSNN_bfd/data/whisker/snn_test3.h5')
    train_data0, train_data1, train_data2 = train_dataset.type_specific_data()
    test_data0, test_data1, test_data2 = test_dataset.type_specific_data()
    data0, data1, data2 = np.concatenate((train_data0,test_data0),axis=0),np.concatenate((train_data1,test_data1),axis=0),np.concatenate((train_data2,test_data2),axis=0)

    data0 = torch.tensor(data0, dtype=torch.float32)
    input = data0.view(data0.size(0), data0.size(1), data0.size(2), T, -1).squeeze()

    outputs, h, _, _ = model(input.cuda())
    h = [r.detach().cpu().numpy() for r in h]
    h_state = np.stack(h, axis=0).transpose(1, 0, 2)  # (batch_size, T, N)

    plot_forwardRaster(h_state, sample=sample, seq_len=550)
    plot_5fr(h_state[sample])

if __name__ == '__main__':
    # CV measure on spiking Whisker sweep dataset
    NAOI_along_weights()

    # show neural dynamics of trained model
    trained_model = torch.load('/data/mosttfzhu/RSNN_bfd/Adp_LIF_RSNN_bfd_seed515_0.04b_0.06w_batchsize128_0.818.pth',
                       map_location='cuda')
    neuro_dymodel(model=trained_model, show='L4',parameter='m')

    # plot weight distribution of initial and trained model
    initial_model = SRNN_bfd(input_size=31 * 3, output_size=3, init_b=0.04, init_w=0.06).cuda()
    trained_weight = cal_conn_matrix(trained_model.cpu())
    initial_weight = cal_conn_matrix(initial_model.cpu())
    plot_degree_distribution(initial_weight, trained_weight)

    # raster plot
    simulate(trained_model)

