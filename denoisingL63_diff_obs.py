#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 23:49:38 2025
训练去噪模型，
@author: shenzheqi
"""
import matplotlib.pyplot as plt
#%% 定义模式方程和积分格式
import numpy as np
def Lorenz63(state,*args):       #Lorenz63模式
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state 
    f = np.zeros(3) 
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f 
def RK4(rhs,state,dt,*args):    # Runge-Kutta格式，输入的rhs代表模式右端方程
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args) # 输出新的一步的状态
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state
#%% Lorenz63模式真值试验和观测构造
sigma = 10.0; beta = 8.0/3.0; rho = 28.0          # 模式参数值   
dt = 0.01                                         # 模式积分步长
n = 3                                             # 模式的状态维数，即有多少个状态变量
m = 3                                             # 模式的观测维数，即多少个变量可以被观测到，这里先假定所有变量都能被观测
tm = 1000                                         # 同化试验时间窗口
nt = int(tm/dt)                                   # 总积分步数
t = np.linspace(0,tm,nt+1)                        # 模式时间网格
x0True = np.array([1.508870, -1.531271, 25.46091])# 真实值的初值
np.random.seed(seed=1)                            # 设置随机种子，由于电脑的随机数是伪随机，记录了随机种子之后，每次运行这个脚本产生的“随机”的观测误差都是一样的。
sig_m= 0.5                                        # 观测误差标准差
R = sig_m**2*np.eye(n)                            # 观测误差协方差矩阵，设为对角阵使得不同变量的误差互不相干
dt_m = 0.2                                        # 观测之间的时间间隔（即每20模式步观测一次）
tm_m = 1000                                       # 最大观测时间（即多少时间之后停止同化，可小于同化试验时间窗口）
nt_m = int(tm_m/dt_m)                             # 进行同化的总次数
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)   # 这是有观测的时刻在总时间网格中的位置指标
t_m = t[ind_m]                                                      # 观测网格
def h(x):                                         # 定义观测算子：观测算子用于构建模式变量和观测之间的关系。当所有变量被观测时，使用单位阵。
    H = np.eye(n)                                 # 观测矩阵为单位阵。
    yo = H@x                                      # 单位阵乘以状态变量，即所有变量直接被观测。
    return yo
Xtrue = np.zeros([n,nt+1])                        # 真值保存在xTrue变量中
Xtrue[:,0] = x0True                               # 初始化真值
km = 0                                            # 观测计数
yo = np.zeros([3,nt_m])                           # 观测保存在yo变量中
for k in range(nt):                               # 按模式时间网格开展模式积分循环
    Xtrue[:,k+1] = RK4(Lorenz63,Xtrue[:,k],dt,sigma,beta,rho)       # 真实值积分
    if (km<nt_m) and (k+1==ind_m[km]):                              # 用指标判断是否进行观测
        yo[:,km] = h(Xtrue[:,k+1]) + np.random.normal(0,sig_m,[3,]) # 通过判断，在观测时间取出真值作为观测值，同时叠加高斯分布随机噪声
        km = km+1                                                   # 观测计数，用于循环控制   
#%% 三维变分同化算子
def Lin3dvar(xb,w,H,R,B):                  # 三维变分同化算法
    A = R + H@B@(H.T)
    b = (w-H@xb)
    xa = xb + B@(H.T)@np.linalg.solve(A,b) # 求解线性方程组
    return xa
#%% 三维变分同化实验
x0b = np.array([1,-1,20])                  # 背景积分的初值
Xctl = np.zeros([3,nt+1]); Xctl[:,0] = x0b # 控制试验结果存在xctl中
# --------------- 背景积分实验 ------------------------
for k in range(nt):                        # 模式积分循环
    Xctl[:,k+1] = RK4(Lorenz63,Xctl[:,k],dt,sigma,beta,rho)   # 不加同化的背景积分结果，后面和同化结果进行对比
# --------------- 数据同化实验 ------------------------
sig_b= 1.0                                 # 设定初始的背景误差
B = sig_b**2*np.eye(3)                     # 设定初始背景误差协方差矩阵，B矩阵的取值对于变分同化比较重要，这个简单模式可以使用简单的对角阵
#
Xa = np.zeros([3,nt+1]); Xa[:,0] = x0b     # 同化试验结果存在Xa中，第一步没同化，所以数值也是x0b
Xprior = np.zeros([3,nt_m])                # 观测时刻的先验
Xposterior = np.zeros([3,nt_m])            # 观测时刻的后验
Xtrue_km = np.zeros([3,nt_m])              # 观测时刻的真值也取出
km = 0                                     # 同化次数计数
H = np.eye(3)                              # 如前述，观测算子是单位阵
for k in range(nt):                        # 模式积分循环
    Xa[:,k+1] = RK4(Lorenz63,Xa[:,k],dt,sigma,beta,rho)     # 没有遇到观测的时候，就是正常积分模式
    if (km<nt_m) and (k+1==ind_m[km]):     # 当有观测时，使用3dvar同化
        Xprior[:,km] = Xa[:,k+1]
        Xa[:,k+1] = Lin3dvar(Xa[:,k+1],yo[:,km],H,R,B)      # 调用3dvar，更新状态和协方差
        Xposterior[:,km] = Xa[:,k+1]
        Xtrue_km[:,km] = Xtrue[:,k+1]
        km = km+1
#%% 变分同化效果:图1
# 变分同化结果中的部分
RMSEb = np.sqrt(np.mean((Xctl-Xtrue)**2,0))
RMSEa = np.sqrt(np.mean((Xa-Xtrue)**2,0))

tRMSEpri =  np.sqrt(np.mean((Xprior-Xtrue_km)**2,1))
tRMSEpost = np.sqrt(np.mean((Xposterior-Xtrue_km)**2,1))
tRMSEobs = np.sqrt(np.mean((yo-Xtrue_km)**2,1))
tRMSEpri_o =  np.sqrt(np.mean((Xprior-yo)**2,1))
tRMSEpost_o = np.sqrt(np.mean((Xposterior-yo)**2,1))
RMSEpri = np.sqrt(np.mean((Xprior-Xtrue_km)**2,0))
RMSEpost = np.sqrt(np.mean((Xposterior-Xtrue_km)**2,0))

# plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# bar_index = np.arange(3)
# bar_width = 0.25
# # 绘制条形图
# bars1 = plt.bar(bar_index, tRMSEpri, bar_width, color = '#004488', label='Prior', alpha=0.8)
# bars2 = plt.bar(bar_index + bar_width, tRMSEobs, bar_width, color = '#006600', label='Observation', alpha=0.8)
# bars3 = plt.bar(bar_index + 2 * bar_width, tRMSEpost, bar_width, color = '#880000', label='Posterior', alpha=0.8)
# # 添加数值标签
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2.+0.1, height,
#                  f'{height:.2f}',
#                  ha='center', va='bottom', rotation=30,fontsize=11)
# add_labels(bars1)
# add_labels(bars2)
# add_labels(bars3)
# plt.legend(ncol=2,fontsize=13)
# plt.title('RMSE',fontsize=17)
# plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
# plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
# plt.grid(axis='y',alpha=0.4)
# plt.subplot(2,2,2)
# bar_index = np.arange(3)
# bar_width = 0.3
# # 绘制条形图
# bars1 = plt.bar(bar_index, tRMSEpri_o, bar_width, color = '#004488', label='Prior', alpha=0.8)
# bars2 = plt.bar(bar_index +  bar_width, tRMSEpost_o, bar_width, color = '#880000', label='Posterior', alpha=0.8)

# add_labels(bars1)
# add_labels(bars2)
# plt.legend(ncol=2,fontsize=13)
# plt.title('RMSE w.r.t Obs.',fontsize=17)
# plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
# plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
# plt.grid(axis='y',alpha=0.4)

# plt.subplot(2,1,2)
# plt.plot(t_m,RMSEpri,color='#004488',alpha=0.8,label='Prior')
# plt.plot(t_m,RMSEpost,color='#880000',alpha=0.8,label='Posterior')
# plt.plot(t,RMSEb,'--',color='C1',alpha=0.4,label='NoAssim')
# plt.ylabel('RMSE',fontsize=15)
# plt.xlabel('Time/TUs',fontsize=15)
# plt.xticks(fontsize=15);plt.yticks(np.arange(0,31,10),fontsize=15)
# plt.legend(ncol=2, loc='upper right',fontsize=13)
# plt.xlim(0,1000);plt.ylim(-0.1,30)
#%% 训练分数匹配模型，从先验去噪得到后验，观测用于约束
tr_steps = 4500    # 用于训练的步数
tt_steps = nt_m - tr_steps # 用于测试的

import torch
import torch.nn as nn
import torch.optim as optim
# 转换为 PyTorch 张量，可用不同数据训练，比较结果
#noisy_data = torch.tensor(Xtrue_km[:,range(0,tr_steps)].T, dtype=torch.float32)
noisy_data = torch.tensor(Xposterior[:,range(0,tr_steps)].T, dtype=torch.float32)
#noisy_data = torch.tensor(Xprior[:,range(0,tr_steps)].T, dtype=torch.float32)
observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
# 分数模型
class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # 输入维度为6，因为输入是 [noise_state, observation]
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def score_matching_loss(score_model, noisy_data, observation, sigma=0.1,lambda_reg=0.01):
    noise = torch.randn_like(noisy_data) * sigma
    perturbed_data = noisy_data + noise
    inputs = torch.cat([perturbed_data, observation], dim=1)  # 输入是 [noise_state, observation]
    score = score_model(inputs)
    #loss = torch.mean(torch.sum(score ** 2, dim=1) / 2 + torch.sum(score * noise, dim=1) / sigma ** 2)
    loss = 0.5 * torch.mean(torch.sum((score*(sigma**2)+ noise)**2,dim=1)/(sigma ** 2))
    return loss

# 初始化模型和优化器
score_model = ScoreModel()
optimizer = optim.Adam(score_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = score_matching_loss(score_model, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 定义去噪同化过程
def denoise(score_model, noisy_sample, observation, num_steps=100, step_size=0.01):
    sample = noisy_sample.clone().requires_grad_(True)
    denoised_steps = [sample.clone().detach()]  # 保存每一步的结果
    for _ in range(num_steps):
        inputs = torch.cat([sample, observation], dim=1)  # 输入是 [noise_state, observation]
        score = score_model(inputs)
        sample = sample + step_size * score
        denoised_steps.append(sample.clone().detach())  # 保存每一步的结果
    return sample.detach(), denoised_steps
#%% 模型测试：去噪能力
yo_val = yo[:,range(tr_steps,nt_m)]
Xprior_val = Xprior[:,range(tr_steps,nt_m)]
Xposterior_val = Xposterior[:,range(tr_steps,nt_m)]
Xtrue_val = Xtrue_km[:,range(tr_steps,nt_m)]
X_est = np.zeros_like(Xprior_val)
# for k in range(tt_steps):
#     noisy_sample = torch.tensor(Xprior_val[:,k].T, dtype=torch.float32).unsqueeze(0)
#     observation_sample = torch.tensor(yo_val[:,k].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
#     denoised_sample,denoised_steps = denoise(score_model, noisy_sample, observation_sample)
#     X_est[:,k] = denoised_sample.squeeze(0)
# #%% 先验去噪结果和真值比较：图2
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,8))
# lbs = ['x','y','z']
# for j in range(3):
#     plt.subplot(3,2,2*j+1)
#     plt.plot(0.2*np.arange(0,500),X_est[j]-Xtrue_val[j],color='#444444',lw=1.5,label='Denoise', alpha=0.6)
#     plt.plot(0.2*np.arange(0,500),Xprior_val[j]-Xtrue_val[j],color='#004488',lw=1.5,label='Prior', alpha=0.6)
#     plt.plot(0.2*np.arange(0,500),yo_val[j]-Xtrue_val[j],color='#006600',lw=1.5,label='Obs.', alpha=0.6)
#     plt.plot(0.2*np.arange(0,500),Xposterior_val[j]-Xtrue_val[j],color='#880000',lw=1.5,label='Posterior', alpha=0.6)
#     plt.ylabel(lbs[j],fontsize=15)
#     plt.yticks(np.arange(-4,5,2),fontsize=15)
#     if j==0:
#         plt.title('Bias',fontsize=17)
#         plt.legend(ncol=2,fontsize=13)
#     if j==2:
#         plt.xlabel('Time',fontsize=15)
#         plt.xticks(np.arange(0,100,20),fontsize=15);
#     else:
#         plt.xticks(np.arange(0,101,20),[],fontsize=15);
#     plt.ylim(-4,4);plt.xlim(0,100)
# rmsed = np.sqrt(np.mean((X_est-Xtrue_val)**2,axis=1))
# rmsep = np.sqrt(np.mean((Xprior_val-Xtrue_val)**2,axis=1))
# rmseo = np.sqrt(np.mean((yo_val-Xtrue_val)**2,axis=1))
# rmses = np.sqrt(np.mean((Xposterior_val-Xtrue_val)**2,axis=1))
# plt.subplot(1,2,2)
# bar_index = np.arange(3)
# bar_width = 0.15
# bars1 = plt.bar(bar_index, rmsed, bar_width,color="#444444", label='Denoise', alpha=0.8)
# bars2 = plt.bar(bar_index + bar_width, rmsep, bar_width, color="#004488", label='Prior', alpha=0.8)
# bars3 = plt.bar(bar_index + 2 * bar_width, rmseo, bar_width, color="#006600",label='Observation', alpha=0.8)
# bars4 = plt.bar(bar_index + 3 * bar_width, rmses, bar_width, color="#880000",label='Posterior', alpha=0.8)
# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                  f'{height:.2f}',
                  ha='center', va='bottom', rotation=45,fontsize=11)
# add_labels(bars1)
# add_labels(bars2)
# add_labels(bars3)
# add_labels(bars4)
# plt.legend(fontsize=13)
# plt.ylim(0,1.6);plt.yticks(np.arange(0,1.51,0.5),fontsize=15)
# plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
# plt.title('RMSE',fontsize=17)
# plt.legend(ncol=2,fontsize=13)
# plt.grid(axis='y')
#%% 只使用验证数据的初值，基于denoise模型进行全程的同化循环。
yo_test = yo[:,range(tr_steps,nt_m)]
x0b_test = Xa[:,ind_m[tr_steps-1]]
Xtrue_test = Xtrue[:,ind_m[tr_steps-1]:ind_m[nt_m-1]+1]
Xa_3dvar = Xa[:,ind_m[tr_steps-1]:ind_m[nt_m-1]+1]
nt_test = tt_steps*20
t_test = np.linspace(0,nt_test*dt,nt_test+1)
t_mtest = t_test[ind_m[range(tt_steps)]]  
#%% Denoise同化
# --------------- 数据同化实验 ------------------------
Xa_test = np.zeros([3,nt_test+1]); Xa_test[:,0] = x0b_test   # 同化试验结果存在Xa中，第一步没同化，所以数值也是x0b
Xprior_test = np.zeros_like(Xprior_val);
Xposterior_test = np.zeros_like(Xposterior_val);
km = 0                                   # 同化次数计数
for k in range(nt_test):                      # 模式积分循环
    Xa_test[:,k+1] = RK4(Lorenz63,Xa_test[:,k],dt,sigma,beta,rho)   # 没有遇到观测的时候，就是正常积分模式
    if (km<nt_m) and (k+1==ind_m[km]):   
        Xprior_test[:,km] = Xa_test[:,k+1]
        noisy_sample = torch.tensor(Xa_test[:,k+1].T, dtype=torch.float32).unsqueeze(0)
        observation_sample = torch.tensor(yo_test[:,km].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
        denoised_sample,denoised_steps = denoise(score_model, noisy_sample, observation_sample)
        analysis_sample = denoised_sample.squeeze(0)
        Xa_test[:,k+1] = analysis_sample.numpy()
        Xposterior_test[:,km] = Xa_test[:,k+1]
        km = km+1
#%% 图3：顺序同化结果画图
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,6))
lbs = ['x','y','z']
gs = gridspec.GridSpec(4, 4)
for j in range(3):
    ax1 = fig.add_subplot(gs[j, :3])
    ax1.plot(t_test,Xtrue_test[j],linestyle=":",color = 'black',lw=1,label='True')
    ax1.plot(t_mtest,yo_test[j], marker='o', linestyle='None',color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=5,label='Observation')
    ax1.plot(t_test,Xa_3dvar[j],color = '#FF8800',lw=1,label='3D-Var')
    ax1.plot(t_test,Xa_test[j],color='#004488',lw=1,label='Denoise')    
    plt.ylabel(lbs[j],fontsize=15)
    plt.xticks(np.arange(0,100,20),[]);plt.yticks(fontsize=15)
    if j==0:
        plt.legend(ncol=4, loc='upper center',fontsize=13)
        plt.title("DA with Denoise Model",fontsize=17)
    plt.xlim(19.9,nt_test*dt)
RMSE3dvar = np.sqrt(np.mean((Xa_3dvar-Xtrue_test)**2,0))
RMSEtest = np.sqrt(np.mean((Xa_test-Xtrue_test)**2,0))
rmse_3dvar = np.sqrt(np.mean((Xa_3dvar-Xtrue_test)**2,axis=1))
rmse_test = np.sqrt(np.mean((Xa_test - Xtrue_test) ** 2, axis=1))
ax1 = fig.add_subplot(gs[3, :3])
ax1.plot(t_test,RMSE3dvar,color='#FF8800',lw=1,label='3D-Var')
ax1.plot(t_test,RMSEtest,color='#004488',lw=1,label='Denoise')
plt.ylabel('RMSE',fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.xticks(np.arange(20,101,20),fontsize=15);plt.yticks(fontsize=15)
plt.legend(ncol=2, loc='center left',fontsize=13)
plt.xlim(19.9,nt_test*dt)
for x in [56.4,  82.6]:
    plt.axvline(x=x, color='gray', linestyle=':', lw=2)
    plt.text(x-3, 7, f'{x}', ha='center', va='top', fontsize=12, color='black')

ax2 = fig.add_subplot(gs[:4, 3])
bar_index = np.arange(3)
bar_width = 0.3
# 绘制条形图
bars1 = plt.bar(bar_index, rmse_3dvar, bar_width, color = '#FF8800', label='3D-Var', alpha=0.8)
bars2 = plt.bar(bar_index +  bar_width, rmse_test, bar_width, color = '#004488', label='Denoise', alpha=0.8)
ax2.yaxis.set_ticks_position('right')
add_labels(bars1)
add_labels(bars2)
plt.legend(ncol=1,fontsize=13)
plt.title('RMSE',fontsize=17)
plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
plt.grid(axis='y',alpha=0.4)
#%%
# 添加噪声生成观测数据
def add_noise(data, noise_level=0.1):
    return data + noise_level * np.random.randn(*data.shape)
# yo2 = add_noise(Xtrue_km, noise_level=0.5)  # 观测数据
# yo2_test = yo2[:,range(tr_steps,nt_m)]
#%%
def assimilation_experiment_diffY(yo_test,score_model):
    """
    同化实验函数
    参数:
    yo_test: 观测数据
    score_model: 用于去噪的模型

    返回值:
    rmse_test: 同化结果的均方根误差
    """
    # 初始化同化结果数组
    Xa_test = np.zeros([3, nt_test + 1])
    Xa_test[:, 0] = x0b_test  # 初始背景场
    # 同化次数计数
    km = 0
    # 模式积分循环
    for k in range(nt_test):
        # 正常积分模式
        Xa_test[:, k + 1] = RK4(Lorenz63, Xa_test[:, k], dt, sigma, beta, rho)
        # 如果当前时间步是观测时间步
        if (km < nt_m) and (k + 1 == ind_m[km]):
            # 将当前状态转换为张量
            noisy_sample = torch.tensor(Xa_test[:, k + 1].T, dtype=torch.float32).unsqueeze(0)
            observation_sample = torch.tensor(yo_test[:, km].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
            # 去噪过程
            denoised_sample, denoised_steps = denoise(score_model, noisy_sample, observation_sample)
            analysis_sample = denoised_sample.squeeze(0)
            # 更新同化结果
            Xa_test[:, k + 1] = analysis_sample.numpy()
            # 更新同化次数计数
            km += 1
    # 计算均方根误差
    rmse_test = np.sqrt(np.mean((Xa_test - Xtrue_test) ** 2, axis=1))
    return rmse_test
#%% 同一个模型，不同的观测进行去噪同化实验。
rmse_testS = np.zeros([4,3])
noise_levels = np.array([1,0.5,0.1,0.05])
for j in range(4):
    yos = add_noise(Xtrue_km, noise_level=noise_levels[j])  # 观测数据
    rmse_testS[j] = assimilation_experiment_diffY(yos[:,range(tr_steps,nt_m)],score_model)  
#%%
"""
验证阶段的真值是Xtrue_test
初值是x0b_test
"""
def assimilation_dif_freq(dt_m):
    # 定义全局变量
    global dt, Xtrue_test, x0b_test, Lorenz63, RK4, score_model, denoise
    
    # 观测和同化相关参数
    tm_m = 100  # 最大观测时间
    nt_m = int(tm_m / dt_m)  # 进行同化的总次数
    ind_m = (np.linspace(int(dt_m / dt), int(tm_m / dt), nt_m)).astype(int)  # 观测时刻在总时间网格中的位置指标
    tm = 100  # 同化试验时间窗口
    nt = int(tm / dt)  # 总积分步数
    t = np.linspace(0, tm, nt + 1)  # 模式时间网格
    t_m = t[ind_m]
    
    # 添加噪声
    yo = add_noise(Xtrue_test[:, ind_m], noise_level=0.5)
    
    # 初始化同化结果数组
    Xa_test = np.zeros([3, nt + 1])
    Xa_test[:, 0] = x0b_test  # 初始背景场
    
    # 同化次数计数
    km = 0
    
    # 模式积分循环
    for k in range(nt):
        # 正常积分模式
        Xa_test[:, k + 1] = RK4(Lorenz63, Xa_test[:, k], dt, sigma, beta, rho)
        
        # 如果当前时间步是观测时间步
        if (km < nt_m) and (k + 1 == ind_m[km]):
            noisy_sample = torch.tensor(Xa_test[:, k + 1].T, dtype=torch.float32).unsqueeze(0)
            observation_sample = torch.tensor(yo[:, km].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
            denoised_sample, denoised_steps = denoise(score_model, noisy_sample, observation_sample)
            analysis_sample = denoised_sample.squeeze(0)
            Xa_test[:, k + 1] = analysis_sample.numpy()
            km += 1
    
    # 计算均方根误差
    rmse_test = np.sqrt(np.mean((Xa_test - Xtrue_test) ** 2, axis=1))
    
    return rmse_test
#%%
rmse_test0 = assimilation_dif_freq(0.25)
rmse_test1 = assimilation_dif_freq(0.1)
rmse_test2 = assimilation_dif_freq(0.05)


#%%
plt.figure(figsize=(12, 8))
# 设置条形图的索引和宽度
bar_width = 0.2  # 条形图的宽度
index = np.arange(rmse_testS.shape[1])  # 3个分类
plt.subplot(2,1,1)
# 绘制条形图
bars1 = plt.bar(index , rmse_testS[0, :], bar_width, color='#004488', label=r'$\sigma$ = '+str(noise_levels[0]))
bars2 = plt.bar(index +  bar_width, rmse_testS[1, :], bar_width, color='#FF8800', label=r'$\sigma$ = '+str(noise_levels[1]))
bars3 = plt.bar(index +  2*bar_width, rmse_testS[2, :], bar_width, color='#006600', label=r'$\sigma$ = '+str(noise_levels[2]))
bars4 = plt.bar(index +  3*bar_width, rmse_testS[3, :], bar_width, color='#880000', label=r'$\sigma$ = '+str(noise_levels[3]))
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
# 添加图例
plt.ylim(0,2)
plt.legend(ncol=2,fontsize=15, loc='best')
# 添加标题
plt.title(r'Denoise with $\delta$t=0.2', fontsize=17)
# 设置y轴范围和刻度
plt.yticks(np.arange(0,1.6,0.3),fontsize=15)
# 设置x轴刻度
plt.xticks(index + bar_width * (rmse_testS.shape[0] - 1) / 2, [], fontsize=15)
# 添加网格线
plt.grid(axis='y', alpha=0.4)

plt.subplot(2,1,2)
# 绘制条形图
bars1 = plt.bar(index , rmse_test0, bar_width, color='#004488', label=r'$\delta$t=0.25')
bars2 = plt.bar(index +  bar_width, rmse_testS[1, :], bar_width, color='#FF8800', label=r'$\delta$t=0.2')
bars3 = plt.bar(index +  2*bar_width, rmse_test1, bar_width, color='#006600', label=r'$\delta$t=0.1')
bars4 = plt.bar(index +  3*bar_width, rmse_test2, bar_width, color='#880000', label=r'$\delta$t=0.05')
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
# 添加图例
plt.ylim(0,1.5)
plt.legend(ncol=2,fontsize=15, loc='best')
# 添加标题
plt.title(r'Denoise with $\sigma=0.5$', fontsize=17)
# 设置y轴范围和刻度
plt.yticks(np.arange(0,1.3,0.3),fontsize=15)
# 设置x轴刻度
plt.xticks(index + bar_width * (rmse_testS.shape[0] - 1) / 2, ['x', 'y', 'z'], fontsize=15)
# 添加网格线
plt.grid(axis='y', alpha=0.4)

plt.savefig('fig8.pdf', format='pdf', dpi=300, bbox_inches='tight')