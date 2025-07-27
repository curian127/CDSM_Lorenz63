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

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
bar_index = np.arange(3)
bar_width = 0.25
# 绘制条形图
bars1 = plt.bar(bar_index, tRMSEpri, bar_width, color = '#004488', label='Prior', alpha=0.8)
bars2 = plt.bar(bar_index + bar_width, tRMSEobs, bar_width, color = '#006600', label='Observation', alpha=0.8)
bars3 = plt.bar(bar_index + 2 * bar_width, tRMSEpost, bar_width, color = '#880000', label='Posterior', alpha=0.8)
# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.+0.1, height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=30,fontsize=11)
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
plt.legend(ncol=2,fontsize=13)
plt.title('RMSE',fontsize=17)
plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
plt.grid(axis='y',alpha=0.4)
plt.subplot(2,2,2)
bar_index = np.arange(3)
bar_width = 0.3
# 绘制条形图
bars1 = plt.bar(bar_index, tRMSEpri_o, bar_width, color = '#004488', label='Prior', alpha=0.8)
bars2 = plt.bar(bar_index +  bar_width, tRMSEpost_o, bar_width, color = '#880000', label='Posterior', alpha=0.8)

add_labels(bars1)
add_labels(bars2)
plt.legend(ncol=2,fontsize=13)
plt.title('RMSE w.r.t Obs.',fontsize=17)
plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
plt.grid(axis='y',alpha=0.4)

plt.subplot(2,1,2)
plt.plot(t_m,RMSEpri,color='#004488',alpha=0.8,label='Prior')
plt.plot(t_m,RMSEpost,color='#880000',alpha=0.8,label='Posterior')
plt.plot(t,RMSEb,'--',color='C1',alpha=0.4,label='NoAssim')
plt.ylabel('RMSE',fontsize=15)
plt.xlabel('Time/TUs',fontsize=15)
plt.xticks(fontsize=15);plt.yticks(np.arange(0,31,10),fontsize=15)
plt.legend(ncol=2, loc='upper right',fontsize=13)
plt.xlim(0,1000);plt.ylim(-0.1,30)
#plt.savefig('fig1.pdf', format='pdf', dpi=300, bbox_inches='tight')

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
for k in range(tt_steps):
    noisy_sample = torch.tensor(Xprior_val[:,k].T, dtype=torch.float32).unsqueeze(0)
    observation_sample = torch.tensor(yo_val[:,k].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
    denoised_sample,denoised_steps = denoise(score_model, noisy_sample, observation_sample)
    X_est[:,k] = denoised_sample.squeeze(0)
#%% 先验去噪结果和真值比较：图2
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors = ['#1a5c8a',  '#cc6600', '#1f801f',  '#a61c1c' ]
linestyles = ['-', ':', '--', '--']
labels = ['Denoise', 'Prior', 'Obs.', 'Posterior']
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
lbs = ['x','y','z']
for j in range(3):
    plt.subplot(3,2,2*j+1)
    plt.plot(0.2*np.arange(0,500),X_est[j]-Xtrue_val[j],color=colors[0],linestyle=linestyles[0],lw=2,label=labels[0], alpha=0.8)
    plt.plot(0.2*np.arange(0,500),Xprior_val[j]-Xtrue_val[j],color=colors[1],linestyle=linestyles[1],lw=2,label=labels[1], alpha=0.8)
    plt.plot(0.2*np.arange(0,500),yo_val[j]-Xtrue_val[j],color=colors[2],linestyle=linestyles[2],lw=2,label=labels[2], alpha=0.8)
    plt.plot(0.2*np.arange(0,500),Xposterior_val[j]-Xtrue_val[j],color=colors[3],linestyle=linestyles[3],lw=2,label=labels[3], alpha=0.8)
    plt.ylabel(lbs[j],fontsize=15)
    plt.yticks(np.arange(-4,5,2),fontsize=15)
    if j==0:
        plt.title('Bias',fontsize=17)
        plt.legend(ncol=3,fontsize=12)
        plt.ylim(-3,2)
        plt.xticks(np.arange(0,101,10),[],fontsize=15);
    if j==1:
        plt.ylim(-2,2)
        plt.xticks(np.arange(0,101,10),[],fontsize=15);
    if j==2:
        plt.xlabel('Time',fontsize=15)
        plt.xticks(np.arange(0,101,10),fontsize=15);
        plt.ylim(-2,2)
        
    plt.xlim(70,100)
rmsed = np.sqrt(np.mean((X_est-Xtrue_val)**2,axis=1))
rmsep = np.sqrt(np.mean((Xprior_val-Xtrue_val)**2,axis=1))
rmseo = np.sqrt(np.mean((yo_val-Xtrue_val)**2,axis=1))
rmses = np.sqrt(np.mean((Xposterior_val-Xtrue_val)**2,axis=1))
plt.subplot(1,2,2)
bar_index = np.arange(3)
bar_width = 0.15
bars1 = plt.bar(bar_index, rmsed, bar_width,color=colors[0], label='Denoise', alpha=0.8)
bars2 = plt.bar(bar_index + bar_width, rmsep, bar_width, color=colors[1], label='Prior', alpha=0.8)
bars3 = plt.bar(bar_index + 2 * bar_width, rmseo, bar_width, color=colors[2],label='Observation', alpha=0.8)
bars4 = plt.bar(bar_index + 3 * bar_width, rmses, bar_width, color=colors[3],label='Posterior', alpha=0.8)
# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=45,fontsize=11)
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
plt.legend(fontsize=13)
plt.ylim(0,1.6);plt.yticks(np.arange(0,1.51,0.5),fontsize=15)
plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
plt.title('RMSE',fontsize=17)
plt.legend(ncol=2,fontsize=13)
plt.grid(axis='y')
#plt.savefig('fig2.pdf', format='pdf', dpi=300, bbox_inches='tight')
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,6))
lbs = ['x','y','z']
gs = gridspec.GridSpec(4, 4)
for j in range(3):
    ax1 = fig.add_subplot(gs[j, :3])
    ax1.plot(t_test,Xtrue_test[j],linestyle=":",color = 'black',lw=1,label='True')
    ax1.plot(t_mtest,yo_test[j], marker='o', linestyle='None',color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=5,label='Observation')
    ax1.plot(t_test,Xa_3dvar[j],color = '#FF8800',lw=1,label='3D-Var')
    ax1.plot(t_test,Xa_test[j],color='#004488',lw=1,label='CDSM')    
    plt.ylabel(lbs[j],fontsize=15)
    plt.xticks(np.arange(0,100,20),[]);plt.yticks(fontsize=15)
         
    plt.xlim(19.9,nt_test*dt)
    if j==0:
        plt.ylim(-25,25)
        plt.title("DA results with CDSM",fontsize=17)
    if j==1:
        plt.ylim(-30,30)
    if j==2:
        plt.legend(ncol=4, loc='upper left',fontsize=13)
        plt.ylim(0,80)
RMSE3dvar = np.sqrt(np.mean((Xa_3dvar-Xtrue_test)**2,0))
RMSEtest = np.sqrt(np.mean((Xa_test-Xtrue_test)**2,0))
rmse_3dvar = np.sqrt(np.mean((Xa_3dvar-Xtrue_test)**2,axis=1))
rmse_test = np.sqrt(np.mean((Xa_test - Xtrue_test) ** 2, axis=1))
ax1 = fig.add_subplot(gs[3, :3])
ax1.plot(t_test,RMSE3dvar,color='#FF8800',lw=1,label='3D-Var')
ax1.plot(t_test,RMSEtest,color='#004488',lw=1,label='CDSM')
plt.ylabel('RMSE',fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.xticks(np.arange(20,101,20),fontsize=15);plt.yticks(fontsize=15)
plt.legend(ncol=2, loc='upper left',fontsize=13)
plt.xlim(19.9,nt_test*dt)
for x in [56.4,  82.6]:
    plt.axvline(x=x, color='gray', linestyle=':', lw=2)
    plt.text(x+3, 7, f'{x}', ha='center', va='top', fontsize=12, color='black')

ax2 = fig.add_subplot(gs[:4, 3])
bar_index = np.arange(3)
bar_width = 0.3
# 绘制条形图
bars1 = plt.bar(bar_index, rmse_3dvar, bar_width, color = '#FF8800', label='3D-Var', alpha=0.8)
bars2 = plt.bar(bar_index +  bar_width, rmse_test, bar_width, color = '#004488', label='CDSM', alpha=0.8)
ax2.yaxis.set_ticks_position('right')
add_labels(bars1)
add_labels(bars2)
plt.legend(ncol=1,fontsize=13,loc="upper left")
plt.title('RMSE',fontsize=17)
plt.ylim(0,1.6);plt.yticks(np.arange(0,1.5,0.5),fontsize=15)
plt.xticks([0.2,1.2,2.2],['x','y','z'],fontsize=15)
plt.grid(axis='y',alpha=0.4)
#plt.savefig('fig3.pdf', format='pdf', dpi=300, bbox_inches='tight')
#%% 图4：诊断图3同化差异原因，判断是否相位变换
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(Xtrue_test[0], Xtrue_test[1], Xtrue_test[2],color='gray',alpha=0.2)
ax.plot(Xtrue_test[0,5580:5661], Xtrue_test[1,5580:5661],Xtrue_test[2,5580:5661],'k:', lw=1.5,label='True')
ax.plot(Xa_test[0,5580:5641], Xa_test[1,5580:5641],Xa_test[2,5580:5641],color='#004488', lw=2,label='CDSM')
ax.plot(Xa_3dvar[0,5580:5641], Xa_3dvar[1,5580:5641],Xa_3dvar[2,5580:5641],color='#FF8800',linestyle='--', lw=2, label='3D-Var')
ax.plot(yo_val[0,278:283], yo_val[1,278:283],yo_val[2,278:283],linestyle='None',marker='o', color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=5,label='Obs.')
plt.xlabel('x');plt.ylabel('y');
ax.set_zlabel('z')
plt.legend(ncol=1,loc="upper right",fontsize=14)
plt.title(r'$t\in[55.8,56.4]$',fontsize=14)

ax = fig.add_subplot(122, projection='3d')
ax.plot(Xtrue_test[0], Xtrue_test[1], Xtrue_test[2],color='gray',alpha=0.2)
ax.plot(Xtrue_test[0,8180:8281], Xtrue_test[1,8180:8281],Xtrue_test[2,8180:8281],'k:', lw=1.5,label='True')
ax.plot(Xa_test[0,8180:8261], Xa_test[1,8180:8261],Xa_test[2,8180:8261],color='#004488', lw=2,label='CDSM')
ax.plot(Xa_3dvar[0,8180:8261], Xa_3dvar[1,8180:8261],Xa_3dvar[2,8180:8261],color='#FF8800',linestyle='--', lw=2, label='3D-Var')
ax.plot(yo_val[0,408:414], yo_val[1,408:414],yo_val[2,408:414],linestyle='None',marker='o', color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=5,label='Obs.')
plt.xlabel('x');plt.ylabel('y');
ax.set_zlabel('z')
plt.title(r'$t\in[81.8,82.8]$',fontsize=14)
# plt.savefig('fig4.pdf', format='pdf', dpi=300, bbox_inches='tight')

#%% 图5：用其中一次同化查看不同机制，使用Xprior_test[:,280]，即t=56这个时间的先验进行变分同化或者去噪
x0test = Xprior_test[:,280];y0test=yo_val[:,280]
x0tr = Xtrue_val[:,280]
#
noisy_sample = torch.tensor(x0test, dtype=torch.float32).unsqueeze(0)
observation_sample = torch.tensor(y0test, dtype=torch.float32).unsqueeze(0)
denoised_sample,denoised_steps  = denoise(score_model, noisy_sample , observation_sample )
x1test = denoised_sample.squeeze(0)

x1var = Lin3dvar(x0test, y0test, H, R, B)
#% 画路径
numpy_arrays = [step.numpy() for step in denoised_steps]
# 将所有NumPy数组堆叠成一个101x3的数组
steps_array = np.stack(numpy_arrays)
steps_array= np.squeeze(steps_array, axis=1)
# 创建图形
fig = plt.figure(figsize=(12, 6))

# 左边三个子图，分别绘制三个变量的变化
variables = ['x', 'y', 'z']
colors = ['#004488', '#004488', '#004488']  # 为每个变量指定颜色
markers = ['o', 's', 'D']  # 为每个变量指定不同的 marker

for j in range(3):
    plt.subplot(3, 2, 2 * j + 1)
    plt.plot(steps_array[:, j], color=colors[j], lw=2.5, marker=markers[j], markevery=5, markersize=6, label=f'{variables[j]}')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 20), fontsize=14)
    plt.yticks(fontsize=14)
    # 添加两条横线
    plt.axhline(y=x0test[j], color='gray', linestyle='--', lw=1.5, label=f'Prior {variables[j]}')
    plt.axhline(y=y0test[j], color='red', linestyle='-.', lw=1.5, label=f'Observation {variables[j]}')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'{variables[j]}', fontsize=14)
    plt.legend(fontsize=12)

# 右边的子图，绘制轨迹
plt.subplot(1, 2, 2)
plt.plot(x0test[0], x0test[2],marker='o', color='white', markeredgecolor='#004488', markeredgewidth=2, markersize=10, label='Prior')
plt.plot(x1var[0], x1var[2], marker='s', color='white', markeredgecolor='#FF8800', markeredgewidth=2, markersize=10, label='3DVAR analysis')
plt.plot(x1test[0], x1test[2], marker='s', color='white', markeredgecolor='#004488', markeredgewidth=2, markersize=10, label='Denoise analysis')
plt.plot(steps_array[:, 0], steps_array[:, 2], color='#004488',marker='o', markevery=5, markersize=5,linestyle='--',lw=2, label='Denoising Trace')
plt.scatter(y0test[0], y0test[2], marker='x', color='red', s=150,lw=2, label='Observation')
plt.scatter(x0tr[0], x0tr[2], marker='+', color='black', s=150, lw=2,label='True')
plt.xlabel('x', fontsize=14)
plt.ylabel('z', fontsize=14)
plt.xlim(-2.5,0.5),plt.xticks(np.arange(-2,0.6,1),fontsize=14)
plt.ylim(9.4,10.4),plt.yticks(np.arange(9.5,10.3,0.3),fontsize=14)
plt.legend(fontsize=12)

plt.annotate('', xy=(x1var[0], x1var[2]), xytext=(x0test[0], x0test[2]),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
plt.grid()
plt.tight_layout()
#plt.savefig('fig5.pdf', format='pdf', dpi=300, bbox_inches='tight')
#%%
import matplotlib.pyplot as plt

# 假设以下变量是你的数据
# x0test: 初始先验点
# x1var: 3D-Var分析结果点
# x1test: 去噪分析结果点
# steps_array: 去噪轨迹
# y0test: 观测点
# x0tr: 真值点

# 示例代码（请用你的实际数据替换）
# x0test = np.array([[...], [...], [...]])
# x1var = np.array([[...], [...], [...]])
# x1test = np.array([[...], [...], [...]])
# steps_array = np.array([[...], [...], [...]])
# y0test = np.array([[...], [...], [...]])
# x0tr = np.array([[...], [...], [...]])

# 创建一个图形和两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1]})

# 左边的子图：3D-Var分析结果和箭头
ax1.plot(x0test[0], x0test[2], marker='o', color='white', markeredgecolor='#004488', markeredgewidth=3, markersize=10, label='Prior')
ax1.plot(x1var[0], x1var[2], marker='s', color='white', markeredgecolor='#FF8800', markeredgewidth=3, markersize=10, label='3DVAR analysis')
ax1.annotate('', xy=(x1var[0], x1var[2]), xytext=(x0test[0], x0test[2]),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax1.scatter(y0test[0], y0test[2], marker='x', color='red', s=300, lw=3, label='Observation')
ax1.scatter(x0tr[0], x0tr[2], marker='+', color='black', s=300, lw=3, label='True')

#ax1.set_xlabel('x', fontsize=14)
#ax1.set_ylabel('z', fontsize=14)
ax1.set_xlim(-2.5, 0.5)
ax1.set_ylim(9.4, 10.4)
ax1.set_xticks([])   # 去掉 x 轴刻度
ax1.set_yticks([])   # 去掉 y 轴刻度
#ax1.legend(fontsize=12)

# 右边的子图：整个Xa_test轨迹
# 左边的子图：3D-Var分析结果和箭头
ax2.plot(x0test[0], x0test[2], marker='o', color='white', markeredgecolor='#004488', markeredgewidth=3, markersize=10, label='Prior')
ax2.plot(x1test[0], x1test[2], marker='s', color='white', markeredgecolor='#004488', markeredgewidth=3, markersize=10, label='Denoise analysis')
ax2.plot(steps_array[:, 0], steps_array[:, 2], color='#004488', marker='o', markevery=3, markersize=5, linestyle='--', lw=3, label='Denoising Trace')
ax2.scatter(y0test[0], y0test[2], marker='x', color='red', s=300, lw=3, label='Observation')
ax2.scatter(x0tr[0], x0tr[2], marker='+', color='black', s=300, lw=3, label='True')
#ax2.set_xlabel('x', fontsize=14)
#ax2.set_ylabel('z', fontsize=14)
ax2.set_xlim(-2.5, 0.5)
ax2.set_ylim(9.4, 10.4)
ax2.set_xticks([])   # 去掉 x 轴刻度
ax2.set_yticks([])   # 去掉 y 轴刻度
#ax2.legend(fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
#%% 训练其他去噪模型：模型1，DM-ctrl，背景训练去噪
noisy_data = torch.tensor(Xctl[:,20:90001:20].T, dtype=torch.float32)
observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
score_model1 = ScoreModel()
optimizer1 = optim.Adam(score_model1.parameters(), lr=0.001)
num_epochs  = 1000
for epoch in range(num_epochs):
    optimizer1.zero_grad()
    loss = score_matching_loss(score_model1, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model1.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer1.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 模型2，DM-3dvarBck，先验训练去噪
noisy_data = torch.tensor(Xprior[:,range(0,tr_steps)].T, dtype=torch.float32)
# observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
score_model2 = ScoreModel()
optimizer2 = optim.Adam(score_model2.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer2.zero_grad()
    loss = score_matching_loss(score_model2, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model2.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer2.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 模型3，DM-3dvarAna，后验训练去噪
score_model3 = score_model
#%% 模型4，DM-enkfBck，EnKF先验训练去噪
npz_file = np.load("EnKF_analysis.npz")  
XpriorEnkf = npz_file["Xprior"] 
noisy_data = torch.tensor(XpriorEnkf[:,range(0,tr_steps)].T, dtype=torch.float32)
# observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
score_model4 = ScoreModel()
optimizer4 = optim.Adam(score_model4.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer4.zero_grad()
    loss = score_matching_loss(score_model4, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model4.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer4.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 模型5，DM-enkfAna，EnKF后验训练去噪
XposteriorEnkf = npz_file["Xposterior"] 
noisy_data = torch.tensor(XposteriorEnkf[:,range(0,tr_steps)].T, dtype=torch.float32)
# observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
score_model5 = ScoreModel()
optimizer5 = optim.Adam(score_model5.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer5.zero_grad()
    loss = score_matching_loss(score_model5, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model5.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer5.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 模型5，DM-True，真值训练去噪
noisy_data = torch.tensor(Xtrue_km[:,range(0,tr_steps)].T, dtype=torch.float32)
# observation = torch.tensor(yo[:,range(0,tr_steps)].T, dtype=torch.float32)
score_model6 = ScoreModel()
optimizer6 = optim.Adam(score_model6.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer6.zero_grad()
    loss = score_matching_loss(score_model6, noisy_data, observation, sigma=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_model6.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer6.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
#%% 接下来，我将计算各模型进行去噪同化的结果RMSE,先定义实验函数
def assimilation_experiment(score_model):
    """
    同化试验函数

    参数:
    score_model: 用于去噪的模型
    Xa_test: 同化试验结果存储变量
    Xprior_val: 先验值存储变量
    Xposterior_val: 后验值存储变量
    nt_test: 时间步数
    nt_m: 观测次数
    ind_m: 观测时间索引
    yo_test: 观测数据
    Xtrue_test: 真值
    dt: 时间步长
    sigma, beta, rho: Lorenz63模型的参数
    Lorenz63: Lorenz63模型函数
    RK4: 四阶龙格-库塔积分函数
    denoise: 去噪函数

    返回值:
    rmse_test: 同化试验结果的均方根误差
    Xa_test: 同化试验结果
    Xprior_test: 先验值
    Xposterior_test: 后验值
    """
    # 初始化变量
    Xprior_test = np.zeros_like(Xprior_val)
    Xposterior_test = np.zeros_like(Xposterior_val)
    km = 0  # 同化次数计数

    # 模式积分循环
    for k in range(nt_test):
        Xa_test[:, k + 1] = RK4(Lorenz63, Xa_test[:, k], dt, sigma, beta, rho)  # 正常积分模式
        if (km < nt_m) and (k + 1 == ind_m[km]):
            Xprior_test[:, km] = Xa_test[:, k + 1]
            noisy_sample = torch.tensor(Xa_test[:, k + 1].T, dtype=torch.float32).unsqueeze(0)
            observation_sample = torch.tensor(yo_test[:, km].T, dtype=torch.float32).unsqueeze(0)  # 使用独立的观测数据
            denoised_sample, denoised_steps = denoise(score_model, noisy_sample, observation_sample)
            analysis_sample = denoised_sample.squeeze(0)
            Xa_test[:, k + 1] = analysis_sample.numpy()
            Xposterior_test[:, km] = Xa_test[:, k + 1]
            km += 1

    # 计算均方根误差
    rmse_test = np.sqrt(np.mean((Xa_test - Xtrue_test) ** 2, axis=1))
    return rmse_test, Xa_test, Xprior_test, Xposterior_test
#%% 计算各个结果
rmse1,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model1)
rmse2,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model2)
rmse3,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model3)
rmse4,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model4)
rmse5,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model5)
rmse6,Xa_test, Xprior_test, Xposterior_test = assimilation_experiment(score_model6)
# 本体质量
rmseana1 = np.sqrt(np.mean((Xctl[:,20::20] - Xtrue_km) ** 2, axis=1))
rmseana2 = np.sqrt(np.mean((Xprior - Xtrue_km) ** 2, axis=1))
rmseana3 = np.sqrt(np.mean((Xposterior - Xtrue_km) ** 2, axis=1))
rmseana4 = np.sqrt(np.mean((XpriorEnkf - Xtrue_km) ** 2, axis=1))
rmseana5 = np.sqrt(np.mean((XposteriorEnkf - Xtrue_km) ** 2, axis=1))
rmseana6 = np.array([0,0,0])
# 和观测差
rmseo1 = np.sqrt(np.mean((Xctl[:,20::20] - yo) ** 2, axis=1))
rmseo2 = np.sqrt(np.mean((Xprior - yo) ** 2, axis=1))
rmseo3 = np.sqrt(np.mean((Xposterior - yo) ** 2, axis=1))
rmseo4 = np.sqrt(np.mean((XpriorEnkf - yo) ** 2, axis=1))
rmseo5 = np.sqrt(np.mean((XposteriorEnkf - yo) ** 2, axis=1))
rmseo6 = np.sqrt(np.mean((Xtrue_km - yo) ** 2, axis=1))
#%% 图6：不同模型的去噪
# 设置图形大小
epsilon = 1e-3
rmseana6 = np.array([epsilon,epsilon,epsilon])
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
# 设置条形图的索引和宽度
bar_width = 0.1  # 条形图的宽度
index = np.arange(3)  # 3个分类
# 绘制条形图
bars1 = plt.bar(index - 2.5 * bar_width, rmseana1, bar_width, color='#004488', label='CDSM-free', alpha=0.7)
bars2 = plt.bar(index - 1.5 * bar_width, rmseana2, bar_width, color='#FF8800', label='CDSM-3dvarBck', alpha=0.7)
bars3 = plt.bar(index - 0.5 * bar_width, rmseana3, bar_width, color='#006600', label='CDSM-3dvarAna', alpha=0.7)
bars4 = plt.bar(index + 0.5 * bar_width, rmseana4, bar_width, color='#880000', label='CDSM-enkfBck', alpha=0.7)
bars5 = plt.bar(index + 1.5 * bar_width, rmseana5, bar_width, color='#884400', label='CDSM-enkfAna', alpha=0.7)
bars6 = plt.bar(index + 2.5 * bar_width, rmseana6, bar_width, color='#660066', label='CDSM-true', alpha=0.7)
# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height < 1e-3:  # 将非常小的值显示为0
            height = 0
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=45, fontsize=12)
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
add_labels(bars5)
add_labels(bars6)
# 设置对数坐标
plt.yscale('log')
# 添加图例
# plt.legend(fontsize=14, loc='upper left')
# 添加标题
plt.title('RMSE of Training Data', fontsize=17)
# 设置y轴范围和刻度
plt.yticks([1e-3, 0.5, 2, 10, 100], ['0', '0.5',  '2', '10','100'], fontsize=14)
plt.ylim(0.5e-3,200)
# 设置x轴刻度
plt.xticks(index, [ ], fontsize=15)
# 添加网格线
plt.grid(axis='y', alpha=0.4)
##########
plt.subplot(2,1,2)
# 绘制条形图
bars1 = plt.bar(index - 2.5 * bar_width, rmse1, bar_width, color='#004488', label='CDSM-free', alpha=0.7)
bars2 = plt.bar(index - 1.5 * bar_width, rmse2, bar_width, color='#FF8800', label='CDSM-3dvarBck', alpha=0.7)
bars3 = plt.bar(index - 0.5 * bar_width, rmse3, bar_width, color='#006600', label='CDSM-3dvarAna', alpha=0.7)
bars4 = plt.bar(index + 0.5 * bar_width, rmse4, bar_width, color='#880000', label='CDSM-enkfBck', alpha=0.7)
bars5 = plt.bar(index + 1.5 * bar_width, rmse5, bar_width, color='#884400', label='CDSM-enkfAna', alpha=0.7)
bars6 = plt.bar(index + 2.5 * bar_width, rmse6, bar_width, color='#660066', label='CDSM-true', alpha=0.7)
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
add_labels(bars5)
add_labels(bars6)
# 设置对数坐标
plt.yscale('log')
# 添加图例
plt.legend(ncol = 3, fontsize=13, loc='best')
# 添加标题
plt.title('RMSE of Denoising Results', fontsize=17)
# 设置y轴范围和刻度
plt.yticks( [0.1,0.5, 2, 10, 100, 500], [ '0,1','0.5',  '2', '10','100',' '], fontsize=14)
# plt.ylim(1e-5,200)
# 设置x轴刻度
plt.xticks(index, [ ], fontsize=15)
# 添加网格线
plt.grid(axis='y', alpha=0.4)
# ####
# plt.subplot(3,1,3)
# # 绘制条形图
# bars1 = plt.bar(index - 2.5 * bar_width, rmseo1, bar_width, color='#004488', label='CDSM-free', alpha=0.7)
# bars2 = plt.bar(index - 1.5 * bar_width, rmseo2, bar_width, color='#FF8800', label='CDSM-3dvarBck', alpha=0.7)
# bars3 = plt.bar(index - 0.5 * bar_width, rmseo3, bar_width, color='#006600', label='CDSM-3dvarAna', alpha=0.7)
# bars4 = plt.bar(index + 0.5 * bar_width, rmseo4, bar_width, color='#880000', label='CDSM-enkfBck', alpha=0.7)
# bars5 = plt.bar(index + 1.5 * bar_width, rmseo5, bar_width, color='#884400', label='CDSM-enkfAna', alpha=0.7)
# bars6 = plt.bar(index + 2.5 * bar_width, rmseo6, bar_width, color='#660066', label='CDSM-true', alpha=0.7)
# add_labels(bars1)
# add_labels(bars2)
# add_labels(bars3)
# add_labels(bars4)
# add_labels(bars5)
# add_labels(bars6)
# # 设置对数坐标
# plt.yscale('log')
# # 添加图例
# # plt.legend(ncol = 5, fontsize=13, loc='best')
# # 添加标题
# plt.title('RMSE of Training Data w.r.t OBS', fontsize=17)
# 设置y轴范围和刻度
# plt.yticks( [0.1,0.5, 2, 10, 100], [ '0,1','0.5',  '2', '10','100'], fontsize=14)
# 设置x轴刻度
plt.xticks(index, ['x', 'y', 'z'], fontsize=15)
# 添加网格线
plt.grid(axis='y', alpha=0.4)
# 显示图形
# plt.show()
# plt.savefig('fig6.pdf', format='pdf', dpi=300, bbox_inches='tight')
#%%图7
colors2 = ['#1a5c8a','#B22222','#1f801f','#8B4513','#6a0dad','#b8860b' ]
x0test = Xprior_test[:,280];y0test=yo_val[:,280]
x0tr = Xtrue_val[:,280]
x1var = Lin3dvar(x0test, y0test, H, R, B)
# 输入的张量
noisy_sample = torch.tensor(x0test, dtype=torch.float32).unsqueeze(0)
observation_sample = torch.tensor(y0test, dtype=torch.float32).unsqueeze(0)
#1
denoised_sample,denoised_steps  = denoise(score_model1, noisy_sample , observation_sample )
x1test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array1 = np.stack(numpy_arrays)
steps_array1= np.squeeze(steps_array1, axis=1)
#2
denoised_sample,denoised_steps  = denoise(score_model2, noisy_sample , observation_sample )
x2test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array2 = np.stack(numpy_arrays)
steps_array2= np.squeeze(steps_array2, axis=1)
#3
denoised_sample,denoised_steps  = denoise(score_model3, noisy_sample , observation_sample )
x3test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array3 = np.stack(numpy_arrays)
steps_array3= np.squeeze(steps_array3, axis=1)
#4
denoised_sample,denoised_steps  = denoise(score_model4, noisy_sample , observation_sample )
x4test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array4 = np.stack(numpy_arrays)
steps_array4= np.squeeze(steps_array4, axis=1)
#5
denoised_sample,denoised_steps  = denoise(score_model5, noisy_sample , observation_sample )
x5test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array5 = np.stack(numpy_arrays)
steps_array5= np.squeeze(steps_array5, axis=1)
#6
denoised_sample,denoised_steps  = denoise(score_model6, noisy_sample , observation_sample )
x6test = denoised_sample.squeeze(0)
numpy_arrays = [step.numpy() for step in denoised_steps]
steps_array6 = np.stack(numpy_arrays)
steps_array6= np.squeeze(steps_array6, axis=1)
#画图
plt.figure(figsize=(6,6))
plt.plot(steps_array1[:, 0], steps_array1[:, 2], color=colors2[0],marker='.', markevery=5, markersize=6,linestyle='-',lw=1.5, label='CDSM-free')
plt.plot(steps_array2[:, 0], steps_array2[:, 2], color=colors2[1],marker='d', markevery=5, markersize=6,linestyle='-.',lw=1.5, label='CDSM-3dvarBck')
plt.plot(steps_array3[:, 0], steps_array3[:, 2], color=colors2[2],marker='o', markevery=5, markersize=6,linestyle='-',lw=1.5, label='CDSM-3dvarAna')
plt.plot(steps_array4[:, 0], steps_array4[:, 2], color=colors2[3],marker='d', markevery=5, markersize=6,linestyle='-.',lw=1.5, label='CDSM-enkfBck')
plt.plot(steps_array5[:, 0], steps_array5[:, 2], color=colors2[4],marker='o', markevery=5, markersize=6,linestyle='-',lw=1.5, label='CDSM-enkfAna')
plt.plot(steps_array6[:, 0], steps_array6[:, 2], color=colors2[5],marker='>', markevery=5, markersize=6,linestyle='-',lw=1.5, label='CDSM-True')
plt.plot(x0test[0], x0test[2],marker='o', color='white', markeredgecolor='#004488', markeredgewidth=2, markersize=10, label='Prior')
plt.plot(x1var[0], x1var[2], marker='s', color='white', markeredgecolor='blue', markeredgewidth=2, markersize=10, label='3DVAR analysis')
plt.annotate('', xy=(x1var[0], x1var[2]), xytext=(x0test[0], x0test[2]),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

plt.scatter(y0test[0], y0test[2], marker='x', color='red', s=150,lw=2, label='Observation')
plt.scatter(x0tr[0], x0tr[2], marker='+', color='black', s=150, lw=2,label='True')
plt.xlabel('x', fontsize=15)
plt.ylabel('z', fontsize=15)
plt.xlim(-3.2,0.5),plt.xticks(np.arange(-3,0.6,1),fontsize=15)
plt.ylim(9.3,10.8),plt.yticks(np.arange(9.5,11.0,0.5),fontsize=15)
plt.legend(ncol=2,fontsize=13)
plt.grid()
plt.tight_layout()
#plt.savefig('fig7.pdf', format='pdf', dpi=300, bbox_inches='tight')
#%% 求ACC
def computeACC(X1,X2):
    clim1 = np.mean(X1,axis=0)
    clim2 = np.mean(X2,axis=0)
    X1_a = X1 - clim1[np.newaxis,:]
    X2_a = X2 - clim2[np.newaxis,:]
    ACC = np.zeros(len(X1))
    for j in range(len(X1)):
        ACC[j] = np.corrcoef(X1_a[j,:],X2_a[j,:])[0,1]
    return ACC

acc1 = computeACC(Xctl[:,20::20],yo)
acc2 = computeACC(Xprior,yo)
acc3 = computeACC(Xposterior,yo)
acc4 = computeACC(XpriorEnkf,yo)
acc5 = computeACC(XposteriorEnkf,yo)
acc6 = computeACC(Xtrue_km,yo)

print([np.mean(acc1),np.mean(acc2),np.mean(acc3),np.mean(acc4),np.mean(acc5),np.mean(acc6)])
