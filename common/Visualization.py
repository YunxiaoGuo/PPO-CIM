import os.path
from common.utils import *
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_cruve(envname,algo):
    Data = []
    path = './data/'+envname+'/'+algo+'/'
    npy_data = os.listdir(path)
    for i in range(len(npy_data)):
        data = np.load(os.path.join(path,npy_data[i]))
        dta = []
        for j in range(0,len(data)-200):
            da = np.mean(data[j:j+200])
            dta.append(da)
        Data.append(dta)
    return np.array(Data)

def get_data(envname):
    METHOD = ["CIM", "CIM-1", "CIM-2"]
    cim = get_cruve(envname,METHOD[0])
    cim1 = get_cruve(envname, METHOD[1])
    cim2 = get_cruve(envname, METHOD[2])
    return cim,cim1,cim2


if __name__ == '__main__':
    algos = ["CIM", "CIM-1", "CIM-2"]
    label = ['PPO-CIM', 'PPO-CIM-1','PPO-CIM-2']
    path = './data/'
    envs = ['Swimmer-v2','Reacher-v2','Hopper-v2','Humanoid-v2','Ant-v2','Walker2d-v2']
    envname = envs[0]
    path = os.path.join(path,envname)
    data = get_data(envname)
    df=[]
    data = np.array(data)
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='Timestep',value_name='Reward'))
        df[i]['Algorithms']= label[i]
    df=pd.concat(df)
    sns.lineplot(data=df,x="Timestep", y="Reward", hue="Algorithms", style="Algorithms")
    plt.title(envname)
    plt.tick_params(labelsize=12)
    plt.savefig('./figures/' + envname + '.svg', format='svg')
    plt.clf()