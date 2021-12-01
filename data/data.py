#!/usr/bin/env python

import numpy as np

def bostonHousing():
    from tensorflow.keras.datasets import boston_housing
    return boston_housing.load_data(test_split=0.)[0]

def concreteData(path='./'):
    d = np.loadtxt(path + 'Concrete_Data.csv', delimiter=',')
    return (d[:, :-1], d[:, -1])

def energyEfficiency(path='./'):
    d = np.loadtxt(path + 'ENB2012_data.txt')
    return (d[:, :-2], d[:, -1])

#def navalPropulsion(path='./'):
#    d = np.loadtxt(path + 'UCI_CBM_Dataset/data.txt')
#    return (d[:, :-2], d[:, -1])

def navalPropulsion(path='./'):
    d = np.loadtxt(path + 'UCI_CBM_Dataset/data.txt')
    return (np.concatenate((d[:, :7],d[:,9:11],d[:,13:-2]),axis=1), d[:, -1])

def BioCon(path='./'):
    d = np.loadtxt(path + 'Grisoni_et_al_2016_EnvInt88.csv', delimiter=',',dtype=str)
    return (d[:, 3:-2].astype('float'), d[:, -1].astype('float'))

def proteinStructure(path='./'):
    d = np.loadtxt(path + 'CASP.csv', delimiter=',')
    return (d[:, 1:], d[:, 0])

def yachtHydrodynamics(path='./'):
    d = np.loadtxt(path + 'yacht_hydrodynamics.data')
    return (d[:, :-1], d[:, -1])

def wine(path='./'):
    d = np.loadtxt(path + 'winequality-red.csv', delimiter=';')
    return (d[:, :-1], d[:, -1])

def RCL(path='./'):
    d = np.loadtxt(path + 'RCL_current.txt')
    return (d[:, :-1], d[:, -1])

def WheatStoneBridge(path='./'):
    d = np.loadtxt(path + 'Wheatstone_Bridge.txt')
    return (d[:, :-1], d[:, -1])

def Ising(path='./'):
    d = np.loadtxt(path + 'ising_energys_flat.txt', delimiter=',')
    return (d[:, :-1], d[:, -1])

def splitData(all_x, all_y, val_pct=0.05, test_pct=0.05):

    N_total = all_x.shape[0]
    N_val = int(N_total * val_pct)
    N_test = int(N_total * test_pct)

    all_indices = np.random.permutation(N_total)
    split1 = N_total - N_val - N_test
    split2 = N_total - N_test    
    # return train, val, and test
    train_indices = all_indices[:split1]
    val_indices = all_indices[split1:split2]
    test_indices = all_indices[split2:]
    return (all_x[train_indices], all_y[train_indices]), \
           (all_x[val_indices], all_y[val_indices]), \
           (all_x[test_indices], all_y[test_indices])

def polyFunction(Number_of_datapoints,Number_of_variables):
    
    # np.random.seed(seed=1337)
    if Number_of_variables != 1:
        coeff0=np.array([0.26202468])
        coeff1=np.array([-0.68263206, -0.44374696, -0.08136623, -0.35799892,  0.03678564])
        coeff2=np.array([[-4.76114149e-01,  6.04586464e-01,  5.26933128e-01,
        	            -6.96917368e-01, -2.00583630e-01],
                       [ 6.04586464e-01, -7.49884147e-01,  3.44810178e-01,
                        -2.68607984e-01, -1.12464228e-02],
                       [ 5.26933128e-01,  3.44810178e-01, -1.67792121e-01,
                         2.54476986e-01,  2.91871177e-01],
                       [-6.96917368e-01, -2.68607984e-01,  2.54476986e-01,
                        -7.03477256e-04,  1.09393802e-02],
                       [-2.00583630e-01, -1.12464228e-02,  2.91871177e-01,
                         1.09393802e-02, -6.29498110e-01]])
    else:
        coeff0=np.random.random_sample(1)*2-1
        coeff1=np.random.random_sample([Number_of_variables])*2-1
        coeff2=np.random.random_sample([Number_of_variables,Number_of_variables])*2-1   

    def f0(x):
        return coeff0

    def f1(x):
        output=0
        for i in range(Number_of_variables):
            output+=coeff1[i]*x[i]
        return output

    def f2(x):
        output=0
        for i in range(Number_of_variables):
            for j in range(Number_of_variables):
                output+=coeff2[i,j]*x[i]*x[j]
        return output   

    if Number_of_variables == 1:
        coeff3=np.random.random_sample([Number_of_variables,Number_of_variables, Number_of_variables])*2-1
        def f3(x):   
            output = 0  
            for i in range(Number_of_variables):
                for j in range(Number_of_variables):
                    for k in range(Number_of_variables):
                        output+=coeff3[i,j,k]*x[i]*x[j]*x[k]
            return output  
        def f(x):
            return f0(x)+f1(x)+f2(x)+f3(x)

    else:
        def f(x):
            return f0(x) + f1(x) + f2(x)

    x_full=np.random.sample([Number_of_datapoints,Number_of_variables])*2-1
    y_full=(np.array([f(x) for x in x_full])).flatten()
    if Number_of_variables == 1:
        x_full = x_full.reshape((x_full.shape[0], 1))
        return (x_full,y_full), f
    return (x_full, y_full)

def getData(key, path='./data', n_points=1000):
    keymap = {'bostonHousing': bostonHousing(),
          'concreteData': concreteData(path),
          'energyEfficiency': energyEfficiency(path),
          'navalPropulsion': navalPropulsion(path),
          'proteinStructure': proteinStructure(path),
          'yachtHydrodynamics': yachtHydrodynamics(path),
          'randomFunction': polyFunction(n_points, 5),
          'ising': Ising(path),
          'RCL': RCL(path),
          'WheatStoneBridge': WheatStoneBridge(path),
          'wine': wine(path),
          'Biocon': BioCon(path)
         }
    return keymap[key]


'''
def yearOfSong(path='./'):
    d = np.loadtxt(path + 'YearPredictionMSD.csv', delimiter=',')
    return (d[:, 1:], d[:, 0])
        '''
