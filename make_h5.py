import h5py
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
def readcsv(filename):
    data=pd.read_csv(filename, header=None, encoding="utf-8").values
    index1=[]
    index2=[]
    for i in range(data.shape[0]):
        if data[i,1] < 20000:
            index1.append(i)
        if data[i,1] > 2980000:
            index1.append(i)
    idex = index1+index2
    data = np.delete(data, idex, axis=0)
    return data

def main(args):
    outfile=h5py.File(args.output,"w")
    trainf=outfile.create_group("train") 
    validf=outfile.create_group("test") 
    
    data=readcsv(args.input)
    n_sample = data.shape[0]
    n_valid = int(n_sample*args.split)
    print(f"Number of samples:{n_sample}")
    print(f"Number of samples:{n_valid}")
    nitrs=0
    for i in range(data.shape[0]):
        file=data[i][0]
        point=int(data[i][1])
        # snr=int(data[i][2])
        path="G:/labeq_20220315/"+file.split("_")[0]+"/"+file
        dd=sio.loadmat(path)["data"]
        toto = dd[point-20000:point+20000]
        pt={"p_pick":20000, "s_pick":20000}
        
        # iden = np.random.randint(1000,5000)
        # ides = iden+1000
        # toto=dd[(point-iden):(point+12000-iden),:]
        # pt={"p_pick":iden, "s_pick":iden}
        # toto = np.insert(toto, 1, toto[:,0], axis=1)
        # toto = np.insert(toto, 2, toto[:,1], axis=1)
        if nitrs < n_valid:
            group=trainf
        else:
            group=validf
        fkey, skey= file.split("_")
        name=file+"."+str(i)
        if fkey not in group:
            subgroup =group.create_group(fkey)
            if args.compression:
                dt = subgroup.create_dataset(name, data=toto, compression="gzip" )
            else:
                dt = subgroup.create_dataset(name, data=toto)
        else:
            subgroup = group[fkey]
            if args.compression:
                dt = subgroup.create_dataset(name, data=toto, compression="gzip" )
            else:
                dt = subgroup.create_dataset(name, data=toto)
        for key in pt:
            dt.attrs[key] = pt[key]
        nitrs += 1
    outfile.close()

import argparse
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(description="Make HDF5 file")
    
    ##添加参数
    parser.add_argument('-i', '--input', default="data/ML_train.csv", type=str, help="Path to diting")       
    parser.add_argument('-o', '--output', default="data/AE_train.h5", type=str, help="Output name")     
    parser.add_argument('-s', '--split', default=1, type=float, help="Partion of training data")         
    parser.add_argument('-c', '--compression', default=True, type=bool, help="Compression")
    
    ##解析参数                                                       
    args = parser.parse_args()  
    main(args)     