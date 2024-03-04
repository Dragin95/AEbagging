import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os 
import scipy.signal as signal
import scipy.io as sio
import torch.nn.functional as F
import time

seqlen =6144
batchstride=6144-512

def read_path(filename):
    ph="Pick-example\event_4"
    chs=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10','ch11','ch12']  
    file=pd.read_csv(filename, header=None, encoding="utf-8").values[:,0]
    pth=[]
    for i in range(1):
        name = file[i]
        for j in range(len(chs)):
            name_= chs[j]+"_"+str(name)+".mat"
            path=os.path.join(ph, name_)
            path=path.replace('\\','/')
            pth.append(path)
    return pth

def main(args):
    all_path=read_path(args.input)
    print(len(all_path))
    result_path=args.output
    for i in range(len(all_path)):
        path_ = all_path[i]
        ph_=path_.strip().split("/")[2]
        rest=os.path.join(result_path, ph_)
        data = sio.loadmat(path_)["data"]
        device=torch.device("cpu")
        data=torch.tensor(data/1.0, dtype=torch.float, device=device)
        T, C = data.shape
        batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
        idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
        idx = idx.clamp(min=0, max=T-1).long() 
        wave = data[idx, :]
        wave = wave.permute(0, 2, 1)
        wave -= torch.mean(wave, dim=2, keepdim=True)
        max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
        wave /= (max + 1e-6)
        with torch.no_grad():
            start =time.perf_counter()
            blstm = torch.jit.load(args.model_1)
            blstm.eval()
            blstm.to(device)
            eqt = torch.jit.load(args.model_2)
            eqt.eval()
            eqt.to(device)
            with torch.no_grad():
                oc_1 = blstm(wave)
                oc_2 = eqt(wave)
                oc = (oc_1+oc_2)/2
                B, C, T = oc.shape
                tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * 1 + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
                oc = oc.permute(0, 2, 1).reshape(-1, C)
                ot = tgrid.squeeze() 
                ot = ot.reshape(-1)
                output=[]
                
                pc = oc[:,1]  ##第二个为p点的概率值
                time_sel = torch.masked_select(ot, pc>0.2)     #pc>0.2的值bool值,保留下pc>0.2的时间点
                score = torch.masked_select(pc, pc>0.2)        #保留pc>0.2对应的概率值
                _, order = score.sort(0, descending=True)      #需要根据概率值进行，从大到小排列，True
                ntime = time_sel[order]   
                nprob = score[order]
                select = -torch.ones_like(order) 
                selidx = torch.arange(0, order.numel(), 1, dtype=torch.long, device=device)   # 大于0.3的个数的排列
                count = 0
                while True:
                    if nprob.numel()<1:
                        break
                    ref = ntime[0]
                    idx = selidx[0]
                    select[idx] = 1
                    count += 1
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>3000)  # 根据概率值高的依次保留大于3000
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>3000)    # 概率值数量进行循环判断
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>3000)    # 依次取ntime的第一个值进行比较
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * 1 
                print(ph_+":", p_time.shape)
                ##删除信噪比小于4
                toto=data-torch.mean(data)
                p_snr=[]
                for ind in range(p_time.shape[0]):
                    point = int(p_time[ind])
                    if point < 100:
                        snr=1
                        p_snr.append(snr)
                        continue
                    if point>3000000:
                        snr=1
                        p_snr.append(snr)
                        continue
                    if point + 6000 > 3000000:
                        pts = 3000000-point
                        signal = toto[int(point):int(point)+int(pts)]
                    else:
                        signal=toto[int(point):int(point)+6000]
                    if point-6000 < 0:
                        noise = toto[1:int(point)]
                    else:
                        noise = toto[int(point)-6000:int(point)]
                    n=torch.sqrt(sum(noise**2)/len(noise))
                    s=torch.sqrt(sum(signal**2)/len(signal))
                    snr=20*torch.log10(s/n)
                    p_snr.append(snr)
                p_snr=torch.tensor(p_snr)
                p_snrs=torch.masked_select(p_snr, p_snr>4.0)
                p_prob=torch.masked_select(p_prob, p_snr>4.0)
                p_type=torch.masked_select(p_type, p_snr>4.0)
                p_time=torch.masked_select(p_time, p_snr>4.0)
                y = torch.stack([p_type, p_time, p_snrs, p_prob], dim=1)
                print(ph_+":", y.shape)
                y = y.cpu().numpy()
                # output.append(y) 
        sio.savemat(rest,{"data":y})
        ##保存每次加载时间
        # end = time.perf_counter()
        # acc_time = end-start
        # print(acc_time)
        # file_=open("time.csv", "a")
        # file_.write(f"{(acc_time)} \n")
        # file_.close()
        
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pick AEs")
    ##添加参数
    parser.add_argument("--input", default="pick_example.csv", type=str, help="Picking Example")
    parser.add_argument("--model_1", default="ckpt/AEBagging/AE.blstm.jit", type=str, help="Jit model name")
    parser.add_argument("--model_2", default="ckpt/Transformer/AE.transformer.jit", type=str, help="Jit model name")
    parser.add_argument("--plot", default=True, type=bool, help="whether plot picking figure")
    parser.add_argument("--output", default="pick_results/event_4", type=str, help="pick results")
    #解析参数
    args=parser.parse_args()
    main(args)