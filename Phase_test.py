import torch 
import numpy as np
import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.data import DitingDataTestThread
from utils.accurate import find_phase
# from utils.post import find_phase_lppn, find_phase_point2point 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

def main(args):
    #######
    model_name_1 = args.model_1
    if "one" in model_name_1.lower():
        stride = 8
        find_pre = find_phase
    else:
        stride=1
        find_pre = find_phase
    device = torch.device("cpu")
    model_1 = torch.jit.load(model_name_1)
    model_1.eval()
    model_1.to(device)
    
    #######
    # model_name_2 = args.model_2
    # if "one" in model_name_2.lower():
    #     stride = 8
    #     find_pre = find_phase
    # else:
    #     stride = 1
    #     find_pre = find_phase
    # model_2 = torch.jit.load(model_name_2)
    # model_2.eval()
    # model_2.to(device)
    
    #######
    # model_name_3 = args.model_3
    # if "one" in model_name_3.lower():
    #     stride = 8
    #     find_pre = find_phase
    # else:
    #     stride = 1
    #     find_pre = find_phase
    # model_3 = torch.jit.load(model_name_3)
    # model_3.eval()
    # model_3.to(device)
    
    #######
    m=0
    data_tool = DitingDataTestThread(file_name=args.input, stride=stride, n_length=6144, padlen=512)
    out_path = args.output
    outfile = open(out_path, "w", encoding="utf-8")
    result_pre=[]
    fig_path = "ckpt\Transformer\AE.transformer.jpg"
    for step in range(1000):
        m=m+1
        a1, a2, a3 = data_tool.batch_data(batch_size=50)
        print(len(a2))
        a1 = torch.tensor(a1, dtype=torch.float32, device=device)
        with torch.no_grad():
            output_1 = model_1(a1)
            avg_probs=output_1
            # output_2 = model_2(a1)
            # # output_3 = model_3(a1)
            # avg_probs = (output_1+output_2)/2
            if stride == 1:
                phase = find_pre(avg_probs.cpu().numpy(), height=0.3, dist=50)
            else:
                phase = find_pre([out.cpu().numpy() for out in avg_probs], height=0.3, dist=50)
            n=0
            print(len(phase))
            for ind in range(len(phase)):
                if len(phase[ind])!=0:
                    n=n+1
            sum=len(phase)
            pre=n/sum
            result_pre.append(pre)
            print(f"accurate:{pre}")
            
            for idx in range(len(a2)):
                pt = a2[idx][0]
                snr= a3[idx]
                outfile.write(f"#phase, {pt}, {snr}\n")
                for p in phase[idx]:
                    outfile.write(f"{p[1]},{p[2]}\n")
                outfile.flush()
            
            if args.plot:
                gs = gridspec.GridSpec(2,1)
                fig = plt.figure(1, figsize=(9,6), dpi=100)
                if stride>1:
                    p=avg_probs[0].detach().cpu().numpu()[0]
                else:
                    p = avg_probs.detach().cpu().numpy()
                w = a1.cpu().numpy()[0, 0, :]
                w = w/np.max(w)
                # result=np.vstack((w,p))
                # name = str(m)+".mat"
                
                # sio.savemat(name, {"data":p})
                # print(p.shape,w.shape,result.shape)
                for i in range(2):
                    ax = fig.add_subplot(gs[i,0])
                    ax.plot(np.repeat(p[i, :], stride), alpha=0.5, c="r")
                    ax.plot(w, alpha=0.3, c="k")
                plt.savefig(fig_path)
                plt.cla() 
                plt.clf()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test with AEs")
    parser.add_argument("-i", "--input", default="data/AE_test.h5", type=str, help="Path to h5 data")
    parser.add_argument("-o", "--output", default="test_results\Transformer\Transformer.txt", type=str, help="Path to h5 data")
    # parser.add_argument("-m", "--model_1", default="ckpt/AE.transformer.jit", type=str, help="Jit model name")
    # parser.add_argument("-n", "--model_2", default="ckpt/AE.blstm.jit", type=str, help="Jit model name")
    parser.add_argument("-p", "--model_1", default="ckpt/Transformer/AE.transformer.jit", type=str, help="Jit model name")
    parser.add_argument('--plot', default=True, type=bool, help="Whether plot training figure")
    args=parser.parse_args()
    main(args)