clc;clear;close all

cd('C:\Users\Administrator.DESKTOP-Q5G5OQ4\Desktop\AEbagging\manuscript_figure\fig_08\sta-lta')
toto =load('event_3.mat').data;

fs =3e6;
totos=toto(:)/max(toto(:));

for j = 1
    x=totos(:);
    data=length(x);
    CF=zeros(1,data);
    for i=2:data-1        
       CF(i)=x(i)^2;
    end
    fs=3e6;
    stw=0.00005; %30
    ltw = 0.001; %15000
    num1=fix(stw*fs);
    num2= fix(ltw*fs);
    STA=zeros(1,data);
    LTA=zeros(1,data);
    k=zeros(1,data);                                              
    
    for i=1:num1
        STA(i)=1;
        LTA(i)=1;
        k(i)=STA(i)/LTA(i);
    end

    for i=(num1+1):num2
             STA(i)=sum(abs(CF((i-(num1-1)):i)))/double(num1);
             LTA(i)=sum(abs(CF(1:i)))/double(length(1:i));
             k(i)=STA(i)/LTA(i);
    end

    for i=(num2+1):data
             STA(i)=sum(abs(CF((i-(num1-1)):i)))/double(num1);
             LTA(i)=sum(abs(CF((i-(num2-1)):i)))/double(num2);
             k(i)=STA(i)/LTA(i);
    end
         sra(:,1)=k;
end

figure
plot(toto(:))
figure
plot(sra)
save('sta_lta3.mat', 'sra')
