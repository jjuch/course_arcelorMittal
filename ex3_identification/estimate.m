
close all
clear all

load iddemo_heatexchanger_data

data = iddata(pt,ct,Ts);
data.InputName  = '\Delta CTemp';
data.InputUnit  = 'C';
data.OutputName = '\Delta PTemp';
data.OutputUnit = 'C';
data.TimeUnit   = 'minutes';
figure
plot(t,ct)

figure
plot(t,pt)

sysTF = tfest(data,3,2,nan,'Ts',Ts,'Feedthrough',1)
%sysTF = tfest(data,3,2,nan)

%clf
figure
compare(data,sysTF)

na = 3;
nb = 3;
nc = 3;
nk = 30;
sys = armax(data,[na nb nc nk])
figure
compare(data,sys)