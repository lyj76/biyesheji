%% pam
clear;
close all;

%% add paths
addpath('fns');
addpath(fullfile('fns','fns2'));

%% parameters
Ft =200e9;   
Osamp_factor = 2;
NumSymbols = 2^18;
NumPreamble = 0;            
NumSym_total = NumSymbols+NumPreamble;
M = 4;                                                                                      
 
s = RandStream.create('mt19937ar', 'seed',529558);
prevStream = RandStream.setGlobalStream(s);

%% MQAM modulate
[xsym, xm] = PAMSource(M,NumSymbols);
xs = xm;
% xs(find(xs==3))=3;
%  figure;plot(xs(:),'o');grid on;
%% pluse shaping

rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor,'Raised Cosine','N,Beta',N,rolloff);
Hd = design(h);
sqrt_ht = Hd.Numerator;
sqrt_ht = sqrt_ht./max(sqrt_ht);
x_upsamp = upsample(xs,Osamp_factor);%upsampling
x_shape = conv(sqrt_ht,x_upsamp);
x_shape =x_shape ./ sqrt(mean(abs(x_shape).^2));



% load('all1km.mat')
NN=101;
% NN=11;
% NN=3:2:25;
% NN=11:10:201;
% Numdata=15;
% Ndata=[7,10,13,1,12,14,6,9,4,11];%btb rop=5
% Ndata=[6,9,10,7,1,8,2,4,5,11];%1km rop=5
file_list = {'rop3dBm_1.mat', 'rop5dBm_1.mat'};
BERall=zeros(length(file_list),length(NN))  ;
for n1=1:length(file_list)
% for n1=3:3
if 0


DSO160G
ReData = resample(channel1data6(2,:),100,160);
ReData = ReData-mean(ReData);

end          


n1
load(file_list{n1},'ReData')

% close all
% close all
ReData=-ReData;
%% synchronization
th = 0.3;
[TE,FE] = TEFEMMM2(ReData,1024,th);
abc = TE+0;
% abc = 100+0+0;
TE =abc;
ysync = ReData(1024+20+TE:end); 
ysync = ysync(1:length(x_shape));


for m1 = 1:length(NN)

%% Match filtering
% yt_filter = ysync(1:end);
yt_filter = ysync(1+N/2:length(ysync)-N/2);
% % yt_filter = conv(sqrt_ht,ysync);
% yt_filter = ysync(1+N/2:length(ysync)-N/2);
% figure;pwelch(yt_filter(:),[],[],[],Ft,'twosided');

%% RLS LE/NE
% if 1
    xTx = xs;
    xRx = yt_filter;
    NumPreamble_TDE = 10000;
    
    N1 = 111; %98
    N2 = 21;%%78
    WL= 1;%%13
    N3 = 11;%%78
    


% P=500; % P=500 for S-IWDFE
% sp=;
sp=0;
% P=535;
% P=280;
% sp=75;

    D1 = 25; %%26
    D2 = 0;%%18
     WD=1;%%9.

     
aa=0.3;
bb=1;


 %[hffe,ye] = FFE_2pscenter(xRx,xTx,NumPreamble_TDE,N1,0.9999); 
%  [hffe,ye] = VNLE2_2pscenter(xRx,xTx,NumPreamble_TDE,N1,N2,0.9999,WL);  % % 2nd
 
  [hffe,hdfe,ye] = LE_FFE2ps_centerDFE_new( xRx,xTx,NumPreamble_TDE,N1,D1,0.9999,M,M/2);  
% [hffe,hdfe,ye] = DP_VFFE2pscenter_VDFE( xRx,xTx,NumPreamble_TDE,N1,N2,D1,D2,0.9999,WL,WD,M,M/2);

%     figure;plot(hffe); hold on;plot(hdfe);
%% Normalize
ym = Normalizepam(ye,M);
% eyediagram(ym(2000:3000),4) 
% figure;hist(ym(:),1000);grid on;
ytemp=ym(NumPreamble_TDE+1:end);
figure;plot(ytemp(:),'o');grid on;
figure;hist(ytemp(:),1000);grid on;
% hold on;hist(xm(1:3000),1000);grid on;



%% MQAM demodulation
ysym = dePAMSource(M,ym);


%% BER/SNR
[ErrCount BER_sub1] = biterr(ysym(NumPreamble_TDE+1:end), xsym(NumPreamble_TDE+1:end), log2(M));
[ErrorSym SER_sub1] = symerr(ysym(NumPreamble_TDE+1:end), xsym(NumPreamble_TDE+1:end));         
% figure;plot(ysym(NumPreamble_TDE+1:end)- xsym(NumPreamble_TDE+1:end))
[SNRdB_sub1,SNR1 ] = snr( xm(NumPreamble_TDE+1:end),ym(NumPreamble_TDE+1:end) );


%%%%
disp([num2str(1),' BER_sub1 = ',num2str(BER_sub1)])
disp([num2str(1),' SNRdB_sub1 = ',num2str(SNRdB_sub1)])


BERall(n1,m1)=BER_sub1;
end
end
 mean(BERall,1)
if 0
[aaa1 bbb1] = sort(BERall(:,end));

avernber = mean(BERall(bbb1(1:10),:),1)
% figure;plot(NN,avernber)

end


berm=mean(BERall,1);

figure;semilogy(NN,berm);
% figure;semilogy(NN,berm);
% berm