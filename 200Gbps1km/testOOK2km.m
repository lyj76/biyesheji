%% pam
clear;
close all;

%% parameters
Ft =200e9;   
% Ft = 63.999980e9;% 
% Ft = 55.999980e9;% 
Osamp_factor = 2;
NumSymbols = 2^17;
NumPreamble = 0;            
NumSym_total = NumSymbols+NumPreamble;
M = 4;                                                                                      
 
s = RandStream.create('mt19937ar', 'seed',529558);
prevStream = RandStream.setGlobalStream(s);

%% MQAM modulate
[xsym, xm] = PAMSource(M,NumSymbols);
xs = xm;

%% pluse shaping
% x_upsamp = upsample(xs,Osamp_factor);%upsampling
% rectangular=ones(1,Osamp_factor);
% x_shape = conv(rectangular,x_upsamp);
% x_shape =x_shape(1:end-Osamp_factor+1);
% x_shape =x_shape ./ sqrt(mean(abs(x_shape).^2));

rolloff = 0.1;
N = 128;
h = fdesign.pulseshaping(Osamp_factor,'Raised Cosine','N,Beta',N,rolloff);
Hd = design(h);
% fvtool(Hd, 'impulse')
% fvtool(Hd)
sqrt_ht = Hd.Numerator;
x_upsamp = upsample(xs,Osamp_factor);%upsampling
x_shape = conv(sqrt_ht,x_upsamp);
x_shape =x_shape ./ sqrt(mean(abs(x_shape).^2));

x_shape=[zeros(1,100),x_shape,zeros(1,100)];

% SNR = 16;
% SNR = SNR-10*log10(Osamp_factor);%%rect无过采样
% x_shape = awgn(x_shape,SNR,'measured');




% figure;plot(x_shape(1:100))
figure;pwelch(x_shape(:),[],[],[],Ft,'twosided');

%% SSMF channel
Amp=1;
% EXR = 10*log10((Bias+Amp)/(Bias));
EXR=1;%dB
Bias=Amp/(10^(EXR/10)-1);
EPower =(x_shape./max(abs(x_shape))+1)/2*Amp+Bias;
figure;plot(EPower)


N_iterations=0;
Dispersion =            16.8e-12/1e-9/1e3;            % 16.8 for 50/100km Dispersion 16.3 for 40km
lambda     =            1550.12e-9;
fiberLength =           2e3;                       % Fiber Length

EPower=EPower.';
Pin = EPower;                  % START with ideal Tx
PHI = zeros(length(Pin),1);    % START with zero phase 
PHI_pi = pi*ones(length(Pin),1); 

time = (0:length(EPower)-1).*1/Ft;     % Time vector
time = time.';
mse=zeros(N_iterations,1);
 
XPower = EPower;


figure;pwelch(XPower(:),[],[],[],Ft,'twosided');


figure;plot(mse)
figure;plot((EPower-mean(EPower))./ sqrt(mean(abs(EPower-mean(EPower)).^2)))
hold on;plot((XPower-mean(XPower))./ sqrt(mean(abs(XPower-mean(XPower)).^2)))
axis([0 2e4 -5 5])
Ein = sqrt(XPower).*exp(1i.*PHI);
Efbr = SMF(time,Ein,Dispersion,fiberLength,lambda);

yPD = (abs (Efbr).^2).';
y=yPD-mean(yPD);

      
SNR = 20;
SNR = SNR-10*log10(Osamp_factor);%%rect无过采样
y = awgn(y,SNR,'measured');
a=0;
% yt_filter=y(101+a:end-100+a);
yt_filter=y(101+a+N/2:end-100+a-N/2);
figure;pwelch(yt_filter(:),[],[],[],Ft,'twosided');



%% RLS LE/NE
      xTx = xs;
    xRx = yt_filter;
    NumPreamble_TDE = 10000;
    
    N1 = 101; %98 
    N2 = 7;%%78
    WL= 3;%%13
    N3 = 0;%%78
    


% P=500; % P=500 for S-IWDFE
% sp=;
sp=0;
% P=535;
% P=280;
% sp=75;

    D1 = 11; %%26
    D2 = 0;%%18
     WD=1;%%9.

     
aa=0.3;
bb=1;


%  [hffe,ye] = FFE_2pscenter(xRx,xTx,NumPreamble_TDE,N1,0.9999); %FFE
%   [hffe,ye] = VNLE2_2pscenter(xRx,xTx,NumPreamble_TDE,N1,N2,0.9999,WL);% 2nd VNLE VFFE
 
%  [hffe,hdfe,ye] = LE_FFE2ps_centerDFE_new( xRx,xTx,NumPreamble_TDE,N1,D1,0.9999,M,M/2);  %FFE-DFE
[hffe,hdfe,ye] = DP_VFFE2pscenter_VDFE(xRx,xTx,NumPreamble_TDE,N1,N2,D1,D2,0.9999,WL,WD,M,M/2);% VFFE-VDFE

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