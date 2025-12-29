function [ ReData ] = awg_channel( TrData,Ft )
%% AWG/OSA
InputSig = TrData;

% InputSig = 1i*TrData;
% 
% at = TrData;
% bt = [at(end/2+10:end),at(1:end/2)];
% InputSig = TrData+1i*[bt,zeros(1,length(at)-length(bt))];%%%2channel

OneShotAWG;               
pause(4)
OneShotDSA_Ele;
System.RowADC = OutputSig-mean(OutputSig);
% System.RowADC = -1*System.RowADC;
% OneShotDSA;
% System.RowADC = OutputSig-mean(OutputSig);
% System.RowADC = -1*System.RowADC;
% 10*log10(mean((System.RowADC).^2))

 if 0
        tt=System.RowADC;
        
        bandwidth = 0.125e9;
        System.RowADC = tt;
        N = length(System.RowADC);
        df = 1.25e9 ./ N;
        f = (-N/2:N/2-1) * df;
        Hf3 = ifftshift(myfilter('ideal',f,bandwidth)).';
        System.RowADC = (ifft((fft(System.RowADC(1,:)).*Hf3)));
         figure;pwelch(System.RowADC,[],[],[],12.5e9,'twosided');
    end
 figure;pwelch(System.RowADC,[],[],[],1.25e9,'twosided');
%% downsampling
% ReData = resample(System.RowADC,1,5); 
% ReData = resample(System.RowADC,8,125); 
% ReData = resample(System.RowADC,2,25); 
% ReData = resample(System.RowADC,1,25);
% figure;pwelch(System.RowADC,[],[],[],12.5e9,'twosided');
% ReData = resample(System.RowADC,2,25);
% ReData = resample(System.RowADC,1,5);
% ReData = resample(System.RowADC,2,5);
ReData = resample(System.RowADC,1,5);

% ReData = resample(System.RowADC,1,1);
% ReData = downsample(System.RowADC,2);
% ReData = System.RowADC;
%% signal spectrum
% figure;plot(ReData);
% figure;pwelch(ReData,[],[],[],Ft,'twosided');
% save('F:\zhangjunwei\ReData.mat','ReData');
end

