function [ TrainingSequence ] = Preamble_sync( NumSubCarr,NumFFT,NumGI,fc_digital )
%PREAMBLE_SYNC Summary of this function goes here
Seq1 = SeqZadoffChu(NumSubCarr/2 ,1);
S2PSymbol =  zeros(NumFFT,1);
S2PSymbol(1:2:NumSubCarr/2,1) = Seq1(1:end/2)*sqrt(2);
S2PSymbol(NumFFT-NumSubCarr/2+1:2:end,1) = Seq1(end/2+1:end)*sqrt(2);
IDFTSequence = ifft(S2PSymbol);
CPSequence = [IDFTSequence(end-NumGI+1:end,:); IDFTSequence];

%% %%%%%%%%%%%%%   Parallel 2 Serial
TrainingSequence = reshape(CPSequence, 1, []);   
TrainingSequence = real(TrainingSequence.*exp(1i*2*pi*fc_digital*(1:length(TrainingSequence))));
TrainingSequence = TrainingSequence ./ sqrt(mean(abs(TrainingSequence).^2));
end

