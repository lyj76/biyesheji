function [ SNRmeandB,SNR ] = snr( x_qam,y_qam_scale )
for msub  = 1:size(x_qam,1);
y_qam_scale(msub,:) = y_qam_scale(msub,:)./ sqrt(mean(abs(y_qam_scale(msub,:)).^2));
x_qam(msub,:) = x_qam(msub,:)./ sqrt(mean(abs(x_qam(msub,:)).^2));
SNR(msub) = var(y_qam_scale(msub,:))./var((y_qam_scale(msub,:)-x_qam(msub,:)));
end
SNRmean = mean(SNR);
% mean(10*log10(SNR))
% 10*log10(SNRmean)
SNRmeandB = 10*log10(SNRmean);
end

