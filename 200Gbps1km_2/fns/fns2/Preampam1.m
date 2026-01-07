function [ XX,Preamble_X ] = Preampam1( xx_qam,NumPreamble )
if NumPreamble == 0
    XX = xx_qam;Preamble_X = [];
else
[xsym2, x2] = PAMSource(2,NumPreamble);
Preamble_X = x2./sqrt(mean(abs(x2(:)).^2));
XX = [Preamble_X,xx_qam];
end
end
