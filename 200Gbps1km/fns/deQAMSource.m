function [ demodsignal ] = deQAMSource(M,x)
% M = 2^BitPerSymbol;
% hdemod = modem.qamdemod('M', M, 'SymbolOrder', 'Gray');
if M == 8
    demodsignal = demodulate_8QAM(x);
else
%     demodsignal = demodulate(hdemod,x);
    demodsignal = qamdemod(x,M,'gray');
end
end

