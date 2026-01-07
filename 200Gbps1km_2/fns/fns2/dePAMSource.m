function [ demodsignal ] = dePAMSource(M,x)
% M = 2^BitPerSymbol;
% hdemod = modem.pamdemod('M', M, 'SymbolOrder', 'Gray');
% 
% demodsignal = demodulate(hdemod,x);
demodsignal = pamdemod(x,M,0,'gray');
end
