function [RowSym, RowCst] = PAMSource(M,SymLen)

% M = 2^BitPerSymbol;
% hInt2Cst = modem.pammod('M', M, 'SymbolOrder', 'Gray');
% RowSym = randi([0 M-1], 1, SymLen);
% 
% RowCst = modulate(hInt2Cst,RowSym);


RowSym = randi([0 M-1],1,SymLen);
RowCst = pammod(RowSym,M,0,'gray');

end