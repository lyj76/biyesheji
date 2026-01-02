function [RowSym, RowCst] = QAMSource(M,SymLen)

% M = 2^BitPerSymbol;
% hInt2Cst = modem.qammod('M', M, 'SymbolOrder', 'Gray');
% RowSym = randi([0 M-1], 1, SymLen);

RowSym = randi([0 M-1],1,SymLen);



if M == 8
    RowCst = modulate_8QAM(RowSym);
else
%     RowCst = modulate(hInt2Cst,RowSym);
    RowCst = qammod(RowSym,M,'gray');
end
end

