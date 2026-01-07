function [ XX,Preamble_X ] = Pream( xx_qam,NumPreamble,NumSubCarr_used,c)
if NumPreamble == 0
    XX = xx_qam;Preamble_X = [];
else
switch c
    case 'zc'
%%%SeqZadoffChu
Preamble_X = SeqZadoffChu(NumSubCarr_used);
for k = 2:NumPreamble
Preamble_X(:,k) = circshift(Preamble_X(:,k-1),1);
end
    case 'dpsk'
%%%dpsk
% Preamble_dpsk = modem.dpskmod();
Preamble_matrix = randi([0 1], 1, NumPreamble*NumSubCarr_used);
% Preamble = modulate(Preamble_dpsk,Preamble_matrix);

Preamble = pammod(Preamble_matrix,2,0,'gray');
Preamble_X = reshape(Preamble,NumSubCarr_used,NumPreamble); 

    case 'qpsk'
%%%QPSK
% Preamble_matrix = randi([0 1], 2, NumPreamble*NumSubCarr_used);
% Preamqamm = modem.qammod('M', 4, 'SymbolOrder', 'Gray','InputType', 'Bit');
% Preamble = modulate(Preamqamm,Preamble_matrix);
Preamble_matrix = randi([0 3], 1, NumPreamble*NumSubCarr_used);
Preamble = qammod(Preamble_matrix,4,'gray');
Preamble_X = reshape(Preamble,NumSubCarr_used,NumPreamble); 
end
%%%
% Preamble_X = Preamble_X./sqrt(mean((abs(Preamble_X(:))).^2));
Preamble_X = Preamble_X./sqrt(mean((abs(Preamble_X(:))).^2)).*sqrt(mean((abs(xx_qam(:))).^2));
%%%%
XX = [Preamble_X,xx_qam];
end
end

