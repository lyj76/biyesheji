function [ x_qam ] = qam_modulate( M,Digital_Baseband_Signal,k )

%QAM_MODULATE Summary of this function goes here
bit_matrix = reshape(Digital_Baseband_Signal,k,length(Digital_Baseband_Signal)/k);
qamm = modem.qammod('M', M, 'SymbolOrder', 'Gray','InputType', 'Bit');
x_qam = modulate(qamm,bit_matrix);

end

