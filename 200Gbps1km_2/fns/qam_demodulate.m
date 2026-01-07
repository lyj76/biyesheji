function [ Re_Digital_Baseband_Signal ] = qam_demodulate( y_qam_scale,M )
%QAM_DEMODULATE Summary of this function goes here
qamd = modem.qamdemod('M', M, 'SymbolOrder', 'Gray','OutputType', 'Bit');
demod_out = demodulate(qamd,y_qam_scale);    
demod_out_s = demod_out(:);                                 %²¢´®×ª»»
Re_Digital_Baseband_Signal = demod_out_s.';

end

