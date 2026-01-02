function SeqTD  = TEFETrainMMM(NumSubCarr,NumFFT,NumGI,Type,fc_digital)
% refer to A Robust Timing amd Frequency Synchronization for OFDM Systems
% Hlaing Minn, Vijay K. Bhargave, Khaled ben Letaief

L = 8;
M = NumSubCarr/L;
M1 = NumFFT/L;
switch L
    case 4
        p = [-1 1 -1 -1];
    case 8
        p = [1 1 -1 -1 1 -1 -1 -1];
    case 16
        p = [1 -1 -1 1 1 1 -1 -1 1 -1 1 1 -1 1 -1 -1];
end

% [tmp Seq] = Golay(2,Obj.M);
Seq = SeqZadoffChu(M);
Seq = [Seq(1:M/2,:);zeros(M1-M,1);Seq(M/2+1:M,:)];

switch Type
    case 'FD'
        SeqTD = sqrt(size(Seq,1))*ifft(Seq, [], 1);
    case 'TD'
        SeqTD = Seq;
    otherwise
        error('');
end
SeqTD = reshape((diag(p) * repmat(SeqTD.', L, 1)).',1,[]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SeqTD = [SeqTD SeqTD(1:NumGI)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SeqTD = real(SeqTD.*exp(1i*2*pi*fc_digital*(1:length(SeqTD))));
SeqTD = SeqTD ./ sqrt(mean(abs(SeqTD).^2));
end