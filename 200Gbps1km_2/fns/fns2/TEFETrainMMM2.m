function SeqTD  = TEFETrainMMM2(NumFFT,NumGI)
% refer to A Robust Timing amd Frequency Synchronization for OFDM Systems
% Hlaing Minn, Vijay K. Bhargave, Khaled ben Letaief

%%
Obj.L = 8;
Obj.M = NumFFT/Obj.L;
Obj.NumGI = NumGI;
Obj.Type = 'FD';
switch Obj.L
    case 4
        p = [-1 1 -1 -1];
    case 8
        p = [1 1 -1 -1 1 -1 -1 -1];
    case 16
        p = [1 -1 -1 1 1 1 -1 -1 1 -1 1 1 -1 1 -1 -1];
end

% [tmp Seq] = Golay(2,Obj.M);
Seq = ZadoffChu(Obj.M);
switch Obj.Type
    case 'FD'
        SeqTD = sqrt(size(Seq,1))*ifft(Seq, [], 1);
    case 'TD'
        SeqTD = Seq;
    otherwise
        error('');
end
SeqTD = reshape((diag(p) * repmat(SeqTD.', Obj.L, 1)).',1,[]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SeqTD = [SeqTD SeqTD(1:Obj.NumGI)];

end