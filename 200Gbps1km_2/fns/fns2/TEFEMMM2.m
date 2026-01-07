function [TimingEstimate FrequencyOffsetEstimate] = TEFEMMM2(Signal,NumFFT,th)
% refer to A Robust Timing amd Frequency Synchronization for OFDM Systems
% Hlaing Minn, Vijay K. Bhargave, Khaled ben Letaief

% Threshold = 0.5;
% plot(real(Signal));
L = 8;
M = NumFFT/L;
Threshold = th;
switch L
    case 4
        p = [-1 1 -1 -1];
    case 8
        p = [1 1 -1 -1 1 -1 -1 -1];
    case 16
        p = [1 -1 -1 1 1 1 -1 -1 1 -1 1 1 -1 1 -1 -1];
end

%% %%%%%%%%%%%%%%%%%%   Timing Estimate
d = 1;
tmp = Signal(d:d+L*M-1);
E(d) = sum(tmp.*conj(tmp));
tmp = reshape(tmp, M, L).';
P(d) = 0;
for k = 1:L-1
	P(d) = P(d) + sum(conj(tmp(k,:)).* tmp(k+1,:))*p(k)*p(k+1);
end
Lambda2(d) = (L/(L-1)*abs(P(d))/E(d))^2;

while(Lambda2(d)<Threshold && (d+L*M)<length(Signal))
    d = d+1;
    tmp = Signal(d:d+L*M-1);
    E(d) = sum(tmp.*conj(tmp));
    tmp = reshape(tmp, M, L).';
    P(d) = 0;
    for k = 1:L-1
        P(d) = P(d) + sum(conj(tmp(k,:)).* tmp(k+1,:))*p(k)*p(k+1);
    end
    Lambda2(d) = (L/(L-1)*abs(P(d))/E(d))^2;
end
ind = d+1;
for d = ind:min(ind+M*L-1,length(Signal)-L*M+1)
    tmp = Signal(d:d+L*M-1);
    E(d) = sum(tmp.*conj(tmp));
    tmp = reshape(tmp, M, L).';
    P(d) = 0;
    for k = 1:L-1
        P(d) = P(d) + sum(conj(tmp(k,:)).* tmp(k+1,:))*p(k)*p(k+1);
    end
    Lambda2(d) = (L/(L-1)*abs(P(d))/E(d))^2;
end
% figure;plot(Lambda2);
% close;
TimingEstimate = find(Lambda2 == max(Lambda2));disp(['TimingEstimate = ',num2str(TimingEstimate)])  
% TimingEstimate = 3;
%%  %%%%%%%%%%%%%%%%%%   FrequencyOffsetEstimate
L = 2 * floor(L/2);
H = L/2;
N = M*L;
y = Signal(TimingEstimate : TimingEstimate+M*L-1);
y = reshape(reshape(y, M, L)*diag(p(1:L)), 1, L*M);

u = 3 * ((L - [1:H]) .* (L - [1:H] + 1) - H*(L-H))/H/(4*H*H - 6*L*H + 3*L*L -1);
for m = 1:H+1
    Ry(m) = 1/(N - (m-1)*M)* sum(conj(y( 1:(N-(m-1)*M) )) .* y((m-1)*M+1:N));
end
phi = wrapToPi(mod(angle(Ry(2:H+1)) - angle(Ry(1:H)), 2*pi)) ;
FrequencyOffsetEstimate = L/2/pi*sum(u.*phi);

end