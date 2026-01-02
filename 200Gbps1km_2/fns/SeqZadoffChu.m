function seq = SeqZadoffChu(N, p)

% p should be relatively prime with repect to N

if nargin < 2; p = 1; end

k = 0:N-1;
if mod(N,2)==0
    phase = pi*k.*k*p/N;
else
    phase = pi*(k+1).*k*p/N;
end
seq = exp(1i*phase).';

% seq = seq .* exp(1i*pi/4);
end


% if nargin < 2; p = 1; end
% xxx=10;
% k = 0:xxx*N-1;
% if mod(xxx*N,2)==0
%     phase = pi*k.*k*p/xxx/N;
% else
%     phase = pi*(k+1).*k*p/xxx/N;
% end
% seq = exp(1i*phase).';
% bbb=500;
% seq =seq(bbb+1:end/xxx+bbb);
% figure;plot(abs(seq))
% % seq = seq .* exp(1i*pi/4);
% end