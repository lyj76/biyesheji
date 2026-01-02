function seq = ZadoffChu(N)
p = 1; % p should be relatively prime with repect to N
k = 0:N-1;
if mod(N,2)==0
phase = pi*k.*k*p/N;
else
phase = pi*(k+1).*k*p/N;
end
seq = exp(1i*phase).';
end