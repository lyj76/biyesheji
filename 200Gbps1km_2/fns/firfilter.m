function [ y ] = firfilter( x )
%%%%%
fsamp = 0.2e9;
fcuts = [19e6 21e6];
mags = [1 0];
devs = [0.003162 0.003162];
[n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,fsamp);
b = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
% freqz(b)
y = filter(b,1,x);%fir
%%%%%
end

