function [ ydata,h ] = RLS( xTx,xRx,NumPreamble_TDE,N, Lambda)
xTx = xTx./sqrt(mean(abs(xTx(:)).^2));
xRx = xRx./sqrt(mean(abs(xRx(:)).^2));
Rxdata = xRx(:);

xTx = xTx(:,1:NumPreamble_TDE);
xRx = xRx(:,1:NumPreamble_TDE);

%% Linear_filter_N_tap_l +
xTx0 = xTx(:);
xRx0 = xRx(:);
y = zeros(size(xTx0));
P = eye(N);
e = [];
h = zeros(1,N).';
xTx0 = [zeros(N-1,1);xTx0];
xRx0 = [zeros(N-1,1);xRx0];
for n = N:length(xTx0)     
    x1=xTx0(n:-1:n-N+1);
    k = P*x1./(Lambda+x1.'*P*x1);
    y(n)=h.'*x1;     
    e(n)=xRx0(n)-y(n); 
    h=h+k*e(n);
    P=(P-k*x1.'*P)/Lambda;
end

stem(h./max(h),'*r');
figure;plot(e(N:end))

ydata = [];

end

