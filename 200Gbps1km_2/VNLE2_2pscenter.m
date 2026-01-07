function [h,ye ] = VNLE2_2pscenter( xTx,xRx,NumPreamble_TDE,N1,N2,Lambda,WL)
xTx =xTx(:);
xRx =xRx(:);
% xTx = xTx./sqrt(mean(abs(xTx(:)).^2));
% xRx = xRx./sqrt(mean(abs(xRx(:)).^2));
xTx = xTx./mean(abs(xTx(:)));
xRx = xRx./mean(abs(xRx(:)));
Rxdata = xRx;
Txdata = xTx;
xTx = xTx(1:NumPreamble_TDE*2);
xRx = xRx(1:NumPreamble_TDE);
% xtemp = xRx;
L = size(xTx,1); 
L1=(N1-1)/2;
L2=(N2-1)/2;
N=(N1+1); 


%% Linear_filter_N_tap_l
xTx0 = xTx(:);
xRx0 = xRx(:);
y = zeros(size(xRx0));
% N = 32;
P = eye(N1+(2*N2-WL+1)*WL/2)*0.01;
e = [];
h = zeros(1,N1+(2*N2-WL+1)*WL/2).';

% u = 0.008;
% xTx0 = [zeros(N1-1,1);xTx0];
% xRx0 = [zeros(N-1,1);xRx0];
xTx0 = [zeros(L1,1);xTx0;zeros(L1,1)];
% xRx0 = [zeros(L1,1);xRx0;zeros(L1,1);];



for n = 1:length(xRx0)   

    x1=xTx0(2*n+(N1-1)/2+L1:-1:2*n-(N1-1)/2+L1);
    x2=[];
    for m=1:WL
        x2=[x2;xTx0(2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)).*xTx0([2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)]-(m-1))];
    end
          x = [x1;reshape(x2,[],1)];
    

    k = P*x./(Lambda+x.'*P*x);
    y(n)=h.'*x;      
    e(n)=xRx0(n)-y(n);  
    h=h+k*e(n);
    P=(P-k*(x.'*P))/Lambda;
end

h1 = h(1:N1);
h2 = h(N1+1:N1+(2*N2-WL+1)*WL/2);

%% 1

% figure;plot(h1);hold on;plot(h2);
% figure;surf(abs(h2));
% view(2)


% save d d;
% save h h
% figure;plot(h);hold on;plot(d);
% % figure;plot(20*log10(abs(mse)))
%         figure;plot(e(N1+1:end))
disp(['mse = ',num2str((abs(e(end))).^2)])




%%%%%%%%%%%%%%
%  hInt2Cst = modem.pammod('M', M, 'SymbolOrder', 'Gray');
%         %% 1
% xTx1 = [zeros(2*N1-1,1);Txdata];
xTx1 = [zeros(L1,1);Txdata;zeros(L1,1)];
xRx1 = Rxdata;
ye = zeros(size(Rxdata));
for n =  1:length(xRx1)   
%     x1=xTx1(n:-1:n-N1+1);
%     x2 = xTx1(n:-1:n-N2+1)*xTx1(n:-1:n-N2+1).';
    x1=xTx1(2*n+(N1-1)/2+L1:-1:2*n-(N1-1)/2+L1);
%     x2 = xTx0(2*n:-1:2*n-N2+1)*xTx0(2*n:-1:2*n-N2+1).';
    x2=[];
    for m=1:WL
        x2=[x2;xTx1(2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)).*xTx1([2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)]-(m-1))];
    end
          x = [x1;reshape(x2,[],1)];
    ye(n)=h.'*x;       
end
% ye =ye(2*N:end);
ye =ye(:).';
end


