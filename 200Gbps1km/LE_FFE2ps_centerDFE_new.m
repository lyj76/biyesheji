function [h,d,ye ] = LE_FFE2ps_centerDFE_new( xTx,xRx,NumPreamble_TDE,N1,D1,Lambda,M,scale)
xTx =xTx(:);
xRx =xRx(:);
xTx = xTx./mean(abs(xTx(:)));
xRx = xRx./mean(abs(xRx(:)));
Rxdata = xRx;
Txdata = xTx;
xTx = xTx(1:NumPreamble_TDE*2);
xRx = xRx(1:NumPreamble_TDE);
% xtemp = xRx;

% N=N1/2;
N=1000;
K1=(N1-1)/2;
%% Linear_filter_N_tap_l
xTx0 = xTx(:);
xRx0 = xRx(:);
y = zeros(size(xRx0));
% N = 32;
P = eye(N1+D1)*0.01;
e = [];
h = zeros(1,N1).';
d = zeros(1,D1).';
ha=[h;d];
% u = 0.008;
xTx0 = [zeros(N-1,1);xTx0;zeros(K1,1)];
xRx0 = [zeros(N/2-1,1);xRx0];
for n = N:length(xRx0)
    x1=xTx0(2*n+K1:-1:2*n-K1);
    xd1 = xRx0(n-1:-1:n-D1);
    x=[x1;xd1];
    k = P*x./(Lambda+x.'*P*x);
    y(n)=ha.'*x;      
    e(n)=xRx0(n)-y(n);  
    ha=ha+k*e(n);
    P=(P-k*(x.'*P))/Lambda;
end
h=ha(1:length(h));
d=ha(1+length(h):end);

% save d d;
% save h h
% figure;plot(h);hold on;plot(d);
% % figure;plot(20*log10(abs(mse)))
%         figure;plot(e(N1+1:end))
disp(['mse = ',num2str((abs(e(end))).^2)])



%         %% 1
xTx1 = [zeros(N-1,1);Txdata;zeros(K1,1)];
xRx1 = [zeros(N/2-1,1);Rxdata];
ye = zeros(size(Rxdata));
% hInt2Cst = modem.pammod('M', M, 'SymbolOrder', 'Gray');
for n = N:length(xRx1)
    x1=xTx1(2*n+K1:-1:2*n-K1);
    if 0
        xDFE = xRx1(n-1:-1:n-D1);
    else
        if n <= NumPreamble_TDE+N
%             if n <= NumPreamble_TDE+1
            xDFE = xRx1(n-1:-1:n-D1);
        else
        xnew = ye(n-1);
        xnew = xnew.*scale;
        
%         xnew = dePAMSource(M,xnew);        
%         xnew = modulate(hInt2Cst,xnew);
if M==2
        if xnew<=0
            xnew=-1;
        else xnew=1;
        end
end
if M==8
        if xnew<=-6
            xnew=-7;
        else if xnew<=-4
                xnew=-5;
            else if xnew<=-2
                    xnew=-3;
                else if xnew<=0
                        xnew=-1;
                    else if xnew<=2
                            xnew=1;
                        else if xnew<=4
                                xnew=3;
                            else if xnew<=6
                                    xnew=5;
                                else xnew=7;
                                end
                            end
                        end
                    end
                end
            end
        end
end
if M==4
xnew = (xnew<-2)*(-3) +  (xnew<0 && xnew>=-2)*(-1) + (xnew<2 && xnew>=0)*(1) ...
     + (xnew>=2)*(3);
end  
if M == 16
    xnew = (xnew<-14)*(-15) + (xnew<-12 && xnew>=-14)*(-13) + (xnew<-10 && xnew>=-12)*(-11) ...
     + (xnew<-8 && xnew>=-10)*(-9) + (xnew<-6 && xnew>=-8)*(-7) + (xnew<-4 && xnew>=-6)*(-5) ...
     + (xnew<-2 && xnew>=-4)*(-3) + (xnew<0 && xnew>=-2)*(-1) + (xnew<2 && xnew>=0)*(1) ...
     + (xnew<4 && xnew>=2)*(3) + (xnew<6 && xnew>=4)*(5) + (xnew<8 && xnew>=6)*(7) ...
     + (xnew<10 && xnew>=8)*(9) + (xnew<12 && xnew>=10)*(11) + (xnew<14 && xnew>=12)*(13) + (xnew>=14)*(15);
end

        
        xnew = xnew./scale;
        xDFE=[xnew;xDFE(1:end-1)];
        end
    end
    xd1 = xDFE;
    x=[x1;xd1];
    ye(n)=ha.'*x;       
end
ye =ye(N/2:end);
ye =ye(:).';
end

