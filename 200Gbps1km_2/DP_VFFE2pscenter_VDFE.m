function [h,d,ye ] = DP_VFFE2pscenter_VDFE( xTx,xRx,NumPreamble_TDE,N1,N2,D1,D2,Lambda,WL,WD,M,scale)
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
% N=N1/2;
% N=1000;


%% Linear_filter_N_tap_l
xTx0 = xTx(:);
xRx0 = xRx(:);
y = zeros(size(xRx0));
% N = 32;
P = eye(N1+(2*N2-WL+1)*WL/2+D1+(2*D2-WD+1)*WD/2)*0.01;
e = [];
h = zeros(1,N1+(2*N2-WL+1)*WL/2).';
d = zeros(1,D1+(2*D2-WD+1)*WD/2).';
ha=[h;d];
% save ha ha
% u = 0.0001;

xTx0 = [zeros(L1,1);xTx0;zeros(L1,1)];
xRx00 = [zeros(D1,1);xRx0;zeros(D1,1)];
% % xRx0 = [zeros(L1/2,1);xRx0;zeros(L1/2,1)];
% xTx0 = [zeros(N-1,1);xTx0];
% xRx0 = [zeros(N/2-1,1);xRx0];
% for n1=1:40
for n = 1:length(xRx0)   
%     x1=xTx0(n:-1:n-N1+1);
% x2 = xTx0(n:-1:n-N2+1)*xTx0(n:-1:n-N2+1).';
       x1=xTx0(2*n+(N1-1)/2+L1:-1:2*n-(N1-1)/2+L1);
%     x2 = xTx0(2*n:-1:2*n-N2+1)*xTx0(2*n:-1:2*n-N2+1).';
    x2=[];
    for m=1:WL
        x2=[x2;xTx0(2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)).*xTx0([2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)]-(m-1))];
    end
    xt = [x1;reshape(x2,[],1)];
    
    xd1 = xRx00(D1+n-1:-1:D1+n-D1);
    xd2 = [];
%     for m=1:WD
%         xd2=[xd2;xRx0(n-1:-1:n-D2+m-1).*xRx0((n-1:-1:n-D2+m-1)-(m-1))];
%     end
    for m=0:WD-1
        xd2=[xd2;xRx00(D1+n-1:-1:D1+n-(D2-m)).*xRx00(D1+n-1-m:-1:D1+n-(D2-m)-m)];
    end
%     xd2 = xRx0(n-1:-1:n-D2)*xRx0(n-1:-1:n-D2).';
    xd = [xd1;reshape(xd2,[],1)];
    x=[xt;xd];
    
    k = P*x./(Lambda+(x.'*P)*x);
    y(n)=ha.'*x;      
    e(n)=xRx0(n)-y(n);  
    ha=ha+k*e(n);
    P=(P-k*(x.'*P))/Lambda;

%  y(n)=ha.'*x;    
%  e(n)=xRx0(n)-y(n);  
%  ha=ha+u*e(n)*x;
end

% figure;plot(h);hold on;plot(d);
% % figure;plot(20*log10(abs(mse)))
%         figure;plot(e(N1+1:end))
h1 = ha(1:N1);
h2 = ha(N1+1:N1+(2*N2-WL+1)*WL/2);
d1 = ha(N1+(2*N2-WL+1)*WL/2+1:N1+(2*N2-WL+1)*WL/2+D1);
d2 = ha(N1+(2*N2-WL+1)*WL/2+D1+1:end);

xTx1 = [zeros(L1,1);Txdata;zeros(L1,1)];
xRx1 = [zeros(D1,1);Rxdata;zeros(D1,1)];

ye = zeros(size(Rxdata));
% hInt2Cst = modem.pammod('M', M, 'SymbolOrder', 'Gray');
for n = 1:length(Rxdata)
%     x1=xTx1(n:-1:n-N1+1);
%     x2 = xTx1(n:-1:n-N2+1)*xTx1(n:-1:n-N2+1).';
    x1=xTx1(2*n+(N1-1)/2+L1:-1:2*n-(N1-1)/2+L1);
%     x2 = xTx0(2*n:-1:2*n-N2+1)*xTx0(2*n:-1:2*n-N2+1).';
    x2=[];
    for m=1:WL
        x2=[x2;xTx1(2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)).*xTx1([2*n+(N2-1)/2+L1:-1:2*n-(N2-1)/2+L1+(m-1)]-(m-1))];
    end
    xt = [x1;reshape(x2,[],1)];
    
    if 0
%         xDFE = xRx1(n-1:-1:n-D1);
        xd1 = xRx1(D1+n-1:-1:D1+n-D1);
        xd2 = [];
        for m=1:WD
          xd2=[xd2;xRx1(D1+n-1:-1:D1+n-D2+m-1).*xRx1((D1+n-1:-1:D1+n-D2+m-1)-(m-1))];
        end
%         xd2 = xRx1(n-1:-1:n-D2)*xRx1(n-1:-1:n-D2).';
        xd = [xd1;reshape(xd2,[],1)];
    else
        if n <= NumPreamble_TDE+D1
            xDFEt = xRx1(D1+n-1:-1:D1+n-D1);
%             xDFE = xRx1(n-1:-1:n-D1);
            xd1 = xRx1(D1+n-1:-1:D1+n-D1);
%             xd2 = xRx1(n-1:-1:n-D2)*xRx1(n-1:-1:n-D2).';
            xd2 = [];
%             for m=1:WD
%                xd2=[xd2;xRx1(n-1:-1:n-D2+m-1).*xRx1((n-1:-1:n-D2+m-1)-(m-1))];
%             end
    for m=0:WD-1
        xd2=[xd2;xRx1(D1+n-1:-1:D1+n-(D2-m)).*xRx1(D1+n-1-m:-1:D1+n-(D2-m)-m)];
    end
            xd = [xd1;reshape(xd2,[],1)];
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


% if M==4
% xnew = (xnew<B1)*(A1) +  (xnew<B2 && xnew>=B1)*(A2) + (xnew<B3 && xnew>=B2)*(A3) ...
%      + (xnew>=B3)*(A4);
% end 
% if M==4
% xnew = (xnew<B1)*(-3) +  (xnew<B2 && xnew>=B1)*(-1) + (xnew<B3 && xnew>=B2)*(1) ...
%      + (xnew>=B3)*(3);
% end 



if M==4
xnew = (xnew<-2)*(-3) +  (xnew<0 && xnew>=-2)*(-1) + (xnew<2 && xnew>=0)*(1) ...
     + (xnew>=2)*(3);
end 

% if M==8
%         if xnew<=-6
%             xnew=-7;
%             
%              else if xnew<=-4
%             xnew=-5;
%              else if xnew<=-2
%             xnew=-3;
%              else if xnew<=0
%             xnew=-1;
%               else if xnew<=2
%             xnew=1;
%               else if xnew<=4
%             xnew=3;
%               else if xnew<=6
%             xnew=5;
%               else if xnew<=8
%             xnew=7;
%                   end
%                   end
%                   end
%                  end
%                  
%                 end
%             end
%                  end
%         end
% end    
        xnew = xnew./scale;
        xDFEt=[xnew;xDFEt];
        xd1 = xDFEt(1:D1);
%         xDFE=[xnew;xDFE(1:end-1)];
%         xd1 = xDFE;
        xd2 = [];
        for m=1:WD
           xd2=[xd2;xDFEt(m:D2).*xDFEt((m:D2)-(m-1))];
%            xd2=[xd2;xDFE(m:D2).*xDFE((m:D2)-(m-1))];
        end
%         xd2 = xDFE(1:D2)*xDFE(1:D2).';
        xd = [xd1;reshape(xd2,[],1)];
        end
    end

    x=[xt;xd];
    ye(n)=ha.'*x;       
end
% ye =ye(N/2:end);
ye =ye(:).';


end
