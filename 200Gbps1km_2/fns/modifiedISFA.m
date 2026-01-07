function [ Hout ] = modifiedISFA( m,H )
   if 0
%% modified ISFA(left)    
    kmax = length(H);
    kmin = 1;
    for k = 1:kmax
        if (k<=m)
            Htemp = H(1:2*k-1);
            Htemp = averaging_window( k-1,Htemp );
            HH(k) = Htemp(k);
            else
        Hg = 0;
        for kk = k-m:1:k+m
            if ((kk<1)||(kk>kmax))
                temp = 0;
            else
                temp = H(kk);
            end
                Hg = Hg+temp;
        end
        num = min(kmax,k+m)-max(kmin,k-m)+1;
        HH(k) = Hg./num;
        end
    end
    Hout = HH(:);
   else
   %% modified ISFA2(left&right)    
    kmax = length(H);
    kmin = 1;
    for k = 1:kmax
        if (k<=m)
            Htemp = H(1:2*k-1);
            Htemp = averaging_window( k-1,Htemp );
            HH(k) = Htemp(k);
        else if (k>=kmax-m+1&&k<=kmax)
            Htemp = H(k-(kmax-k):end);
            Htemp = averaging_window( k-1,Htemp );
            HH(k) = Htemp(kmax-k+1);
            else
        Hg = 0;
        for kk = k-m:1:k+m
            if ((kk<1)||(kk>kmax))
                temp = 0;
            else
                temp = H(kk);
            end
                Hg = Hg+temp;
        end
        num = min(kmax,k+m)-max(kmin,k-m)+1;
        HH(k) = Hg./num;
            end
        end
    end
    Hout = HH(:);
end

end