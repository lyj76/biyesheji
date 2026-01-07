function [ Hout ] = averaging_window( m,H )
   
%% ISFA
    kmax = length(H);
    kmin = 1;
    for k = 1:kmax
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
    Hout = HH(:);
end

