function y = Normalize (x, mn)
%NORMALIZE Normalize signal along the column direction by the mean
% of signal's absolute value.
tmp = mean(abs(x));
ncl = size(x,1);
while (ncl > 1)
    tmp = mean(tmp);
    ncl = ncl - 1;
end
switch mn
    case {1,2}
        y = x / tmp;
    case 4
        y = x / tmp * sqrt(2);
    case 8
        y = x / tmp * 2.070206586286458;
    case 16
        y = x / tmp * 2.99535239245729;
    case 32
        y = x / tmp * 4.23016904833816;
    case 64
        y = x / tmp * 6.0868904690454;
    case 128    
        y = x / tmp * 8.531455769697674;
    case 256    
        y = x / tmp * 12.225287369849347;
    case 512
        y = x / tmp * 17.100207972244853;
    case 1024
        y = x / tmp * 24.477212133509510;
end