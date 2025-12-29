function y = Normalizepam (x, mn)
%NORMALIZE Normalize signal along the column direction by the mean

tmp = mean(abs(x));
ncl = size(x,1);
while (ncl > 1)
    tmp = mean(tmp);
    ncl = ncl - 1;
end
switch mn
    case 2
        y = x / tmp;
    case 4
        y = x / tmp * 2;
    case 8
        y = x / tmp * 4;
    case 16
        y = x / tmp * 8;
end