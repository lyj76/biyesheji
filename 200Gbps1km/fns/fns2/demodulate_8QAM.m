function MatSym_hat = demodulate_8QAM(Cst_hat)
    input = reshape(Cst_hat,1,[]);
    const = [1+1i -1+1i 1-1i -1-1i 1+sqrt(3) 1i*(1+sqrt(3)) -(1+sqrt(3)) -1i*(1+sqrt(3))];
    input = repmat(input,8,1);
    [mindis MatSym_hat] = min(abs(input - repmat(const.',1,size(input,2))));
    MatSym_hat = MatSym_hat - 1;
end