function [ Y_e,H_average ] = CE( YY,Preamble_X,NumPreamble,NumSymbols,m )
if NumPreamble == 0
    Y_e = YY;H_average = [];
else
Y = YY(:,NumPreamble+1:end);
% Y = YY;
Hy = YY(:,1:NumPreamble)./Preamble_X;
H_average = mean(Hy,2);
%         amp = abs(H_average);
%         angles = smooth(unwrap(angle(H_average)),5);
%         H_average = amp .* exp(1i.*angles);
%         H_average = smooth(H_average,3);
% H_average = averaging_window( m,H_average );
% H_average = modifiedISFA( m,H_average );
H_inverse = 1./H_average;
H_invertible_matrix = repmat(H_inverse,1,size(Y,2));
Y_e = Y.*H_invertible_matrix;

end
end

