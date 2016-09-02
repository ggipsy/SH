function [ U ] = computeU( K,Gamma,q )


for j = 1:size(Gamma,3)
    U(j,1) = q'*K*Gamma(:,:,j)*K*q;
end

end

