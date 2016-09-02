function [ V ] = computeV( K, Gamma, M )

% Compute V(x) = tr(K*Sigmai*K*Gamma(j))

for j = 1:size(Gamma,3)
    V(j,1) = trace(M*K*Gamma(:,:,j)*K);
end

end

