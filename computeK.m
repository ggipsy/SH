function [ K,Ki ] = computeK( x, Gamma )

Ki = zeros(size(Gamma,1),size(Gamma,2));
for j = 1:length(x)
    Ki = Ki + x(j)*Gamma(:,:,j);
end
K = Ki\eye(size(Ki));
end

