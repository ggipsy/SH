function [ Phi ] = createPhi( data,n )

[N,m] = size(data.u);
u = [zeros(n,m); data.u];
p = size(data.y,2);

Phi = zeros(N*p,n*m*p);


for j = 1:p
    for l = 1:m
        for k = 1:n
            Phi((j-1)*N+1:N*j,(l-1)*n+n*m*(j-1)+k) = u(n-k+1:n-k+N,l);
        end
    end
end



end

