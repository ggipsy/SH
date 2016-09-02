function [ P ] = createP( m,p,n,l,k)

%% P*h=vec(H')
P = zeros(l*p*k*m,m*p*n);
Ppart2 = [];

for s = 1:p
    Ppart1 = zeros(m*k,m*n);
    for i = 1:k
        for j = 1:m
            Ppart1(m*(i-1)+j,:) = [zeros(1,(j-1)*n+(i-1)) 1 zeros(1,(m-j+1)*n-i)];
        end
    end
    Ppart2 = blkdiag(Ppart2,Ppart1);
end

for i = 1:l
    P((i-1)*p*m*k+1:i*p*m*k,:) = [zeros(p*m*k,i-1) Ppart2(:,1:n*p*m-i+1)];
end




end

