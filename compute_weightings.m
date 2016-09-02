
% compute row and column weightings for the Hankel matrix

% Hb = W2'*Hb*W1';


function [W1,W2] = compute_weightings(y,u,P,h,n,nr_H,nc_H,p,m);

kf = round(nr_H/p);
kp = round(nc_H/m);
kp0 = n-kp;
Hf = zeros(p*kf,m*kf);

H = reshape(P*h,nc_H,nr_H)';

for i=2:kf,
Hf((i-1)*p+1:end,(i-2)*m+1:(i-1)*m) = H(1:end-(i-1)*p,1:m);
end;

Hp0 = zeros(kf*p,kp0*m);
for i=1:kf-1,
Hp0(1:end-(i)*p,end-i*m+1:end-i*m+m) = H((i)*p+1:end,end-m+1:end);
end;


N = size(y,1);

yv = reshape(y',N*p,1);
uv = reshape(u',N*m,1);


Y =  block_Hankel(yv(1:(kp0+kp+kf)*p),y((kp0+kp+kf-1)+1:end,:)',p,1);
U =  block_Hankel(uv(1:(kp0+kp+kf)*m),u((kp0+kp+kf-1)+1:end,:)',m,1);
Yf = Y((kp0+kp)*p+1:end,:);
Uf = U((kp0+kp)*m+1:end,:);

for j=1:kp
    Up((j-1)*m+1:j*m,:) = U((kp0+kp)*m-j*m+1:(kp0+kp)*m-(j-1)*m,:);
end;

Up0 = U(1:kp0*m,:);

Z = [Up0;Up;Yf-Hf*Uf];
[Q,L]=qr(Z',0);
L = L';


L32 = L((kp+kp0)*m+1:end,kp0*m+1:kp0*m+kp*m);
L33 = L((kp+kp0)*m+1:end,kp0*m+kp*m+1:end);
L22 = L((kp0)*m+1:(kp0+kp)*m,(kp0)*m+1:(kp0+kp)*m);
W2 = L32*L32'+L33*L33';
W2 = pinv(chol(W2));

W1 = L22';











