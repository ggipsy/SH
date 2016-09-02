function [Qs,Qn,threshold,Us] = initQ_decoupled(h,Ndata,P,nr_H,nc_H,W1,W2,p,m,iter)


H = reshape(P*h,nc_H,nr_H)';
H = W2'*H*W1';

idx_trunc = 1+iter;
[Ur,Sr,~]=svd(H*H');
threshold = Sr(idx_trunc);

Us = Ur(:,1:idx_trunc);
Un = Ur(:,idx_trunc+1:end);
Qs = Us*Us';
Qn = Un*Un';

end






