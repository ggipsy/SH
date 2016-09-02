function [ Mreg,P_chol,K_chol,hp,Mreg_t,P_chol_t,K_chol_t,hp_t,idx_trunc ] = rfir_truncated( data, orders, kernelType, tol, min_order )

m = size(data.u,2);
p = size(data.y,2);

n = orders(:,1);
nk = orders(:,2:end);

[Mreg,P_chol,K_chol,hp] = rfir(iddata([zeros(n,p); data.y],[zeros(n,m); data.u]),[n nk],kernelType);


% truncate the kernel according to the condition number of each block
cond_K = zeros(1,p);
for ii = 1:p
    cond_K(ii) = cond(K_chol{ii});
end
n_new = n;
kk = 1;
while sum(cond_K > tol) ~= 0 && n_new > min_order
    for jj = 1:p
        K_chol_trunc_tmp=[];
        for im=1:size(data.u,2),
            K_chol_trunc_tmp = blkdiag(K_chol_trunc_tmp,K_chol{jj}((im-1)*n+1:(im-1)*n+n-kk,(im-1)*n+1:(im-1)*n+n-kk));
        end;
        cond_K(jj) = cond(K_chol_trunc_tmp);
    end
    n_new = n -kk;
    kk = kk +1;
end
idx_trunc = n_new;


[Mreg_t,P_chol_t,K_chol_t,hp_t] = rfir(iddata([zeros(idx_trunc,p); data.y],[zeros(idx_trunc,m); data.u]),[idx_trunc nk],kernelType);

cond_K_t = zeros(1,p);
for ii = 1:p
    cond_K_t(ii) = cond(K_chol_t{ii});
end

%     if sum(cond_K_t > tol) ~= 0
%         disp('wrong')
%         cond_K_t
%     end



end

