function [ h, M, covar_post, noisev, n, DeltaQ, l0, l1, l2 ] = sh( data,n,min_n, kernel )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUTS:
% data = iddata object with the estimation data
% n = initial length of the estimated impulse response
% min_n = minimal length of the estimated impulse response
% kernel = string containing the kernel type (e.g. 'TC','DC','SS')

% OUTPUTS:
% h(:,i) = impulse response coefficients at the i-th iteration of the
% procedure 
% M{i} = FIR model estimated at the i-th iteration of the procedure
% covar_post{i} = posterior covariance at the i-th iteration of the procedure
% noisev = noise variance estimated through a LS FIR estimate
% n = length of the estimated impulse response

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nest = size(data.y,1);
m = size(data.u,2);
p = size(data.y,2);

n_ini = n;
nk = 1;
tol = 1e6;
n_iter = 50;

% Initialization
M = cell(1,n_iter+1);
covar_post = cell(1,n_iter+1);



[~,~,~,~,M{1},P_chol,K_chol,hp_tmp,n] = rfir_truncated(data,[n_ini nk],kernel,tol,min_n);

K = [];
for j =1:p
    Ktmp = P_chol{j}*P_chol{j}'; % already contains the kernel scaling factor
    K = blkdiag(K,Ktmp);
end
noisev = M{1}.NoiseVariance;

h = zeros(n*m*p, n_iter+1);

%% Determine size of H


[ ll,kk ] = determineHsize( m,p,n );
nr_H = ll*p;
nc_H = m*kk;

while nr_H>nc_H;
    ll=ll-1;
    kk=kk+1;
    nr_H = ll*p;
    nc_H = m*kk;
end;

P = createP(m,p,n,ll,kk);

%% Initialize algorithm

Y = reshape(data.y,numel(data.y),1);
Y = Y./kron(sqrt(diag(noisev)),ones(Nest,1));

Phi = createPhi(data,n);
Phi = Phi./kron(kron(sqrt(diag(noisev)),ones(Nest,1)),ones(1,size(Phi,2)));

R = Phi'*Phi;
F = Phi'*Y;
Ki= (K\eye(size(K,1)));

%% Stable Spline Solution

lambda0 = ones(p,1); % start with SS solution
T = kron(diag(lambda0),eye(m*n))*Ki;
covar_post{1} = (R + T)\eye(size(R,1));
h(:,1) = covar_post{1}*F;
lambda0_vec(1) = lambda0(1);

[W1,W2] = compute_weightings(data.y,data.u,P,h(:,1),n,nr_H,nc_H,p,m);


%% SSR

%%%%%%%%%%%%%%%% Iter 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i=2;


[Qs,Qn,threshold] = initQ_decoupled(h(:,i-1),Nest,P,nr_H,nc_H,W1,W2,p,m,i-2);

Qs = zeros(size(Qs));
Qn = eye(size(Qn));


l0=0.5;
l1 = 1;
l2 = threshold;
vars = [l0; l1; l2];

Gamma(:,:,1) = Ki;
Gamma(:,:,2) = P'*kron(W2*Qs*W2',W1'*W1)*P;
Gamma(:,:,3) = P'*kron(W2*Qn*W2',W1'*W1)*P;


[ MLval,xFinalValue,fun,x,grad ] = sgp(vars,Gamma,Phi,Y);

ML(i) = MLval;
l0 = xFinalValue(1);
lambda0 = l0*ones(p,1);
l1 = xFinalValue(2);
l2 = xFinalValue(3);

lambda1_vec(i) = l1;
lambda0_vec(i) = l0;
lambda2_vec(i) = l2;



T = kron(diag(lambda0),eye(m*n))*(Ki);
covar_post{i} = (R + l1*Gamma(:,:,2) + l2*Gamma(:,:,3) + T)\eye(size(R,1));
h(:,i) = covar_post{i}*F;


Btmp = reshape(h(:,i),n,m,p);
B = cell(p,m);

for k = 1:p
    for j = 1:m
        B{k,j} = [zeros(1,nk) Btmp(:,j,k)'];
    end
end
M{i} = idpoly(1,B);



%%%%%%%%%%%%%%%%% Iter 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=3;
DeltaQ=0;
cont_ciclo_ext=1;
while i<=n_iter+1 & cont_ciclo_ext
    cont=1;
    iold=i;
    while cont & i<=n_iter+1
        
        [W1,W2] = compute_weightings(data.y,data.u,P,h(:,i-1),n,nr_H,nc_H,p,m);
        
        [Qs,Qn] = initQ_decoupled(h(:,i-1),Nest,P,nr_H,nc_H,W1,W2,p,m,DeltaQ);
        
        
        Gamma(:,:,2) = P'*kron(W2*Qs*W2',W1'*W1)*P;
        Gamma(:,:,3) = P'*kron(W2*Qn*W2',W1'*W1)*P;
        vars = [l0; l1; l2];
        
        
        [ MLval,xFinalValue,fun,x,grad ] = sgp(vars,Gamma,Phi,Y);
        
        
        if MLval > ML(i-1)-(1e-5)*ML(i-1)
            cont=0;
            DeltaQ=DeltaQ+1;
            break
            
        end
        
        ML(i) = MLval;
        l0 = xFinalValue(1);
        lambda0 = l0*ones(p,1);
        l1 = xFinalValue(2);
        l2 = xFinalValue(3);
        
                
        lambda1_vec(i) = l1;
        lambda0_vec(i) = l0;
        lambda2_vec(i) = l2;
        
        
        
        T = kron(diag(lambda0),eye(m*n))*(Ki);
        covar_post{i} = (R + l1*Gamma(:,:,2) + l2*Gamma(:,:,3) + T)\eye(size(R,1));
        h(:,i) = covar_post{i}*F;
        
        Btmp = reshape(h(:,i),n,m,p);
        B = cell(p,m);
        
        for k = 1:p
            for j = 1:m
                B{k,j} = [zeros(1,nk) Btmp(:,j,k)'];
            end
        end
        M{i} = idpoly(1,B);
        
        
        i=i+1;
        
    end;
    if iold == i & DeltaQ>1
        cont_ciclo_ext=0;
    end;
    
   
    
end;


M = M(1:i-1);
h = h(:,1:i-1);
covar_post = covar_post(1:i-1);



end

