function [ fun,K,L,S,Si,Ki ] = computeObjFun( x, Gamma, Phi, Y )

% Compute objective funtion using formula
% (||Y||^2 - ||inv(S)*L'*Phi'*Y||^2)/sigma2 + N*log(sigma2) + 2log|S|

N = size(Phi,1);
n = size(Phi,2);
[K,Ki] = computeK(x,Gamma);
try
%     L = chol(K,'lower');
    [UU,SS,~] = svd(K);
    L = UU*sqrt(SS);
    S = chol(eye(n)+L'*(Phi'*Phi)*L,'lower');
    Si = S\eye(size(S,1));
    
    fun = Y'*Y - Y'*Phi*L*(Si'*Si)*L'*Phi'*Y + 2*sum(log(diag(S)));
    
catch
    
    A =  Ki;
    Sigmai = eye(size(Phi,1))-Phi*((Phi'*Phi + A)\eye(size(A,1)))*Phi';
    try
        L = chol(Sigmai)';
        Yb = L'*Y;
        fun = Yb'*Yb-2*sum(log(diag(L)));
        L = 0;
        S = 0;
        Si = 0;
    catch
        fun = 1e10;
        L = 0;
        S = 0;
        Si = 0;
    end
end



end

