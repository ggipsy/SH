function [ funFinalValue,xFinalValue,fun,x,grad ] = sgp( x0,Gamma,Phi,Y )


% Scaled Gradient Projection Method
tol = 1e-9;
max_iter = 5000;

% Set parameters
beta = 1e-4; 
gamma = 0.4;
alpha_min = 1e-7;
alpha_max = 1e2;
L_min = 1e-5;
L_max = 1e10;
M_alpha = 3;
tau = 0.5;

% Initialization
fun = zeros(1,max_iter);
x = zeros(size(x0,1),max_iter);
grad = zeros(size(x0,1), max_iter);
r = zeros(size(x0,1), max_iter);
w = zeros(size(x0,1), max_iter);
alpha_BB1 = zeros(1, max_iter);
alpha_BB2 = zeros(1, max_iter);
z = zeros(size(x0,1), max_iter);
Delta_x = zeros(size(x0,1), max_iter);


k = 1;
alpha(k) = 1;
x(:,k) = x0;
n = size(Phi,2);
N = size(Phi,1);
R = Phi'*Phi;
Yt = Phi'*Y;

while k<=max_iter 
    
    [fun(k),K,L,S,Si] = computeObjFun(x(:,k),Gamma,Phi,Y);
    
    if k>1
       if fun(k-1)-fun(k)<tol*abs(fun(k))
           break
       end
    end

    if L == 0
       Ki = K\eye(size(K));
       Z = (Ki + Phi'*Phi)\eye(size(Ki));
    else
       Z = L*(Si'*Si)*L';
    end
    %Sigmai = eye(N) - Phi*Z*Phi';
    M = R - R*Z*R;
    q = (eye(n)-R*Z)*Yt; %/sigma2;
    V = computeV(K,Gamma,M);
    U = computeU(K,Gamma,q);
    grad(:,k) = U - V;
    
    
    %% Step 1: Compute scaling matrix D
    
    D = diag(min([max([L_min*ones(1,size(x,1));(x(:,k)./V)']); L_max*ones(1,size(x,1))]));
    Di = D\eye(size(D,1));

    
    %% Step 1: Compute step-size alpha
    if k > 1
        r(:,k-1) = x(:,k)-x(:,k-1);
        w(:,k-1) = grad(:,k)-grad(:,k-1);
        alpha_BB1_tmp = (r(:,k-1)'*Di*Di*r(:,k-1))/(r(:,k-1)'*Di*w(:,k-1));
        alpha_BB2_tmp = (r(:,k-1)'*D*w(:,k-1))/(w(:,k-1)'*D*D*w(:,k-1));
        % Alternation strategy
        if r(:,k-1)'*Di*w(:,k-1) <= 0
            alpha_BB1(k) = alpha_max;
        else
            alpha_BB1(k) = min([alpha_max max([alpha_min alpha_BB1_tmp])]);
        end
        if r(:,k-1)'*D*w(:,k-1) <= 0
            alpha_BB2(k) = alpha_max;
        else
            alpha_BB2(k) = min([alpha_max max([alpha_min alpha_BB2_tmp])]);
        end
        if alpha_BB2(k)/alpha_BB1(k) <= tau
            alpha(k) = min(alpha_BB2(max([2,k-M_alpha]):k));
            tau = 0.9*tau;
        else
            alpha(k) = alpha_BB1(k);
            tau = 1.1*tau;
        end
    end
   
    %% Step 2: Projection
    z_tmp = x(:,k)-alpha(k)*D*grad(:,k);
    
    z(:,k) = z_tmp;   
    if sum(z_tmp<0) > 0
        z((z_tmp<0), k) = 0;
    end
    
    
    %% Step 3: Descent direction
    Delta_x(:,k) = z(:,k) - x(:,k);
    
    %% Step 4
    lambda = 1;
    
    %% Step 5: Backtracking
    
    while computeObjFun(x(:,k)+lambda*Delta_x(:,k),Gamma,Phi,Y) > fun(k)+beta*lambda*grad(:,k)'*Delta_x(:,k)
        lambda = gamma*lambda;
    end
    
    x(:,k+1) = x(:,k) + lambda*Delta_x(:,k);
    fun(k+1) = computeObjFun(x(:,k)+lambda*Delta_x(:,k),Gamma,Phi,Y);
    k = k+1;

    
end

fun = fun(1:k);
x = x(:,1:k);
funFinalValue = fun(k);
xFinalValue = x(:,k);


end

