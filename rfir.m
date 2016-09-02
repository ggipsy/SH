function  [M,P_chol,K_chol,hp] = rfir(varargin)
%new_rfir	Construct an FIR model using regularization method.
%
%   M = new_rfir(z,nb,kernel) or
%   M = new_rfir(z,[nb,nk],kernel) or
%   M = new_rfir(z,{nb,nk},kernel)
%
%   estimate an FIR model represented by:
%   y(t) = B(q) u(t-nk) +  e(t)
%   where:
%       nb = order of B polynomial + 1
%       nk = input delay (in number of samples)
%       Ny = number of output
%       Nu = number of inputs
%
%   The estimated model is delivered as an IDPOLY object. Type "help
%   idpoly" for more information on IDPOLY objects.
%
%   Output:
%       M : IDPOLY model containing estimated values for B
%       polynomials along with their structure information. When Ny = 1,
%       M is a multi-input idpoly model. When Ny > 1, Jl is temporarily
%       a Ny by 1 cell array with each element as a multi-input idpoly model.
%       In the future version, Jl will be a multi-input-multi-ouput idpoly model.
%
%   Inputs:
%       z : The estimation data as an IDDATA. Use IDDATA object for input-output
%       time domain signals. Type "help iddata" for more information.
%
%       nb and nk: Orders and delays of the FIR model. nb must be specified
%       but nk can be omitted and the default value of nk is 1. Both nb and
%       nk can be scalar, which assumes all FIR models will have the same
%       order and input delay. nb can also be a Ny by 1 column vector, each
%       element of which indicates the order of all FIR models associated
%       with that ouput. nk can also be a Ny by Nu matrix, through which the
%       delays of each FIR model can be set individually.
%
%       kernel: the structure of the regularization matrix. The following
%       kernels are supported:
%            CS: qubic spline kernel
%            SE: squared exponential kernel
%            SS: stable spline kernel
%            HF: high frequency stable spline kernel
%            DI: diagonal kernel
%            TC: tuned and correlated kernel
%            DC: diagonal and correlated kernel
%            ITC: integrated TC kernel
%       kernel can be a char (e.g., kernel = 'DC'), which assumes all FIR models
%       use the same kernel. kernel can also be a Ny by Nu cell array,
%       through which the kernels of each FIR model can be set individually.
%       More information about the structure of the regularization matrix and
%       the regulariation method can be found in literature.
%
%

% check No. of inputs
error(nargchk(3,3,nargin,'struct')); % data, [nb nk], kernel

% front matter
% decomopse the inputs and check applicability
for j = 1:length(varargin)
    if isa(varargin{j},'iddata')
        data = varargin{j};
    else if isscalar(varargin{j}) || isnumeric(varargin{j})
            nbnk = varargin{j};
        else if ischar(varargin{j})
                kernel = upper(varargin{j});
            else if iscell(varargin{j})
                    tmp = varargin{j};
                    if isnumeric(tmp{1})
                        nbnk = tmp;
                    else
                        kernel = upper(tmp);
                    end
                end
            end
        end
    end
end

if realdata(data)
    [Nd Ny Nu Ne] = size(data);
    dom = pvget(data,'Domain');
else
    error('Error in the data: only real-valued signals supported.');
end



if isscalar(nbnk)
    nb = nbnk*ones(Ny,1); nk = ones(Ny,Nu);
else if isnumeric(nbnk)
        if isequal(size(nbnk),[1 2])
            nb = nbnk(1)*ones(Ny,1); nk = nbnk(2)*ones(Ny,Nu);
        else if isequal(size(nbnk),[Ny 1])
                nb = nbnk; nk = ones(Ny,Nu);
            else if isequal(size(nbnk),[Ny Nu+1])
                    nb = nbnk(:,1); nk = nbnk(:,2:end);
                else
                    error('Error in the order (nb) and delay (nk) parameter matrix: Dimension mismatch with that of the MIMO system.')
                end
            end
        end
    else if iscell(nbnk)
            if isequal(size(nbnk{1}),[1 1]) && isequal(size(nbnk{2}),[Ny Nu])
                nb = nbnk{1}*ones(Ny,1); nk = nbnk{2};
            else
                error('Error in the order (nb) and delay (nk) parameter matrix: Dimension mismatch with that of the MIMO system.')
            end
        end
    end
end




if ischar(kernel)
    tmp = kernel;
    kernel = cell(Ny,Nu);
    for i = 1:Nu
        for j = 1:Ny
            kernel{j,i} = tmp;
        end
    end
else if iscell(kernel)
        if ~isequal(size(kernel),[Ny Nu])
            error('Error in the kernel allocation matrix: Dimension mismatch with that of the MIMO system.')
        end
    end
end

indth = zeros(Ny,Nu+1);
for j = 1:Ny
    ind = cumsum((nb(j)+1)*ones(1,Nu)-nk(j,:));  % ind: index of the estimated FIR parameters for each of Nu blocks
    indth(j,:) = [0 ind];
    
    i = 1;
    if strcmp(dom(1),'t') || strcmp(dom(1),'T')
        if sum(Nd-nb(j)*ones(1,Ne)) - ind(end) <=0 || ~isempty(find(Nd-nb(j)*ones(1,Ne)<=0,1))
            i = 0;
        end
    else if strcmp(dom(1),'f') || strcmp(dom(1),'F')
            if  (sum(Nd)-1)*4 -ind(end)  <= 0
                i = 0;
            end
        end
    end
    if i == 0
        error(['Error in ouput channel No.' num2str(j) ' and experiment No.' num2str(i) ' : More data samples required to estimate the noise variance.']);
    end
    
    for i = 1:Nu
        if ~strcmp(kernel{j,i},{'CS' 'SE' 'DI' 'SS' 'HF' 'TC' 'DC' 'ITC'})
            error(['Error in output channel No.' num2str(j) ': No such kernel ' kernel{j,i} ' supported.']);
        end
    end
end



% main body

th = cell(Ny,1);
hp = cell(Ny,1);
hpini = cell(Ny,1);
obj = zeros(Ny,1);
exflg = zeros(Ny,1);
sigma = zeros(Ny,1);
indhp = zeros(Ny,Nu+1);
gradient = cell(Ny,1);
hessian = cell(Ny,1);
otpt = cell(Ny,1);
condNb = zeros(Ny,1);

alg = {'trust-region-reflective','active-set', 'interior-point','sqp'};
options = optimset('GradObj','on','Hessian','on','Hessian','user-supplied','Algorithm',alg{1},'Display','off');
optionsITC = optimset('GradObj','on','Hessian','on','Hessian','user-supplied','Algorithm',alg{4},'Display','off');

for j = 1:Ny
    z = data(:,j,:);
    if strcmp(kernel{j,1}, 'ITC')
         [hpini{j}, Rt, Ney, sigma(j), indhp(j,:),lb,ub] = ini_rfir_miso(z,nb(j),nk(j,:),kernel(j,:),optionsITC);
    else
        [hpini{j}, Rt, Ney, sigma(j), indhp(j,:),lb,ub] = ini_rfir_miso(z,nb(j),nk(j,:),kernel(j,:),options);
    end
    ff = @(x)nglglklhd(x,Rt, Ney, kernel(j,:),sigma(j),indth(j,:), indhp(j,:));
    if strcmp(kernel{j,1}, 'ITC')
        
        Aconstr = [];
        for jj = 1:Nu
            Aconstr = blkdiag(Aconstr, [1 -1 0]);
        end
        
       [hp{j} obj(j) exflg(j) otpt{j}, ~, gradient{j} hessian{j}] = fmincon(ff,hpini{j},Aconstr,zeros(Nu,1),[],[],lb,ub,[],optionsITC);
    else
        [hp{j} obj(j) exflg(j) otpt{j}, ~, gradient{j} hessian{j}] = fmincon(ff,hpini{j},[],[],[],[],lb,ub,[],options);
    end
%      ff = @(x)nglglklhd_simp(x,Rt, Ney, kernel(j,:),sigma(j),indth(j,:), indhp(j,:));
%      [hp{j} obj(j) exflg(j) otpt{j}] = fminsearch(ff,hpini{j});
    [~, condNb(j), th{j}, P_chol{j}, K_chol{j}] = nglglklhd_simp(hp{j},Rt, Ney, kernel(j,:),sigma(j),indth(j,:), indhp(j,:));
end


% output the estimated model and associated structure information
if Ny == 1
    M =  idpoly();
else
    M = idarx();
end

for j = 1:Ny
    
    thtmp = th{j};
    mtmp = idpoly();
    
    for i = 1:Nu
        md = idpoly(1,[zeros(1,nk(j,i)) thtmp(indth(j,i)+1:indth(j,i+1))']);
        mtmp = [mtmp md];
    end
    
    mtmp.NoiseVariance =  sigma(j);
    estinfo = mtmp.estimationinfo;
    estinfo.Status =  ['estimated ' 'model'];
    estinfo.Method =  'new_rfir';
    estinfo.DataLength =  Nd;
    estinfo.DataDomain =  'Time';
    estinfo.InitialState ='Zero';
    estinfo.IniHyperparameter = hpini{j}';
    estinfo.Hyperparameter = hp{j}';
    estinfo.Hyperpartition = indhp(j,:)';
    estinfo.Kernel = [kernel{j,:}];
    estinfo.Optimizationinfo = struct('Minimizer','fmincon', 'options', options, 'Gradient',gradient{j},'Hessian', hessian{j},'Exitflag', exflg(j),'Optinfo', otpt{j});
    estinfo.obj = obj(j);
    estinfo.cond =  condNb;
    try
    mtmp.estimationinfo = estinfo;
    catch
    end
    if Ny == 1
        M = mtmp;
    else
        if j == 1
            M = idarx(mtmp);
        else
            M = [M;idarx(mtmp)];
        end
    end
end



function [hpini, Rt, Ney, sigma, ind2, lb, ub] = ini_rfir_miso(z,nb,nk,kernel,options)

Nu = size(z,3);

% construct the regression matrix Phi for sys-id
[Rt Ney sigma] = qrfactor_Phi(z,nb,nk);

% Initialize the prior hyperparameter
ind2 = zeros(1,Nu);
hpini = [];
lb = [];
ub = [];

for ni = 1:Nu
    
    beta =[0.5:0.1:0.9 0.99]';
    hpr = [-6 -4 -2 0:1:5];
%     if strcmp(kernel{ni}, 'CS')
%         beta = 1;
%         lbi = -inf;
%         ubi = +inf;
%     else if strcmp(kernel{ni}, 'SE')
%             lbi = [eps  -inf]';
%             ubi = [+inf +inf]';
%         else if strcmp(kernel{ni}, 'SS')  || strcmp(kernel{ni}, 'HF')  || strcmp(kernel{ni}, 'DI') || strcmp(kernel{ni}, 'TC')
%                 lbi = [eps -inf]';
%                 ubi = [1-eps +inf]';
%             else if strcmp(kernel{ni}, 'DC')
%                     lbi = [eps -1+eps -inf]';
%                     ubi = [1-eps 1-eps +inf]';
%                 end
%             end
%         end
%     end
  K = kernel{ni};
  tol = sqrt(eps);
   if strcmp(K,'CS')
      beta = 1;
      lbi = -inf;
      ubi = +inf;
   elseif strcmp(K, 'SE')
      lbi = [tol -inf]';
      ubi = [2.5 +inf]';
   elseif strcmp(K, 'SS')
%       lbi = [0.9 -inf]';
      lbi = [tol -inf]';
      ubi = [0.999 +inf]';
   elseif any(strcmp(K, {'HF','DI','TC'}))
%        lbi = [0.7 -inf]'; 
      lbi = [tol -inf]';
      ubi = [1-tol +inf]';
   elseif strcmp(K, 'DC')
      %lbi = [0.72 -0.99 -inf]';
      lbi = [tol -1+tol -inf]';
      %ubi = [1-tol 0.99 +inf]';
      ubi = [1-tol 1-tol +inf]';
   elseif strcmp(K, 'ITC')
       lbi = [tol tol -inf]';
       ubi = [1-tol 1-tol +inf]';

   end
    
    obj = zeros(size(beta,1),size(hpr,2));
    
    
    if Nu == 1
        Rti = Rt;
    else
        Rti = qrfactor_Phi(z,nb,nk,ni);
    end
    
    for nj = 1:size(beta,1)
        for nm = 1:size(hpr,2)
            hpstart = [beta(nj)*ones(1,~strcmp(kernel{ni}, 'CS')) 0.99*ones(1,strcmp(kernel{ni}, 'DC')) 0.99*ones(1,strcmp(kernel{ni}, 'ITC')) hpr(nm)]';
            obj(nj,nm) = nglglklhd_simp(hpstart,Rti,Ney,kernel(ni),sigma,[0 nb+1-nk(ni)], [0 length(hpstart)]);
        end
    end
    
    [~, ind_bbeta] = min(obj(:));
    [indr indc] = ind2sub([nj nm],ind_bbeta);
    hptmp = [beta(indr)*ones(1,~strcmp(kernel{ni}, 'CS')) 0.99*ones(1,strcmp(kernel{ni}, 'DC')) 0.99*ones(1,strcmp(kernel{ni}, 'ITC')) hpr(indc)]';
    
    
    if Nu > 1
        hpstart = hptmp;
        ff = @(x)nglglklhd(x,Rti,Ney,kernel(ni),sigma,[0 nb+1-nk(ni)], [0 length(hpstart)]);
        
        if strcmp(K, 'ITC')
            hptmp = fmincon(ff,hpstart,[1 -1 0],0,[],[],lbi,ubi,[],options);
        else
            
            hptmp = fmincon(ff,hpstart,[],[],[],[],lbi,ubi,[],options);
        end
    end
    
    hpini = [hpini; hptmp];
    lb = [lb; lbi];
    ub = [ub; ubi];
    ind2(ni) = length(hpstart);
    
end
ind2 = cumsum(ind2); ind2=[0 ind2];


function [Rt Ney sigma] = qrfactor_Phi(z,nb,nk,whichu)

[Nd, ~, Nu, Ne] = size(z);

if exist('whichu','var')  && ~isempty(whichu)
    Nu = whichu;
    ind = [zeros(1,whichu) nb+1-nk(whichu)]; % index of the estimated FIR parameters for each of Nu blocks
else
    whichu = 1;
    ind = cumsum((nb+1)*ones(1,Nu)-nk); ind = [0 ind]; % index of the estimated FIR parameters for each of Nu blocks
end

Nth = ind(end);
Rt = zeros(0,Nth+1);
Neind = 1:Ne;
maxsize = 25000;

dom = pvget(z,'Domain');

if lower(dom(1)) == 't'
    
    Ney = sum(Nd-nb*ones(1,Ne));
    
    if Ney < Nth+1
        error('Error in the data: not enough samples.');
    end
    
    if Nth > (maxsize-1)/2
        error('Error in the parameter to be estimated: to many parameters.');
    end
    
    y = pvget(z,'OutputData');
    u = pvget(z,'InputData');
    
    maxsize = 25000-Nth-1;
    
    NylMind = find(Nd <= maxsize); % index of the Exp that has number of samples smaller than the maxize
    NygMind = find(Nd > maxsize);  % index of the Exp that has number of samples larger than the maxize
    
    
    if size(NylMind,2) > 0
        
        nt = cumsum(Nd(NylMind)-nb*ones(1,length(NylMind))); nt = [0 nt];
        nqr = ceil(nt(end)/maxsize);
        nqrind = zeros(nqr,1);nqrind(nqr) = length(NylMind);
        
        for nqri = 1:nqr-1
            nqrind(nqri) = find(nt>nqri*maxsize,1,'first') - 1;
        end
        
        for nqri = 1:nqr
            
            yt = zeros(nt(nqrind(nqri)),1);
            Phi = zeros(Nth,nt(nqrind(nqri)));
            
            for ne = 1:nqrind(nqri)
                for nj = 1:Nd(NylMind(ne))-nb
                    for ni = whichu:Nu
                        Phi(ind(ni)+1:ind(ni+1),nj+nt(ne)) = flipud(u{Neind(NylMind(ne))}(nj:(nj+nb-nk(ni)),ni));
                    end
                end
                yt(nt(ne)+1:nt(ne+1),1) = y{Neind(NylMind(ne))}(nb+1:end);
            end
            
            % Calculate the QR factor of Phi and yt
            Rt = triu(qr([Rt;Phi' yt]));
            Rt = Rt(1:Nth+1,:);
        end
        
    end
    
    
    if size(NygMind,2) > 0
        
        for nNygM = 1:length(NygMind)
            
            nqr = ceil((Nd(NygMind(nNygM))-nb)/maxsize);
            nqrind = [0 (1:nqr-1)*maxsize Nd(NygMind(nNygM))-nb];
            
            for nqri = 1:nqr
                Phi = zeros(Nth,nqrind(nqri+1)-nqrind(nqri));
                for nj = 1:nqrind(nqri+1)-nqrind(nqri)
                    for ni = whichu:Nu
                        Phi(ind(ni)+1:ind(ni+1),nj) = flipud(u{Neind(NygMind(nNygM))}(nj+nqrind(nqri):nj+nqrind(nqri)+nb-nk(ni),ni));
                    end
                end
                yt = y{Neind(NygMind(nNygM))}(nb+1+nqrind(nqri):nb+nqrind(nqri+1));
                % Calculate the QR factor of Phi and yt
                Rt = triu(qr([Rt;Phi' yt]));
                Rt = Rt(1:Nth+1,:);
            end
        end
        
    end
    
    
    
else if lower(dom(1)) == 'f'
        
        nz = complex(z);
        Nd = size(nz,1);
        Ney = sum(2*Nd);
        
        if Ney < Nth+1
            error('Error in the data: not enough samples.');
        end
        
        if Nth > (maxsize-1)/2
            error('Error in the parameter to be estimated: to many parameters.');
        end
        
        y = pvget(nz,'OutputData');
        u = pvget(nz,'InputData');
        w = pvget(nz,'Radfreqs');
        Ts = pvget(nz,'Ts');
        
        
        maxsize = floor((25000-Nth-1)/2)*2;
        
        NylMind = find(Nd <= maxsize/2); % index of the Exp that has number of samples smaller than the maxize
        NygMind = find(Nd > maxsize/2);  % index of the Exp that has number of samples larger than the maxize
        
        
        if size(NylMind,2) > 0
            
            
            nt = cumsum(2*Nd(NylMind)); nt = [0 nt];
            nqr = ceil(nt(end)/maxsize);
            nqrind = zeros(nqr,1);nqrind(nqr) = length(NylMind);
            
            for nqri = 1:nqr-1
                nqrind(nqri) = find(nt>nqri*maxsize,1,'first') - 1;
            end
            
            for nqri = 1:nqr
                
                yt = zeros(nt(nqrind(nqri)),1);
                Phi = zeros(Nth,nt(nqrind(nqri)));
                
                for ne = 1:nqrind(nqri)
                    for ni = whichu:Nu
                        OM = exp(-1i*(nk(ni):nb)'*w{Neind(NylMind(ne))}'*Ts{Neind(NylMind(ne))});
                        tempM = (OM.').*(u{Neind(NylMind(ne))}(:,ni)*ones(1,nb-nk(ni)+1));
                        Phi(ind(ni)+1:ind(ni+1),nt(ne)+1:nt(ne+1)) = [real(tempM)' imag(tempM)'];
                    end
                    yt(nt(ne)+1:nt(ne+1),1) = [real(y{Neind(NylMind(ne))});imag(y{Neind(NylMind(ne))})];
                end
                
                % Calculate the QR factor of Phi and yt
                Rt = triu(qr([Rt;Phi' yt]));
                Rt = Rt(1:Nth+1,:);
                
            end
        end
        
        if size(NygMind,2) > 0
            
            for nNygM = 1:length(NygMind)
                
                nqr = ceil(2*Nd(NygMind(nNygM))/maxsize);
                nqrind = [0 (1:nqr-1)*maxsize/2 Nd(NygMind(nNygM))];
                
                for nqri = 1:nqr
                    Phi = zeros(Nth,2*(nqrind(nqri+1)-nqrind(nqri)));
                    for ni = whichu:Nu
                        OM = exp(-1i*(nk(ni):nb)'*w{Neind(NygMind(nNygM))}(nqrind(nqri)+1:nqrind(nqri+1),ni)'*Ts{Neind(NygMind(nNygM))});
                        tempM = (OM.').*(u{Neind(NygMind(nNygM))}(nqrind(nqri)+1:nqrind(nqri+1),ni)*ones(1,nb-nk(ni)+1));
                        Phi(ind(ni)+1:ind(ni+1),:) = [real(tempM)' imag(tempM)'];
                    end
                    yt = [real(y{Neind(NygMind(nNygM))}(nqrind(nqri)+1:nqrind(nqri+1),ni)); imag(y{Neind(NygMind(nNygM))}(nqrind(nqri)+1:nqrind(nqri+1),ni))];
                    
                    % Calculate the QR factor of Phi and yt
                    Rt = triu(qr([Rt;Phi' yt]));
                    Rt = Rt(1:Nth+1,:);
                end
            end
            
        end
        
    end
end

if nargout > 2
    % Calculate the variance of the measurement noise
    X = (Rt(:,1:Nth)'*Rt(:,1:Nth))\Rt(:,1:Nth)'*Rt(:,end);
    sigma = sum((Rt(:,end)-Rt(:,1:Nth)*X).^2)/(Ney-Nth);
end



function [obj condNb xest P_chol K_chol] = nglglklhd_simp(hp,Rt,Ney, kernel,sigma,ind,ind2)

warning('off', 'MATLAB:nearlySingularMatrix');

Nth = size(Rt,2) -1;
Nu = size(kernel,2);
P_chol = zeros(Nth,Nth);


chol_flag = 1;
cond_flag = 1;

for di = 1:Nu
    
    hyper = hp(ind2(di)+1:ind2(di+1));
    nPi = ind(di+1)-ind(di);
    Pi = zeros(nPi,nPi);
    
    switch kernel{di}
        
        case 'CS'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = dk*dj*min(dk,dj)/2-min(dk,dj)^3/6;
                end
            end
            
        case 'SE'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = exp(-(dk-dj)^2/(2*hyper(1)^2));
                end
            end
            
            
        case 'DI'
            t=abs(hyper(1)).^(1:nPi);%exp(-beta*[1:nb])
            Pi=diag(t);
            
        case 'SS'
            t=abs(hyper(1)).^(1:nPi);%exp(-beta*[1:nb])
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = t(dk)*t(dj)*min(t(dk),t(dj))/2-min(t(dk),t(dj))^3/6;
                end
            end
            
        case 'HF'
            t=abs(hyper(1)).^(1:nPi);%exp(-beta*[1:nb])
            for dk=1:nPi
                for dj=1:nPi
                    if mod(dk+dj,2) == 0
                        Pi(dk,dj) = min(t(dk),t(dj));
                    else
                        Pi(dk,dj) = -min(t(dk),t(dj));
                    end
                end
            end
            
        case 'TC'
            t=abs(hyper(1)).^(1:nPi); %exp(-beta*[1:nb])
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) =min(t(dk),t(dj));
                end
            end
            
        case 'DC'
            for dk = 1:nPi
                for dj = 1:nPi
                    Pi(dk,dj) = abs(hyper(1))^((dk+dj)/2)*hyper(2)^(abs(dk-dj));
                end
            end
            
        case 'ITC'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = (hyper(2)^(max(dk,dj)+1) - hyper(1)^(max(dk,dj)+1))/(max(dk,dj)+1);
                end
            end            
    end
    
    if cond(Pi) > 1e+100
        cond_flag = 0;
    end
    
    try
        U = chol(Pi)'*sqrt(exp(hyper(end)));
    catch
        chol_flag = 0;
        U=chol(Pi+1e-4*eye(nPi))'*sqrt(exp(hyper(end)));
    end
    P_chol(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = U;
    
    
    try
        UU = chol(Pi)';
    catch
        chol_flag = 0;
        UU=chol(Pi+1e-4*eye(nPi))';
    end
    K_chol(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = UU;
    
end


R1 = triu(qr([Rt(:,1:Nth)*P_chol Rt(:,Nth+1); sqrt(sigma)*eye(Nth) zeros(Nth,1)]));
R1 = R1(1:Nth+1,:);



if  cond_flag ~= 1 || chol_flag ~=1
    obj = 1/eps;
else
    obj = R1(end,Nth+1)^2/sigma + 2*sum(log(abs(diag(R1(1:Nth,1:Nth)))))+ (Ney-Nth)*log(sigma);
end

if nargout >1
condNb = cond((R1(1:Nth,1:Nth)'*R1(1:Nth,1:Nth)));
xest = P_chol/R1(1:Nth,1:Nth)*R1(1:Nth,Nth+1);
end


function [obj grad hesn P_chol P_invsqr] = nglglklhd(hp,Rt, Ney, kernel,sigma,ind,ind2)

warning('off', 'PiATLAB:nearlySingularPiatrix');

Nth = size(Rt,2)-1;
Nu = size(kernel,2);
P = zeros(Nth,Nth);
P_chol = zeros(Nth,Nth);
P_invsqr = zeros(Nth,Nth); % square root of the inverse of P
Grd = cell(Nu,1);
Hsn = cell(Nu,1);
grad = zeros(length(hp),1);
hesn = zeros(length(hp),length(hp));

chol_flag = 1;
cond_flag = 1;

for di = 1:Nu
    
    hyper = hp(ind2(di)+1:ind2(di+1));
    nPi = ind(di+1)-ind(di);
    Pi = zeros(nPi,nPi);
    J1 = Pi; J2 = Pi; H11 = Pi; H12 = Pi; H22 = Pi;
    
    switch kernel{di}
        
        case 'CS'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = dk*dj*min(dk,dj)/2-min(dk,dj)^3/6;
                end
            end
            
        case 'SE'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = exp(-(dk-dj)^2/(2*hyper(1)^2));
                    J1(dk,dj) = exp(-(dk-dj)^2/(2*hyper(1)^2))*(dk-dj)^2/hyper(1)^3;
                    H11(dk,dj) = exp(-(dk-dj)^2/(2*hyper(1)^2))*(dk-dj)^2/hyper(1)^4*((dk-dj)^2/hyper(1)^2 - 3);
                end
            end
            Grd{di} = exp(hyper(end))*J1;
            Hsn{di} = exp(hyper(end))*H11;
            
        case 'DI'
            t = hyper(1).^(1:nPi);%exp(-beta*[1:nb])
            td = [1 (2:nPi).*hyper(1).^(1:nPi-1)];
            tdd = [0 2 (2:nPi-1).*((3:nPi).*hyper(1).^(1:nPi-2))];
            Pi = diag(t);
            J1 = diag(td);
            H11 = diag(tdd);
            Grd{di} = exp(hyper(end))*J1;
            Hsn{di} = exp(hyper(end))*H11;
            
        case 'SS'
            for dk=1:nPi
                for dj=1:nPi
                    if dk >= dj
                        Pi(dk,dj) = hyper(1)^(2*dk+dj)/2 - hyper(1)^(3*dk)/6;
                        J1(dk,dj) = (2*dk+dj)*hyper(1)^(2*dk+dj-1)/2 - dk*hyper(1)^(3*dk-1)/2;
                        H11(dk,dj) = (2*dk+dj-1)*(2*dk+dj)*hyper(1)^(2*dk+dj-2)/2 - (3*dk-1)*dk*hyper(1)^(3*dk-2)/2;
                    else
                        Pi(dk,dj) = hyper(1)^(2*dj+dk)/2 - hyper(1)^(3*dj)/6;
                        J1(dk,dj) = (2*dj+dk)*hyper(1)^(2*dj+dk-1)/2 - dj*hyper(1)^(3*dj-1)/2;
                        H11(dk,dj) = (2*dj+dk-1)*(2*dj+dk)*hyper(1)^(2*dj+dk-2)/2 - (3*dj-1)*dj*hyper(1)^(3*dj-2)/2;
                    end
                end
            end
            Grd{di} = exp(hyper(end))*J1;
            Hsn{di} = exp(hyper(end))*H11;
            
        case 'HF'
            t = hyper(1).^(1:nPi);%exp(-beta*[1:nb])
            td = [1 (2:nPi).*hyper(1).^(1:nPi-1)];
            tdd = [0 2 (2:nPi-1).*((3:nPi).*hyper(1).^(1:nPi-2))];
            for dk=1:nPi
                for dj=1:nPi
                    if mod(dk+dj,2) == 0
                        Pi(dk,dj) = min(t(dk),t(dj));
                        J1(dk,dj) = min(td(dk),td(dj));
                        H11(dk,dj) = min(tdd(dk),tdd(dj));
                    else
                        Pi(dk,dj) = -min(t(dk),t(dj));
                        J1(dk,dj) = -min(td(dk),td(dj));
                        H11(dk,dj) = -min(tdd(dk),tdd(dj));
                    end
                end
            end
            Grd{di} = exp(hyper(end))*J1;
            Hsn{di} = exp(hyper(end))*H11;
            
        case 'TC'
            t = hyper(1).^(1:nPi);%exp(-beta*[1:nb])
            td = [1 (2:nPi).*hyper(1).^(1:nPi-1)];
            tdd = [0 2 (2:nPi-1).*((3:nPi).*hyper(1).^(1:nPi-2))];
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) = min(t(dk),t(dj));
                    J1(dk,dj) = min(td(dk),td(dj));
                    H11(dk,dj) = min(tdd(dk),tdd(dj));
                end
            end
            Grd{di} = exp(hyper(end))*J1;
            Hsn{di} = exp(hyper(end))*H11;
            
        case 'ITC'
            for dk=1:nPi
                for dj=1:nPi
                    Pi(dk,dj) =  (hyper(2)^(max(dk,dj)+1) - hyper(1)^(max(dk,dj)+1))/(max(dk,dj)+1);
                    J1(dk,dj) = -hyper(1)^max(dk,dj);
                    J2(dk,dj) = hyper(2)^max(dk,dj);
                    
                    H11(dk,dj) = -max(dk,dj)*hyper(1)^(max(dk,dj)-1);
                    H12(dk,dj) = 0;
                    H22(dk,dj) = max(dk,dj)*hyper(2)^(max(dk,dj)-1);

                end
            end
            Grd{di} = exp(hyper(end))*[J1 J2];
            Hsn{di} = exp(hyper(end))*[H11 H12 H22];
            
        case 'DC'
            for dk = 1:nPi
                for dj = 1:nPi
                    Pi(dk,dj) = hyper(1)^((dk+dj)/2)*hyper(2)^(abs(dk-dj));
                    J1(dk,dj) = (dk+dj)/2*hyper(1)^((dk+dj)/2-1)*hyper(2)^(abs(dk-dj));
                    if dk == dj
                        J2(dk,dj) = 0;
                        H12(dk,dj) = 0;
                    else
                        J2(dk,dj) = abs(dk-dj)*hyper(1)^((dk+dj)/2)*hyper(2)^(abs(dk-dj)-1);
                        H12(dk,dj) = abs(dk-dj)*(dk+dj)/2*hyper(1)^((dk+dj)/2-2)*hyper(2)^(abs(dk-dj)-1);
                    end
                    H11(dk,dj) = ((dk+dj)/2-1)*(dk+dj)/2*hyper(1)^((dk+dj)/2-2)*hyper(2)^(abs(dk-dj));
                    if dk == dj || abs(dk-dj) == 1
                        H22(dk,dj) =0;
                    else
                        H22(dk,dj) = (abs(dk-dj)-1)*abs(dk-dj)*hyper(1)^((dk+dj)/2)*hyper(2)^(abs(dk-dj)-2);
                    end
                end
            end
            Grd{di} = exp(hyper(end))*[J1 J2];
            Hsn{di} = exp(hyper(end))*[H11 H12 H22];
    end
    
    P(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = exp(hyper(end))*Pi;
    
    % Terminite if the condition number of the regularization marix is too large
    if cond(Pi) > 1e+100
        cond_flag = 0;
    end
    
    try
        U = chol(Pi)';
        P_invsqr(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = sqrt(1/exp(hyper(end)))*eye(nPi)/U;
        U = U*sqrt(exp(hyper(end)));
    catch
        % Terminite if the cholesky decomposition fails
        chol_flag = 0;
        U = chol(Pi+1e-4*eye(nPi))';
        P_invsqr(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = sqrt(1/exp(hyper(end)))*eye(nPi)/U;
        U = U*sqrt(exp(hyper(end)));
    end
    P_chol(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1)) = U;
end


R1 = triu(qr([Rt(:,1:Nth)*P_chol Rt(:,Nth+1); sqrt(sigma)*eye(Nth) zeros(Nth,1)]));
R1 = R1(1:Nth+1,:);

if  cond_flag ~= 1 || chol_flag ~=1
    obj = 1/eps;
else
    obj = R1(end,Nth+1)^2/sigma + 2*sum(log(abs(diag(R1(1:Nth,1:Nth)))))+ (Ney-Nth)*log(sigma);
end




%  calculate gradient and hessian
tmp = eye(Nth)/R1(1:Nth,1:Nth)';
for di = 1:Nu
    
    nh = ind2(di+1) - ind2(di);
    Pi = P(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1));
    Pi_inv = P_invsqr(ind(di)+1:ind(di+1),ind(di)+1:ind(di+1));
    
    tmp1 = tmp(1:Nth,ind(di)+1:ind(di+1))*Pi_inv;
    tmp2 = sigma*(tmp1'*tmp1);
    
    X1 = (Pi_inv'*Pi_inv) - tmp2;
    tmp3 = R1(1:Nth,end)'*tmp1;
    X2 = (tmp3'*tmp3);
    Xd = X1 -X2;
    
    if nh == 1
        grad(ind2(di)+1) = Xd(:)'*Pi(:);
        
        X4 = X2*Pi*tmp2;
        Htmp = X2 + X2' - X1'*Pi*X1 - X4 - X4';
        hesn(ind2(di)+1,ind2(di)+1) = Htmp(:)'*Pi(:);
        
    else if nh == 2
            J1 = Grd{di};
            grad(ind2(di)+1) = Xd(:)'*J1(:);
            grad(ind2(di)+2) = Xd(:)'*Pi(:);
            
            H11 = Hsn{di};
            
            tmp4 = X2*J1;
            X3 = tmp4*(Pi_inv'*Pi_inv);
            X4 = tmp4*tmp2;
            Htmp = X3 + X3' - X1'*J1*X1 - X4 - X4';
            hesn(ind2(di)+1,ind2(di)+1) = Xd(:)'* H11(:) + Htmp(:)'*J1(:);
            hesn(ind2(di)+1,ind2(di)+2) = Xd(:)'* J1(:) + Htmp(:)'*Pi(:);
            
            X4 = X2*Pi*tmp2;
            Htmp = X2 + X2' - X1'*Pi*X1 - X4 - X4';
            hesn(ind2(di)+2,ind2(di)+2) = Htmp(:)'*Pi(:);
            hesn(ind2(di)+2,ind2(di)+1) = hesn(ind2(di)+1,ind2(di)+2);
            
        else if nh == 3
                J = Grd{di};
                J1 = J(:,1:ind(di+1)-ind(di));
                J2 = J(:,ind(di+1)-ind(di)+1:end);
                grad(ind2(di)+1) = Xd(:)'*J1(:);
                grad(ind2(di)+2) = Xd(:)'*J2(:);
                grad(ind2(di)+3) = Xd(:)'*Pi(:);
                H = Hsn{di};
                H11 = H(:,1:ind(di+1)-ind(di)); H12 = H(:,ind(di+1)-ind(di)+1:2*(ind(di+1)-ind(di))); H22 = H(:,2*(ind(di+1)-ind(di))+1:end);
                
                
                tmp4 = X2*J1;
                X3 = tmp4*(Pi_inv'*Pi_inv);
                X4 = tmp4*tmp2;
                Htmp = X3 + X3' - X1'*J1*X1 - X4 - X4';
                hesn(ind2(di)+1,ind2(di)+1) = Xd(:)'* H11(:) + Htmp(:)'*J1(:);
                hesn(ind2(di)+1,ind2(di)+2) = Xd(:)'* H12(:) + Htmp(:)'*J2(:);
                hesn(ind2(di)+1,ind2(di)+3) = Xd(:)'* J1(:) + Htmp(:)'*Pi(:);
                
                tmp4 = X2*J2;
                X3 = tmp4*(Pi_inv'*Pi_inv);
                X4 = tmp4*tmp2;
                Htmp = X3 + X3' - X1*J2*X1 - X4 - X4';
                hesn(ind2(di)+2,ind2(di)+2) = Xd(:)'* H22(:) + Htmp(:)'*J2(:);
                hesn(ind2(di)+2,ind2(di)+3) = Xd(:)'* J2(:) + Htmp(:)'*Pi(:);
                
                X4 = X2*Pi*tmp2;
                Htmp = X2 + X2' - X1'*Pi*X1 - X4 - X4';
                hesn(ind2(di)+3,ind2(di)+3) = Htmp(:)'*Pi(:);
                
                hesn(ind2(di)+2,ind2(di)+1) = hesn(ind2(di)+1,ind2(di)+2);
                hesn(ind2(di)+3,ind2(di)+1) = hesn(ind2(di)+1,ind2(di)+3);
                hesn(ind2(di)+3,ind2(di)+2) = hesn(ind2(di)+2,ind2(di)+3);
                
            end
        end
    end
    
end



