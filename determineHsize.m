function [ l,k ] = determineHsize( m,p,n )

l_lb = round((m*n+m-p)/(p+m));
l_ub = round((m*n+m)/(p+m));

k_lb = n+1-l_lb;
k_ub = n+1-l_ub;

if ( abs(p*l_lb-m*k_lb) <= abs(p*l_ub-m*k_ub) )
    l = l_lb;
    k = k_lb;
else
    l = l_ub;
    k = k_ub;
end
end

