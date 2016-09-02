function H = block_Hankel(c,r,p,m)
nr = round(size(c,1)/p);
nc = round(size(r,2)/m);
H=[c];
for i = 2:nc,
    c = [c(p+1:end,:); r(:,(i-1)*m+1:i*m)];
    
    H = [H c];

end
