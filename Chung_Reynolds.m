function objval = Chung_Reynolds(x)%[-5.12,5.12]
 n = size(x,2); 
f = 0;
for i = 1:n;
    f = f+(x(i).^2).^2;
end
objval = f;
