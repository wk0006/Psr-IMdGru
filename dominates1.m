% Function that returns 1 if x dominates y and 0 otherwise
function d = dominates1(x,y)
    d = all(x<=y,2) & any(x<y,2);
end
