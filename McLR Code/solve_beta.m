function new_beta = solve_beta_1(Clinical,V,beta0,kappa,phi)
    function F = func(beta) 
        [m,k] = size(V);
        E = Clinical(:,2);
        T = Clinical(:,1);
        rank_triangle = tril(ones(m,m-1),-1);
        A = repmat(exp(V*beta),1,m-1).*rank_triangle; 
        B = -kappa*sum(repmat(E(1:end-1)',k,1).*(V(1:end-1,:)' - ((A')*V)'./(repmat(sum((A),1),k,1))),2);
        C = (phi/m) * sum( (  (V' - repmat(sum(V',2)/m,1,m))'.*repmat((T - mean(T)),1,k)  ).*repmat(E,1,k),1);
        F = B + C';
    end
new_beta = fsolve(@func,beta0,options);
end
