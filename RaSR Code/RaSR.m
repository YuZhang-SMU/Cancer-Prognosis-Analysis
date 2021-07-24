%% Input
% X_G,X_H     n_g*c1,n_p*c2  Normalized original feature matrices
% X           n*(c1+c2)
%             n is the number of training samples and (c1+c2) is the dimension of feature
% Status      n*1           The event indicator vector 
% Time        n*1           The observed time
% Y           n*q           The clinical attribute matrix
%             q is the number of clinical variables
% H           n*q           The indicator matrix for Y
% Z           n*(q+1)       The label-attribute matrix
% S           n(q+1)        The indicator matrix for Z
% Y_val       n_val*q       The clinical attribute matrix for validation samples, and the 
%                           validation set is randomly selected from the training set 
% H_val       n_val*q       The indicator matrix for Y_val
% Y_tr       (n-n_val)*q    Y with Y_val removed
% H_tr       (n-n_val)*q    The indicator matrix for Y_tr

%% Parameter Settings 
kappa1=1e-5;    % The range is [5e-8,1e-7,...,1e-3,5e-3]
kappa2=1e-3;    % The range is [5e-4,1e-3,...,1e1,5e1]
kappa3=1e-5;    % The range is [5e-5,1e-4,...,1e0,5e0]
gamma1=1e-5;    % The range is [1e-6,1e-5,1e-4]
gamma2=1e-5;    % The range is [1e-6,1e-5,1e-4]
gamma3=1e-5;    % The range is [1e-6,1e-5,1e-4]
epsilon1=1e-5;  % The range is [5e-9,1e-8,...,1e-4,5e-4]
epsilon2=1e-5;  % The range is [5e-9,1e-8,...,1e-4,5e-4]
epsilon3=1;     % The range is [0.2,0.4,...,1.6,1.8]
k=8;            % The feature dimension for the shared representations 
rho=1e-4;       % The scale factor 
tau=2;          % The training strip length    

%% Procedure
P = randn(c,k); % projection matrix, c=c1+c2
U = randn(k,c); % reconstruction matrix 
B = randn(k-1,k);
D = randn(k,q);
V = randn(n,k); % shared representations matrix 
lambda=abs(randn(1,q));
beta = randn((k+q),1); % the coefficient vector for Cox roportional hazard model 
muk=mean(lambda)*ones(1,q);

%% Calculate the initial loss for subsequent lambda updates 
loss_part1_train_1=zeros(1,q);
loss_part1_val_1=zeros(1,q);
for temp1=1:q
    loss_part1_train_1(1,temp1)=epsilon3*norm(H_tr(:,temp1).*(V_tr*D(:,temp1)-Y_tr(:,temp1)),'fro')^2; % V_tr is the shared representations of the samples corresponding to Y_1 
    loss_part1_val_1(1,temp1)=epsilon3*norm(H_val(:,temp1).*(V_val*D(:,temp1)-Y_val(:,temp1)),'fro')^2; % V_val is the shared representations of the samples corresponding to Y_val 
end
E_train_all=zeros(Iter_Time+1,q); % Iter_Time is the total number of iterations 
E_val_all=zeros(Iter_Time+1,q);
E_train_all(1,:)=loss_part1_train_1;
E_val_all(1,:)=loss_part1_val_1;

%% Iterative update        
for iter_time = 1:Iter_Time
    V_H=V(Location_H,:); %Location_H is the location of the patients with missing histopathological images
    
    %% Step1--P
    P_G=(kappa1*(X_G'*X_G)+gamma1*eye(c1))\(kappa1*X_G'*V);% Calculate P_G
    P_H=(kappa1*(X_P'*X_H)+gamma1*eye(c2))\(kappa1*X_H'*V_H);% Calculate P_H
    P_new=[P_G;P_H];
    
    %% Step2--U
    U_G=(kappa2*(V'*V)+gamma2*eye(k))\(kappa2*V'*X_G);% Calculate U_G 
    U_H=(kappa2*(V_H'*V_H)+gamma2*eye(k))\(kappa2*V_H'*X_H);% Calculate U_H
    U_new=[U_G,U_H];
    
    %% Step3--B
    V_B_1=zscore(V);
    delta_B=zeros(k-1,k);
    for temp=1:k
        V_j_rest=V_B_1;
        v_j=V_j_rest( :,temp);
        V_j_rest( :,temp)=[];
        delta_B(:,temp)= 2*V_j_rest'*(v_j*v_j')*V_j_rest*B( :,temp);
    end
    B_new=B-epsilon1*delta_B;
    
    %% Step4--D
    %part1
    D_part1=zeros(k,q);
    for temp=1:q
        D_part1(:,temp)=2*epsilon3*lambda(1,temp)*V'*(eye(n).*(H_train(:,temp)*H_train(:,temp)'))*V*d_j_all(:,temp)-2*epsilon3*lambda(1,temp)*V'*(eye(n).*(H(:,temp)*H(:,temp)'))*Y(:,temp) ;
    end
    
    %part2
    M_1=zeros(n,k+q);
    for temp=1:n
        if Status(temp,1)==0
            M_1(temp,:)=zeros(1,k+q);
        else
            M_1(temp,:)=beta';
        end
    end
    D_part2=zeros(k,q);
    for temp=1:q
        D_part2( :,temp)=V'*M_1( :,k+temp);
    end
    delta_D=D_part1-kappa3*D_part2;
    D_new=D-delta_D;

    %% Step5--V
    % part1
    U_G=U(:,1:c1);
    P_G=P(1:c1,:);
    
    V_XP_GP=-2*kappa1*X_GP*P'; % X_GP is the total feature matrix of the sample with complete multi-modality data
    V_XP_G_only=-2*kappa1*X_G_only*U_G'; % X_G_only is the feature matrix of patients with only gene data
    V_XP=zeros(n,k);
    V_XP(Location,:)=V_XP_GP; % Location is the location of the patients with complete multi-modality data
    V_XP(Location_H,:)=V_XP_G_only;
    
    V_XU_GP=-2*kappa2*X_GP*Q;
    V_XU_G_only=-2*kappa2*X_G_only*P_G;
    V_XU=zeros(n,k);
    V_XU(Location,:)=V_XU_GP;
    V_XU(Location_H,:)=V_XQ_G_only;
    
    V_part1_1=zeros(n,k);
    for temp=1:k
        V_j_rest=V;
        v_j=V_j_rest( :,temp);
        V_j_rest( :,temp)=[];
        V_part1_1(:,temp)=2*V_j_rest*B( :,temp)*B( :,temp)'* V_j_rest'*v_j;
    end
    
    V_part1_2=zeros(n,k);
    for temp=1:q+1
        V_part1_2_1=(Z( :,temp).*S(:,temp))*(Z( :,temp)'.*S(:,temp)')*V;
        V_part1_2=V_part1_2+V_part1_2_1;
    end
    V_part1_2=2*(1/(q+1))*V_part1_2;
    
    V_part1=2*kappa2*V*(U*U')+V_XP+2*kappa1*V+V_XU+2*gamma3*V+epsilon1*V_part1_1-epsilon2*V_part1_2;
    
    % part2
    V_part2=zeros(n,k);
    for temp=1:q
        V_part2_1=lambda(1,temp)*epsilon3*(eye(n).*(H(:,temp)*H(:,temp)'))*V*D(:,temp)*D(:,temp)'-lambda(1,temp)*epsilon3*(eye(n)*(H(:,temp)*H(:,temp)'))*Y(:,temp)*D(:,temp)';
        V_part2=V_part2+V_part2_1;
    end
    V_part2=(2*q)*V_part2;
    
    % part3
    M_2=zeros(n,k+q);
    G=[eye(k),D];
    for temp=1:n
        if Status(temp,1)==0
            M_2(temp,:)=zeros(1,k+q);
        else
            M_2(temp,:)=beta';
        end
    end
    V_part3=M_2*G';
  
    delta_V=V_part1+V_part2-kappa3*V_part3;
    V_new=V-delta_V;
    
    %% Step6--beta
    M=V*G;
    M_3=(mapminmax(M',-1,1))';
    try
        beta_new = solve_beta(Status,M_3,kappa3,beta);
    catch
        disp('error')
        break;
    end
    
    %% Step7--lambda and muk
    loss_train_part=zeros(1,q);
    loss_val_part=zeros(1,q);
    for temp=1:q
        loss_train_part(1,temp)=epsilon3*norm(H_tr(:,temp).*(V_tr*D(:,temp)-Y_tr(:,temp)),'fro')^2;
        loss_val_part(1,temp)=epsilon3*norm(H_val(:,temp).*(V_val*D(:,temp)-Y_val(:,temp)),'fro')^2;
    end
    
    E_train_all(iter_time+1,:)=loss_train_part;
    E_val_all(iter_time+1,:)=loss_val_part;
    if (iter_time-tau)>=0
        for temp=1:q
            E_part1=(E_val_all(iter_time-t+1,temp)-E_val_all(iter_time+1,temp))/(E_val_all(iter_time-t+1,temp));
            E_part2=(E_train_all(iter_time-t+1,temp)-E_train_all(iter_time+1,temp))/(E_train_all(iter_time+1,temp));
            muk(1,temp)=rho*E_part1*E_part2;% update muk
        end
        part=zeros(1,q);
        for temp=1:q
            part(1,temp)=epsilon3*lambda(1,temp)*norm(H(:,temp).*(V*D(:,temp)-Y(:,temp)),'fro')^2;
            lambda(1,temp)=min(1,max(phi,muk(1,temp)-part(1,temp)));% update lambda; phi is a constant lowerbound close to zero
        end
    end
    
    P=P_new; Q=Q_new; B=B_new; D=D_new; V=V_new; beta=beta_new;
end













