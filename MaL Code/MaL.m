%% Input
% T             n*1          The observed time (n is the number of training samples)
% Status     n*1           The censoring status (1 for the uncensored sample while 0 for the censored sample, respectively))
% X             n*m_x      Original mono-modality feature matrix derived from genomic data or histopathological data
% Z             n*m_z      Original multi-modality feature matrix.

% Parameter Settings
eta=1e-5;                   % The range is [1e-5,1e-4,1e-3]
alpha=1e-5;                % The range is [1e-5,1e-4,1e-3]
epsilon=1e-2;              % The range is [0.01,0.1,1]
rho=1;                         % The range is [0.01,0.1,1]
gamma=1e-3;              % The range is [1e-5,1e-4,1e-3]
theta=1e-4;                  % The range is [1e-5,1e-4,1e-3]
beta=1e-3;                   % The range is [1e-5,1e-4,1e-3]
kappa=1e-5;                 % The range is [1e-5,1e-4,1e-3]

% Parameters of Augumented Lagarange Multiplier
sigma_1=1e-4;              % Corresponding to U=Y*Psi
sigma_2=1e-6;              % Corresponding to Y=J

m_y=20;                        % The dimension of mono-modality latent representations with the range of [20,25,30,35]
m_u=20;                        % The dimension of multi-modality latent representations with the range of [5,10,15,20]
C=4;                              % The sample collection is partitioned into C groups according to samples' survival time in ascending order
iter_times=50;                % The iterative times
  

%% Procedure
% Calculate the similarity matrix S_Z and Laplacian matrix L
sigma=1;
[m_z,n_z]=size(Z);
[m_x,n_x]=size(X);

for j=1:m_x
    for k=1:m_x
        S_Z(j,k)=exp((-1)*((norm((Z(j,:)-Z(k,:)),2))^2)/(2*sigma^2));
    end
end

M=diag(sum(S_Z,2));
L=M-S_Z;

% Calculate the feature matrices of the reference points in the original feature space: R_X, R_Z
R_X=calR(X,C);
R_Z=calR(Z,C);

% Calculate the weight coefficient matrices determined by the Gaussian distance: H_X, H_Z
H_X=calH(sigma,X,R_X);
H_Z=calH(sigma,Z,R_Z);

% Random initialization
rng('default');
W=rand(n_x,m_y);
Y=rand(m_x,m_y);
P=rand(n_z,m_u);
U=rand(m_z,m_u);
Psi=rand(m_y,m_u);
J=rand(m_x,m_y);
K=rand(m_y,1);
E=rand(m_z,1);
V_1=rand(m_z,m_u);
V_2=rand(m_z,m_y);

%% Iterative update
for iter=1:iter_times    
    % Calculate the feature matrices of the reference points in the latent representations: R_Y,R_U 
    R_Y=calR(Y,C); 
    R_U=calR(U,C);

    % Calculate the Euclidean distance matrices between samples and reference points within latent feature subspace: D_Y,D_U
    D_Y=calD(Y,R_Y);
    D_U=calD(U,R_U);
    
    % Step1 -- Optimization of Y
    [m_D_Y,n_D_Y]=size(D_Y);
    [m_Y,n_Y]=size(Y);
    O=get_O(Status);
    
    delta_Y_part1=alpha*(Y-X*W);
    delta_Y_part2=2*rho*L*Y;
    
    SUM1=0;
    for m=1:n_D_Y
    sum1=((D_Y(:,m).*H_X(:,m)-D_U(:,m).*H_Z(:,m))*H_X(:,m)').*eye(m_D_Y)*(Y-R_Y(m,:).*ones(m_Y,n_Y));
    SUM1=SUM1+sum1;
    end
    delta_Y_part3=4*gamma*SUM1;
    
    delta_Y_part4=(-1)*theta*U*R_U'*R_Y;
    delta_Y_part5=beta*(T-(Status-1).*E-Y*K)*(-K');
    delta_Y_part6=kappa*O'*K';
    delta_Y_part7=(-1)*V_1*Psi'+sigma_1*(Y*Psi-U)*Psi';
    delta_Y_part8=V_2+sigma_2*(Y-J);
    delta_Y=delta_Y_part1+delta_Y_part2+delta_Y_part3+delta_Y_part4+delta_Y_part5 +delta_Y_part6+delta_Y_part7+delta_Y_part8;
    Y_new=Y-delta_Y;
    
    % Step2 -- Optimization of U
    [m_D_U,n_D_U]=size(D_U);
    [m_U,n_U]=size(U);
    
    delta_U_part1=eta*(U-Z*P);
    delta_U_part2=(-1)*theta*(Y*R_Y'*R_U);    
    delta_U_part3=V_1+sigma_1*(U-Y*Psi);
    
    SUM3_1=0;
    for m=1:n_D_U
        sum3_1=((D_U(:,m).*H_Z(:,m)-D_Y(:,m).*H_X(:,m))*H_Z(:,m)').*eye(m_D_U)*(U-R_U(m,:).*ones(m_U,n_U));
        SUM3_1=SUM3_1+sum3_1;
    end

    delta_U_part4=4*gamma*SUM3_1;
    delta_U=delta_U_part1+delta_U_part2+delta_U_part3+delta_U_part4;
    U_new=U-delta_U;
    
    % Step3 -- Optimization of W
    W_new=(X'*X)\(X'*Y);
    
    % Step4 -- Optimization of P    
    P_new=(Z'*Z)\(Z'*U);
    
    % Step5 -- Optimization of Psi        
    Psi_new=(Y'*Y)\((Y'*V_1)/sigma_1+Y'*U);
    
    % Step6 -- Optimization of K     
    K_new=(Y'*Y)\Y'*(T-(Status-1).*E-kappa/beta*O');    

    % Step7 -- Optimization of J
    temp=Y+V_2/sigma_2;
    r=rank(temp);
    [U_,S_,V_]=svd(temp);
    U_=U_(:,1:r);
    V_=V_(:,1:r);
    S_=S_(1:r,1:r);
    S_=S_-epsilon/sigma_2;
    S_(S_<0)=0;
    J_new=U_*S_*V_';
    
    % Step8 -- Optimization of E
    [m_T,~]=size(T);
    for m=1:m_T
        if Status(m,:)==0
            E_new(m,:)=kappa/beta*O(:,m)-T(m,:)+Y(m,:)*K;
            if E_new(m,:)<=0
                E_new(m,:)=0;
            end
        else
            E_new(m,:)=0;
        end
    end
      
    % Update variables
    Y=Y_new;
    Y=(mapminmax(Y',0,1))';
    U=U_new;
    U=(mapminmax(U',0,1))';
    W=W_new; 
    W=mapminmax(W,0,1);
    P=P_new;
    P=mapminmax(P,0,1);
    Psi=Psi_new;
    Psi=mapminmax(Psi,0,1);
    K=K_new;
    K=mapminmax(K',0,1)';
    J=J_new;
    J=(mapminmax(J',0,1))';
    E=E_new;
    E=(mapminmax(E',0,1))';
    
    % Step9 -- Optimization of V_1, V_2
    V1=sigma_1*(U-Y*Psi)+V_1;
    V2=sigma_2*(Y-J)+V_2;
    V1=(mapminmax(V1',0,1))';
    V2=(mapminmax(V2',0,1))';
    
    % Step10 -- Optimization of sigma_1, sigma_2
    miu=1.25;
    sigma_max=1E06;
    sigma_1=sigma_1*miu;
    sigma_2=sigma_2*miu;
    if sigma_1>=sigma_max
        sigma_1=sigma_max;
    end
    if sigma_2>=sigma_max
        sigma_2=sigma_max;
    end
end
   

%% Function of calculating the feature matrix R of the reference points 
function R=calR(X,N)
% X is the input feature matrix; N is the predefined number of reference points
[m1,~]=size(X);
num1=floor(m1/N);
    for i=1:(N-1)
        R(i,:)=mean(X(num1*(i-1)+1:num1*i,:));
    end
R(N,:)=mean(X(num1*(N-1)+1:end,:));
end


%% Function of calculating the Euclidean distance matrix D between samples and reference points
function D=calD(X,R)
% X is the input feature matrix; R is the reference point matrix
[m1,~]=size(X);
[m2,~]=size(R);
    for j=1:m1
        for k=1:m2
            D(j,k)=(norm((X(j,:)-R(k,:)),2))^2;
        end
    end
end


%% Function of calculating the weight coefficient matrix H determined by the Gaussian distance
function H=calH(sigma,X,R)
% sigma is the standard deviation of the Gaussian function; X is the input feature matrix; R is the reference point matrix
[m1,~]=size(X);
[m2,~]=size(R);
    for j=1:m1
        for k=1:m2
            H(j,k)=exp((-1)*(norm((X(j,:)-R(k,:)),2)^2)/(2*sigma^2));
        end
    end
end


%% Function of calculating the variable O
function c=get_O(status)
[m,~]=size(status);
c=zeros(1,m);
    if status(1,:)==0
        c(1,1)=0;
    else
        c(1,1)=m-1;
    end
    for i=2:m
        if status(i,:)==0
            b=0;
            a=sum(status(1:i,:)==1);
            c(1,i)=b-a;
        end
        if status(i,:)==1
            b=m-i;
            a=sum(status(1:(i-1),:)==1);
            c(1,i)=b-a;
        end
    end
end
