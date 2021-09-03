%% Input
% Clinical   m*2             The first column is the observed time, the second column is Status.
%            m is the number of training samples.
% X_H,X_G    m*n1,m*n2       Original features matrices drived from histopathological images and gene data
% X          m*(n1+n2)       Original multi-modality features matrix

switch mode
    case 1 % histopathological data
        X_train = X_H;
    case 2 % gene data
        X_train = X_G;
    case 3 % multi-modality data
        X_train = X;
end

% Parameter Settings
epsilon=1e-3;  % The range is [1e-6,5e-6,...,1e-2,5e-2]
theta=1e-4;    % The range is [1e-6,5e-6,...,1e-2,5e-2]
kappa=1e-4;    % The range is [1e-6,5e-6,...,5e-2,1e-3]
nu=1e-4;       % The range is [1e-6,5e-6,...,5e-3,1e-2]
phi=80;        % The range is [10,20,...,640,1280]
gamma=1e-4;    % The range is [5e-7,1e-6,...,1e-3,5e-3]
yita=1e-4;     % The range is [5e-7,1e-6,...,1e-3,5e-3]
k=8;           % The feature dimension for the latent feature matrix
               % The range is [6,8,...,22,24]
ALPHA=3;       
THRESHOLD=3;   

%% Procedure
P = randn(k,n); % reconstruction matrix
V = randn(m,k); % latent feature matrix
Q = randn(n,k); % projection matrix
beta = randn(k,1); % the coefficient vector for Cox roportional hazard model 

%% Iterative update
for iter_time = 1:Iter_Time
%% Step1--V
% part1
delta_V_1 = -2*X*P'+2*V*(P)*P';

% part2       
alpha = ALPHA;
part = fix(m/alpha);
part4_a = zeros(m,k);
delta_V_2 = zeros(m,k);
threshold = THRESHOLD;% Threshold
V_1 = V(1:part,:); 
V_2 = V(part+1:2*part,:);
V_3 = V(2*part+1:end,:);
            
V_1_var = std(V_1,0,1).^2;
V_2_var = std(V_2,0,1).^2;
V_3_var = std(V_3,0,1).^2;% Variance of each Segment
            
V_var = V_1_var+V_2_var+V_3_var;
            
for i=1:size(V,2)
    if V_var(i)<threshold
        delta_V_2(:,i) = zeros(m,1);  
        part4_a(:,i) = zeros(m,1);  
    else
        V_1(:,i) = 2*(V_1(:,i)-mean(V_1(:,i)))*(1-1/part);
        V_2(:,i) = 2*(V_2(:,i)-mean(V_2(:,i)))*(1-1/part);
        V_3(:,i) = 2*(V_3(:,i)-mean(V_3(:,i)))*(1-1/part); 
        part4_1 = (V_1(:,i)-mean(V_1(:,i))).^2;
        part4_2 = (V_2(:,i)-mean(V_2(:,i))).^2;
        part4_3 = (V_3(:,i)-mean(V_3(:,i))).^2;   
        delta_V_2(:,i) = [V_1(:,i);V_2(:,i);V_3(:,i)];
        part4_a(:,i) = [part4_1;part4_2;part4_3];
        part4_a(:,i) = part4_a(:,i) - threshold*(1/m);  
    end
end
        part4_b = sum(part4_a(:));
            
%part3
delta_V_3 = zeros(m,k);
for i = 1:m
    if Clinical(i,2) == 0
        delta_V_3(i,:) = zeros(1,k);
    else
        delta_V_3(i,:) = beta';
    end
end

%part4
delta_V_4 = 2*(V-X*Q);
            
delta_V = epsilon*delta_V_1 + theta*delta_V_2 - kappa*delta_V_3 + nu*delta_V_4;
Z = V - delta_V;
V_new = sign(Z).*max(abs(Z)-gamma,0);   %soft threshold function

%% Step2--P
S = V'*V;
s = V'*X;
S1 = S-diag(diag(S));
u = s-S1*P;
u_norm = sum(abs(u).^2,2).^(1/2);
Srr_inv = 1./(diag(S)+1e-16);
frame = 1-yita*1./(u_norm+1e-16);
first_twoterm = repmat(Srr_inv.*max(frame,0),1,n); 
P_new = first_twoterm .* u;

%% Step3--Q
Q_new = (X'*X)\(X'*V);

%% Step4--beta
try
    beta_new = solve_beta_1(Clinical,V,beta,kappa,phi);
catch
    disp('error')
    break;
end
beta_new = beta_new / std(V * beta_new); 
V=V_new; P=P_new; Q=Q_new; beta = beta_new; 
end
