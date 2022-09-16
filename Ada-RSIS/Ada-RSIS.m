
%% Input
% X1             m_x1*n      Original feature matrix derived from genomic data 
% X2             m_x2*n      Original feature matrix derived from histopathological data

% Parameter Settings
r=4;                           % The number of the risk subgroups, determined by the number of cancer stages (i.e., r = 4)
m=15;                        % The subspace dimension with the range of [15,25]           
epsilon=1E-05;        % The regularization coefficient with the range of [1e-7,1e-5]
iter_times=50;            % The iterative times

%% Procedure
[m_x1,n]=size(X1);
[m_x2,~]=size(X2);

% Random initialization 
rng('default');
U=rand(m,n);
Y1=rand(m,n);
Y2=rand(m,n);
Q1=rand(m_x1,m);
Q2=rand(m_x2,m);
P1=rand(m_x1,m);
P2=rand(m_x2,m);
W1=rand(m,m_x1);
W2=rand(m,m_x2);

%% Iterative update
for iter=1:iter_times
    % Calculate the adaptive weights theta_ij (i=1,...,6; j=1,2) by the estimation of variance
    [theta_11,theta_21,theta_31,theta_41,theta_51,theta_61]= calculate_theta(W1,W2,X1,X2,Q1,Q2,P1,U,Y1,Y2,r);
    [theta_12,theta_22,theta_32,theta_42,theta_52,theta_62]= calculate_theta(W2,W1,X2,X1,Q2,Q1,P2,U,Y2,Y1,r);

    % Step1 -- Optimization of U
    delta_U_1=2*theta_11*(U+Y1-W1*X1)+2*theta_12*(U+Y2-W2*X2);
    
    delta_U_2=2*theta_21*(-Q1')*(X1-Q1*U)+2*theta_22*(-Q2')*(X2-Q2*U);
    
    SUMU3={};
    [~,n_y1]=size(Y1);
    for i=1:r-1
    subgroup=floor(n_y1/r);    
    U_j=U(:,subgroup*(i-1)+1:subgroup*i);
    Y1_j=Y1(:,subgroup*(i-1)+1:subgroup*i);
    Y2_j=Y2(:,subgroup*(i-1)+1:subgroup*i);
    sumu3=2*theta_51*(U_j*(eye(subgroup)-Y1_j'*Y1_j)*(eye(subgroup)-Y1_j'*Y1_j)-2*U_j*Y1_j'*(Y1_j-Y1_j*(U_j'*U_j)))+...
        2*theta_52*(U_j*(eye(subgroup)-Y2_j'*Y2_j)*(eye(subgroup)-Y2_j'*Y2_j)-2*U_j*Y2_j'*(Y2_j-Y2_j*(U_j'*U_j)));
    SUMU3(i)={sumu3};
    end
    n_y11=n_y1-subgroup*(r-1);
    U_j=U(:,subgroup*(r-1)+1:end);
    Y1_j=Y1(:,subgroup*(r-1)+1:end);
    Y2_j=Y2(:,subgroup*(r-1)+1:end);
    sumu3=2*theta_51*(U_j*(eye(n_y11)-Y1_j'*Y1_j)*(eye(n_y11)-Y1_j'*Y1_j)-2*U_j*Y1_j'*(Y1_j-Y1_j*(U_j'*U_j)))+...
        2*theta_52*(U_j*(eye(n_y11)-Y2_j'*Y2_j)*(eye(n_y11)-Y2_j'*Y2_j)-2*U_j*Y2_j'*(Y2_j-Y2_j*(U_j'*U_j)));
    SUMU3(r)={sumu3};
    delta_U_3=cell2mat(SUMU3);
    
    delta_U=delta_U_1+delta_U_2+delta_U_3;
    
    % Step2-1 -- Optimization of Y1
    delta_Y1_1=2*theta_11*(U+Y1-W1*X1);
    
    delta_Y1_2=2*theta_31*(-P1')*(X1-P1*Y1);
    
    SUMY3={};
    [~,n_y1]=size(Y1);
    for i=1:r-1
    subgroup=floor(n_y1/r);    
    U_j=U(:,subgroup*(i-1)+1:subgroup*i);
    Y1_j=Y1(:,subgroup*(i-1)+1:subgroup*i);
    sumu3=2*theta_51*(-2*Y1_j*U_j'*(U_j-U_j*(Y1_j'*Y1_j))+Y1_j*(eye(subgroup)-U_j'*U_j)*(eye(subgroup)-U_j'*U_j));
    SUMY3(i)={sumu3};
    end
    n_y11=n_y1-subgroup*(r-1);
    U_j=U(:,subgroup*(r-1)+1:end);
    Y1_j=Y1(:,subgroup*(r-1)+1:end);
    sumu3=2*theta_51*(-2*Y1_j*U_j'*(U_j-U_j*(Y1_j'*Y1_j))+Y1_j*(eye(n_y11)-U_j'*U_j)*(eye(n_y11)-U_j'*U_j));
    SUMY3(r)={sumu3};
    delta_Y1_3=cell2mat(SUMY3);
    
    delta_Y1=delta_Y1_1+delta_Y1_2+delta_Y1_3;
    
    % Step2-2 -- Optimization of Y2
    delta_Y2_1=2*theta_12*(U+Y2-W2*X2);
    
    delta_Y2_2=2*theta_32*(-P2')*(X2-P2*Y2);
    
    SUMY3={};
    [~,n_y1]=size(Y1);
    for i=1:r-1
    subgroup=floor(n_y1/r);    
    U_j=U(:,subgroup*(i-1)+1:subgroup*i);
    Y2_j=Y2(:,subgroup*(i-1)+1:subgroup*i);
    sumu3=2*theta_52*(-2*Y2_j*U_j'*(U_j-U_j*(Y2_j'*Y2_j))+Y2_j*(eye(subgroup)-U_j'*U_j)*(eye(subgroup)-U_j'*U_j));
    SUMY3(i)={sumu3};
    end
    n_y11=n_y1-subgroup*(r-1);
    U_j=U(:,subgroup*(r-1)+1:end);
    Y2_j=Y2(:,subgroup*(r-1)+1:end);
    sumu3=2*theta_52*(-2*Y2_j*U_j'*(U_j-U_j*(Y2_j'*Y2_j))+Y2_j*(eye(n_y11)-U_j'*U_j)*(eye(n_y11)-U_j'*U_j));
    SUMY3(r)={sumu3};
    delta_Y2_3=cell2mat(SUMY3);
    
    delta_Y2=delta_Y2_1+delta_Y2_2+delta_Y2_3;
    
    % Step3-1 -- Optimization of Q1
    Q1_new=theta_21*X1*U'/(theta_21*(U*U')+epsilon*eye(m));
    
    % Step3-2 -- Optimization of Q2
    Q2_new=theta_22*X2*U'/(theta_22*(U*U')+epsilon*eye(m));
    
    % Step4-1 -- Optimization of P1
    P1_new=theta_31*X1*Y1'/(theta_31*(Y1*Y1')+theta_41*(Y2*Y2')+epsilon*eye(m));
    
    % Step4-2 -- Optimization of P2
    P2_new=theta_32*X2*Y2'/(theta_32*(Y2*Y2')+theta_42*(Y1*Y1')+epsilon*eye(m));
    
    % Step5-1 -- Optimization of W1    
    W1_new=theta_11*(U*X1'+Y1*X1')/(theta_11*(X1*X1')+epsilon*eye(m_x1));
   
    % Step5-2 -- Optimization of W2    
    W2_new=theta_12*(U*X2'+Y2*X2')/(theta_12*(X2*X2')+epsilon*eye(m_x2));
     
    % Update variables
    if norm(P1_new-P1,'inf')<1E-07 && norm(P2_new-P2,'inf')<1E-07 && norm(Q1_new-Q1,'inf')<1E-07 && norm(Q2_new-Q2,'inf')<1E-07
      break;
    end
    U=U-delta_U;
    U=mapminmax(U,0,1);      
    Y1=Y1-delta_Y1;
    Y1=mapminmax(Y1,0,1);
    Y2=Y2-delta_Y2;
    Y2=mapminmax(Y2,0,1);   
    Q1=Q1_new;
    Q1=mapminmax(Q1',0,1)';   
    Q2=Q2_new;
    Q2=mapminmax(Q2',0,1)';  
    P1=P1_new;
    P1=mapminmax(P1',0,1)';  
    P2=P2_new;
    P2=mapminmax(P2',0,1)';  
    W1=W1_new;
    W1=mapminmax(W1,0,1);  
    W2=W2_new;
    W2=mapminmax(W2,0,1);  

end

%% Function of calculating the variance sigma_C and the adaptive weights theta_ij (i=1,...,6; j=1,2)
function [theta_1i,theta_2i,theta_3i,theta_4i,theta_5i,theta_6i]= calculate_theta(W1,W2,X1,X2,Q1,Q2,Pi,U,Yi,Y_ba,r)

[~,n_yi]=size(Yi);

% Calculate sigma_Wi
sigma_Wi=norm(W1*X1-U-Yi,'fro');

% Calculate sigma_Qi
sigma_Qi=norm(X1-Q1*U,'fro');

% Calculate sigma_Pi
sigma_Pi=sqrt(norm(X1-Pi*Yi,'fro')^2+norm(Pi*Y_ba,'fro')^2);

% Calculate sigma_Ui
sigma_Ui_part31=0;
sigma_Ui_part32=0;
for i=1:r-1
subgroup=floor(n_yi/r);    
U_j=U(:,subgroup*(i-1)+1:subgroup*i);
Y_j=Yi(:,subgroup*(i-1)+1:subgroup*i);
sigma_Ui_part3_01=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Ui_part31=sigma_Ui_part31+sigma_Ui_part3_01;
end
U_j=U(:,subgroup*(r-1)+1:end);
Y_j=Yi(:,subgroup*(r-1)+1:end);
sigma_Ui_part3_01=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Ui_part31=sigma_Ui_part31+sigma_Ui_part3_01;

for i=1:r-1
U_j=U(:,subgroup*(i-1)+1:subgroup*i);
Y_j=Y_ba(:,subgroup*(i-1)+1:subgroup*i);
sigma_Ui_part3_02=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Ui_part32=sigma_Ui_part32+sigma_Ui_part3_02;
end
U_j=U(:,subgroup*(r-1)+1:end);
Y_j=Y_ba(:,subgroup*(r-1)+1:end);
sigma_Ui_part3_02=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Ui_part32=sigma_Ui_part32+sigma_Ui_part3_02;

sigma_Ui=sqrt(norm(W1*X1-U-Yi,'fro')^2+norm(W2*X2-U-Y_ba,'fro')^2+norm(X1-Q1*U,'fro')^2+norm(X2-Q2*U,'fro')^2+...
   +sigma_Ui_part31+sigma_Ui_part32);

% Calculate sigma_Yi
sigma_Yi_part3=0;
for i=1:r-1
U_j=U(:,subgroup*(i-1)+1:subgroup*i);
Y_j=Yi(:,subgroup*(i-1)+1:subgroup*i);
sigma_Yi_part3_0=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Yi_part3=sigma_Yi_part3+sigma_Yi_part3_0;
end
U_j=U(:,subgroup*(r-1)+1:end);
Y_j=Yi(:,subgroup*(r-1)+1:end);
sigma_Yi_part3_0=norm(U_j-U_j*(Y_j'*Y_j),'fro')^2+norm(Y_j-Y_j*(U_j'*U_j),'fro')^2;
sigma_Yi_part3=sigma_Yi_part3+sigma_Yi_part3_0;
sigma_Yi=sqrt(norm(W1*X1-U-Yi,'fro')^2+norm(X1-Pi*Yi,'fro')^2+sigma_Yi_part3);

% Calculate the adaptive weights theta_ij (i=1,2; j=1,...,6)
theta_1i=1/(2*sigma_Wi^2)+1/(2*sigma_Ui^2)+1/(2*sigma_Yi^2);
theta_2i=1/(2*sigma_Qi^2)+1/(2*sigma_Ui^2);
theta_3i=1/(2*sigma_Pi^2)+1/(2*sigma_Yi^2);
theta_4i=1/(2*sigma_Pi^2);
theta_5i=1/(2*sigma_Ui^2)+1/(2*sigma_Yi^2);
theta_6i=log(sigma_Wi)+log(sigma_Qi)+log(sigma_Pi)+0.5*log(sigma_Ui)+log(sigma_Yi);
end

