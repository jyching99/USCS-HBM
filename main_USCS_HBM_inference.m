clc;clear;close all;
%%
m = 2;
%% load targer site data
xlsFile = 'Malamocco_nL_7.xlsx'; % load Malamocco test site data (nL = 7)
[matrix2, text2, rawData2] = xlsread(xlsFile);
label_site = text2(2:end,4);
for i = 1:length(label_site)
    if strcmp(label_site{i}, 'C')
        L_site(i,:) = [0 0 0 1 0];
    elseif strcmp(label_site{i},'M')
        L_site(i,:) = [0 0 1 0 0];
    elseif strcmp(label_site{i},'S')
        L_site(i,:) = [0 1 0 0 0];
    elseif strcmp(label_site{i},'G')
        L_site(i,:) = [1 0 0 0 0];
    elseif strcmp(label_site{i},'O')
        L_site(i,:) = [0 0 0 0 1];
    else
        L_site(i,:) = [NaN NaN NaN NaN NaN];
    end
end
Y_site = matrix2(:,[1,2]); Y_site = log(Y_site);
%% Inference Stage
load("hyper_para_samples.mat"); % load hyper-parameter samples
TT = 2000; T1 = 120; burn_in1 = 100; dT1 = 20; TT1 = (T1-burn_in1)/dT1;
nn = 1;
for t=1:TT,t
    [mu_tg,C_tg,P_tg] = USCS_HBM_inference(Y_site,L_site,mu0(:,:,t),C0(:,:,:,t),Ksi0(:,:,:,t),nu0(:,t),T1,burn_in1,dT1);
for p=1:TT1
    mu_tg_c(:,:,nn) = mu_tg(:,:,p);
    C_tg_c(:,:,:,nn) = C_tg(:,:,:,p);
    P_tg_c(nn,:) = P_tg(p,:);
    nn = nn + 1;
end
end
%% Predicted USCS categorical probability at CPTU19
xlsFile = 'CPTU19_data.xlsx'; % load CPTU19 Qtn and Fr data
[matrix2, text2, rawData2] = xlsread(xlsFile);
Depth = matrix2(:,1)'; CPT_tg = matrix2(:,[2,3]); CPT_tg = log(CPT_tg);
for n=1:length(CPT_tg) % 
    for k=1:5 %
        for i=1:size(mu_tg_c,3)
            log_like(i) = log(P_tg_c(i,k)) + (-log(2*pi) - 0.5*log(det(C_tg_c(:,:,k,i))) - 0.5*((CPT_tg(n,:)'-mu_tg_c(:,k,i))' * inv(C_tg_c(:,:,k,i)) * (CPT_tg(n,:)'-mu_tg_c(:,k,i))));
        end
        max_log_like = max(log_like);
        ave_log_like(n,k) = log(mean(exp(log_like-max_log_like)))+max_log_like;
    end
    for k=1:5
        max_log_like = max(ave_log_like(n,:));
        P_pred(n,k) = exp(ave_log_like(n,k)-max_log_like)/sum(exp(ave_log_like(n,:)-max_log_like)); % P_pred contains the predicted USCS categorical probabilities
    end
end
%% plotting
color = {[0.2,0.2,0.2],[0.24,0.35,0.67],[0.24,0.57,0.25],[0.96,0.64,0.38],[0.7 0.13 0.13]};
figure('Position', [0.5, 0.5, 560, 1200]); 
P_c = cumsum(P_pred,2)'.*100;
hold on;
set(gca, 'YDir', 'reverse');
fill([P_c(1,:) fliplr(zeros(1,size(P_c,2)))], [Depth fliplr(Depth)],color{1}, 'EdgeColor', 'none');
for k=2:5
    fill([P_c(k,:) fliplr(P_c(k-1,:))], [Depth fliplr(Depth)],color{k}, 'EdgeColor', 'none');
end
ylim([10,56]);xlim([0 100]);
xlabel('Probability (%)'); ylabel('Depth (m)')
hold off;