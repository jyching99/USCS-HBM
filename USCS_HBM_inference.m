function [mu_tg,C_tg,P_tg] = USCS_HBM_inference(target_data,target_L,mu0,C0,Ksi0,nu0,T,burn_in,dT)
% This is the function code for the inference stage of the USCS_HBM
% Input: 
%       target_data(n x 2): target-site data of CPT parameters (ln(Qtn), ln(Fr)), where n
%       denotes the total number of target-site data.
%       target_L(n x 5): target-site categorical record ;
%       burn_in(integer): burn-in period for Gibbs sampler;
%       T(integer): total interval for Gibbs sampler;
%       dT(integer): sampling interval for Gibbs sampler;
% Output:
%       mu_tg(2 x 5 x TT): posterior mean samples for the quasi-site-specific model, where TT = (T-burn_in)/dT;
%       C_tg(2 x 2 x 5 x TT): posterior covariance samples for the quasi-site-specific model;
%       P_tg(TT x 5): posterior probability distributions for the quasi-site-specific model.
%% simulate data
m = size(target_data,2); n_data = size(target_data,1); old_data = target_data; old_L = target_L;
for k=1:5
    inv_C0(:,:,k) = inv(C0(:,:,k));
end
%% GS initialization
known_data_type = nansum(target_L);
p_pi = [1,1,1,1,1];
for k=1:5
    mu_tg(:,k) = zeros(m,1); C_tg(:,:,k,1) = eye(m); inv_C_tg(:,:,k,1) = inv(C_tg(:,:,k,1));
end
P_tg = drchrnd(p_pi,1);
for j=1:n_data
    f_ind = find(isnan(old_data(j,:))); ff_ind = find(~isnan(old_data(j,:)));
    if isnan(old_L(j,:))
       target_L(j,:) = mnrnd(1, P_tg); % multinomial distribution
    end
    fff_ind = find(target_L(j,:)>0);
    if length(f_ind) > 0
        mu_x = mu_tg(f_ind,fff_ind) + C_tg(f_ind,ff_ind,fff_ind)*inv(C_tg(ff_ind,ff_ind,fff_ind))*(target_data(j,ff_ind)'-mu_tg(ff_ind,fff_ind));
        var_x = C_tg(f_ind,f_ind,fff_ind) - C_tg(f_ind,ff_ind,fff_ind)*inv(C_tg(ff_ind,ff_ind,fff_ind))*C_tg(f_ind,ff_ind,fff_ind)';
        target_data(j,f_ind) = (mu_x + sqrtm(var_x)*randn(length(f_ind),1));
    end
end
data_tg(:,:,1) = target_data;
data_L(:,:,1) = target_L;
%% GS
Ic_order = [5, 4, 3, 2, 1];
MH_sample = 1500;
power = 4;
for t=2:T
    if t>2
        MH_sample = 150;
    end
    n_data_type = zeros(1,5);
    label_data_type = cell(1,5);
    for k=1:5
        idx = 1;
        for j=1:n_data
            if target_L(j,k) == 1
                n_data_type(k) = n_data_type(k)+1;
                label_data_type{k}(idx, :) = target_data(j, :);
                idx = idx + 1;
            end
        end
        if n_data_type(k) ~= 0
            C_mu_u(:,:,k) = inv(inv_C0(:,:,k) + n_data_type(k)*inv_C_tg(:,:,k));
            mu_mu_u(:,k) = C_mu_u(:,:,k)*(inv_C0(:,:,k)*mu0(:,k)+ n_data_type(k)*inv_C_tg(:,:,k)*mean(label_data_type{k},1)');
        else
            C_mu_u(:,:,k) = C0(:,:,k); mu_mu_u(:,k) = mu0(:,k);
        end
        L_mu_u(:,:,k) = sqrtm(C_mu_u(:,:,k));
    end
    % Metropolis-Hastings
    accept_count = 0;
    X = mu0;
    for k=1:5
        log_f(k) = MH_reject(X(:,k),L_mu_u(:,:,k),mu_mu_u(:,k));
    end
    for z = 1:MH_sample
        if z > 100
            power = 1;
        end
        % Sample
        for k=1:5
            % New Sample
            X_c(:,k) = X(:,k) + power*L_mu_u(:,:,k)*randn(2,1);
            log_fc(k) = MH_reject(X_c(:,k), L_mu_u(:,:,k), mu_mu_u(:,k));
            acc = rand < min(1, exp(log_fc(k)-log_f(k)));
            X_cc(:,k) = acc*X_c(:,k) + (1-acc)*X(:,k);
            log_fcc(k) = acc*log_fc(k) + (1-acc)*log_f(k); % sample candidate
            Qtn(k) = exp(X_cc(1,k));
            Fr(k) = exp(X_cc(2,k));
            Ic(k) = sqrt( ( 3.47 - log10(Qtn(k)) ).^2 + ( log10(Fr(k)) + 1.22 ).^2 );
        end
        if all(diff(Ic(Ic_order)) <= 0)
            X = X_cc; log_f = log_fcc;
        end
        if all(diff(Ic(Ic_order)) <= 0) && acc == 1
            accept_count = accept_count + 1;
        end
    end
    mu_tg(:,:,t) = X;
    for k=1:5
        if n_data_type(k) ~= 0
            % sample C
            nu0_u =  n_data_type(k)+nu0(k);
            Ksi0_u = Ksi0(:,:,k) + (label_data_type{k}'-mu_tg(:,k,t)*ones(1, n_data_type(k)))*(label_data_type{k}'-mu_tg(:,k,t)*ones(1, n_data_type(k)))';
            C_tg(:,:,k,t) = iwishrnd(real(Ksi0_u),nu0_u);
            inv_C_tg(:,:,k) = inv(C_tg(:,:,k,t));
        else
            % sample C
            nu0_u = nu0(k); Ksi0_u = Ksi0(:,:,k);
            C_tg(:,:,k,t) = iwishrnd(real(Ksi0_u),nu0_u);
            inv_C_tg(:,:,k) = inv(C_tg(:,:,k,t));
        end
    end
    % update L
    for j=1:n_data
        if isnan(old_L(j,:))
            B = zeros(1,5);
            for k=1:5
                B(k) = (P_tg(k)*mvnpdf(target_data(j,:),mu_tg(:,k,t)',C_tg(:,:,k,t)));
            end
            B = B/sum(B);
            target_L(j,:) = mnrnd(1, B);
        end
    end
    % sample P
    P_tg(t,:) = drchrnd(p_pi + known_data_type, 1);
    % update hidden data
    for j=1:n_data
        f_ind = find(isnan(old_data(j,:))); ff_ind = find(~isnan(old_data(j,:)));
        fff_ind = find(target_L(j,:)>0);
        if length(f_ind) > 0
            mu_x = mu_tg(f_ind,fff_ind,t) + C_tg(f_ind,ff_ind,fff_ind,t)*inv(C_tg(ff_ind,ff_ind,fff_ind,t))*(target_data(j,ff_ind)'-mu_tg(ff_ind,fff_ind,t));
            var_x = C_tg(f_ind,f_ind,fff_ind,t) - C_tg(f_ind,ff_ind,fff_ind,t)*inv(C_tg(ff_ind,ff_ind,fff_ind,t))*C_tg(f_ind,ff_ind,fff_ind,t)';
            target_data(j,f_ind) = (mu_x + sqrtm(var_x)*randn(length(f_ind),1))';
        end
    end
    data_tg(:,:,t) = target_data;
    data_L(:,:,t) = target_L;
end
mu_tg = mu_tg(:,:,burn_in+1:dT:T); C_tg = C_tg(:,:,:,burn_in+1:dT:T); P_tg = P_tg(burn_in+1:dT:T,:);