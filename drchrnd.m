function P = drchrnd(alpha, n)
    % drchrnd: 生成 Dirichlet 分佈樣本
    % alpha: Dirichlet 分佈的參數 (1xk)
    % n: 生成樣本的數量
    % r: 生成的樣本矩陣 (nxk)
    
    p = length(alpha); % Dirichlet 分佈的參數數量
    P = gamrnd(repmat(alpha, n, 1), 1, n, p); % 使用 Gamma 分佈生成樣本
    P = P ./ sum(P, 2); % 將 Gamma 分佈樣本歸一化為 Dirichlet 分佈
end