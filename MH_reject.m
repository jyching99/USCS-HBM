function log_pdf = MH_reject(x,L,mu)

v = L\(x-mu);
log_pdf = -0.5*2*log(2*pi) - sum(log(diag(L))) - 0.5*v'*v;