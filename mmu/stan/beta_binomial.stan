data {
    int<lower=1> n;
    int<lower=1> y[4];
    
    real<lower=0> tpr_prior[2];
    real<lower=0> tnr_prior[2];
    real<lower=0> phi_prior[2];
}

parameters {
    real<lower=0,upper=1> tpr;
    real<lower=0,upper=1> tnr;
    real<lower=0,upper=1> phi;
}

transformed parameters{
    real<lower=0,upper=1> theta_tp = phi * tpr;
    real<lower=0,upper=1> theta_tn = (1-phi) * tnr;
    real<lower=0,upper=1> theta_fn = phi * (1-tpr);
    real<lower=0,upper=1> theta_fp = (1-phi) * (1-tnr);
}

model {  
    phi ~ beta(phi_prior[1], phi_prior[2]);
    tpr ~ beta(tpr_prior[1], tpr_prior[2]);
    tnr ~ beta(tnr_prior[1], tnr_prior[2]);
    
    y[1] ~ binomial(n, theta_tn);
    y[2] ~ binomial(n, theta_tp);
    y[3] ~ binomial(n, theta_fn);
    y[4] ~ binomial(n, theta_fp);
}

generated quantities{
    int<lower=0> y_hat[4];
    y_hat[1] = binomial_rng(n, theta_tn);
    y_hat[2] = binomial_rng(n, theta_tp);
    y_hat[3] = binomial_rng(n, theta_fn);
    y_hat[4] = binomial_rng(n, theta_fp);
}
