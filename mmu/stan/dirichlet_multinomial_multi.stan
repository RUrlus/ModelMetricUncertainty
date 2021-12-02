data {
    int N; // num observations
    real<lower=0> v;
    real<lower=0> alpha;
    int<lower=2> total_count;
    int y[N, 4]; // multinomial observations
}

parameters {
    vector[4] phi;
    simplex[4] theta;
}

model {
    phi ~ dirichlet(rep_vector(alpha, 4));
    theta ~ dirichlet(1 + phi * v);
    {
        for (n in 1:N) {
            y[n] ~ multinomial(theta);
        }
    }
}

generated quantities {
    vector[4] theta_hat = dirichlet_rng(phi * v);
    int y_hat[4] = multinomial_rng(theta, total_count);
}
