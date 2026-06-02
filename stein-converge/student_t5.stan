
data { real<lower=0> nu; }
parameters { real theta; }
model { theta ~ student_t(nu, 0, 1); }
