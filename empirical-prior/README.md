# Fast MCMC with unknown high-dimensional prior

This is a brute-force implementation of Stan of the empirical prior defined in the
following paper.

* Chenyang Zhong, Shouxuan Ji, and  Tian Zheng. 2024. [Graph-Enabled Fast MCMC Sampling with an Unknown High-Dimensional Prior Distribution](https://arxiv.org/abs/2408.02122). arXiv 2408.02122.

## File structure

There are three files:

* `sim.py`:  Python simulation and fit code for Ohio and NY
* `flat-logistic.stan`:  Stan logistic regression model with flat priors
* `empirical-logisic.stan`:  Stan logistic regression model with empirical priors

# Running the simulation

```
$ cd case-studies/empirical-prior
$ python3 sim.py
```

which should print something that looks like this:

```
empirical-prior (master)$ python3 sim.py
beta=array([ 0.32292202, -1.67712417,  0.80797451,  0.23766868,  0.86741335,
       -1.506818  ])
np.mean(y_NY)=np.float64(0.368)
17:08:34 - cmdstanpy - INFO - compiling stan file /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/flat-logistic.stan to exe file /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/flat-logistic
17:08:42 - cmdstanpy - INFO - compiled model executable: /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/flat-logistic
17:08:42 - cmdstanpy - INFO - CmdStan start processing
chain 1 |██████████████████████████████████████████████████████████████████████████████████████████| 00:02 Sampling completed
chain 2 |██████████████████████████████████████████████████████████████████████████████████████████| 00:02 Sampling completed
chain 3 |██████████████████████████████████████████████████████████████████████████████████████████| 00:02 Sampling completed
chain 4 |██████████████████████████████████████████████████████████████████████████████████████████| 00:02 Sampling completed
                                    
17:08:44 - cmdstanpy - INFO - CmdStan done processing.
               Mean      MCSE    StdDev       MAD          5%         50%         95%  ESS_bulk  ESS_tail    R_hat
lp__    -557.382000  0.025521  1.737420  1.560440 -560.661000 -557.052000 -555.203000   4741.12   6418.16  1.00046
beta[1]    0.366390  0.000772  0.072327  0.072928    0.248282    0.366051    0.486274   8806.16   7336.19  1.00016
beta[2]   -1.790090  0.001243  0.105562  0.105524   -1.965540   -1.788830   -1.619120   7238.00   7597.53  1.00023
beta[3]    0.919031  0.000900  0.079263  0.078493    0.789709    0.917736    1.050000   7791.26   7036.17  1.00041
beta[4]    0.245745  0.000778  0.073334  0.073553    0.126556    0.245181    0.366897   8934.58   6160.00  1.00052
beta[5]    0.826686  0.000882  0.078421  0.077470    0.699096    0.826170    0.957079   7927.63   7081.54  1.00079
beta[6]   -1.578750  0.001177  0.097490  0.097021   -1.739350   -1.578330   -1.416750   6878.93   7327.66  1.00054
17:08:45 - cmdstanpy - INFO - compiling stan file /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/empirical-logistic.stan to exe file /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/empirical-logistic
17:08:54 - cmdstanpy - INFO - compiled model executable: /Users/bcarpenter/github/bob-carpenter/case-studies/empirical-prior/empirical-logistic
17:08:54 - cmdstanpy - INFO - CmdStan start processing
chain 1 |██████████████████████████████████████████████████████████████████████████████████████████| 00:34 Sampling completed
chain 2 |██████████████████████████████████████████████████████████████████████████████████████████| 00:34 Sampling completed
chain 3 |██████████████████████████████████████████████████████████████████████████████████████████| 00:34 Sampling completed
chain 4 |██████████████████████████████████████████████████████████████████████████████████████████| 00:34 Sampling completed
                                  
17:09:29 - cmdstanpy - INFO - CmdStan done processing.
               Mean      MCSE    StdDev       MAD          5%         50%         95%  ESS_bulk  ESS_tail     R_hat
lp__    -559.122000  0.053302  1.702570  1.478890 -562.379000 -558.779000 -557.001000   1033.39   1246.12  1.003030
beta[1]    0.350714  0.001253  0.051488  0.051961    0.267381    0.350895    0.437376   1712.84   1476.49  1.001330
beta[2]   -1.690070  0.002133  0.071637  0.071335   -1.806060   -1.689900   -1.574170   1139.18   1286.46  1.001020
beta[3]    0.821465  0.001461  0.055462  0.054288    0.732546    0.821911    0.912160   1473.30   1639.19  1.000200
beta[4]    0.309187  0.001171  0.052614  0.052405    0.224269    0.308911    0.395340   2035.31   1384.39  1.003350
beta[5]    0.811029  0.001536  0.056688  0.057730    0.719682    0.809859    0.905025   1383.98   1141.46  0.999504
beta[6]   -1.550230  0.001952  0.068913  0.068681   -1.670110   -1.549630   -1.440560   1254.39   1543.84  1.002220
empirical-prior (master)$ 
```

Running it again should produce the same result because I have fixed a seed chosen at random in the file for reproduciblity.  These results are from my 2017 iMac with Intel Xeons.  A newer machine would be faster, especially an ARM-based Mac.
