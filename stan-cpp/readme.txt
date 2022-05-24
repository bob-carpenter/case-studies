Clone CmdStan repo and update submodules

> cd <dir_stan>
> git clone https://github.com/stan-dev/cmdstan.git
> cd cmdstan
> make stan-update


Clone case-studies repo: 

> cd <dir_studies>
> git clone https://github.com/bob-carpenter/case-studies.git 


Build example as CmdStan application

> cd <dir_stan>/cmdstan
> make -j4 <dir_studies>/case-studies/stan-cpp/eg

Run example

> cd <dir_studies>/case-studies/stan-cpp/eg
> ./eg sample data file=eg.data.R







