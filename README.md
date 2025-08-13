# Project Description

... going to add "abstract" etc once it is done

# Reproducing this repository (on HPC)

## Install pixi

Install pixi on login node with `curl -fsSL https://pixi.sh/install.sh | bash`

Make it executable and findable
``` export PATH="$HOME/.pixi/bin:$PATH"``` and  ```echo 'export
PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc```

Check that it is installed with `pixi --version`

## Pull this project and install its environment

Pull the project
```git clone https://github.com/anthano/psycourse.git```
```cd psycourse````

Then install the environment with 
```pixi install```

## Add the data to the right directory
Everything should be self-handled except for the data, that is confidential of course. 
If you obtain the data folder named "data", move it into 
```mv data src/psycourse/data```

# Executing the repository
Now, on a local computer, you should be able to just run 
```pixi run pytask```or parallelize it with ```pixi run pytask -n 3``` and you should get the whole analysis. 
Since some of the analysis steps are quite analysis-heavy, here is a slurm.job script to run it on a HPC. 

