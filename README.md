# CS9692 Machine Learning and Optimization Project: Bilevel Optimization with F$`^2`$SA and Adam

This repository was created for CS9692 Machine Learning and Optimization Project. 

This repository contains the implementation and experiments for a project focused on bilevel optimization using the Fully First-Order Bilevel Stochastic Approximation (F$`^2`$SA) algorithm with the Adam optimizer.

# Introduction
Bilevel optimization involves minimizing a problem that depends on the optimal solutions of another problem. In this project, we specifically address a bilevel optimization problem represented as follows:

```math
\begin{align}
\min_{x}~ F(x):= f(x,y^*(x)), ~~\text{ s.t. }~~ y^*(x)=\arg\min_y g(x,y)
\end{align}
```

Bilevel optimization have real-world application such as hyperparameter optimization, meta-learning, representation learning, reinforcement learning, continual learning, adversarial learning, and neural architecture search.
## Background
From the nested optimization perspective, the bilevel objective $`F(x)`$ is an implicit function of $`x`$. The project utilizes the implicit function theorem and advanced Hessian inversion estimation techniques to approximate the gradient of the bilevel objective, allowing for the use of gradient-based optimization algorithms.
In this project, we boost the fully first-order bilevel method via Adam, aiming at improving the convergence performance. We experimented with 2 main tasks, namely data hyper-cleaning and Regularization selection. The results demonstrate that F$`^2`$SA aided with Adam accelerates the convergence to a target accuracy and the performance of it is relatively stable to both theparameters and the settings.

### F$`^2`$SA Algorithm 
The project utilizes the Fully First-Order Bilevel Stochastic Approximation (F$`^2`$SA) algorithm proposed in ICML 2023 by Kwon et al. (2023a), which aims to solve bilevel optimization problems efficiently without requiring second-order information. The algorithm is designed to handle both upper-level and lower-level learning rates, presenting a challenge due to potential correlations between these rates.
### F$`^2`$SA with Adam
To enhance the F$`^2`$SA algorithm, the project proposes F$`^2`$SA with Adam, incorporating the Adam optimizer to improve convergence performance. Adam is an adaptive gradient method that approximates second-order information using first-order terms, making it suitable for bilevel optimization.

## Numerical Experiments
The project conducts experiments on two common bilevel tasks: data hyper-cleaning and regularization selection. It compares F$`^2`$SA with Adam against other bilevel methods, including F$`^2`$SA, StocBiO, and SABA, on datasets such as MNIST, FashionMNIST, and Ijcnn1.
## Usage
We utilized the repository bult by Dagr{\'e}ou et al. [2] in this project.

To use the provided code, follow the instructions below.

Use git clone to clone the repository.
```
git clone https://github.com/MeltemTatli/ML_and_Opt_Project.git
```

Install necessary libraries.
```
pip install -U benchopt

pip install jax

pip install libsvmdata
```

```
cd ML_and_Opt_Project
```

Run the following line to solve regularization selection task for ijcnn1 dataset.
```
benchopt run benchmark_bilevel-main -s f2sa_adam -s f2sa -s SABA -s StocBiO -d ijcnn1
```

Run the following line to solve data hyper-cleaning task for mnist dataset.
```
benchopt run benchmark_bilevel-main -s f2sa_adam -s f2sa -s SABA -s StocBiO -d mnist
```

Change the the string after -d to change the dataset. Add or substract a solver by using -s command.

The plots related to accuracy and convergence can be seen in the html file created under outputs folder.

## Results
The project presents results showcasing the performance of F$`^2`$SA with Adam in terms of convergence speed and stability compared to other bilevel optimization methods.
## Conclusion
In conclusion, this project introduces and explores the application of F$`^2`$SA with Adam in bilevel optimization. The proposed algorithm demonstrates promising results in terms of convergence performance and stability, addressing some of the limitations of existing bilevel optimization methods.


Our implementation of the F$`^2`$SA, SABA, and StocBiO algorithm is based on the code authored by (Dagr ́eou et al., 2022), as made available in their code repository https://github.com/benchopt/benchmark_bilevel. We added a new FashionMNIST dataset and the proposed optimizer – F$`^2`$SA with Adam to it. We also corrected the definition of the accuracy in their repository, as the original one accounted for error percentage instead.


References 
----------
```
Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In Proc. International Conference on Machine Learning, Honolulu, HI, 2023a.

Mathieu Dagre ́ou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. In Proc. Advances in Neural Information Processing Systems, New Orleans, LA, 2022.
```
