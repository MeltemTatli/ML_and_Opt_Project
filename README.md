# CS9692 Machine Learning and Optimization

This repository was created for CS9692 Machine Learning and Optimization Project. 

The focus of our project was adding an adaptive gradient descent method (ADAM) to F2SA algorithm to increase the convergence speed while solving bilevel optimization problems.

Our implementation of the F2SA, SABA, and StocBiO algorithm is based on the code authored by (Dagr ́eou et al., 2022), as made available in their code repository https://github.com/benchopt/benchmark_bilevel. We added a new FashionMNIST dataset and the proposed optimizer – F2SA with Adam to it. We also corrected the definition of the accuracy in their repository, as the original one accounted for error percentage instead.

Use git clone to clone the repository.
```
git clone
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



References 
----------
```
@inproceedings{saba,
      title = {A Framework for Bilevel Optimization That Enables Stochastic and Global Variance Reduction Algorithms},
      booktitle = {Advances in {{Neural Information Processing Systems}} ({{NeurIPS}})},
      author = {Dagr{\'e}ou, Mathieu and Ablin, Pierre and Vaiter, Samuel and Moreau, Thomas},
      year = {2022}
   }
```
