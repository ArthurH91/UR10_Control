# To test the solver

In a terminal window : 

```
meshcat-server 
```

In another terminal window : 

``` 
python test_functions.py 
``` 

This python file shows that the solver works with a rosenbrock function and quadratic functions. 

# Description of this repository

The **create_visualizer.py** file is a module to create a meshcat.Visualizer of a pinocchio model.

The **RobotWrapper.py** file is a module to unwrap an URDF and add a end_effector frame and if the user wants, a sphere that can be used as a target along its target frame.

The **Solver.py** file is a module solving a unconstrained optimization problem by gradient method or newton method. A BFGS method is planned to be developped but not done. 

The **OptimizationProblem.py** is a module computing the cost function of a certain quradratic problem, the gradient and the hessian matrix. 

The **Ball_reaching_with_gradient_descent.py** is the "main" function of this repo as it is combining all the modules to solve the following quadratic problem : 

- An UR10 has a random initial configuration, with a sphere as a target. It must reach the target. Hence, the cost function is the squared distance between the target and the end effector.


# To solve this QP


In a terminal window : 

```
meshcat-server 
```

In another terminal window : 

``` 
python Ball_reaching_with_gradient_descent.py 
``` 

## Note :
Pinocchio 3 was used while developping, it may be needed to compile the code. It remains to be seen whether the code can be compiled with pino2.
