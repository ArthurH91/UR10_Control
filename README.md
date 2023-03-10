# UR10 Inverse Kinematics Solver 

This repository contains 6 Python files that compute the inverse kinematics of a UR10 robot to reach a target. The main function is the **Ball_reaching_with_gradient_descent.py** file which combines all the modules to solve a quadratic problem.

![image](https://user-images.githubusercontent.com/106057062/224975355-e9bff0a2-8c18-47fd-8722-87852da0360a.png)



## File description
The **create_visualizer.py** file is a module to create a meshcat.Visualizer of a pinocchio model.

The **RobotWrapper.py** file is a module to unwrap an URDF and add a end_effector frame and if the user wants, a sphere that can be used as a target along its target frame.

The **Solver.py** file is a module solving a unconstrained optimization problem by gradient method or newton method. A BFGS method is planned to be developped but not done. 

The **OptimizationProblem.py** is a module computing the cost function of a certain quradratic problem, the gradient and the hessian matrix. 

The **Ball_reaching_with_gradient_descent.py** is the "main" function of this repo as it is combining all the modules to solve the following quadratic problem : 

- An UR10 has a random initial configuration, with a sphere as a target. It must reach the target. Hence, the cost function is the squared distance between the target and the end effector.

## Maths behind this problem

This quadratic problem can be formulated like 

$$ \min_{q \in \mathbb{R}^n} \frac{1}{2} | f(q) - y |_2^2$$

where $f(q)$ is the forward kinematics function and $y$ is the desired target.


## To test the solver

In a terminal window : 

```
meshcat-server 
```

In another terminal window : 

``` 
python test_functions.py 
``` 

This python file shows that the solver works with a rosenbrock function and quadratic functions. 

## Solving this QP

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
