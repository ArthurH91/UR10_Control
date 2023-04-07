# UR10 Inverse Kinematics Solver 

This repository contains 9 Python files that compute the inverse kinematics of a UR10 robot to reach a target. The main function is the **ComparisonMarcToussaintNewtonLinesearch.py** file which combines all the modules to solve a quadratic problem.

![image](https://user-images.githubusercontent.com/106057062/224975355-e9bff0a2-8c18-47fd-8722-87852da0360a.png)



## File description

The **create_visualizer.py** file is a module to create a meshcat.Visualizer of a pinocchio model.

The **RobotWrapper.py** file is a module to unwrap an URDF and add a end_effector frame and if the user wants, a sphere that can be used as a target along its target frame.

The **QuadraticProblemInverseKinematics.py** is a module computing the cost function of a certain quradratic problem, the gradient and the hessian matrix. 

The **Solver.py** file is a module solving an unconstrained optimization problem by gradient method or newton method.

The **test_functions.py** file is a module testing the functions of the module **Solver.py**

The **SolvingQPWithSolverGradientNewtonMethod.py** file is a module using the solver **Solver.py** to solve the QP, both with a gradient method and a newton method and compares both of them.

The **NewtonMethodMarcToussaint.py** file is a module solving an unconstrained optimization problem by using a trust region method on a Newton algorithm.

The **SolvingQPWithSolverMarcToussaint.py** file is a module using the solver **NewtonMethodMarcToussaint** to solve the QP.

The **ComparisonMarcToussaintNewtonLinesearch.py** is the "main" module of this repo as it is combining all the modules to solve the following quadratic problem : 

- An UR10 has a random initial configuration, with a sphere as a target. It must reach the target. Hence, the cost function is the squared distance between the target and the end effector.

This module compares the methods from Newton and Marc Toussaint. 


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
ComparisonMarcToussaintNewtonLinesearch.py
``` 

## Note :
Pinocchio 3 was used while developping, it is needed to compile the code. It does not work with pinocchio 2.
