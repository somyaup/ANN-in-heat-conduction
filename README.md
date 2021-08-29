# ANN-in-heat-conduction
ANN used to plot 2-D steady state conduction for fixed boundry conditions in a cylinder
Artificial neural network application in solving heat conduction equation


Somya Upadhyay*a, P. Srinivsanb^b
*a Birla Institute of Technology & Science (BITS), Pilani, Rajasthan, India, f20170962@pilani.bits-pilani.ac.in
^b Birla Institute of Technology & Science (BITS),, Pilani, Rajasthan, India, psrinivasan@pilani.bits-pilani.ac.in
*Corresponding Author
 
This paper presents the Artificial Neural Network technique's viability in solving the Heat conduction equation with fixed boundary conditions that depend upon the environment. This study aims to find an accurate method to simulate heat maps other than numerical technique like FEM with reusable learning. The objective is to make a qualitative and quantitative comparison of the same thermal system's Temperature function. Numerical approximation methods such as the Finite element method (FEM) are popular because of their time and memory-efficient results and easy applicability. However, the error in estimation due to discretization used in FEM is a trade-off. The technique to solve the equation using artificial neural networks (ANN) to back-calculate the temperature function can address this issue. We assume our loss function to be as shown in the equation where ψ (x, y, z, t) is the heat conduction equation. T i is the ith boundary condition that is specified for the system at position and time (x i ,y i ,z i ,t i) given n constraints due to the system..
Loss= mean{ψ (x, y ,z ,t) }2 + ∑{T i- T(x i ,y i ,z i ,t i)}2  ……..(i)                                          

The power of backpropagation in differentiation and a suitable network are used in this work to  achieve a plot that would theoretically have no approximation error if the loss is minimized to zero. Backpropagation uses a combination of symbolic differentiation and Jacobian matrix to find the value of the partial differential equation for heat conduction equation. For simplicity, we begin with modelling a 2-Dimensional problem in steady-state condition to visualize it on a three-dimensional plot for position and temperature profile.
With a longer initial training time which pays off with transferable weights to variable mesh size without losing information are obtained. Comparing to FEM results, ANN may acquire localized inconsistency due to the black box effect that cannot be explained mathematically. The temperature for boundary conditions assigned for a radial system as given in table 1.  8000 epoch of training over a 1:32:32:1 sized network result in a comparable grid that gives an mean deviation of ±10.81857462730 units per cell. 

|----------------------------------------------------------------------------------------------------------------------------|
|Technique   	        | ANN                                                 	|FEM                                           |
|----------------------------------------------------------------------------------------------------------------------------|
|Initial time	        | 60 to 120 minutes	                                    |2 to 3 minutes                                |
|----------------------------------------------------------------------------------------------------------------------------|
|Plot size constrains | 601 x 601 (or higher with nearly no computation cost) |101x101(System memory allocation limit in GPU)|
|----------------------------------------------------------------------------------------------------------------------------|
|Accuracy           	| Rregional error . blackbox error.                     |Approximation error.                          |
|----------------------------------------------------------------------------------------------------------------------------|
                            Table1: Result Analysis for problem statement
                            
ANN is an evolving technique finding more application as the processing power advances. Obtaining higher degree equations accuracy is possible with no architectural changes to the program or computational power. We only change the loss function for required ψ (x, y,z,t). At this stage, the FEM solution displays accurate results for considerable discretization in a lower degree of energy function (non-zero constant in the tested model). ANN result is independent of the mesh size. From the Mechanical standpoint, the results by ANN overestimate temperature values compared to the FEM solution. With hyperparameter adjustments and more epochs of training, the results resemble realistic conditions closely.
