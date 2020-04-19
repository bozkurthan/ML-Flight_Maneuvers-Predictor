# ML-Flight_Maneuvers-Predictor (CMP-712)

This project is developing under the Hacettepe University CMP-712 Machine Learning Lecture.
This repository contains prediction of Basic Flight Maneuvers based on Machine Learning which runs on Gazebo-PX4-Plane.

## Flight Maneuver Prediction with Machine Learning Applications in the Gazebo Simulation

<p align="center">
  <b>Proposal</b><br>
</p>


### 1. Project Idea

The rapid growth of Aviation, the flight tests have become an indispensable part of production and
maintenance process to validate the aircraft flights. There are many techniques are being implement
partially using Machine Learning to overcome this problem. However, many of them still needs
mankind help. The project idea was born from this perspective. In this project, we shall try to use
fully Machine Learning technique to predict Fixed-Wing aircraft maneuvers which shall run on
Gazebo Simulation Environment. Gazebo supports many airframes those are runs on PX4. “PX4 is the
Professional Autopilot. Developed by world-class developers from industry and academia, and
supported by an active worldwide community, it powers all kinds of vehicles from racing and cargo
drones through to ground vehicles and submersibles.” is said by PX4 developer crew. We created
different scenario with some common maneuvers (i.e. Take-off, Climb, Loiter, Cruise, Descend,
Land) to obtain flight data. To perform this process, we set up fully autonomous flight missions. Per
flight took approximately 2-3 minutes. Furthermore, we added more challenge using physics motor of
Gazebo Simulation. Some of data are obtained under the wind condition. After this step we shall
make arrangements data to get rid of irrelevant part. The raw data does not only contain useful
information but it also covers useless hundreds line which can be cause inaccurate prediction.
After whole steps, we expect to predict all maneuvers at acceptable level while data comes from
Simulation in a real-time.

### 2. Relevant Papers
Jones B., & Jenkins K. “A Machine Learned Model of a Hybrid Aircraft,” CS229 Machine Learning, Fall 2016,
Stanford University
Johansen, T. A., Cristofaro, A., Sørensen, K., Hansen, J. M., & Fossen, T. I. (2015, June). On estimation of
wind velocity, angle-of-attack and sideslip angle of small UAVs using standard sensors. In 2015 International
Conference on Unmanned Aircraft Systems (ICUAS) (pp. 510-519). IEEE.

### 3. Approach to Problem & Computational Work

We decided to implement Support Vector Machines. After we obtain and preprocess the data for
machine learning model, firstly we will fit the size of all the flight data to size of the minimum flight
with down-sampling. Secondly we will normalize the data with PCA. After normalizing we will get
the matrix transposition of the data and vectorized the data with response matrix as each flight to be a
vector. Then we train the vectorized data with SWM, validate the model and finally test the model
with test data. If accuracy values lover than we expect we will play with the features as the result of
correlation matrix.

### 4. Dataset

Will be add soon
