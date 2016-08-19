---
layout:     post
title:      Extending the OpenAI Gym for robotics
date:       2016-08-19 
summary:    Benchmarking in robotics remains an unsolved issue, this article proposes an extension of the OpenAI Gym for robotics using the Robot Operating System (ROS) and the Gazebo simulator to address the benchmarking problem.
categories: robots, simulation, ai, rl, reinforcement learning
mathjax: true
---

<p style="border: 2px solid #000000; padding: 10px; background-color: #E5E5E5; color: black; font-weight: light;">
Content based on Erle Robotics's whitepaper: [Extending the OpenAI Gym for robotics: a toolkit for reinforcement learning using ROS and Gazebo](http://erlerobotics.com/whitepaper/robot_gym.pdf).
</p>

The [OpenAI Gym](http://gym.openai.com) is a is a toolkit for reinforcement learning research that has recently gained popularity in the machine learning community. The work presented here follows the same baseline structure displayed by researchers in the OpenAI Gym, and builds a gazebo environment on top of that. OpenAI Gym focuses on the episodic setting of RL, aiming to maximize the expectation of total reward each episode and to get an acceptable level of performance as fast as possible. This toolkit aims to integrate the Gym API with robotic hardware, validating reinforcement learning algorithms in real environments. Real-world operation is achieved combining [Gazebo simulator](http://gazebosim.org), a 3D modeling and rendering tool, with the [Robot Operating System](http://ros.org), a set of libraries and tools that help software developers create robot applications.

As [discussed previously](http://blog.deeprobotics.es/robots,/ai,/deep/learning,/rl,/reinforcement/learning/2016/07/06/rl-intro/), the main problem with RL in robotics is the high cost per trial, which is not only the economical cost but also the long time needed to perform learning operations. Another known issue is that learning with a real robot in a real environment can be dangerous, specially with flying robots like quad-copters. In order to overcome this difficulties, advanced robotics simulators like Gazebo have been developed which help saving costs, reducing time and speeding up the simulation.

<div id='architecture'/>
## Architecture

![](gym_architecture.png)

The architecture consits of three main software blocks: OpenAI Gym, ROS and Gazebo. Environments developed in OpenAI Gym interact with the Robot Operating System, which is the connection between the Gym itself and Gazebo simulator. Gazebo provides a robust physics engine, high-quality graphics, and convenient programmatic and graphical interfaces.

