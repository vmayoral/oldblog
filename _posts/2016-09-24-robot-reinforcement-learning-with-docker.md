---
layout:     post
title:      Robot Reinforcement Learning with Docker
date:       2016-09-24
summary:    Robot Reinfocement Learning is becoming more and more popular. This article covers how to use the gym-gazebo toolbox for reinforcement learning is Docker to easily set up a reinforcement learning infrastructure for robots.
categories: robots, simulation, ai, rl, reinforcement learning
mathjax: true
---

Robot Reinfocement Learning is becoming more and more popular however setting up the infrastructure to do reinforcement learning with popular tools like [Gazebo](http://gazebosim.org) and [ROS](http://ros.org) can take quite a bit of time, specially if you have to do it in tenths of servers to automate the learning process in a robot. 

 This article will walk you through the process of how to use the [gym-gazebo](https://github.com/erlerobot/gym-gazebo) toolbox for reinforcement learning via a Docker container that has everything cooked already.


 ## Getting gym-gazebo as a Docker container

 Assuming that you've got [Docker installed](https://docs.docker.com/engine/installation/) in your system, here's what you need to do:

 ```
 docker pull erlerobotics/gym-gazebo:latest # takes about
 docker run -it gym-gazebo

 ```

 That easy! The overall process of fetching the container takes about 15 minutes but afterwards you can deploy as many as you want with a single command line. Depending on your simulations (typically when using plugins like cameras), you might need to set a fake screen on your server/dev. machine. Here's how to do it:
 
```
 xvfb-run -s "-screen 0 1400x900x24" bash
 
 ```

 ## Getting a local front-end

 Sometimes, you may want to supervise the learning process using `gzclient`. Assuming that the docker container is running locally, here's what you'd do:

 ```

 export GAZEBO_MASTER_IP=$(sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' "id of running container")
export GAZEBO_MASTER_URI=$GAZEBO_MASTER_IP:11345
gzclient
 
 ```

