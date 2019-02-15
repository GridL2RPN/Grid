
The goal of this challenge is to to solve the electric power grid control problem using a reinforcement learning based approach. Participants are asked to design a reinforcement learning agent using tools and algorithms of their choice. The designed agents are supposed to learn a policy that maximizes the final score returned by the simulator, the possible actions at a given timestep being either switching a line status or changing the line interconnections. Indeed, we provide a simulation environement consisting of a power grid simulator along with a set of chronics. We use pypownet, an open-source power grid simulator developed by Marvin Lerousseau, that simulates the behaviour of a power grid of any given characterics subject to a set of external constraints governed by the given chronics. Samples of such chronics can be found under the sample_data directory. 

We also provide a set of baselines solutions:
- A Do Nothing agent, which does not take actions at all. 
- A Partial Brute Force agent, which takes the best actions after trying to disconnect and connect one line after another.

## Installation:

### Using Docker 

Pull the docker image
```bash
docker pull mok0na/l2rpn:2.0
```

Download starting-kit
```
unzip starting-kit
cp starting-kit ~/aux
```

Mount shared volume on docker and run the jupyter notebook for the first time
```
docker run --name l2rpn -it -p 5000:8888 -v ~/aux:/home/aux mok0na/l2rpn:2.0 jupyter notebook --ip 0.0.0.0 --notebook-dir=/home/aux --allow-root
```

Open the link and replace the port 5000 instead of 8888.

e.g. : http://127.0.0.1:5000/?token=2b4e492be2f542b1ed5a645fa2cfbedcff9e67d50bb35380

To reuse the docker container
```
docker start l2rpn
```

The submission zip is in your local directory `~/aux/example_submission.zip`

### Without Docker

Requirements:

* Python >= 3.6
* Octave >= 4.0.6
* Matpower >= 6.0
 
[Full instructions here](https://github.com/MarvinLer/pypownet) 

## Usage
 
* The file README.ipynb contains step-by-step instructions on how to create a custom agent and use it on the simulator.  
* modify `example_submission` to provide a better model
* run the agent
```
python ingestion_program/ingestion.py input_data input_data/res ingestion_program example_submission
```
* zip the contents of `example_submission` (without the directory, but with metadata) 


References and credit: 
- The pypownet project was created Marvin Lerousseau. (https://github.com/MarvinLer/pypownet)
- The competition protocol was designed by Isabelle Guyon. 
- This work was mentored by Balthazar Donon and Antoine Marot.
- The baseline methods were inspired by work performed by Kimang Khun. 

