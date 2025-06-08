## Location of files

Code created for this project can be found in the following directories:

- [`examples/our_rl/dqn`](https://github.com/vasyagladush/ml-ns3/tree/main/examples/our_rl/dqn) - contains files for Deep Q-Learning (DQN)
- [`examples/our_rl/ql`](https://github.com/vasyagladush/ml-ns3/tree/main/examples/our_rl/ql) - contains files for Q-Learning (QL)

Other directories in the `examples/` directory contain examples provided by ns3-gym.

## Results

Graphs showing obtained results can be found in:
- [`examples/our_rl/baseline/results`](https://github.com/vasyagladush/ml-ns3/tree/main/examples/our_rl/baseline/results) - Baseline graphs with CW min and CW max set to 15 and 1023 respectively
- [`examples/our_rl/dqn`](https://github.com/vasyagladush/ml-ns3/tree/main/examples/our_rl/dqn) - DQN result graphs
- [`examples/our_rl/ql/results`](https://github.com/vasyagladush/ml-ns3/tree/main/examples/our_rl/ql/results) - Q-Learning result graphs
## Use intructions

1. Build image with ns3-gym env:

**There is no benefit to performing this step if the container has already been built and is not missing.**

```
docker build -t ns3-gym-env .
```

2. Run the container and enter Bash:

```console
docker run -it --rm -v ./examples:/ns-allinone-3.40/ns-3.40/contrib/opengym/examples ns3-gym-env bash
```

3. Activate python venv (probably we can automate this step in the 2nd command):

```console
source ../ns3gym-venv/bin/activate
```

4. Change directory to appropriate directory:

```console
cd ./our_rl/ql
```
OR

```console
cd ./our_rl/dqn
```

5. Ensure all required Python modules are installed:

```console
pip install -r requirements.txt
```

6. Run the thing, e.g.

_Both programs can be run as many times as desired without having to restart the container._

```console
./qlearn.py
```

OR

```console
./dqn.py
```

## Sources:

- ns3-gym: https://github.com/tkn-tub/ns3-gym
