## Use intructions

1. Build image with ns3-gym env:

```
docker build -t ns3-gym-env .
```

2. Run the container and enter Bash:

```console
docker run -it --rm -v ./examples:/ns-allinone-3.40/ns-3.40/contrib/opengym/examples ns3-gym-env bash
```

3. Activate python venv (probably we can automate this step in the 2nd command):

```console
cd .. && source ../ns3gym-venv/bin/activate
```

4. Run the thing, e.g.

```console
cd opengym && ./test.py
```

5. Stop and exit the container: Ctrl + D

## Sources:

- ns3-gym: https://github.com/tkn-tub/ns3-gym
