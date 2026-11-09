# Two-intersection CityFlow corridor

Independent DQN agents control signalized intersections **J1** and **J2** in a
software-only CityFlow simulation. Each agent observes its four incoming
approach queues, the four queues shared by its arterial neighbor, its current
phase, and elapsed green time. Actions keep or switch between a north/south
phase and an east/west phase. A wrapper-controlled 3-second yellow clearance
separates every phase change.

Each incoming and outgoing road has two lanes. Each signal movement explicitly
connects both input lanes to both output lanes in CityFlow's road network.

The minimum green is 15 seconds and the maximum is 60 seconds. Action masks
prevent early switching and require a switch at the maximum. All DQN training
uses CUDA. `demo.py` and evaluation can perform inference on either CUDA or
CPU. CityFlow itself runs on CPU.

## Install

Use WSL2 Ubuntu with Python 3.10+ and an NVIDIA driver that supports CUDA.
CityFlow builds its Python extension from source. On a minimal Ubuntu image,
create a user-local environment with a C++ compiler, CMake, and Ninja:

```bash
mkdir -p "$HOME/.local/bin" "$HOME/mamba"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2
python3 -c 'import bz2,sys; sys.stdout.buffer.write(bz2.decompress(open(sys.argv[1],"rb").read()))' /tmp/micromamba.tar.bz2 | tar -x -C "$HOME/.local" bin/micromamba
export MAMBA_ROOT_PREFIX="$HOME/mamba"
"$HOME/.local/bin/micromamba" create -y -c conda-forge \
  -p "$HOME/.venvs/traffic-cityflow" \
  python=3.13 pip cmake ninja make cxx-compiler git
```

Install CUDA PyTorch and project dependencies in that environment:

```bash
"$HOME/.local/bin/micromamba" run -p "$HOME/.venvs/traffic-cityflow" \
  python -m pip install torch==2.11.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
"$HOME/.local/bin/micromamba" run -p "$HOME/.venvs/traffic-cityflow" \
  python -m pip install gymnasium matplotlib numpy
```

Build CityFlow from its upstream source. The upstream CMake files require a
small version update for modern CMake, and its old bundled pybind11 must be
updated for Python 3.13:

```bash
mkdir -p "$HOME/src"
git clone --depth 1 --recurse-submodules \
  https://github.com/cityflow-project/CityFlow.git "$HOME/src/CityFlow"
sed -i 's/cmake_minimum_required(VERSION 3.0)/cmake_minimum_required(VERSION 3.5)/' \
  "$HOME/src/CityFlow/CMakeLists.txt"
git -C "$HOME/src/CityFlow/extern/pybind11" fetch --depth 1 \
  https://github.com/pybind/pybind11.git v2.13.6
git -C "$HOME/src/CityFlow/extern/pybind11" checkout FETCH_HEAD
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/' \
  "$HOME/src/CityFlow/extern/pybind11/CMakeLists.txt" \
  "$HOME/src/CityFlow/extern/pybind11/tools/pybind11Tools.cmake"
"$HOME/.local/bin/micromamba" run -p "$HOME/.venvs/traffic-cityflow" \
  python -m pip install "$HOME/src/CityFlow"
```

Finally, install this project in editable mode and verify CUDA:

```bash
"$HOME/.local/bin/micromamba" run -p "$HOME/.venvs/traffic-cityflow" \
  python -m pip install --no-deps -e .
"$HOME/.venvs/traffic-cityflow/bin/python" -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

`run.sh` automatically uses `$HOME/.venvs/traffic-cityflow/bin/python` when
that environment exists. Set `PYTHON` to override it.

Run the unit tests after installation:

```bash
python -m unittest discover -s tests -v
```

## Run with the launcher

From WSL2 Ubuntu, in this directory, the launcher can check the environment,
generate the scenario, train both agents, export a demo replay, and run the
benchmark:

```bash
bash run.sh all
```

This full workflow requires CUDA and uses 100 training episodes by default.
For a quick smoke test, run `EPISODES=2 MAX_SECONDS=120 EVAL_EPISODES=1
bash run.sh all`. Individual commands are available too:

```bash
bash run.sh check
bash run.sh generate
bash run.sh test
bash run.sh train --episodes 100 --max-seconds 1800
bash run.sh demo --policy fixed
bash run.sh evaluate --episodes 5
```

Run `bash run.sh help` to see options. Set `PYTHON` to select a Python
executable or `EPISODES`, `MAX_SECONDS`, and `EVAL_EPISODES` to tune the full
workflow.

## Generate the scenario

```bash
python generate_scenario.py
```

The deterministic corridor road network and traffic flows are written to
`data/corridor_2x1/roadnet.json` and `flow.json`; the same folder also contains
`cityflow_config.json`. Each junction has north, south, east, and west
approaches. J1 and J2 share a two-way arterial link. Multiple through and
cross-street flows create demand in both arterial directions and at both
junctions.

## Train

```bash
python train.py --episodes 100
```

Training runs for 1,800 simulated seconds per episode by default and saves
both independent agents in `models/dqn_traffic_2x1.pt`, plus episode metrics
in `results/training_metrics.jsonl`. CUDA is mandatory for learning. Adjust
`--episodes`, `--seed`, `--max-seconds`, or `--output` as needed.

## Run a demo and export a replay

```bash
python demo.py --policy dqn
```

The DQN demo uses CUDA when available and otherwise CPU. Alternatives are
`--policy fixed` and `--policy max-pressure`. CityFlow's visualizer replay is
exported as `data/corridor_2x1/replay.txt` alongside its road network. Override
the destination with `--replay-output` if needed.

## Compare policies

```bash
python evaluate.py --episodes 5
```

This evaluates fixed-time, Max-Pressure, and (if its checkpoint exists) DQN
under matching scenario seeds, reports average travel time and queue lengths,
and writes `results/evaluation.json` and `results/evaluation.png`. Use
`--episodes` and `--max-seconds` to adjust the benchmark. A missing DQN
checkpoint is reported rather than silently omitted.

## Method and interpretation

- Independent agents have separate online/target networks and replay buffers;
  shared neighbor queues enable coordination without hard-coded progression
  offsets.
- The reward is a negative squared normalized maximum approach queue, with a
  small total-queue penalty and a positive reward for vehicles completing the
  corridor. The queue penalty discourages starvation while the peak penalty
  prioritizes severe spillback.
- Max-Pressure uses upstream-minus-downstream lane queue pressure by phase.
  Fixed-Time alternates phases on a 30-second green.
- A short demo is not evidence that DQN outperforms either baseline. Use
  multiple seeds, inspect training convergence, and calibrate demand and
  vehicle parameters before making research or deployment claims.
- This is a research baseline, not a safety-certified traffic controller.

## Published artifacts

The repository includes a snapshot of the trained model and generated
evaluation and simulation outputs under `artifacts/`, grouped by category.
These are reference outputs; the scripts above can regenerate them.
