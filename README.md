# microRTS 4
My fourth attempt at building a microRTS AI. This one builds on RLlib.

## Requirements
- Python 3.6


## Installation
1. Optionally create a Python3 virtualenv called `venv` (separate project dependencies)

        virtualenv -p python3 venv

1. Activate the virtualenv (if you created one)

        source venv/bin/activate

1. Install dependencies

        pip install -r requirements.txt


## Usage
Activate the virtualenv (if used)

    source venv/bin/activate

Run using main.py and configure via optional command line arguments

    python main.py {options}

### Command Line Options

| Flag | Parameters | Description | Required | Default Value | 
| ---- | ---------- | ----------- | -------- | ------------- |
| policy | str | The agent type to use | N | PPO |
| map | str | The map file to use | N | 4x4_melee_light2 |
| use-cnn | bool | Whether to use a MLP (default) or CNN | N | MLP |
| debug | bool | Whether to use debugging mode | N | False |
| restore | str | Optionally provide the location of a checkpoint file to load | N | None (will not load) |
