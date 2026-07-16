## Creating a Plugin
Plugins must be placed under the 'plugins' subdirectory, then under a directory named for the author of the plugin, then under a directory named for a plugin.

For example, if someone named John Johnson were to create a plugin for analyzing squats, they would place it under
```text
plugins/
└── john-johnson/
    └── squats/
        ├── plugin goes here
        ├── main.py 
        └── et cetera.
```

### Plugin Structure

Each plugin follows a standard layout:

```text
plugin-name/
├── main.py        # Entry point
├── README.md      # Plugin-specific docs
└── Any other folders or files
```
The plugin _must_ be contained within a single folder and it _must_ have a file in it called `main.py` which is the entry point of the plugin.
Other python files can be used as part of the script, within those restrictions.

---

### Plugin Interface

#### Inputs
The plugin will be given four things as input, _in this order_:
1. the root directory of the trial
2. the absolute path to the .osim file 
3. the absolute path to the .trc file 
4. the absolute path to the .mot file

All these inputs are for one specific trial.

Assumptions about the structure of the trial directory and locations of files within it should be kept to an absolute minimum; prefer using the given paths.

These will be given in the manner of command-line arguments - for an example:
```powershell
python 
    C:\Users\Max\Code\Recapture-IRIS\recapture-lib\main.py 
    C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example-session\walk
    C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example-session\models\LaiUhlrich2022_scaled.osim 
    C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example-session\walk\P06_walkPref.trc 
    C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example-session\walk\P06_walkPref.mot 
```
(Note: I have put these arguments on one line for ease of reading. They would actually be called as a single line.)

To access these command line arguments, access `sys.argv` from the `sys` module inside the python script. This array will contain each of the command line arguments. 
Importantly, the first item in `sys.argv`, `sys.argv[0]`, is always the script being called. The rest of the arguments follow in the order they were given. 
For instance, `sys.argv[2]` would be `C:\Users\Max\Code\Recapture-Plugins\recapture-lib\example-session\walk`.

#### Outputs
- The python script must create a JSON file within the root directory of the trial.
- The name of the file must match `*_graphs.json`. That is, it must be `<some text>_graphs.json`; for example `gait_graphs.json` or `jump_graphs.json`. 
- It _must_ follow the JSON schema given in [new_graph_schema.json](testing/new_graph_schema.json). If it does not, the graphs will render improperly or not at all.

Instructions on how to validate your data structure and write to a json file can be found in [validation_example.py](testing/validation_example.py).

### Plugin environment
The plugin will be run in an environment with the following packages:
- python v3.11
- opensim
- pandas
- scipy
- matplotlib
- spyder-kernels
- requests
- pyyaml
- natsort
- statsmodels
- jsonschema

This environment can be replicated using conda or pixi, by copying [env.yaml](env.yaml) or [pixi.toml](pixi.toml) respectively. No guarantees can be made about the specific versions of packages other than python; contact the developers to resolve problems arising from package versions.

The plugin must have, in its root directory, a file called `main.py`. This will be used as the entry point of the plugin; if it does not have such a file, the plugin will not run. 
