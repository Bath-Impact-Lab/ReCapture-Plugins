# Plugin Details
This document details how a plugin for recapture should work, what it needs to take as input, and what it needs to output. It assumes working knowledge of python and conda.

## Environment
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

## Input
The plugin will be given four things as input, _in this order_:
- the root directory of the trial
- the absolute path to the .osim file
- the absolute path to the .trc file
- the absolute path to the .mot file

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
For instance, `sys.argv[2]` would be `C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example-session\walk`.

## Output
The python script must create a file within the root directory of the trial called `spec.json`. It _must_ follow the schema given in [new_graph_schema.json](new_graph_schema.json). If it does not, the graphs will render improperly or not at all.

Instructions on how to validate your data structure and write to a json file can be found in [validation_example.py](src/validation_example.py).

## Testing
To test whether your plugin will run inside recapture, run the following command: 

`.\test_python_plugin.ps1 <plugin root directory>`

Replace '\<plugin root directory>' with the absolute path to the root directory of your plugin, the one containing `main.py`. The script will then run your plugin on some sample data.
It is a work in progress, but should error in helpful ways and tell you what is wrong. Please contact the developers if this is not the case.

At some point in the future, the recapture app will be updated to include the ability to test plugins on real data.