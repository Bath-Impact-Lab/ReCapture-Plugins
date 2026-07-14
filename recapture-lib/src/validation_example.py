import jsonschema # this package is on pip: https://pypi.org/project/jsonschema/
import sys
import json

# you can specify the paths however you choose; in this script i've just put them on the command-line arguments.
# you would call it like: python validation_example.py C:\Users\Max\Code\Recapture-IRIS\recapture-lib\new_graph_schema.json C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example_line_graph.json C:\Users\Max\Code\Recapture-IRIS\recapture-lib\example_output.json
schema_path = "/path/to/schema/new_graph_schema.json" # this is the path to the schema file

data_path = "/path/to/data/example_line_graph.json" # this is the path to the data, after having been exported to a JSON file.
                                                    # as will be explained below, you can also validate a python object directly.

output_path = "/path/to/output/output.json" # this is where the validated data will be outputted to; dumping the object to a JSON file is the easiest way to transfer it to the frontend for now.


schema_path = sys.argv[1]
data_path = sys.argv[2]
output_path = sys.argv[3]


# this section loads the JSON schema into python. This is required.
print(f"Loading JSON schema from {schema_path}")
try:
    with open(schema_path, "r") as file:
        schema = json.load(file)
except Exception as e:
    print(f"Failed to load schema: {e}")
    exit()

# this section loads the data from a JSON file into a python dictionary. This step is not required; see below.
print(f"Loading JSON data from {data_path}")
try:
    with open(data_path, "r") as file:
        data = json.load(file)
except Exception as e:
    print(f"Failed to load schema: {e}")
    exit()

# if using within an existing script, and you have the data in a python dict (or other datastructure) already, you can simply validate your data by calling jsonschema.validate with the schema as done below.
# Note that when validating directly like this you may encounter the error that numpy's ndarray type cannot be serialized.
# I have included a function at the bottom of this script to recursively go through a large datastructure and turn any ndarrays into python lists.

try:
    jsonschema.validate(instance=data, schema=schema) # replace 'data' with your data if validating directly.
    print("Validation successful, dumping data to file for use in the frontend.")
    with open(output_path, "w") as file:
        json.dump(data, file, indent=2)
except jsonschema.exceptions.ValidationError as e:
    # see https://python-jsonschema.readthedocs.io/en/latest/errors/ for everything this error can tell you. I have printed the basic information below.
    print(f"Error validating json schema: {e.message}")
    print(f"Location of error in data: {e.json_path} (note that '$' means the root object.)")
    exit()


# recursively goes through a large datastructure and converts any numpy arrays to python lists.
def recursive_to_list(obj):
    if isinstance(obj, np.ndarray):
        return [recursive_to_list(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {k:recursive_to_list(v) for k, v in obj.items()}

    try:
        # this will throw an error on non-iterable objects like numbers, in which case the object is returned as-is. also checks for strings first so as not to split them into an array of characters, since strings are iterable.
        return obj if isinstance(obj, str) else [recursive_to_list(item) for item in obj]
    except:
        return obj

