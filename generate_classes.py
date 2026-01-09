import os
import json

# Change this if your dataset folder path is different
data_dir = "Dataset"  

# Collect folder names (each folder = one class)
class_names = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])

# Save them to classes.json
with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("✅ classes.json created with", len(class_names), "classes.")
print(class_names)
=======
import os
import json

# Change this if your dataset folder path is different
data_dir = "Dataset"  

# Collect folder names (each folder = one class)
class_names = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])

# Save them to classes.json
with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("✅ classes.json created with", len(class_names), "classes.")
print(class_names)
