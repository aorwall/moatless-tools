import json

from benchmark.cli import diff_file_names

with open('princeton-nlp-SWE-bench-test.json') as f:
    data = json.load(f)

with open('devin_swe_outputs.json') as f:
    devin_data = json.load(f)

instance_by_id = {}
for instance in data:
    instance_by_id[instance['instance_id']] = instance

for devin_instance in devin_data:
    instance = instance_by_id.get(devin_instance['instance_id'])
    if not instance:
        print(f"Instance {devin_instance['instance_id']} not found")
        continue

    devin_instance["model_patch_files"] = diff_file_names(devin_instance['model_patch'])

    devin_instance.update(instance)

with open('princeton-nlp-SWE-bench-devin.json', 'w') as f:
    json.dump(devin_data, f, indent=2)
