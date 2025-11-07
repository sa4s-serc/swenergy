import json

f = open('swe-bench-verified-mini.jsonl')
mini_data = [json.loads(line) for line in f]
f.close()
f= open('swe-bench-verified.jsonl')
verified_data = [json.loads(line) for line in f]
f.close()

print(f"mini: {len(mini_data)}")
print(f"verified: {len(verified_data)}")

# Create a dictionary mapping instance_id to line number in verified_data
verified_index = {item['instance_id']: idx for idx, item in enumerate(verified_data)}

# Find line numbers for mini_data in verified_data
with open('mini_verified_line_numbers.txt', 'w') as out_f:
    for item in mini_data:
        instance_id = item['instance_id']
        if instance_id in verified_index:
            print(f"{instance_id}: {verified_index[instance_id]}")
            out_f.write(str(verified_index[instance_id]) + '\n')
        else:
            print(f"{instance_id}: not found")