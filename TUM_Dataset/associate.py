import sys

def read_file_list(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    result = {}
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        timestamp = float(parts[0])
        data = parts[1:]  # Keep as strings (file paths)
        result[timestamp] = data
    return result

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    
    potential_matches = []
    for a in first_keys:
        for b in second_keys:
            diff = abs(a - (b + offset))
            if diff < max_difference:
                potential_matches.append((diff, a, b))
    
    potential_matches.sort()
    
    matches = []
    first_used = set()
    second_used = set()
    
    for diff, a, b in potential_matches:
        if a in first_used or b in second_used:
            continue
        first_used.add(a)
        second_used.add(b)
        matches.append((a, b))
    
    matches.sort()
    return matches

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python associate.py rgb.txt depth.txt')
        sys.exit(1)
    
    first_list = read_file_list(sys.argv[1])
    second_list = read_file_list(sys.argv[2])
    
    matches = associate(first_list, second_list)
    
    for a, b in matches:
        print(f'{a} {" ".join(first_list[a])} {b} {" ".join(second_list[b])}')
