from collections import defaultdict
from pathlib import Path
import os
import sys
import argparse
import glob


CLASS_NAMES = {
    0: "ManMade_objects",
    1: "Submerged_metal",
    2: "Submerged_natural_object",
    3: "Submerged_wood",
    4: "plastic_debris",
}

counts = {name: 0 for name in CLASS_NAMES.values()}


def count_instance_class(dataset_dir):
    for filename in glob.iglob(f'{dataset_dir}/*'):
        with open(filename) as f:
            for line in f:
                clean_line = line.lstrip()
                if clean_line:
                    first_char = clean_line[0]
                    label = CLASS_NAMES[int(first_char)]
                    counts[label] += 1
                    
        
def main():
    parser = argparse.ArgumentParser(
        description='Count the Instances of Classes across the Dataset'
    )
    parser.add_argument("--path", required=True,
                       help="Path of dataset to be counted")
    parser.add_argument("--output", action="store_true",
                        help="Output Path if dataset will be split")
    args = parser.parse_args()

    dataset_dir = Path(args.path).resolve()


    if not dataset_dir.exists():
        print("[ERROR] Dataset directory does not exist")
        sys.exit(1)
    
    count_instance_class(dataset_dir/"train"/"labels")
    count_instance_class(dataset_dir/"valid"/"labels")
    count_instance_class(dataset_dir/"test"/"labels")

    for key, value in counts.items():
        print(f"{key}:{value}")
   

if __name__ == "__main__":
    main()