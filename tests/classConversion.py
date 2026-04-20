import os

label_folder = r"C:\Users\Test\Documents\ThesisVideos\VideoDatasetSegmented\PlasticDatasetAugmented\valid\labels"

for file in os.listdir(label_folder):

    if file.endswith(".txt"):
        path = os.path.join(label_folder, file)

        new_lines = []

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            parts[0] = "4"   # change class id

            new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("Class IDs converted.")