import sys
import os

def process_file(input_file):
    # Read all lines from input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("No data found in file.")
        return

    # Extract the first timestamp to normalize time
    start_time = float(lines[0].split(',')[0])

    output_lines = []
    for idx, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) != 7:
            continue  # Skip malformed lines

        timestamp = float(parts[0]) - start_time
        z_abs = abs(float(parts[3]))
        roll = parts[4]
        pitch = parts[5]
        yaw = parts[6]

        image_name = f"{idx + 1:08d}.jpg"
        output_lines.append(f"{image_name},{timestamp:.3f},{z_abs},{roll},{pitch},{yaw}")

    # Save output file in the script's directory
    output_file = os.path.join(os.path.dirname(__file__), "converted_data.txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Processed data saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_data.py input_file.txt")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    process_file(input_path)
