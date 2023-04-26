import argparse
import json

if __name__ == '__main__':
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--input_file', help='Path to the input file')
    parser.add_argument('--output_file', help='Path to the output file')

    # Parse the arguments
    args = parser.parse_args()

    cuboids = []
    # Read the input file
    with open(args.input_file, 'r') as f:
        while True:
            next_line = next(f)
            if next_line is None:
                break
            assert next_line.strip() == 'position:'
            lines = [next(f).strip() for _ in range(8)]

            cuboid = {}
            assert lines[0].startswith('x: ')
            cuboid['x'] = float(lines[0][3:])
            assert lines[1].startswith('y: ')
            cuboid['y'] = float(lines[1][3:])
            assert lines[2].startswith('z: ')
            cuboid['z'] = float(lines[2][3:])

            assert lines[3] == 'orientation:'
            assert lines[4].startswith('x: ')
            cuboid['qx'] = float(lines[4][3:])
            assert lines[5].startswith('y: ')
            cuboid['qy'] = float(lines[5][3:])
            assert lines[6].startswith('z: ')
            cuboid['qz'] = float(lines[6][3:])
            assert lines[7].startswith('w: ')
            cuboid['w'] = float(lines[7][3:])

            print(cuboid)
            cuboids.append(cuboid)

            try:
                next(f)
            except StopIteration:
                break

    # Convert the input text to JSON
    json_data = cuboids

    # Write the JSON data to the output file
    with open(args.output_file, 'w') as f:
        json.dump(json_data, f)

    print('Conversion complete.')