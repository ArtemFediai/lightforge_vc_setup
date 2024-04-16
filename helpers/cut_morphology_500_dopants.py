"""
Optimal cubic shape structure from structurePBC.cml.
Optimal: contains ~ 500 dopants -> okay for Lightforge!
Estimation is made assuming one molecule has a volume of 1 nm**3.
"""

import sys
import numpy as np
from QuantumPatch.Shredder.Parsers.CMLParser import CMLParser, CMLWriter


def calculate_cube_dimensions(dopants_needed, ratio_dopant_host):
    volume_per_molecule = 1000  # Assuming each molecule occupies 1000 cubic units (10x10x10)
    total_molecules_needed = dopants_needed / ratio_dopant_host
    total_volume_needed = total_molecules_needed * volume_per_molecule
    side_length = total_volume_needed ** (1 / 3)  # Cube root to find the cube dimension
    return side_length


def count_dopants(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
        types = [int(line.split()[-1]) for line in data]  # Extract the last column (molecule type)
        minority_type = min(set(types), key=types.count)  # Determine which type is the minority (0 or 1)
        dopant_count = types.count(minority_type)
    return minority_type, dopant_count


def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py inputfile outputfile dopants ratio")
        sys.exit(1)

    filename = sys.argv[1]
    outfilename = sys.argv[2]
    dopants = int(sys.argv[3])
    ratio_dopant_host = float(sys.argv[4])

    cp = CMLParser()
    system = cp.parse_to_system(filename)
    skip_ids = []

    cube_side_length = calculate_cube_dimensions(dopants, ratio_dopant_host)
    print(f"Calculated cube side length to contain approx. {dopants} dopants: {cube_side_length:.2f} A")

    center_z = (np.max(system.cogs, axis=0)[2] + np.min(system.cogs, axis=0)[2]) / 2
    half_side = cube_side_length / 2

    z_upper_bound = center_z + half_side
    z_lower_bound = center_z - half_side
    x_upper_bound = y_upper_bound = half_side
    x_lower_bound = y_lower_bound = -half_side

    for mol_id in range(system.number_of_mols()):
        mol_cog = system.cogs[mol_id]
        if not (z_lower_bound <= mol_cog[2] <= z_upper_bound and
                x_lower_bound <= mol_cog[0] <= x_upper_bound and
                y_lower_bound <= mol_cog[1] <= y_upper_bound):
            skip_ids.append(mol_id)

    CMLWriter.write_cml(system, filename=outfilename, skip_molecules=skip_ids)

    with open('COM.dat', 'r') as file:
        com_data = file.readlines()

    with open('COM_updated.dat', 'w') as file:
        for index, line in enumerate(com_data):
            if index not in skip_ids:
                file.write(line)

    # Counting dopants in the updated COM.dat file
    minority_type, actual_dopants = count_dopants('COM_updated.dat')
    print(f"Actual number of dopant molecules (type {minority_type}): {actual_dopants}")


if __name__ == "__main__":
    main()
