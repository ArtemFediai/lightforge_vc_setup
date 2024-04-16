"""
Starting from the origin (center) of the morphology, makes [0 - xy, 0 - xy, z_c - z] to [0 + xy, 0 + xy, z_c + z]
morphology as specified with the command line arguments xy and z.

Long description:
The Python script filters molecules from a CML file based on spatial criteria in the x,
y, and z directions. It accepts four command-line arguments: the path to the input CML file, the path for the output
file, the maximum vertical distance (z_distance) from the center, and the maximum horizontal distance (xy_distance)
from the origin. The script calculates the center of the morphology in the z-direction and sets bounds based on the
provided distances. Molecules whose COGs fall outside these bounds are excluded from the output CML
file. The script provides feedback on the number of molecules excluded and ensures proper usage with argument checks
"""

import sys
from QuantumPatch.Shredder.Parsers.CMLParser import CMLParser, CMLWriter
import numpy as np


def main():
    if len(sys.argv) < 5:
        print("Usage: python script.py inputfile outputfile z_distance xy_distance")
        sys.exit(1)

    filename = sys.argv[1]
    outfilename = sys.argv[2]
    z_distance = float(sys.argv[3])
    xy_distance = float(sys.argv[4])

    cp = CMLParser()
    system = cp.parse_to_system(filename)
    skip_ids = []

    # Set the center of the morphology in the z-direction
    top_z = np.max(system.cogs, axis=0)[2]
    bottom_z = np.min(system.cogs, axis=0)[2]
    center_z = (top_z + bottom_z) / 2

    # Calculate the range for inclusion based on specified distances
    z_upper_bound = center_z + z_distance
    z_lower_bound = center_z - z_distance
    x_upper_bound = xy_distance  # Since center x and y are always 0
    x_lower_bound = -xy_distance
    y_upper_bound = xy_distance
    y_lower_bound = -xy_distance

    num_mol_before = system.number_of_mols()

    # Checking if new bounds exceed the original bounds of the molecules
    if z_upper_bound > top_z or z_lower_bound < bottom_z:
        print("Warning: Specified z-distance exceeds the original bounds of the molecule data. Nothing will be cut "
              "from z-direction (idle cutoff). The morphology in z direction may be smaller than you wanted it to be.")
    if x_upper_bound > np.max(system.cogs, axis=0)[0] or x_lower_bound < np.min(system.cogs, axis=0)[0] or \
       y_upper_bound > np.max(system.cogs, axis=0)[1] or y_lower_bound < np.min(system.cogs, axis=0)[1]:
        print("Warning: Specified xy-distance exceeds the original bounds of the molecule data. Nothing will be cut "
              "out of xy-direction (idle cutoff). The morphology in xy direction may be smaller than you wanted it to"
              " be.")
    
    # Determine molecules to skip
    for mol_id in range(system.number_of_mols()):
        mol_cog = system.cogs[mol_id]
        if not (z_lower_bound <= mol_cog[2] <= z_upper_bound and
                x_lower_bound <= mol_cog[0] <= x_upper_bound and
                y_lower_bound <= mol_cog[1] <= y_upper_bound):
            skip_ids.append(mol_id)

    print(f"Initial system contained {num_mol_before} moleculed")
    print(f"Cutting {len(skip_ids)} molecules based on specified distances")
    CMLWriter.write_cml(system, filename=outfilename, skip_molecules=skip_ids)
    num_mol_remains = num_mol_before - len(skip_ids)
    print(f"{num_mol_remains=}")


if __name__ == "__main__":
    main()
