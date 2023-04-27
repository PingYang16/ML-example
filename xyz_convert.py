import scipy.io

qm7 = scipy.io.loadmat('qm7.mat')
R = qm7['R']
Z = qm7['Z']
T = qm7['T'][0]

# dictionary for charge to atom type
charge_to_atom = {
    1 : 'H',
    6 : 'C',
    7 : 'N',
    8 : 'O',
    16: 'S'
}

# unit conversion from bohr to angstrom
BohrToA = 0.529177249

for i in range(len(Z)):
    filename = f"/Users/pingyang/Documents/ML example/xyz_file/qm7_xyz_{i+1}.xyz"
    atoms = []
    coordinates = []
    with open(filename, "w") as xyz:
        for j in range(len(Z[i])):
            if int(Z[i][j]) != 0:
                atoms.append(charge_to_atom[int(Z[i][j])])
                coordinates.append(R[i][j]*BohrToA)
        xyz.write(f"{len(atoms)}\n\n")
        for k in range(len(atoms)):
            xyz.write(f"{atoms[k]}  {coordinates[k][0]:.10f} {coordinates[k][1]:.10f} {coordinates[k][2]:.10f}\n")