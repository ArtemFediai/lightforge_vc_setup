"""
Script to change lightforge settings and copy over necessary files
todo: results/experiments/particle_densities/all_data_points/ion_dopants_0.dat
this is where doping is computed.
ion_dop = second / first line.

# todo
# todo
# TODO TODO TODO TODO if you set custom parameters, you still need that the dir exist (Analysis). This has to be fixed!

"""
import ast
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import yaml
import os
import pathlib
import shutil
import sys
import logging
from typing import List, Dict, Union, Tuple, Optional, Set
from collections import Counter
import pandas as pd

from QuantumPatch.Shredder.Parsers.ParserCommon import parse_system

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_PATH = pathlib.Path('/hkfs/work/workspace/scratch/nz8308-VC/vdw_materials/')
LF_SETUP_PATH = BASE_PATH / 'light/setup'
LF_SIM_PATH = BASE_PATH / 'light/sim'
QP_SETUP_PATH = BASE_PATH / 'qp/set_up'
QP_SIM_PATH = BASE_PATH / 'qp/sim'
PARA_SIM_PATH = BASE_PATH / 'para/sim'
DEPO_SIM_PATH = BASE_PATH / 'depo/sim'
MATERIALS = ['aNPD', 'BFDPB', 'BPAPF', 'TCTA']
# MATERIALS = ['BFDPB', 'BPAPF', 'TCTA']
# names of simulations folder whatever was simulated: disorder / ipea / vc. todo: weak point.
LF_SETTINGS_TMPL_PATH = LF_SETUP_PATH / 'lf_settings_tmpl/settings.yml'
# REORGANIZATION_ENERGIES = PARA_SIM_PATH / 'lambda/reorganization_energies.csv'
REORGANIZATION_ENERGIES = PARA_SIM_PATH / 'lambda_LH/reorganization_energies.csv'

MAP_MATERIAL_TO_HOST_DOPANT_NAME = {
    'aNPD': ('aNPD', 'C60F48'),
    'BFDPB': ('BFDPB', 'C60F48'),
    'BPAPF': ('BPAPF', 'C60F48'),
    'TCTA': ('TCTA', 'C60F48')
}

# VC_NAME = 'vc_new_after_dr_0.0.csv' # this will be taken
VC_NAME = 'updated_vc_new_after_dr_0.0.csv' # this will be taken. include effect on vc due to increased eps.
# VC_NAME = 'vc_for_lf.csv'  # do not make a difference between cation-anion and anion-cation.
# VC_NAME = 'vc_for_lf.csv'  # do not make a difference between cation-anion and anion-cation; corrected because of eps. maybe, the estimation is too approx.
# todo: 1. name is verbose. change in qp code base.
#  2. location will be different if generated from within QP.
#  3. the vc computed for 50 pairs with b3lyp is somewhere else.
COM_NAME = 'COM.dat'  # always from the disorder simulations of QP. maybe set an explicit path.

# this is something different because it is intended to submit the script:
SUBMIT_FILE = '/hkfs/work/workspace/scratch/nz8308-VC/vdw_materials/light/setup/submitQP_hk'

experimental_ips = {
    'BFDPB': 5.3,
    'aNPD': 5.45,
    'BPAPF': 5.6,
    'TCTA': 5.85
}

gw_nbs_extrap_ip_vacuum = {
    'aNPD': 6.851309428950864,
    'BFDPB': 6.712514705882352,
    'BPAPF': 6.901298817774153,
    'TCTA': 7.24258977149075
}

gw_cardinal_ip = {
    'aNPD': 6.11296,
    'BFDPB': 6.02801,
    'BPAPF': 6.18888,
    'TCTA': 6.36794
}

# ea of C60F48
gw_cardinal_ea = {
    'aNPD': 4.57809,
    'BFDPB': 4.61549,
    'BPAPF': 4.64139,
    'TCTA': 4.60249
}

experimental_eas = {
    'C60F48': 5.1,
    'C60F36': 4.5
}

# from qp. franz method to evaluate eps.
eps_qp_low = {
    'aNPD': 3.21,
    'BFDPB': 3.27,
    'BPAPF': 3.15,
    'TCTA': 2.90
}

# from qp, franz method to evaluate eps
eps_qp_high = {
    'aNPD': 3.29,
    'BFDPB': 3.87,
    'BPAPF': 3.29,
    'TCTA': 2.95
}

# mean from qp.
eps_qp_mean = {key: 0.5 * (eps_qp_low[key] + eps_qp_high[key]) for key in eps_qp_low}

# this is estimated or experimental data when available.
eps_materials_static = {
    'aNPD': 3.5,
    'BFDPB': 3.8,
    'BPAPF': 3.7,
    'TCTA': 3.46
}

ea_that_fits_bpapf = 5.5135

host_disorder = {
    'BFDPB': (0.111, 0.088),
    'aNPD': (0.099, 0.08),
    'BPAPF': (0.06, 0.08),
    'TCTA': (0.117, 0.084)
}

# below, all evaluations are wrong.
lh_ip = {
    'BFDPB': 5.451,  # LH: 6.64. GW: 6.531. Delta: 0.11 --> 5.341  |
    'aNPD': 5.55,  # vacuum LH: 6.78. GW: 6.67. Delta = 0.11 --> 5.44 eV (GW cardinal extr.)  | 5.97 if GW.
    'BPAPF': 5.65,  # LH: 6.8, GW: 6.72. Delta = 0.08 --> 5.57
    'TCTA': 5.83   # LH: 7.11, GW: 7.058. Delta = 7.11 - 7.058 = 0.052 --> 5.778
}

# this is the EA of the C60F48
lh_ea = {
    'aNPD': 4.8,  # 4.756 from GW bs limit vacuum + P gsp_free.
    'BFDPB': 4.84,
    'BPAPF': 4.86,
    'TCTA': 4.83
}

d_eps = {
    'aNPD': 0.5,
    'BFDPB': 0.5,
    'BPAPF': 0.5,
    'TCTA': 0.5,
    'C60F48': 0.5
}

# eps_qp_mean --> esp_qp_mean + 0.5
dP_virb = {
    'aNPD': 0.0479,
    'BFDPB': 0.0375,
    'BPAPF': 0.0375,
    'TCTA': 0.0692,
    'C60F48': 0.0950
}

# using eps form other papers or similar for intrinsic materials.
custom_eps_high_freq = {
    'aNPD': 2.79,
    'BFDPB': 3.27,
    'BPAPF': 3.15,
    'TCTA': 2.49
}

# as above plut 0.5 for all materials but aNPD.
custom_eps_low_freq = {
    'aNPD': 3.5,
    'BFDPB': 3.77,
    'BPAPF': 3.65,
    'TCTA': 3.5
}

# see custom_eps_high_freq and custom_eps_low_freq 
dP_vibr_custom_eps = {
    'aNPD': 0.0850,
    'BFDPB': 0.0442,
    'BPAPF': 0.0442,
    'TCTA': 0.0931,
    'C60F48': 0.0543
}

lh_ip_minus_dP_vibr = {mol:(lh_ip[mol] - dP_virb[mol]) for mol in lh_ip.keys()}  # IP decreased because of additional polarization due to vibronics.
# lh_ea_plus_dP_vibr = {
#     'C60F48': lh_ea['C60F48'] + dP_virb['C60F48']
# }  # unused?

gw_cardinal_ip_minus_dP_vibr_custom_eps = {mol:(gw_cardinal_ip[mol] - dP_vibr_custom_eps[mol]) for mol in gw_cardinal_ip.keys()}
gw_cardinal_ea_plus_dP_vibr_custom_eps = {mol:(gw_cardinal_ea[mol] + dP_vibr_custom_eps[mol]) for mol in gw_cardinal_ea.keys()}


def main():
    # todo this specific main has to be a new script. The input above has to go.
    for material in MATERIALS:
        """
        Main players:
        - host_uuid, dopant_uuid.
        - QuantumPatchOutput, short QPO.
        - ChangeSettings(..., QPO)
        Restriction:
        - 2 component systems.
        """

        logging.info(f"\n\n{'=' * 50}\n\tProcessing material: {material}\n{'=' * 50}\n")
        # todo name `material` is misleading. This is just the name of the folder. Or material identificator. It looks as if it is some complex object. But it is a string.
        create_sim_dir(material)
        copy_settings_to_sim(material)
        copy_vc_to_sim(material, VC_NAME)
        copy_com_to_sim(material)
        modify_and_copy_submit_script(material)
        # host_dopant_uuids_and_dmr_and_hmr = return_host_dopant_uuid(material)  # old useless implementation.
        # classes
        # host / dopant molecules are being initialed here based on the structure which is
        host_molecule, dopant_molecule = return_host_dopant_molecules(material,
                                                                      return_as='Molecule',
                                                                      dump=True)
        for mol in (host_molecule, dopant_molecule):
            mol.add_reorganization_energy(material)  # mol knows if it is host or dopant.

        #  all qps outputs
        QPO = QuantumPatchOutputType  # short
        qp_outputs = QuantumPatchOutputs(
            QuantumPatchOutput(QPO.IPEA),
            QuantumPatchOutput(QPO.DISORDER),
            QuantumPatchOutput(QPO.VC),
            QP_SIM_PATH,
            material,
            host_molecule,
            dopant_molecule
        )

        depo_output = DepositOutput(material)

        with ChangeLightforgeSettings(material, host_molecule, dopant_molecule, qp_outputs,
                                      depo_output) as change_settings:
            change_settings.set_host_dopant_uuid()
            # change_settings.set_disorder(source='SystemAnalysis', manual_disorder_host=host_disorder['BPAPF'])
            change_settings.set_disorder(source='SystemAnalysis')
            # change_settings.set_ip_ea_eps(adiabatic_energies=True,
            #                               adiabatic_dopant_ea=ea_that_fits_bpapf,  # adiabatic already!
            #                               adiabatic_host_ip=experimental_ips[material] + 0.548,  #0.452,
            #                               manual_eps=eps_materials_static[material])
            # change_settings.set_ip_ea_eps(adiabatic_energies=True,
            #                               manual_dopant_ea=lh_ea[material],  # was this an error??
            #                               manual_host_ip=lh_ip[material],
            #                               manual_eps=eps_materials_static[material])
            change_settings.set_ip_ea_eps(adiabatic_energies=True,
                                          manual_dopant_ea=gw_cardinal_ea_plus_dP_vibr_custom_eps[material] + 0.4,
                                          manual_dopant_ip=12.0,  # fiction
                                          manual_host_ip=gw_cardinal_ip_minus_dP_vibr_custom_eps[material],
                                          manual_host_ea=1.0,  # fiction
                                          manual_eps=custom_eps_low_freq[material])
            # change_settings.set_ip_ea_eps(adiabatic_energies=False,
            #                               manual_eps=eps_materials_static[material])
            change_settings.set_morphology_size(pbc=False)

            # 
            logging.info(f"Completed processing for material: {material}\n{'-' * 50}")

        compute_and_log_molecule_parameters(host_molecule, dopant_molecule, qp_outputs)

        log_folder_structure(LF_SIM_PATH / material)


class lazy_property:
    def __init__(self, function):
        self.function = function
        self.attribute_name = '_' + function.__name__

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.attribute_name):
            setattr(obj, self.attribute_name, self.function(obj))
        return getattr(obj, self.attribute_name)


@dataclass
class DepositOutput:
    """
    x, y, z are morphology sizes.
    """
    material: str
    PBC: bool = True
    one_point_two: float = 1.2

    def __post_init__(self):
        self._size_computed = False
        self._x = None
        self._y = None
        self._z = None

    @property
    def x(self):
        self._compute_size_if_needed()
        return self._x

    @property
    def y(self):
        self._compute_size_if_needed()
        return self._y

    @property
    def z(self):
        self._compute_size_if_needed()
        return self._z

    def _compute_size_if_needed(self):
        if not self._size_computed:
            self._x, self._y, self._z = self.compute_size_of_morphology()
            self._size_computed = True

    def compute_size_of_morphology(self):
        """
        Compute Lx, Ly, Lz to use in KMC.
        :return: Tuple of (x, y, z) sizes
        """
        # Load simulation parameters
        logging.info(f"Compute size of {self.material} . . .")
        sim_p_path = DEPO_SIM_PATH / self.material / 'simulation_parameters.sml'
        with open(sim_p_path) as fid:
            sim_p = yaml.load(fid, Loader=yaml.CLoader)

        x = sim_p["preprocessor"]["algorithm"]["params"]["grid_size_x"]
        y = sim_p["preprocessor"]["algorithm"]["params"]["grid_size_y"]

        if self.PBC:
            x *= 3
            y *= 3

        if self.PBC:
            infilename = DEPO_SIM_PATH / self.material / 'structurePBC.cml'  # todo: this has to be defined above as consts
        else:
            infilename = DEPO_SIM_PATH / self.material / 'structure.cml'  # todo: this has to be defined above as consts
        system = parse_system(infilename)

        # Compute atomic coordinates size
        atomic_coords = system.coordinates
        atomic_coords_size = np.max(atomic_coords, axis=0) - np.min(atomic_coords, axis=0)

        z = atomic_coords_size[2] + self.one_point_two

        logging.info(f". . . Size of {self.material} computed.")
        logging.info(f"Size of kMC morphology: {x}, {y}, {z}")

        return x, y, z


@dataclass
class LambdaOutput:
    """
    format:
    Compound,Charging State,Vertical Energy,Adiabatic Energy,Reorganization Energy
    aNPD,IP,-49058.1380029,-49057.9977218,-.1402811
    BFDPB,IP,-59610.2446803,-59610.1445158,-.1001645
    TCTA,IP,-62456.4751605,-62456.3541551,-.1210054
    BPAPF,IP,-79416.3745314,-79416.3438503,-.0306811
    C60F48,EA,-192380.1906395,-192380.0061228,-.1845167

    """
    mol_name: str
    lambda_ip: float = field(init=False)
    lambda_ea: float = field(init=False)

    def extract_lambda_from_csv(self):
        df = pd.read_csv(REORGANIZATION_ENERGIES)
        # Filter the dataframe for the current material
        material_df = df[df['Compound'] == self.mol_name]

        # Initialize lambda_ip and lambda_ea as None
        self.lambda_ip = None
        self.lambda_ea = None

        # Check for IP and EA
        if 'IP' in material_df['Charging State'].values:
            self.lambda_ip = material_df[material_df['Charging State'] == 'IP']['Reorganization Energy'].iloc[0]
        if 'EA' in material_df['Charging State'].values:
            self.lambda_ea = material_df[material_df['Charging State'] == 'EA']['Reorganization Energy'].iloc[0]

        # Handle missing data
        if self.lambda_ip is None and self.lambda_ea is None:
            raise ValueError(f"No IP or EA data available for {self.mol_name}")
        elif self.lambda_ip is None:
            self.lambda_ip = self.lambda_ea
            logging.warning(f"IP data missing for {self.mol_name}. Using EA value for IP.")
        elif self.lambda_ea is None:
            self.lambda_ea = self.lambda_ip
            logging.warning(f"EA data missing for {self.mol_name}. Using IP value for EA.")


def create_sim_dir(material: str):
    os.makedirs(LF_SIM_PATH / material, exist_ok=True)
    logging.info(f"Folder {material} is created in {LF_SIM_PATH}")


def copy_settings_to_sim(material: str):
    """
    Copy settings template to the simulation directory of the given material
    """
    source_path = LF_SETTINGS_TMPL_PATH
    destination_path = LF_SIM_PATH / material
    try:
        shutil.copy2(source_path, destination_path)
        logging.info(f"Settings template `{str(LF_SETTINGS_TMPL_PATH).split('/')[-1]}` copied to {destination_path}")
    except IOError as e:
        logging.error(f"Unable to copy file. {e}")
    except Exception as e:
        logging.error("Unexpected error:", exc_info=e)


def copy_vc_to_sim(material: str, vc_name):
    """
    Copy `vc.csv` to the simulation directory of the given material.
    `vc.csv` contains data on Coulomb binding energy typically between host and dopant.
    """
    source_path = QP_SIM_PATH / QuantumPatchOutputType.VC.value / material / vc_name
    destination_path = LF_SIM_PATH / material / 'vc.csv'
    try:
        shutil.copy2(source_path, destination_path)
        logging.info(f"Data on VC for material {material}, `vc`, is copied over to {destination_path}")
    except IOError as e:
        logging.error(f"Unable to copy file. {e}")
    except Exception as e:
        logging.error("Unexpected error:", exc_info=e)


def copy_com_to_sim(material: str):
    """
    Copy `com.csv` to the simulation directory of the given material.
    """
    source_path = QP_SIM_PATH / QuantumPatchOutputType.DISORDER.value / material / 'Analysis/files_for_kmc/COM.dat'
    destination_path = LF_SIM_PATH / material / 'COM.dat'
    try:
        shutil.copy2(source_path, destination_path)
        logging.info(f"Data COM.dat for material {material}, is copied over to {destination_path}")
    except IOError as e:
        logging.error(f"Unable to copy file. {e}")
    except Exception as e:
        logging.error("Unexpected error:", exc_info=e)


def modify_and_copy_submit_script(material):
    """
    Modify the Slurm submit script to change the job name to the given material and copy it to the simulation directory.
    Uses global constants (SUBMIT_FILE etc.) for source and destination paths.
    """
    source_path = SUBMIT_FILE
    filename = os.path.basename(SUBMIT_FILE)
    destination_path = LF_SIM_PATH / material / filename

    try:
        with open(source_path, 'r') as file:
            content = file.readlines()

        modified_content = [line if not line.startswith('#SBATCH -J') else f'#SBATCH -J {material}\n' for line in
                            content]

        with open(destination_path, 'w') as file:
            file.writelines(modified_content)

        logging.info(f"Submit script for material {material} is modified and copied to {destination_path}")
    except IOError as e:
        logging.error(f"Unable to modify or copy the submit script. {e}")
        raise


class MoleculeType(Enum):
    HOST = 'host'
    DOPANT = 'dopant'


class QuantumPatchOutputType(Enum):
    IPEA = 'ipea'
    DISORDER = 'disorder'
    VC = 'vc'


@dataclass
class Molecule:
    name: str
    type: MoleculeType
    uuid: str
    molar_fraction: float
    lambda_ip: float = field(init=False)
    lambda_ea: float = field(init=False)

    def add_reorganization_energy(self, material):
        if self.type == MoleculeType.HOST:
            mol_name = MAP_MATERIAL_TO_HOST_DOPANT_NAME[material][0]
        elif self.type == MoleculeType.DOPANT:
            mol_name = MAP_MATERIAL_TO_HOST_DOPANT_NAME[material][1]
        lout = LambdaOutput(mol_name=mol_name)
        lout.extract_lambda_from_csv()
        self.lambda_ip = lout.lambda_ip
        self.lambda_ea = lout.lambda_ea


def return_host_dopant_molecules(
        material: str,
        return_as: str = 'dict',
        dump: bool = True
) -> Union[Dict[str, Union[str, float]], Tuple[Molecule, Molecule]
]:
    """
    Creates two molecules host and dopant
    or
    Corresponding dictionaries
    Note: from `disorder` simulation folder file `structure.cml` is parsed to identify host/dopant uuid / doping rates.
    :param material:
    :param return_as:
    :param dump: will save it as a dictionary if the dictionary.
    :return:
    """
    infilename = QP_SIM_PATH / 'disorder' / material / 'structure.cml'
    system = parse_system(infilename)
    mol_type_counter = Counter(system.moltypes)
    sorted_mol_types = mol_type_counter.most_common()

    host_uuid = sorted_mol_types[0][0]  # highest count
    dopant_uuid = sorted_mol_types[-1][0]  # lowest count

    dopant_count = mol_type_counter[dopant_uuid]
    host_count = mol_type_counter[host_uuid]
    total_count = sum(mol_type_counter.values())
    dopant_molar_fraction = dopant_count / total_count
    host_molar_fraction = host_count / total_count
    logging.info(f"host is {host_uuid}, dopant is {dopant_uuid}. DMF = {dopant_molar_fraction}")
    # opt: dump ->
    return_dict = {
        'host': host_uuid,
        'dopant': dopant_uuid,
        'DMF': dopant_molar_fraction,
        'HMF': host_molar_fraction
    }
    if dump:
        logging.info("Trying to dump uuid of host and dopant...")
        file_name = f'{material}.yml'
        with open(file_name, 'w') as fid:
            yaml.safe_dump(return_dict, fid)
        logging.info(f"... Dumped uuids into {os.path.relpath(file_name)}")
    # <- opt: dump
    if return_as == 'dict':
        return {
            'host': host_uuid,
            'dopant': dopant_uuid,
            'DMF': dopant_molar_fraction,
            'HMF': host_molar_fraction
        }
    elif return_as == 'Molecule':
        return (
            Molecule(material, MoleculeType.HOST, host_uuid, host_molar_fraction),
            Molecule(material, MoleculeType.DOPANT, dopant_uuid, dopant_molar_fraction)
        )
    else:
        sys.exit(f"f either is the type of the output is `Molecule` or `dict`. it is: {return_as} Exiting")


@dataclass
class QuantumPatchOutput:
    """
    :param path: path to QP simulation of a given type.
    """
    type: QuantumPatchOutputType
    sim_path: pathlib.Path = field(default=None)
    _analysis_path: pathlib.Path = field(init=False)  # use this if later computed with property decorator
    _disorder: Dict[str, Tuple] = field(default_factory=dict, init=False)
    _disorder_computed: bool = field(default=False, init=False)
    _ea_ip_eps: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    _ea_ip_computed: bool = field(default=False, init=False)

    @lazy_property
    def analysis_path(self) -> pathlib.Path:
        return self.sim_path / 'Analysis'

    @lazy_property
    def files_for_kmc_path(self) -> pathlib.Path:
        return self.sim_path / 'Analysis' / 'files_for_kmc'

    def get_id_from_uuid(self, uuid):
        if self.type != QuantumPatchOutputType.DISORDER:
            raise ValueError("This method is only available for DISORDER type")
        dict_id_uuid = {}
        with open(self.files_for_kmc_path / 'mol_type_names.dat') as fid:
            for ini_id, line in enumerate(fid):
                dict_id_uuid[ini_id] = line.strip()  # strip() removes leading and trailing whitespace, including \n
        inverse_dict = {v: k for k, v in dict_id_uuid.items()}
        return inverse_dict[uuid]

    def query_disorder_from_files_for_kmc(self, uuid: str) -> Tuple:
        if self.type != QuantumPatchOutputType.DISORDER:
            raise ValueError("This method is only available for DISORDER type")
        mol_id = self.get_id_from_uuid(uuid)  # 0 or 1
        path_to_disorder = self.files_for_kmc_path / f'sigma_mol_pairs_{mol_id}_{mol_id}.dat'
        with open(path_to_disorder, 'r') as fid:
            lines = fid.readlines()
            homo_disoder, lumo_disorder = tuple(float(line.strip()) for line in lines)
        return homo_disoder, lumo_disorder

    def compute_disorder_from_system_analysis(self) -> None:
        """
        for all uuids. usage: when global disorder is requested.
        :return:
        """
        if self.type != QuantumPatchOutputType.DISORDER:
            raise ValueError("This method is only available for DISORDER type")

        path_to_per_mol_data = self.analysis_path / "SystemAnalysis/PerMoleculeInfo.dat"

        # -> get last_step
        pattern = re.compile(r'homo_lumo_total_by_molid_step_(\d+)\.dat')

        # Initialize a variable to hold the maximum step value
        last_step = -1

        # Iterate over files in the directory
        for file in self.analysis_path.glob('SystemAnalysis/homo_lumo_total_by_molid_step_*.dat'):
            match = pattern.search(str(file.name))
            if match:
                # Extract the step number and update max_step if it's higher
                step = int(match.group(1))
                if step > last_step:
                    last_step = step
        # <- get last_step

        path_to_homo_lumo_total = self.analysis_path / f'SystemAnalysis/homo_lumo_total_by_molid_step_{last_step}.dat'

        # Load first file (UUIDs)
        a = np.loadtxt(path_to_per_mol_data, dtype={'names': ('index', 'uuid'), 'formats': ('i4', 'U32')}, skiprows=2)
        b = np.loadtxt(path_to_homo_lumo_total,
                       dtype={'names': ('index', 'homo', 'lumo', 'total'), 'formats': ('f4', 'f8', 'f8', 'f8')},
                       skiprows=2)

        # Extract unique UUIDs
        unique_uuids = np.unique(a['uuid'])

        # Calculate std for homo and lumo for each UUID
        for uuid in unique_uuids:
            # Find indices in 'b' where 'index' matches any index from 'a' for the current UUID
            indices = [item[0] for item in a if item[1] == uuid]
            relevant_rows = b[np.isin(b['index'], indices)]

            # Calculate standard deviation for homo and lumo
            homo_std = np.std(relevant_rows['homo']) if len(relevant_rows) > 0 else np.nan
            lumo_std = np.std(relevant_rows['lumo']) if len(relevant_rows) > 0 else np.nan

            # Store in dictionary
            self._disorder[uuid] = homo_std, lumo_std

        self._disorder_computed = True

    def query_disorder_from_system_analysis(self, uuid: str) -> Tuple:
        if not self._disorder_computed:
            self.compute_disorder_from_system_analysis()
        return self._disorder.get(uuid)

    def get_ip_ea_eps(self) -> None:
        """
        Extracts and saves EA, IP, and epsilon values for all UUIDs found in Analysis/IPEA_final_*_summary.yml files.
        """
        if self.type != QuantumPatchOutputType.IPEA:
            raise ValueError("This method is only available for IPEA type")

        uuids = self.find_uuids_in_filenames(directory=self.analysis_path)
        print(f"uuids = {uuids}")
        for uuid in uuids:
            with open(self.analysis_path / f'IPEA_final_{uuid}_summary.yml') as fid:
                data = yaml.safe_load(fid)
            self._ea_ip_eps[uuid] = {
                'EA': data['EA']['mean'],
                'IP': data['IP']['mean'],
                'eps': data['epsilon']
            }

        print(self._ea_ip_eps)
        self._ea_ip_computed = True

    def get_ea(self, uuid: str) -> float:
        if not self._ea_ip_computed:
            self.get_ip_ea_eps()
        return self._ea_ip_eps.get(uuid, {}).get('EA')

    def get_ip(self, uuid: str) -> float:
        if not self._ea_ip_computed:
            self.get_ip_ea_eps()
        return self._ea_ip_eps.get(uuid, {}).get('IP')

    def get_epsilon(self, uuid: str) -> float:
        if not self._ea_ip_computed:
            self.get_ip_ea_eps()
        return self._ea_ip_eps.get(uuid, {}).get('eps')

    @staticmethod
    def find_uuids_in_filenames(directory: pathlib.Path, file_pattern: str = 'IPEA_final_*_summary.yml') -> Set[str]:
        """
        Finds UUIDs in file names within a given directory based on a specified pattern.

        :param directory: Path to the directory containing the files.
        :param file_pattern: File name pattern to match. Default is 'IPEA_final_*_summary.yml'.
        :return: A set of unique UUIDs extracted from the file names.
        """
        # Regular expression to extract the UUID
        uuid_pattern = re.compile(r'IPEA_final_([a-f0-9]{32})_summary\.yml')

        # Find all files matching the pattern
        files = directory.glob(file_pattern)

        # Extract UUIDs
        uuids = set()
        for file in files:
            match = uuid_pattern.match(file.name)
            if match:
                uuid = match.group(1)
                uuids.add(uuid)

        return uuids


@dataclass
class QuantumPatchOutputs:
    """
    simply three QP Output objects
    """
    ipea: QuantumPatchOutput
    disorder: QuantumPatchOutput
    vc: QuantumPatchOutput
    qp_sim_path: pathlib.Path
    material: str
    host: Molecule
    dopant: Molecule

    def __post_init__(self):
        for qp_simulation in [self.ipea, self.disorder, self.vc]:
            qp_simulation.sim_path = self.qp_sim_path / qp_simulation.type.value / self.material


@dataclass
class ChangeLightforgeSettings:
    """
    This class is responsible for modifying the settings.yml file for a given material in a Lightforge simulation.
    It utilizes the output from Quantum Patch (QP) simulations (ipea, disorder, vc) to update the settings.

    Attributes:
        material (str): The name of the simulation folder for the specific material.
        host: host molecule.
        dopant: dopant molecule.
        qp_outputs (QuantumPatchOutputs): Outputs from ipea, disorder, and vc QP simulations.
        yaml_file_path (pathlib.Path): The file path to the settings.yml file. This is initialized post object creation.
        settings (dict): A dictionary representing the contents of the Lightforge settings file.
    """
    material: str
    host: Molecule
    dopant: Molecule
    qp_outputs: QuantumPatchOutputs
    depo_output: DepositOutput
    yaml_file_path: pathlib.Path = field(init=False)
    settings: dict = field(default=None, init=False)

    class DisorderExtractionTypes(Enum):
        FILES_FOR_KMC = 'files_for_kmc'
        SYSTEM_ANALYSIS = 'SystemAnalysis'
        MANUAL = 'manual'

    def __post_init__(self):
        self.yaml_file_path = LF_SIM_PATH / self.material / 'settings.yml'

    def __enter__(self):
        """Open the YAML file, load its contents, and then close the file."""
        logging.info(f"Opening YAML file: {self.yaml_file_path}")
        with open(self.yaml_file_path, 'r') as file:
            self.settings = yaml.safe_load(file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Open the file again, write changes, and close the file."""
        logging.info("Writing changes to settings file ...")
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(self.settings, file)
        logging.info("settings.yaml file was successfully changed")

    @lazy_property
    def host_dopant_order(self):
        mol_type_names_path = self.qp_outputs.disorder.files_for_kmc_path / 'mol_type_names.dat'
        with open(mol_type_names_path) as fid:
            mol_names = fid.read().splitlines()

        if mol_names[0] == self.dopant.uuid and mol_names[1] == self.host.uuid:
            dopant_index, host_index = 0, 1
        elif mol_names[1] == self.dopant.uuid and mol_names[0] == self.host.uuid:
            dopant_index, host_index = 1, 0
        else:
            sys.exit("Host and dopant names do not match to previously extracted. Exiting ... ")

        return dopant_index, host_index

    @lazy_property
    def host_index(self):
        """
        Lazy property to determine the index of the host in the material settings list.
        """
        _, host_index = self.host_dopant_order
        logging.info(f"Host id is identified from mol_type_names.dat: {host_index}")
        return host_index

    @lazy_property
    def dopant_index(self):
        """
        Lazy property to determine the index of the dopant in the material settings list.
        """
        dopant_index, _ = self.host_dopant_order
        logging.info(f"Dopant id is identified from mol_type_names.dat: {dopant_index}")
        return dopant_index

    def set_host_dopant_uuid(self):
        """
        Set host and dopant UUIDs in the settings.
        Besides (sorry for naming), it sets:
         1. "is_dopant" flag for both
         2. "name" for both.
        """
        self.settings['materials'][self.host_index]['molecule_parameters']['custom_hash'] = self.host.uuid
        self.settings['materials'][self.dopant_index]['molecule_parameters']['custom_hash'] = self.dopant.uuid

        self.settings['materials'][self.dopant_index]['molecule_parameters']['is_dopant'] = True
        self.settings['materials'][self.host_index]['molecule_parameters']['is_dopant'] = False
        self.settings['materials'][self.dopant_index]['name'] = 'dopant'
        self.settings['materials'][self.host_index]['name'] = 'host'

        logging.info(
            f"Host UUID set to {self.host.uuid}, "
            f"Dopant UUID set to {self.dopant.uuid} "
            f"for material {self.material}"
        )

    def set_disorder(self, source: str = DisorderExtractionTypes.FILES_FOR_KMC.value,
                     manual_disorder_dopant=None, manual_disorder_host=None):
        """
        for LF simulations.
        :return:
        """

        if manual_disorder_host is None or manual_disorder_dopant is None:
            disorder_sources = {
                self.DisorderExtractionTypes.FILES_FOR_KMC.value: self.qp_outputs.disorder.query_disorder_from_files_for_kmc,
                self.DisorderExtractionTypes.SYSTEM_ANALYSIS.value: self.qp_outputs.disorder.query_disorder_from_system_analysis,
            }

            query_disorder_function = disorder_sources[source]

        if manual_disorder_host is None:
            host_disorder = query_disorder_function(self.host.uuid)
        else:
            host_disorder = manual_disorder_host

        if manual_disorder_dopant is None:
            dopant_disorder = query_disorder_function(self.dopant.uuid)
        else:
            dopant_disorder = manual_disorder_dopant

        host_energies_str = self.settings['materials'][self.host_index]['molecule_parameters']['energies']
        dopant_energies_str = self.settings['materials'][self.dopant_index]['molecule_parameters']['energies']

        host_energies_list = self.convert_string_to_list(host_energies_str)
        dopant_energies_list = self.convert_string_to_list(dopant_energies_str)
        # [[orbitals], [disorder], [lambda]]
        # [ 0              1          2    ]

        host_energies_list[1] = list(host_disorder)
        dopant_energies_list[1] = list(dopant_disorder)

        host_energies_str = self.convert_list_to_string_with_brackets(host_energies_list)
        dopant_energies_str = self.convert_list_to_string_with_brackets(dopant_energies_list)

        self.settings['materials'][self.host_index]['molecule_parameters']['energies'] = host_energies_str
        self.settings['materials'][self.dopant_index]['molecule_parameters']['energies'] = dopant_energies_str

        logging.info(
            f"Host disorder set to {host_disorder}, "
            f"Dopant disorder set to {dopant_disorder} "
            f"for material {self.material}"
        )

    @staticmethod
    def convert_string_to_list(string):
        try:
            return ast.literal_eval(string)
        except ValueError as e:
            logging.error(f"Error converting string to list: {e}")
            sys.exit(f"Fatal error: unable to convert string to list. Exiting program.")

    @staticmethod
    def convert_list_to_string_with_brackets(lst):
        return str(lst)

    @staticmethod
    def convert_list_to_string_no_brackets(lst):
        str_list = [str(element) for element in lst]
        return ' '.join(str_list)

    def set_ip_ea_eps(self, adiabatic_energies=False,
                      manual_host_ip=None, manual_host_ea=None,
                      manual_dopant_ip=None, manual_dopant_ea=None,
                      manual_eps=None,
                      adiabatic_dopant_ea=None, adiabatic_host_ip=None):
        """
        Set IP and EA for host and dopant, with an option to manually specify these values.

        :param adiabatic_energies: If True, use adiabatic energies.
        :param manual_host_ip: Manually specified IP for the host.
        :param manual_host_ea: Manually specified EA for the host.
        :param manual_dopant_ip: Manually specified IP for the dopant.
        :param manual_dopant_ea: Manually specified EA for the dopant.
        """
        ipea_data = self.qp_outputs.ipea

        # Use manual values if provided, otherwise get from ipea_data
        host_ip = manual_host_ip if manual_host_ip is not None else ipea_data.get_ip(self.host.uuid)
        host_ea = manual_host_ea if manual_host_ea is not None else -ipea_data.get_ea(self.host.uuid)
        dopant_ip = manual_dopant_ip if manual_dopant_ip is not None else ipea_data.get_ip(self.dopant.uuid)
        dopant_ea = manual_dopant_ea if manual_dopant_ea is not None else -ipea_data.get_ea(self.dopant.uuid)

        print(f"Step 1: dopant_ea: {dopant_ea}")

        print(f"Lambda EA = {self.dopant.lambda_ea}")

        # Apply adiabatic correction if needed
        if adiabatic_energies:
            host_ip -= self.host.lambda_ip
            host_ea += self.host.lambda_ea
            dopant_ip -= self.dopant.lambda_ip
            dopant_ea += self.dopant.lambda_ea

        dopant_ea = adiabatic_dopant_ea if adiabatic_dopant_ea is not None else dopant_ea  # todo hard coded.
        host_ip = adiabatic_host_ip if adiabatic_host_ip is not None else host_ip  # todo hard coded.

        print(f"Step 2: afer lanmda added. dopant_ea + lambda: {dopant_ea}")

        host_ip_ea = [host_ip, host_ea]
        dopant_ip_ea = [dopant_ip, dopant_ea]

        # breakpoint()

        


        eps = manual_eps if manual_eps is not None else 0.5 * (ipea_data.get_epsilon(self.host.uuid) + ipea_data.get_epsilon(self.dopant.uuid))

        # todo: maybe not that good to have it here rather than above!

        # Retrieve existing lambda and disorder values from settings
        host_energies = self.convert_string_to_list(
            self.settings['materials'][self.host_index]['molecule_parameters']['energies'])
        dopant_energies = self.convert_string_to_list(
            self.settings['materials'][self.dopant_index]['molecule_parameters']['energies'])

        # Update settings with IP/EA
        host_energies[0] = host_ip_ea
        dopant_energies[0] = dopant_ip_ea

        # Update settings with lambda_value
        host_energies[2] = [self.host.lambda_ip, self.host.lambda_ea]
        dopant_energies[2] = [self.dopant.lambda_ip, self.dopant.lambda_ea]

        host_energies_str = self.convert_list_to_string_with_brackets(host_energies)
        dopant_energies_str = self.convert_list_to_string_with_brackets(dopant_energies)

        self.settings['materials'][self.host_index]['molecule_parameters']['energies'] = host_energies_str
        self.settings['materials'][self.dopant_index]['molecule_parameters']['energies'] = dopant_energies_str
        self.settings['epsilon_material'] = eps

        logging.info(
            f"Host IP/EA set to {host_ip_ea}, "
            f"Dopant IP/EA set to {dopant_ip_ea} "
            f"Dielectric permittivity set to {eps} "
            f"for material {self.material}"
        )

    def set_morphology_size(self, pbc=False):
        """
        sets the size of the morphology.
        2 x redundant in LF + in x-dim 1 x redundant => set in 3 places
        :return:
        """
        box = self.depo_output
        box.PBC = pbc

        # self.settings['layers']['dimensions']  todo ask Franz if not used??
        # self.settings['layers']['thickness']  todo same

        self.settings['dimensions'] = [float(box.z / 10.0), float(box.x / 10.0),
                                       float(box.y / 10.0)]  # z -> x -> y [nm]

        self.settings['layers'][0]['box1']['dimensions'] = self.convert_list_to_string_no_brackets(
            self.settings['dimensions'])

        self.settings['layers'][0]['thickness'] = float(box.z / 10.0)

        logging.info("Setting morphology size:")
        logging.info(f"\tsettings:dimensions (x, y, z) are set to {box.z / 10.0, box.x / 10.0, box.y / 10.0} nm. ")


# todo make a separate module out of it.
def log_folder_structure(startpath):
    startpath_str = str(startpath)  # Convert PosixPath to string
    for root, dirs, files in os.walk(startpath_str, topdown=True):
        level = root.replace(startpath_str, '').count(os.sep)
        indent = '│   ' * level + '├── '

        if level == 0:
            logging.info(f"{os.path.basename(root)}/")
        else:
            logging.info(f"{indent[:-4]}└── {os.path.basename(root)}/")

        subindent = '│   ' * (level + 1) + '├── '
        for f in files:
            logging.info(f"{subindent}{f}")


def log_parameters_as_table(host_params, dopant_params):
    """
    Log the parameters for host and dopant in a table-like format.

    :param host_params: Dictionary of parameters for the host.
    :param dopant_params: Dictionary of parameters for the dopant.
    """
    # Define the header
    header = "| Parameter     | Host Value                    | Dopant Value                   |"
    separator = "+---------------+-------------------------------+-------------------------------+"

    # Log the header and separator
    logging.info(separator)
    logging.info(header)
    logging.info(separator)

    # Iterate over parameters and log each
    for key in host_params:
        host_value = format_value(host_params[key])
        dopant_value = format_value(dopant_params[key])
        line = f"| {key.ljust(13)} | {host_value.ljust(29)} | {dopant_value.ljust(29)} |"
        logging.info(line)

    # Log the final separator
    logging.info(separator)


def format_value(value):
    """
    Format the value based on its type (UUID or floating point number).
    """
    if isinstance(value, str):  # Assuming UUIDs are strings
        return value[:3]
    elif isinstance(value, float):
        return f"{value:.3f}"  # todo hard-coded
    return str(value)


def compute_and_log_molecule_parameters(host_molecule, dopant_molecule, qp_outputs):
    """
    Compute (actually re-compute, which is a problem) parameters for host and dopant molecules and log them in a table format.
    """
    # todo: IP/EA if manually set will not be correct here. Read everything from the setting file!
    # todo: other parameters are also in danger!
    decimal_points = 3

    # Extract disorder from FILES_FOR_KMC source
    host_disorder = qp_outputs.disorder.query_disorder_from_system_analysis(host_molecule.uuid)  # todo careful
    dopant_disorder = qp_outputs.disorder.query_disorder_from_system_analysis(dopant_molecule.uuid)

    # Adjusted IP and EA values
    try:
        adjusted_host_ip = qp_outputs.ipea.get_ip(host_molecule.uuid) - host_molecule.lambda_ip
        adjusted_host_ea = -qp_outputs.ipea.get_ea(host_molecule.uuid) + host_molecule.lambda_ea

        adjusted_dopant_ip = qp_outputs.ipea.get_ip(dopant_molecule.uuid) - dopant_molecule.lambda_ip
        adjusted_dopant_ea = -qp_outputs.ipea.get_ea(dopant_molecule.uuid) + dopant_molecule.lambda_ea

        # Retrieve dielectric permittivity (eps) for host and dopant
        host_eps = qp_outputs.ipea.get_epsilon(host_molecule.uuid)
        dopant_eps = qp_outputs.ipea.get_epsilon(dopant_molecule.uuid)

        # Calculate the average dielectric permittivity
        average_eps = (host_eps + dopant_eps) / 2

        host_parameters = {
            'UUID': host_molecule.uuid[:3],
            'IP': round(adjusted_host_ip, decimal_points),
            'EA': round(adjusted_host_ea, decimal_points),
            'Disorder HOMO': round(host_disorder[0], decimal_points),
            'Disorder LUMO': round(host_disorder[1], decimal_points),
            'Lambda IP': round(host_molecule.lambda_ip, decimal_points),
            'Lambda EA': round(host_molecule.lambda_ea, decimal_points),
            'Eps': round(average_eps, decimal_points),
        }

        dopant_parameters = {
            'UUID': dopant_molecule.uuid[:3],
            'IP': round(adjusted_dopant_ip, decimal_points),
            'EA': round(adjusted_dopant_ea, decimal_points),
            'Disorder HOMO': round(dopant_disorder[0], decimal_points),
            'Disorder LUMO': round(dopant_disorder[1], decimal_points),
            'Lambda IP': round(dopant_molecule.lambda_ip, decimal_points),
            'Lambda EA': round(dopant_molecule.lambda_ea, decimal_points),
            'Eps': round(average_eps, decimal_points),
        }

        # Log the parameters in a table-like format
        log_parameters_as_table(host_parameters, dopant_parameters)
    except TypeError:
        logging.error("I could not make log info probably because I used the custom data ... ")


if __name__ == '__main__':
    main()
