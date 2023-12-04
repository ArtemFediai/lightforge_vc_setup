"""
Script to change settings

# todo
1. sizes of the box for kmc.
2. correct EA of the dopant molecule.
maybe set reorganization energy. no, this is not relevant.
# lambda is not laambda. thois is the geometry relaxation energy

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
# QP_SIM_PATH = BASE_PATH / 'light/setup/fake_ipea_output'
MATERIALS = ['aNPD', 'BFDPB', 'BPAPF',
             'TCTA']  # names of simulations folder whatever was simulated: disorder / ipea / vc. todo: weak point.
LF_SETTINGS_TMPL_PATH = LF_SETUP_PATH / 'lf_settings_tmpl/settings.yml'
REORGANIZATION_ENERGIES = PARA_SIM_PATH / 'lambda/reorganization_energies.csv'

MAP_MATERIAL_TO_HOST_DOPANT_NAME = {
    'aNPD': ('aNPD', 'C60F48'),
    'BFDPB': ('BFDPB', 'C60F48'),
    'BPAPF': ('BPAPF', 'C60F48'),
    'TCTA': ('TCTA', 'C60F48')
}


def main():
    for material in MATERIALS:
        """
        Plan.
        1/ Identify host and dopants: uuid_dop, uuid_host, DMR, HMR.
        2/ Construct QP output objects: ini + extract info.
        3/ Change setting in LF. The ChangeSettings object has to take an object QuantumPatchOutputs. 
        Not sure I would make it simply a list of QPO objects.
        
        Main players:
        - host_uuid, dopant_uuid.
        - QuantumPatchOutput, short QPO.
        - ChangeSettings(..., QPO)
        Restriction:
        - 2 component systems.
        """

        logging.info(f"\n{'=' * 50}\nProcessing material: {material}\n{'=' * 50}")
        # todo name `material` is misleading. This is just the name of the folder. Or material identificator. It looks as if it is some complex object. But it is a string.
        create_sim_dir(material)
        copy_settings_to_sim(material)
        copy_vc_to_sim(material)
        # host_dopant_uuids_and_dmr_and_hmr = return_host_dopant_uuid(material)  # old useless implementation.
        # classes
        # host / dopant molecules are being initialed here based on the structure which is
        host_molecule, dopant_molecule = return_host_dopant_molecules(material,
                                                                      return_as='Molecule',
                                                                      dump=True)
        for mol in (host_molecule, dopant_molecule):
            mol.add_reorganization_energy(material)  # mol knows, if it is host or dopant. 

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

        with ChangeLightforgeSettings(material, host_molecule, dopant_molecule, qp_outputs, depo_output) as change_settings:
            change_settings.set_host_dopant_uuid()
            change_settings.set_disorder(source='SystemAnalysis')
            change_settings.set_ip_ea_eps(adiabatic_energies=True)
            change_settings.set_morphology_size()
            # (3) todo: change concentration of the dopant, or maybe not required.

            # 
            logging.info(f"Completed processing for material: {material}\n{'-' * 50}")

        # todo: copy vc over
        # todo: neighbours =150 seems to much!!!


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

        infilename = DEPO_SIM_PATH / self.material / 'structurePBC.cml'
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


def copy_vc_to_sim(material: str):
    """
    Copy `vc.csv` to the simulation directory of the given material.
    `vc.csv` contains data on Coulomb binding energy typically between host and dopant.
    """
    source_path = QP_SIM_PATH / QuantumPatchOutputType.VC.value / material
    destination_path = LF_SIM_PATH / material
    try:
        shutil.copy2(source_path, destination_path)
        logging.info(f"Data on VC for material {material}, `vc`, is copied over to {destination_path}")
    except IOError as e:
        logging.error(f"Unable to copy file. {e}")
    except Exception as e:
        logging.error("Unexpected error:", exc_info=e)


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
    def host_index(self):
        """
        Lazy property to determine the index of the host in the material settings list.
        """
        for index, material_settings in enumerate(self.settings['materials']):
            if not material_settings['molecule_parameters']['is_dopant']:
                return index
        logging.error("Host not found in material settings.")
        raise ValueError("Host not found in material settings.")

    @lazy_property
    def dopant_index(self):
        """
        Lazy property to determine the index of the dopant in the material settings list.
        """
        for index, material_settings in enumerate(self.settings['materials']):
            if material_settings['molecule_parameters']['is_dopant']:
                return index
        logging.error("Dopant not found in material settings.")
        raise ValueError("Dopant not found in material settings.")

    def set_host_dopant_uuid(self):
        """
        Set host and dopant UUIDs in the settings.
        """
        self.settings['materials'][self.host_index]['molecule_parameters']['custom_hash'] = self.host.uuid
        self.settings['materials'][self.dopant_index]['molecule_parameters']['custom_hash'] = self.dopant.uuid

        logging.info(
            f"Host UUID set to {self.host.uuid}, "
            f"Dopant UUID set to {self.dopant.uuid} "
            f"for material {self.material}"
        )

    def set_disorder(self, source: str = DisorderExtractionTypes.FILES_FOR_KMC.value):
        """
        for LF simulations.
        :return:
        """

        disorder_sources = {
            self.DisorderExtractionTypes.FILES_FOR_KMC.value: self.qp_outputs.disorder.query_disorder_from_files_for_kmc,
            self.DisorderExtractionTypes.SYSTEM_ANALYSIS.value: self.qp_outputs.disorder.query_disorder_from_system_analysis
        }

        query_disorder_function = disorder_sources[source]

        host_disorder = query_disorder_function(self.host.uuid)
        dopant_disorder = query_disorder_function(self.dopant.uuid)

        host_energies_str = self.settings['materials'][self.host_index]['molecule_parameters']['energies']
        dopant_energies_str = self.settings['materials'][self.dopant_index]['molecule_parameters']['energies']

        host_energies_list = self.convert_string_to_list(host_energies_str)
        dopant_energies_list = self.convert_string_to_list(dopant_energies_str)
        # [[orbitals], [disorder], [lambda]]
        # [ 0              1          2    ]

        host_energies_list[1] = list(host_disorder)
        dopant_energies_list[1] = list(dopant_disorder)

        host_energies_str = self.convert_list_to_string(host_energies_list)
        dopant_energies_str = self.convert_list_to_string(dopant_energies_list)

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
        except ValueError:
            print("Error: The string could not be converted to a list.")
            return None

    @staticmethod
    def convert_list_to_string(lst):
        return str(lst)

    def set_ip_ea_eps(self, adiabatic_energies=False):
        """
        if adiabatic_energies, these will be extracted from the host and dopant molecules.
        """

        ipea_data = self.qp_outputs.ipea

        host_ip_ea = [ipea_data.get_ip(self.host.uuid), -ipea_data.get_ea(self.host.uuid)]
        dopant_ip_ea = [ipea_data.get_ip(self.dopant.uuid),
                        -ipea_data.get_ea(self.dopant.uuid)]  # minus because EA from the output has wrong sign

        if adiabatic_energies:
            logging.info("adiabatic_energies option is set for IP and EA.")

            logging.info(
                f"adiabatic host ip: {host_ip_ea[0] - self.host.lambda_ip} = {host_ip_ea[0]} - {self.host.lambda_ip}")
            logging.info(
                f"adiabatic host ea: {host_ip_ea[1] + self.host.lambda_ea} = {host_ip_ea[1]} + {self.host.lambda_ea}")
            logging.info(
                f"adiabatic dopant ip: {dopant_ip_ea[0] - self.dopant.lambda_ip} = {dopant_ip_ea[0]} - {self.dopant.lambda_ip}")
            logging.info(
                f"adiabatic dopant ea: {dopant_ip_ea[1] + self.dopant.lambda_ea} = {dopant_ip_ea[1]} + {self.dopant.lambda_ea}")

            host_ip_ea[0] -= self.host.lambda_ip
            host_ip_ea[1] += self.host.lambda_ea
            dopant_ip_ea[0] -= self.dopant.lambda_ip
            dopant_ip_ea[1] += self.dopant.lambda_ea

        eps = 0.5 * (ipea_data.get_epsilon(self.host.uuid) + ipea_data.get_epsilon(
            self.dopant.uuid))  # mean eps. formally, these must be the same

        host_energies_str = self.settings['materials'][self.host_index]['molecule_parameters']['energies']
        dopant_energies_str = self.settings['materials'][self.dopant_index]['molecule_parameters']['energies']

        host_energies_list = self.convert_string_to_list(host_energies_str)
        dopant_energies_list = self.convert_string_to_list(dopant_energies_str)
        # [[orbitals], [disorder], [lambda]]
        # [ 0              1          2    ]

        host_energies_list[0] = list(host_ip_ea)
        dopant_energies_list[0] = list(dopant_ip_ea)

        host_energies_str = self.convert_list_to_string(host_energies_list)
        dopant_energies_str = self.convert_list_to_string(dopant_energies_list)

        self.settings['materials'][self.host_index]['molecule_parameters']['energies'] = host_energies_str
        self.settings['materials'][self.dopant_index]['molecule_parameters']['energies'] = dopant_energies_str

        # eps:
        self.settings['epsilon_material'] = eps

        logging.info(
            f"Host IP/EA set to {host_ip_ea}, "
            f"Dopant IP/EA set to {dopant_ip_ea} "
            f"Dielectric permittivity set to {eps}"
            f"for material {self.material}"
        )

    def set_morphology_size(self):
        print("Hey!")

        box = self.depo_output

        # self.settings['layers']['dimensions']  todo ask Franz if not used??
        # self.settings['layers']['thickness']  todo same
        self.settings['dimensions'] = [float(box.z/10.0), float(box.x/10.0), float(box.y/10.0)]  # z -> x -> y
        logging.info("Setting morphology size:")
        logging.info(f"\tsettings:dimensions are set to {box.x/10.0, box.y/10.0, box.z/10.0} nm. ")


if __name__ == '__main__':
    main()
