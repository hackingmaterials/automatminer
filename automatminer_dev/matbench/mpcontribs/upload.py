import wget, json, os, math
from string import capwords
from pybtex.database import parse_string
import pybtex.errors
from mpcontribs.client import Client
from pymatgen import MPRester, Structure
import tqdm
import pprint

# from matminer.datasets.dataset_retrieval import (
#     get_all_dataset_info,
#     get_available_datasets,
#     load_dataset,
# )

from matminer.datasets import load_dataset

from automatminer_dev.config import DIELECTRIC, JDFT2D, PEROVSKITES, STEELS, BENCHMARK_FULL_SET, BENCHMARK_DICT, HAS_STRUCTURE


pybtex.errors.set_strict_mode(False)
api_key = os.environ["MPCONTRIBS_API_KEY"]
client = Client(api_key, host='ml-api.materialsproject.cloud')
mprester = MPRester()


# client.get_project("matbench_steels").pretty()


fn = 'dataset_metadata.json'
if not os.path.exists(fn):
    wget.download(f'https://raw.githubusercontent.com/hackingmaterials/matminer/master/matminer/datasets/{fn}')
metadata = json.load(open(fn, 'r'))
metadata = {k: d for k, d in metadata.items() if "matbench" in k}



# Creating new projects
#######################
# todo: might not have access to add new projects
# for name, info in metadata.items():
#
#     print(f"Uploading {name}")
#
#     columns = {}
#     for col, text in info['columns'].items():
#         k = col.replace('_', '|').replace('-', '|').replace('(', ' ').replace(
#             ')', '')
#         columns[k] = text
#
#     project = {
#         'is_public': True,
#         'owner': 'ardunn@lbl.gov',
#         "name": name,
#         'title': name,  # TODO update and set long_title
#         'authors': 'A. Dunn, A. Jain',
#         'description': info['description'],
#         'other': {
#             'columns': columns,
#             'entries': info['num_entries']
#         },
#         'references': []
#     }
#
#     for ref in info['bibtex_refs']:
#
#         if name == "matbench_phonons":
#             ref = ref.replace(
#                 "petretto_dwaraknath_miranda_winston_giantomassi_rignanese_van setten_gonze_persson_hautier_2018",
#                 "petretto2018")
#
#         bib = parse_string(ref, 'bibtex')
#         for key, entry in bib.entries.items():
#             key_is_doi = key.startswith('doi:')
#             url = 'https://doi.org/' + key.split(':', 1)[
#                 -1] if key_is_doi else entry.fields.get('url')
#             k = 'Zhuo2018' if key_is_doi else capwords(key.replace('_', ''))
#             if k.startswith('C2'):
#                 k = 'Castelli2012'
#             elif k.startswith('Landolt'):
#                 k = 'LB1997'
#             elif k == 'Citrine':
#                 url = 'https://www.citrination.com'
#
#             if len(k) > 8:
#                 k = k[:4] + k[-4:]
#             project['references'].append({"label": k, "url": url})
#
#     try:
#         print(client.projects.create_entry(project=project).result())
#     except Exception as ex:
#         print(
#             ex)  # TODO should use get_entry to check existence -> use update_entry if project exists





# Map of canonical yet non-mpcontribs-compatible tagret nams to compatible (unicode, no punctuation) target names
target_map = {
    "yield strength": "σᵧ",
    "log10(K_VRH)": "log₁₀Kᵛʳʰ",
    "log10(G_VRH)": "log₁₀Gᵛʳʰ",
    "n": "𝑛",
    "exfoliation_en": "Eˣ",
    "gap pbe": "Eᵍ",
    "is_metal": "metallic",
    "e_form": "Eᶠ",
    "gfa": "glass",
    "gap expt": "Eᵍ",
    "last phdos peak": "ωᵐᵃˣ",
}


# # Getting project-level metadata in order
# #########################################
#
# # Add warning to mpcontribs since the results will be stored out of order.
# # Also, fix columns for new mpcontribs deployment
# for name, info in metadata.items():
#     mb_shortname = name.replace("matbench_", "")
#
#     description = info["description"] + f" If you are viewing this on MPContribs-ML interactively, please ensure the order of the identifiers is sequential (mb-{mb_shortname}-0001, mb-{mb_shortname}-0002, etc.) before benchmarking."
#     if "For benchmarking" not in description:
#         print(name, description)
#
#     has_structure = mb_shortname in [ds["name"] for ds in HAS_STRUCTURE]
#     primitive_key = "structure" if has_structure else "composition"
#     target = BENCHMARK_DICT[mb_shortname]["target"]
#
#     print(client.projects.update_entry(
#         pk=name,
#         project={
#             "description": description,
#             'other.columns': {
#                 target_map[target]: metadata[name]["columns"][target],
#                 primitive_key: metadata[name]["columns"][primitive_key]
#             }
#         }).result())




# Entering all contributions to projects
########################################


# steels.........X
# log_kvrh.......
# log_gvrh.......
# dielectric.....
# jdft2d.........X
# expt_gap.......X
# expt_is_metal..X
# phonons........
# mp_is_metal....
# mp_gap.........
# glass..........X
# mp_e_form......
# perovskites....




ds_config = BENCHMARK_DICT["jdft2d"]

name = "matbench_" + ds_config["name"]
print(f"Loading {name}")
df = load_dataset(name)
target = ds_config["target"]
unit = f" {ds_config['unit']}" if ds_config["unit"] else ""


# print(f"Updating 'other' column entries of {name} with unicode.")
# print(client.projects.update_entry(pk=name, project={
#     'other.columns': {
#         target_map[target]: metadata[name]["columns"][target],
#         "structure": metadata[name]["columns"]["structure"]
#         # "composition": metadata[name]["columns"]["composition"]
#     }
# }).result())



print(f"Deleting contributions of {name}")
client.delete_contributions(name)



print(f"Assembling and uploading contributions for {name}")
structure_filename = "/Users/ardunn/Downloads/outfile.cif"
contributions = []
id_prefix = df.shape[0]


id_n_zeros = math.floor(math.log(df.shape[0], 10)) + 1

df = df.iloc[:2]

for i, row in tqdm.tqdm(enumerate(df.iterrows())):
    entry = row[1]
    contrib = {'project': name, 'is_public': True}

    if "structure" in entry.index:
        structures = []
        s = entry.loc["structure"]
        s.to("cif", structure_filename)
        s = Structure.from_file(structure_filename)
        c = s.composition.get_integer_formula_and_factor()[0]
        contrib["structures"] = [s]

    else:
        c = entry["composition"]

    id_number = f"{i+1:0{id_n_zeros}d}"
    identifier = f"mb-{ds_config['name']}-{id_number}"
    contrib["identifier"] = identifier

    contrib["data"] = {target_map[target]: f"{entry.loc[target]}{unit}"}
    contrib["formula"] = c
    contributions.append(contrib)

pprint.pprint(contributions)
client.submit_contributions(contributions)