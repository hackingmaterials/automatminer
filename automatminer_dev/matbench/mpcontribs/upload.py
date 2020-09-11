import wget, json, os
from string import capwords
from pybtex.database import parse_string
import pybtex.errors
from mpcontribs.client import Client
from pymatgen import MPRester
import tqdm


# from matminer.datasets.dataset_retrieval import (
#     get_all_dataset_info,
#     get_available_datasets,
#     load_dataset,
# )

from matminer.datasets import load_dataset

from automatminer_dev.config import DIELECTRIC


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
#     if "phonons" not in name:
#         continue
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




# Entering all contributions to projects
########################################

LIMIT = 100



ds_config = DIELECTRIC
name = "matbench_" + ds_config["name"]
client.delete_contributions(name)
print(f"Loading {name}")
df = load_dataset(name)
target = ds_config["target"]
unit = f" {ds_config['unit']}" if ds_config["unit"] else ""


#todo: PROBLEMATIC ENTRY IS DIELECTRIC INDEX 105

# df = df.iloc[104:]

chunks = (df.shape[0] - 1) // LIMIT + 1
for j in range(chunks):
    print(f"\tBatch {j} of {chunks}")
    batch = df.iloc[j * LIMIT:(j + 1) * LIMIT]

    contributions = []

    for i, row in enumerate(batch.iterrows()):
        entry = row[1]
        contrib = {'project': name, 'is_public': True, 'structures': []}
        s = entry.loc["structure"]
        c = s.composition.get_integer_formula_and_factor()[0]
        identifier = f"mb-{ds_config['name']}-{(j - 1) * LIMIT + i}"
        contrib["identifier"] = identifier
        contrib["data"] = {target: f"{entry.loc[target]}{unit}"}
        contrib["formula"] = c
        contrib["structures"].append(s)
        contributions.append(contrib)
    client.submit_contributions(contributions)