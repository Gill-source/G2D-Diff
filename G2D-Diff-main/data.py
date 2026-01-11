import pandas as pd

# Load master drug-response file (contains real cell names)
meta = pd.read_csv("./data/drug_response_data/DC_drug_response.csv")
cell_names = meta['ccle_name'].values

# genotype CSV length = available genotype count
N = len(pd.read_csv("./data/drug_response_data/original_cell2mut.csv"))

print("genotype available cell count:", N)
print("\nCells that can be tested:\n")

# Print only the first N names
for i in range(N):
    print(i, cell_names[i])
