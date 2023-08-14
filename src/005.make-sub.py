"""
This script generates prediction files

1. Reads two Parquet files, one containing a model's predictions (`sub_model.pqt`) and the other containing popularity information (`sub_popularity.pqt`).
2. Resets the index and concatenates the two DataFrames.
3. Sorts the DataFrame by index and assigns a new bid value according to the index.
4. Writes the predictions to a text file, `prediction.txt`.
5. Reads the text file back and restructures it into a DataFrame, sorting it by bid.
6. Writes the sorted predictions back to `prediction.txt`.
7. Compresses the prediction text file into a ZIP archive with DEFLATED compression.
8. Prints a "Done" message once all tasks are completed.
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from tqdm import tqdm

df_nrms = pd.read_parquet("../predictions/sub_model.pqt")
df_pop = pd.read_parquet("../predictions/sub_popularity.pqt")
df_nrms = df_nrms.reset_index()
df_pop = df_pop.reset_index()
df = pd.concat((df_nrms, df_pop)).reset_index(drop=True)
df = df.sort_index()
df["bid"] = df.index
# print(df.columns)
# df = df.sort_values(by='b_id', ascending=True).reset_index(drop=True)
# print(df.head())
# print(df.tail())
out_path = Path("../predictions/prediction.txt")

with out_path.open(mode="w") as fp:
    # for idx, row in tqdm(df.iterrows()):
    #     fp.writelines(str(int(row['b_id'])) + ' ' + row['preds'] + '\n')
    for row in tqdm(df["preds"].values):
        fp.writelines(row + "\n")
bids = []
preds = []
with open("../predictions/prediction.txt", "r") as f:
    for line in f.readlines():
        bids.append(line.split(" ")[0])
        preds.append(line.split(" ")[1].strip())

df = pd.DataFrame(
    {
        "bid": bids,
        "preds": preds,
    }
)
df.bid = df.bid.astype(int)
df = df.sort_values(by="bid", ascending=True).reset_index(drop=True)

with out_path.open(mode="w") as fp:
    for idx, row in tqdm(df.iterrows()):
        fp.writelines(str(int(row["bid"])) + " " + row["preds"] + "\n")


with ZipFile("../predictions/prediction.txt.zip", "w", ZIP_DEFLATED) as z:
    z.write(out_path)

print("Done")
