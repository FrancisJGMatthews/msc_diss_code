### Notebook naming convention: number-type-name

- **number:** integer number of notebook. To be run in ascending order.
- **type:** letter denoting type of notebook. e = exploratory, d = data processing, m = model training
- **name:** plain text name description of notebook.


### Notebooks overview

| Notebook | Purpose | Inputs | Outputs|
|----------|----------|----------|----------|
| **0-d-process-replication-data.ipynb** | Read in replication data and Zhixiang data, produce characteristics table for congressmembers | **characteristics data:** data/raw/all_congress_members.csv | **plots (optional):** reports/figures/  |
| **1-d-congress-accounts-following-data-process.ipynb**| processes raw CSV following data for each congressperson and compiles into master edgelist dataframe. Node attributes exist as columns for both the congresspeople (party, DW-Nominate score, Chamber, Gender, State, congressperson_active) and the accounts they follow (account created_at, total followers count, total following count). House and Senate dataframes are also produced and saved. | **characteristics data:** data/raw/all_congress_members.csv, <br> **following data:** data/raw/following_data_raw/  | **master following table:** data/interim/congress_master_following_table.csv <br> **House following table:** data/interim/house_master_following_table.csv <br> **Senate following table:** data/interim/senate_master_following_table.csv|
| **2-e-congress-acounts-exploratory.ipynb** | Read in characteristics data on congressmembers and produce a series of exploratory plots: Bar chart of chamber split; DW-Nominate spectrum, boxplot and histogram | **characteristics data:** data/raw/all_congress_members.csv | **plots (optional):** reports/figures/  |
| **3.1-d-congress-accounts-create-networks.ipynb** | Read in Congress, House and Senate following dataframes from 1 and produce network edgelists (training data). Optionality of removing certain nodes below $k^{in}$ threshold when constructing networks. | **House following table:** data/interim/house_master_following_table.csv <br> **Senate following table:** data/interim/senate_master_following_table.csv | **House edgelist:** data/processed/house_edgelist_kout_kin.csv <br> **Senate edgelist:** data/processed/senate_edgelist_kout_kin.csv|
| **3.2-e-congress-accounts-network-exploratory.ipynb** | Compute network statistics for Congress, House and Senate networks: Node/Edge counts, degree distributions | **House edgelist:** data/processed/house_edgelist_kout_kin.csv <br> **Senate edgelist:** data/processed/senate_edgelist_kout_kin.csv | None |
| **4-m-embedding-neural-network.ipynb** | Train neural networks using the Congress, House or Senate edgelists as training data to create embedding representation of congressmembers | **Edgelist training data:** data/processsed/\*edgelist\*.csv | **Trained network :** models/model\*.pth <br> **Embeddings dictionary (pickle):** data/processed/embedding\*.pkl|
| **5-m-principal-components.ipynb** | Take vector representations of congressmembers and compute/plot a specified number of principcal components through PCA  | **Embeddings dictionary (pickle):** data/processed/embedding\*.pkl <br> **characteristics data:** data/raw/all_congress_members.csv | TBC (plots) |
| **6-m-adjacency-principal-components.ipynb** | Run PCA on congressmembers row of biadjacency matrix, i.e a vector representing the users who they follow. | **Embeddings dictionary (pickle):** data/processed/embedding\*.pkl <br> **characteristics data:** data/raw/all_congress_members.csv | TBC (plots) |
| **7-m-k-means-clustering.ipynb** | Perform K-Means clustering on congressmember embeddings  | **Embeddings dictionary (pickle):** data/processed/embedding\*.pkl <br> **characteristics data:** data/raw/all_congress_members.csv | TBC (plots) |
| **execution_notebook.ipynb** | This is an ad-hoc notebook used to iterate over mutltiple runs of parameterised notebook(s) to produce edgelists, cleansed networks etc. | **Embeddings dictionary (pickle):** data/processed/embedding\*.pkl <br> **characteristics data:** data/raw/all_congress_members.csv | TBC (plots) |

