# Data Cleaning
The file that I cleaned is too big to upload to GitHub, but can be found at this link: https://drive.google.com/file/d/1Zps0YssoJbZHrn6iLte2RDLlgruhAX1s/view?usp=sharing

# Prompt

This was part of my final exam for my Python course.

Your boss comes to you Monday morning and says “I figured out our next step; we are going to pivot from an online craft store and become a data center for genetic disease information! I found **ClinVar** which is a repository that contains expert curated data, and it is free for the taking. This is a gold mine! Look at the file and tell me what gene and mutation combinations are classified as dangerous.”

Make sure that you only give your boss the dangerous mutations and include:

1) Gene name

2) Mutation ID number

3) Mutation Position (chromosome & position)

4) Mutation value (reference & alternate bases)

5) Clinical severity 

6) Disease that is implicated

* Limit your output to the first 100 harmful mutations and tell your boss how many total harmful mutations were found in the file


# Notes:
### VCF file description (Summarized from version 4.1)

```
* The VCF specification:

VCF is a text file format which contains meta-information lines, a header line, and then data lines each containing information about a position in the genome. The format also can contain genotype information on samples for each position.

* Fixed fields:

There are 8 fixed fields per record. All data lines are **tab-delimited**. In all cases, missing values are specified with a dot (‘.’). 

1. CHROM - chromosome number
2. POS - position DNA nuceleotide count (bases) along the chromosome
3. ID - The unique identifier for each mutation
4. REF - reference base(s)
5. ALT - alternate base(s)
6. FILTER - filter status
7. QUAL - quality
8. INFO - a semicolon-separated series of keys with values in the format: <key>=<data>

```
### Applicable INFO field specifications

```
GENEINFO = <Gene name>
CLNSIG =  <Clinical significance>
CLNDN = <Disease name>
```

# Assumptions
- I assume I only need GENEINFO, CLNSIG, and CLNDN from the INFO column and I can ignore the rest
- I assume I should convert 'O'to 0 in POS column
- I have ranked the "danger level" in this order:
    1. Pathogenic
    2. Likely_pathogenic
    3. Conflicting
    4. Other / Not_Given
    5. Likely_benign
    6. Benign
- In my final dataframe I will include only ones that are "Pathogenic"and contain the word "cancer" in the "NAME" category (but the rest of the dataframe is ready to show if my boss asks!)
- I will take only these values, sort by the "POS" column, then take the top 100 from there. 
- I will replace missing values in the dataframe with: 'Not_Given'


```python
import pandas as pd
import io
import os
```


```python
# Read in the data
def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'CHROM': str, 'POS': str, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'CHROM': 'CHROM'})

data = read_vcf("clinvar_final.txt")
data.head() #not pretty
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CHROM</th>
      <th>POS</th>
      <th>ID</th>
      <th>REF</th>
      <th>ALT</th>
      <th>FILTER</th>
      <th>QUAL</th>
      <th>INFO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1014O42</td>
      <td>475283</td>
      <td>G</td>
      <td>A</td>
      <td>.</td>
      <td>.</td>
      <td>AF_ESP=0.00546;AF_EXAC=0.00165;AF_TGP=0.00619;...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1O14122</td>
      <td>542074</td>
      <td>C</td>
      <td>T</td>
      <td>.</td>
      <td>.</td>
      <td>AF_ESP=0.00015;AF_EXAC=0.00010;ALLELEID=514926...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1014143</td>
      <td>183381</td>
      <td>C</td>
      <td>T</td>
      <td>.</td>
      <td>.</td>
      <td>ALLELEID=181485;CLNDISDB=MedGen:C4015293,OMIM:...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1014179</td>
      <td>542075</td>
      <td>C</td>
      <td>T</td>
      <td>.</td>
      <td>.</td>
      <td>ALLELEID=514896;CLNDISDB=MedGen:C4015293,OMIM:...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1014217</td>
      <td>475278</td>
      <td>C</td>
      <td>T</td>
      <td>.</td>
      <td>.</td>
      <td>AF_ESP=0.00515;AF_EXAC=0.00831;AF_TGP=0.00339;...</td>
    </tr>
  </tbody>
</table>
</div>




```python
def info_parser(info):
    """Extract relevant information from a cell in the info column of this dataframe.
    Returns the information as a list."""
    info_str = []
    if "GENEINFO=" in info:
        gene = info[info.index("GENEINFO=")+9 :]
        if ";" in gene:
            gene = gene[:gene.index(";")]
        info_str.append(gene)
    else:
        info_str.append("Not_Given")
    if "CLNSIG=" in info:
        sig = info[info.index("CLNSIG=")+7 :]
        if ";" in sig:
            sig = sig[:sig.index(";")]
        info_str.append(sig)
    else:
        info_str.append("Not_Given")
    if "CLNDN=" in info:
        name = info[info.index("CLNDN=")+6 :]
        if ";" in name:
            name = name[:name.index(";")]
        info_str.append(name)
    else:
        info_str.append("Not_Given")

    return info_str
```


```python
# Convert INFO column to lists
data["INFO"] = data["INFO"].apply(info_parser)

#Convert lists into columns and add to dataframe
info_col = data["INFO"].apply(pd.Series)
info_col = info_col.rename(columns = {0:"GENE", 1:"SIG", 2:"NAME"})
data = pd.concat([data[:], info_col[:]], axis = 1)
data = data.drop("INFO", axis = 1)
```


```python
def danger_function(row):
    """Sort the 'SIG' column by how dagnerous it seems to someone who is impersonating a doctor."""
    if "Conflicting" in row:
        return 3
    elif "Likely_pathogenic" in row:
        return 2
    elif "Pathogenic" in row:
        return 1
    elif "Likely_benign" in row:
        return 5
    elif "Benign" in row:
        return 6
    else:
        return 4

data["DANGER"] = data["SIG"].apply(danger_function)
data.sort_values(by = ["DANGER"])

# Convert "." in everything and'O'to 0 in POS column
data = data.replace(to_replace = '.', value = "Not_Given")
for i in range(len(data["POS"])):
    if "O" in data["POS"][i]:
        data["POS"][i] = data["POS"][i].replace("O", "0")
```

    /Users/patriciadegner/opt/anaconda3/envs/r-env/lib/python3.7/site-packages/ipykernel_launcher.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy



```python
### Sort Data
# concat cols
data["CHROM_POS"] = data["CHROM"].map(str) + " : " + data["POS"]
data["REF_ALT"] = data["REF"] + " : " + data["ALT"]

# remove unneded cols
data = data.drop("FILTER", axis = 1)
data = data.drop("QUAL", axis = 1)
data = data.drop("CHROM", axis = 1)
data = data.drop("POS", axis = 1)
data = data.drop("REF", axis = 1)
data = data.drop("ALT", axis = 1)

# reorder cols
data = data[["GENE", "ID", "CHROM_POS", "REF_ALT", "SIG", "NAME", "DANGER"]]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENE</th>
      <th>ID</th>
      <th>CHROM_POS</th>
      <th>REF_ALT</th>
      <th>SIG</th>
      <th>NAME</th>
      <th>DANGER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ISG15:9636</td>
      <td>475283</td>
      <td>1 : 1014042</td>
      <td>G : A</td>
      <td>Benign</td>
      <td>Immunodeficiency_38_with_basal_ganglia_calcifi...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ISG15:9636</td>
      <td>542074</td>
      <td>1 : 1014122</td>
      <td>C : T</td>
      <td>Uncertain_significance</td>
      <td>Immunodeficiency_38_with_basal_ganglia_calcifi...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ISG15:9636</td>
      <td>183381</td>
      <td>1 : 1014143</td>
      <td>C : T</td>
      <td>Pathogenic</td>
      <td>Immunodeficiency_38_with_basal_ganglia_calcifi...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ISG15:9636</td>
      <td>542075</td>
      <td>1 : 1014179</td>
      <td>C : T</td>
      <td>Uncertain_significance</td>
      <td>Immunodeficiency_38_with_basal_ganglia_calcifi...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ISG15:9636</td>
      <td>475278</td>
      <td>1 : 1014217</td>
      <td>C : T</td>
      <td>Benign</td>
      <td>Immunodeficiency_38_with_basal_ganglia_calcifi...</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# How many mutations are harmful in this list?
print("Danger level = 1: %s" %len(data[(data['DANGER'])==1]))
print("Danger level = 2: %s" %len(data[(data['DANGER'])==2]))
print("Total danger <= 2: %s" %len(data[(data['DANGER'])<=2]))
```

    Danger level = 1: 12330
    Danger level = 2: 7144
    Total danger <= 2: 19474



```python
# Make the dataframe I will show my boss -> the 100 most dangerous
df = data[data["DANGER"] == 1 & (data['NAME'].str.contains("cancer"))]
print(len(df))
df.sort_values(by = ["CHROM_POS"])
boss_df = df.head(100)
boss_df.head()
```

    1198





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GENE</th>
      <th>ID</th>
      <th>CHROM_POS</th>
      <th>REF_ALT</th>
      <th>SIG</th>
      <th>NAME</th>
      <th>DANGER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4358</th>
      <td>SDHB:6390</td>
      <td>12781</td>
      <td>1 : 17022648</td>
      <td>C : T</td>
      <td>Pathogenic</td>
      <td>Pheochromocytoma|Gastrointestinal_stroma_tumor...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4359</th>
      <td>SDHB:6390</td>
      <td>186827</td>
      <td>1 : 17022649</td>
      <td>G : A</td>
      <td>Pathogenic</td>
      <td>Pheochromocytoma|Gastrointestinal_stroma_tumor...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4366</th>
      <td>SDHB:6390</td>
      <td>12782</td>
      <td>1 : 17022653</td>
      <td>TAGAG : T</td>
      <td>Pathogenic</td>
      <td>Pheochromocytoma|Hereditary_cancer-predisposin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4373</th>
      <td>SDHB:6390</td>
      <td>233795</td>
      <td>1 : 17022659</td>
      <td>GA : G</td>
      <td>Pathogenic</td>
      <td>Hereditary_cancer-predisposing_syndrome</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4383</th>
      <td>SDHB:6390</td>
      <td>142637</td>
      <td>1 : 17022684</td>
      <td>C : T</td>
      <td>Pathogenic</td>
      <td>Pheochromocytoma|Gastrointestinal_stroma_tumor...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This dataframe contains the following columns:
    1. GENE: Gene name
    2. ID: Mutation ID number
    3. CHROM_POS: Mutation Position (chromosome & position)
    4. REF_ALT: Mutation value (reference & alternate bases)
    5. SIG: Clinical severity
    6. NAME: Disease(s) that is implicated
    7. DANGER: Danger level

The Danger level is assessed from clinical signficance. It was assigned the score below if the clinical signicance value contained that word:
    1. Pathogenic
    2. Likely pathogenic
    3. Conflicting Results
    4. Other / Not_Given
    5. Likely benign
    6. Benign

There are 12,330 mutations that are considered "Pathogenic" and 7,144 that are "Likely_pathogenic", for a total of 19,474 harmful mutations. 

The final dataframe that I submitted has only mutations that are "Pathogenic" and where the implicated disease is cancer. There are 1,198 mutations that meet this critera. I sorted this list by mutation position then pulled the first 100 to show you. 
