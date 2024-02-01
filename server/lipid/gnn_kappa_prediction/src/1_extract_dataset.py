import re

import pandas as pd
import os

def process_nodes(lipid_name):

    try:
    # Define the path to your node feature matrix data file
        node_feature_file_path = os.path.join(node_feature_dir, lipid_name+'.txt')

        # Read the node feature matrix data into a DataFrame
        node_feature_df = pd.read_csv(node_feature_file_path, sep='\t', index_col=0)

        # Define the path to your adjacency matrix data file
        adjacency_matrix_file_path = os.path.join(adjacency_matrix_dir, f"{lipid_name} .txt")
    except Exception as e:
        return ([],[])

    # Read the adjacency matrix data into a DataFrame
    adjacency_df = pd.read_csv(adjacency_matrix_file_path, sep='\t', index_col=0)
    node_features=[]
    edge_features=[]
    # Process the node feature DataFrame
    for index, row in node_feature_df.iterrows():
        features = [column for column in node_feature_df.columns if row[column] > 0.0]
        if features:
            for i in features:
                node_features.append((row.name,i))
    for source_node, row in adjacency_df.iterrows():
        for target_node, value in row.items():
            if value > 0.0:
                edge_features.append((source_node,target_node))

    return (node_features,edge_features)





# Define the directories containing node feature and adjacency matrix files
node_feature_dir = '../data/TextFiles/Node_Features/'  # Replace with the actual directory path
adjacency_matrix_dir = '../data/TextFiles/Adjacency_Matrix/'  # Replace with the actual directory path

def extract_composition_names(composition_str):
    # Regex to find words that are not part of percentages or numerical values
    matches = re.findall(r'\b(?<!\d%)(?<!\d\.\d%)(?<!\d\.)(?<!\d)\b[a-zA-Z]+', composition_str)
    return matches

def extract_percentages(composition_str):
    pattern = r"(\d+(?:\.\d+)?)%"
    percentages = re.findall(pattern, composition_str)
    return [float(p)/100 for p in percentages]

df=pd.read_csv('../data/Merged_and_Sorted_Df.csv')

df['Node Features'] = None
df['Edge List'] = None
df['Graph-Level Features']=None

def sort_tuple(my_list):
    my_list = [(min(i[0], i[1]), max(i[0], i[1])) for i in my_list]
    my_list.sort()
    return list(set(my_list))

for index, row in df.iterrows():
    names = extract_composition_names(row['Composition'])
    node_features = []
    edge_features = []

    for name in names:
        data = process_nodes(name)
        node_features.extend(data[0])  # Assuming data[0] is a list of node features
        edge_features.extend(data[1])  # Assuming data[1] is a list of edge features
    if len(node_features)==0:
        df.drop(index, inplace=True)
        continue

    percentages=extract_percentages(row['Composition'])

    df.at[index, 'Node Features'] = list(set(node_features))
    df.at[index, 'Edge List'] = sort_tuple(edge_features)
    df.at[index,'Graph-Level Features']= percentages

df.to_csv('../data/Final_Dataset_for_Model_Train.csv')

