import re
import pandas as pd
import os

current_directory = os.path.dirname(__file__)
node_feature_dir= os.path.join(current_directory, '../data/TextFiles/Node_Features/' )
adjacency_matrix_dir = os.path.join(current_directory, '../data/TextFiles/Adjacency_Matrix/')
# file_path = os.path.join(current_directory, '../data/Merged_and_Sorted_Df.csv')
# old_df = pd.read_csv(file_path)

def extract_dataset(df):
    # Create a new DataFrame with the same index as the existing DataFrame
    new_columns = ['Node Features', 'Edge List', 'Graph-Level Features']
    new_df = pd.DataFrame(columns=new_columns)

    # Fill the new columns with None values
    new_df['Node Features'] = None
    new_df['Edge List'] = None
    new_df['Graph-Level Features'] = None
    df = pd.concat([df, new_df], axis=1)

    # df['Node Features'] = None
    # df['Edge List'] = None
    # df['Graph-Level Features']=None
    print("init df")
    print(df)
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
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, '../data/Final_Dataset_for_Model_Train.csv')
    print(df.tail())
    df.to_csv(file_path,index=False)

def process_nodes(lipid_name):
    try:
        # Define the path to your node feature matrix data file
        node_feature_file_path = os.path.join(node_feature_dir, f"{lipid_name}.txt")

        # Read the node feature matrix data into a DataFrame
        node_feature_df = pd.read_csv(node_feature_file_path, sep='\t', index_col=0)

        # Define the path to your adjacency matrix data file
        adjacency_matrix_file_path = os.path.join(adjacency_matrix_dir, f"{lipid_name} .txt")
    except Exception as e:
        print("error")
        return ([], [])
    # Read the adjacency matrix data into a DataFrame
    adjacency_df = pd.read_csv(adjacency_matrix_file_path, sep='\t', index_col=0)
    node_features = []
    edge_features = []
    # Process the node feature DataFrame
    for index, row in node_feature_df.iterrows():
        features = [column for column in node_feature_df.columns if row[column] > 0.0]
        if features:
            for i in features:
                node_features.append((row.name, i))
    for source_node, row in adjacency_df.iterrows():
        for target_node, value in row.items():
            if value > 0.0:
                edge_features.append((source_node, target_node))
    return (node_features, edge_features)

def extract_composition_names(composition_str):
    # Regex to find words that are not part of percentages or numerical values
    matches = re.findall(r'\b(?<!\d%)(?<!\d\.\d%)(?<!\d\.)(?<!\d)\b[a-zA-Z]+', composition_str)
    return matches

def extract_percentages(composition_str):
    pattern = r"(\d+(?:\.\d+)?)%"
    percentages = re.findall(pattern, composition_str)
    return [float(p) / 100 for p in percentages]

def sort_tuple(my_list):
    my_list = [(min(i[0], i[1]), max(i[0], i[1])) for i in my_list]
    my_list.sort()
    return list(set(my_list))
