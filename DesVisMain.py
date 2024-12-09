#import random
from utils.LoadConfig import load_config
from utils.VisPipelineDes import visualize_mapper_graph

def main():
    # Configs Paths:
    # Descriptors Euclidean: configs/DesVisEucConfig.yaml
    # Descriptors PoincareBall: configs/DesVisPBConfig.yaml
    config_path = 'configs/DesVisPBConfig.yaml'
    config = load_config(config_path)

    # Define the target column:
    # {Timestamp, Compound, Compound_Concetration, Growth_Condition_Binary}
    target_column = 'Growth_Condition_Binary'  # Replace with your desired target variable

    visualize_mapper_graph(
        config=config, 
        target_column=target_column, 
        space_type=config['space'],
        random_seed=config['random_seed'])

if __name__ == "__main__":
    main()