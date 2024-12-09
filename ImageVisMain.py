from utils.LoadConfig import load_config
from utils.VisPipelineImage import visualize_mapper_graph

def main():
    # Configs Paths:
    # Image Euclidean BYOL: configs/ImgVisEucBYOLConfig.yaml
    # Image Euclidean SIMCLR: configs/ImgVisEucSIMCLRConfig.yaml
    # Image PoincareBall BYOL: configs/ImgVisPBBYOLConfig.yaml
    # Image PoincareBall SIMCLR: configs/ImgVisPBSimCLRConfig.yaml
    config_path = 'configs/ImgVisPBBYOLConfig.yaml'
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