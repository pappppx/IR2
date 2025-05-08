from model_utils import train_simple_model, train_deep_model_tf, train_mlp_model_tf
    
def main():
    train_simple_model("datasets/dataset_05segundo.csv")
    train_mlp_model_tf("datasets/dataset_05segundo.csv")
    train_deep_model_tf("datasets/dataset_05segundo.csv")

if __name__ == '__main__':
    main()