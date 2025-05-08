from model_utils import train_simple_model, train_deep_model_tf, train_mlp_model_tf, train_deep_model_new, train_mlp_model_new, train_wide_and_deep_model_new, train_wide_and_deep_model_tf
    
def main():
    train_simple_model("datasets/merge_05_31k.csv")
    train_mlp_model_tf("datasets/merge_05_31k.csv")
    train_deep_model_tf("datasets/merge_05_31k.csv")
    train_wide_and_deep_model_tf("datasets/merge_05_31k.csv")
    print("Ahora el dataset transformado")
    train_mlp_model_new("datasets/transformed_31k.csv")
    train_deep_model_new("datasets/transformed_31k.csv")
    train_wide_and_deep_model_new("datasets/transformed_31k.csv")

if __name__ == '__main__':
    main()