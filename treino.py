import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from preprocessamento import carregar_dados, criar_preprocessador

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-mlp")

NOME_MODELO = "titanic-mlp"

# Configurações de hiperparâmetros testadas
EXPERIMENTOS = [
    {"nome": "MLP-01", "hidden_layer_sizes": (32,),     "activation": "relu", "learning_rate_init": 0.001,  "solver": "adam", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
    {"nome": "MLP-02", "hidden_layer_sizes": (32, 32),  "activation": "relu", "learning_rate_init": 0.001,  "solver": "adam", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
    {"nome": "MLP-03", "hidden_layer_sizes": (64, 64),  "activation": "relu", "learning_rate_init": 0.001,  "solver": "adam", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
    {"nome": "MLP-04", "hidden_layer_sizes": (64, 64),  "activation": "relu", "learning_rate_init": 0.0001, "solver": "adam", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
    {"nome": "MLP-05", "hidden_layer_sizes": (64, 64),  "activation": "tanh", "learning_rate_init": 0.001,  "solver": "adam", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
    {"nome": "MLP-06", "hidden_layer_sizes": (64, 64),  "activation": "relu", "learning_rate_init": 0.001,  "solver": "adam", "alpha": 0.01,   "batch_size": 64, "max_iter": 300},
    {"nome": "MLP-07", "hidden_layer_sizes": (16,),     "activation": "logistic", "learning_rate_init": 0.01, "solver": "sgd", "alpha": 0.0001, "batch_size": 32, "max_iter": 300},
]

X_train, X_test, y_train, y_test = carregar_dados()

modelos_salvos = {}  # run_id -> endereço do modelo salvo no MLflow

for exp in EXPERIMENTOS:
    with mlflow.start_run(run_name=exp["nome"]):
        # Pipeline = pré-processamento + MLP (salvos juntos no mesmo artifact)
        modelo = Pipeline([
            ("preprocessador", criar_preprocessador()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=exp["hidden_layer_sizes"],
                activation=exp["activation"],
                learning_rate_init=exp["learning_rate_init"],
                solver=exp["solver"],
                alpha=exp["alpha"],
                batch_size=exp["batch_size"],
                max_iter=exp["max_iter"],
                random_state=42,
            )),
        ])
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Parameters
        mlflow.log_param("hidden_layers", str(exp["hidden_layer_sizes"]))
        mlflow.log_param("activation", exp["activation"])
        mlflow.log_param("learning_rate", exp["learning_rate_init"])
        mlflow.log_param("solver", exp["solver"])
        mlflow.log_param("alpha", exp["alpha"])
        mlflow.log_param("batch_size", exp["batch_size"])
        mlflow.log_param("max_iter", exp["max_iter"])

        # Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # Artifact (modelo treinado)
        info = mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle")
        modelos_salvos[mlflow.active_run().info.run_id] = info.model_uri

        print(f"{exp['nome']} -> F1: {f1_score(y_test, y_pred):.4f}")

# Seleção do melhor modelo (entre os runs desta execução): maior F1-score
runs = mlflow.search_runs(experiment_names=["titanic-mlp"], order_by=["metrics.f1_score DESC"])
runs = runs[runs["run_id"].isin(modelos_salvos)]
melhor = runs.iloc[0]

print(f"\nMelhor modelo: {melhor['tags.mlflow.runName']} (F1 = {melhor['metrics.f1_score']:.4f})")

# Registra o melhor modelo no Model Registry do MLflow
mlflow.register_model(modelos_salvos[melhor["run_id"]], NOME_MODELO)
