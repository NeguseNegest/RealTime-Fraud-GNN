"""MLflow wrapper for the GraphSAGE and XGBoost ensemble."""

import mlflow
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from src.training.train_gnn import load_encoder_checkpoint


class FraudEnsemble(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.encoder = load_encoder_checkpoint(context.artifacts["encoder"], "cpu")
        self.encoder.eval()
        self.classifier = XGBClassifier()
        self.classifier.load_model(context.artifacts["xgboost"])

    def predict(self, context, model_input, params=None):
        x = torch.as_tensor(model_input["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(model_input["edge_index"], dtype=torch.long)
        target_nodes = torch.as_tensor(model_input["target_nodes"], dtype=torch.long)

        with torch.no_grad():
            embeddings = self.encoder(x, edge_index)

        features = np.concatenate((x[target_nodes].numpy(), embeddings[target_nodes].numpy()), axis=1)
        with threadpool_limits(limits=1):
            return self.classifier.predict_proba(features)[:, 1]
