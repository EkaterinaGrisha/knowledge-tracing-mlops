"""MLflow "models from code" definition of the packaged DKT model.

MLflow runs this file when the model is logged and loaded instead of
unpickling a Python object, which could execute arbitrary code. The file is
executed as a standalone script, hence the absolute import.
"""

import mlflow

from knowledge_tracing.inference import DKTPyfuncModel

mlflow.models.set_model(DKTPyfuncModel())
