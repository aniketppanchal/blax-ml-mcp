import contextlib
import json
import logging
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from blaxel.core import SandboxInstance

from .config import settings
from .sb_manager import SandboxManager
from .schemas import ModelTrainingConfig

logger = logging.getLogger(__name__)

BLAXEL_BASE_DIR = "/blaxel"

CSV_FILENAME = "dataset.csv"
CSV_PROFILE = "csv_profile.json"

PIPELINE_DIR = "pipeline"
PIPELINE_META = "pipeline_meta.json"
PIPELINE_MODEL = "pipeline.joblib"

CSV_PROFILER_SCRIPT = "csv_profiler.py"
MODEL_TRAINER_SCRIPT = "model_trainer.py"
MODEL_SERVER_SCRIPT = "model_server.py"

CSV_DOWNLOAD_TIMEOUT_SECONDS = settings.csv_download_timeout_seconds
CSV_PROFILE_TIMEOUT_MILLISECONDS = settings.csv_profile_timeout_seconds * 1000
MODEL_TRAINING_TIMEOUT_MILLISECONDS = settings.model_training_timeout_seconds * 1000

GRADIO_SERVER_PROCESS = "gradio_server"


async def generate_request_id() -> dict:
    """Generate a unique request ID for model training workflow.

    This tool creates a new UUID-based request ID and sets up a dedicated directory in
    the training sandbox for a single ML workflow. The request ID must be passed to
    subsequent operations like uploading CSVs, training models, and deploying models.
    Note that reusing a request ID will overwrite the existing CSV, model, and
    deployment.

    Returns:
        dict: Response containing:
            - ok (bool): True if successful, False otherwise
            - response: Request ID (str) on success, or error message (str) on failure
    """
    request_id = str(uuid4())

    try:
        sandbox = await SandboxManager.get_trainer_sandbox()
        await sandbox.fs.mkdir(f"{BLAXEL_BASE_DIR}/{request_id}")
    except Exception:
        logger.exception("Failed to create request dir")
        return {
            "ok": False,
            "response": "Failed to create request dir",
        }

    return {
        "ok": True,
        "response": request_id,
    }


async def upload_and_profile_csv(request_id: str, file_url: str) -> dict:
    """Upload a CSV file from a URL and generate a data profile.

    This tool downloads a CSV file from the provided URL, saves it to the training
    sandbox, and runs the CSV profiler to analyze the dataset. The profile includes
    information about columns, data types, missing values, and statistics.

    Args:
        request_id (str): The request ID from generate_request_id()
        file_url (str): HTTP/HTTPS URL pointing to a CSV file

    Returns:
        dict: Response containing:
            - ok (bool): True if successful, False otherwise
            - response: Profile results (dict) on success, or error message (str) on
                failure
    """
    if not file_url.startswith(("http://", "https://")):
        return {
            "ok": False,
            "response": "Invalid file URL",
        }

    try:
        async with httpx.AsyncClient(timeout=CSV_DOWNLOAD_TIMEOUT_SECONDS) as client:
            response = await client.get(file_url)
            response.raise_for_status()

            is_extension_csv = urlparse(file_url).path.lower().endswith(".csv")

            content_type = response.headers.get("content-type", "").lower()
            is_type_csv = content_type.startswith(("text/csv", "application/csv"))

            if not (is_extension_csv or is_type_csv):
                return {
                    "ok": False,
                    "response": "URL does not point to a CSV file",
                }

            csv_bytes = response.content
    except httpx.TimeoutException:
        logger.exception("Download timed out—file may be too large")
        return {
            "ok": False,
            "response": "Download timed out—file may be too large",
        }
    except Exception:
        logger.exception("Failed to download CSV file")
        return {
            "ok": False,
            "response": "Failed to download CSV file",
        }

    try:
        sandbox = await SandboxManager.get_trainer_sandbox()
        if not await _is_valid_request_id(request_id, sandbox):
            return {
                "ok": False,
                "response": "Invalid request ID",
            }
    except Exception:
        logger.exception("Failed to verify request ID")
        return {
            "ok": False,
            "response": "Failed to verify request ID",
        }

    request_dir_path = f"{BLAXEL_BASE_DIR}/{request_id}"
    csv_path = f"{request_dir_path}/{CSV_FILENAME}"

    try:
        await sandbox.fs.write_binary(csv_path, csv_bytes)
    except Exception:
        logger.exception("Failed to save CSV file")
        return {
            "ok": False,
            "response": "Failed to save CSV file",
        }

    command_parts = [
        f"python {BLAXEL_BASE_DIR}/{CSV_PROFILER_SCRIPT}",
        f"--dataset_path '{csv_path}'",
        f"--output_dir '{request_dir_path}'",
    ]
    command = " ".join(command_parts)

    try:
        process = await sandbox.process.exec(
            {
                "command": command,
                "wait_for_completion": True,
                "timeout": CSV_PROFILE_TIMEOUT_MILLISECONDS,
            }
        )
        if process.status == "failed":
            logger.error(f"Failed to profile CSV - {request_id} - {process.logs}")
            return {
                "ok": False,
                "response": "Failed to profile CSV",
            }
    except Exception:
        logger.exception(f"Failed to profile CSV - {request_id}")
        return {
            "ok": False,
            "response": "Failed to profile CSV",
        }

    try:
        profile_result_str = await sandbox.fs.read(f"{request_dir_path}/{CSV_PROFILE}")
        profile_result = json.loads(profile_result_str)
    except Exception:
        logger.exception(f"Failed to read CSV profiling results - {request_id}")
        return {
            "ok": False,
            "response": "Failed to read CSV profiling results",
        }

    return {
        "ok": True,
        "response": profile_result,
    }


async def train_model(training_config_str: str) -> dict:
    """Train a machine learning model with the uploaded dataset.

    This tool trains a model using the CSV file that was previously uploaded and
    profiled. It requires a request ID from generate_request_id() and accepts various
    configuration parameters for model training, preprocessing, and evaluation.

    Args:
        training_config_str (str): JSON string containing training configuration
            parameters

    Returns:
        dict: Response containing:
            - ok (bool): True if successful, False otherwise
            - response: Training results with model metadata (dict) on success, or error
                message (str) on failure

    JSON Schema:
    {
        "request_id": {
            "title": "Request Id",
            "type": "string"
        },
        "task_type": {
            "description": "Type of machine learning task",
            "enum": [
                "regression",
                "classification"
            ],
            "title": "Task Type",
            "type": "string"
        },
        "target": {
            "description": "Target column name",
            "title": "Target",
            "type": "string"
        },
        "features": {
            "default": "",
            "description": "Comma-separated feature column names",
            "title": "Features",
            "type": "string"
        },
        "model_type": {
            "default": "auto",
            "description": "Type of machine learning model",
            "enum": [
                "auto",
                "random_forest",
                "svm",
                "linear"
            ],
            "title": "Model Type",
            "type": "string"
        },
        "n_estimators": {
            "default": 100,
            "description": "Number of estimators for random forest",
            "exclusiveMinimum": 0,
            "maximum": 1000,
            "title": "N Estimators",
            "type": "integer"
        },
        "svm_kernel": {
            "default": "rbf",
            "description": "Kernel type for SVM",
            "enum": [
                "linear",
                "poly",
                "rbf",
                "sigmoid"
            ],
            "title": "Svm Kernel",
            "type": "string"
        },
        "missing_strategy": {
            "default": "median",
            "description": "Strategy for handling missing values",
            "enum": [
                "drop",
                "mean",
                "median",
                "most_frequent"
            ],
            "title": "Missing Strategy",
            "type": "string"
        },
        "remove_outliers": {
            "default": false,
            "description": "Flag to remove outliers",
            "title": "Remove Outliers",
            "type": "boolean"
        },
        "scaler_type": {
            "default": "none",
            "description": "Type of feature scaler",
            "enum": [
                "none",
                "standard",
                "minmax",
                "robust"
            ],
            "title": "Scaler Type",
            "type": "string"
        },
        "test_size": {
            "default": 0.2,
            "description": "Proportion of dataset for testing",
            "exclusiveMaximum": 0.5,
            "minimum": 0.1,
            "title": "Test Size",
            "type": "number"
        },
        "random_state": {
            "default": 42,
            "description": "Random seed for reproducibility",
            "minimum": 0,
            "title": "Random State",
            "type": "integer"
        },
        "required": [
            "request_id",
            "task_type",
            "target"
        ],
        "type": "object"
    }
    """
    try:
        training_config_dict = json.loads(training_config_str)
        training_config = ModelTrainingConfig(**training_config_dict)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON format")
        return {
            "ok": False,
            "response": "Invalid JSON format",
        }
    except Exception as e:
        logger.exception(f"Validation error - {str(e)}")
        return {
            "ok": False,
            "response": f"Validation error - {str(e)}",
        }

    request_id = training_config.request_id

    try:
        sandbox = await SandboxManager.get_trainer_sandbox()
        if not await _is_valid_request_id(request_id, sandbox):
            return {
                "ok": False,
                "response": "Invalid request ID",
            }
    except Exception:
        logger.exception("Failed to verify request ID")
        return {
            "ok": False,
            "response": "Failed to verify request ID",
        }

    request_dir_path = f"{BLAXEL_BASE_DIR}/{request_id}"
    csv_path = f"{request_dir_path}/{CSV_FILENAME}"

    try:
        request_dir = await sandbox.fs.ls(request_dir_path)
        exists = any(file.name == CSV_FILENAME for file in request_dir.files)
        if not exists:
            return {
                "ok": False,
                "response": "CSV file not found—please upload the CSV first",
            }
    except Exception:
        logger.exception("Failed to access request dir")
        return {
            "ok": False,
            "response": "Failed to access request dir",
        }

    command_parts = [
        f"python {BLAXEL_BASE_DIR}/{MODEL_TRAINER_SCRIPT}",
        f"--dataset_path '{csv_path}'",
        f"--output_dir '{request_dir_path}/{PIPELINE_DIR}'",
        f"--task_type '{training_config.task_type}'",
        f"--target '{training_config.target}'",
        f"--model_type '{training_config.model_type}'",
        f"--n_estimators '{training_config.n_estimators}'",
        f"--svm_kernel '{training_config.svm_kernel}'",
        f"--missing_strategy '{training_config.missing_strategy}'",
        f"--scaler_type '{training_config.scaler_type}'",
        f"--test_size '{training_config.test_size}'",
        f"--random_state '{training_config.random_state}'",
    ]
    if training_config.features:
        command_parts.append(f"--features '{training_config.features}'")
    if training_config.remove_outliers:
        command_parts.append("--remove_outliers")
    command = " ".join(command_parts)

    try:
        process = await sandbox.process.exec(
            {
                "command": command,
                "wait_for_completion": True,
                "timeout": MODEL_TRAINING_TIMEOUT_MILLISECONDS,
            }
        )
        if process.status == "failed":
            logger.error(f"Failed to train model - {request_id} - {process.logs}")
            return {
                "ok": False,
                "response": "Failed to train model",
            }
    except Exception:
        logger.exception(f"Failed to train model - {request_id}")
        return {
            "ok": False,
            "response": "Failed to train model",
        }

    try:
        pipeline_meta_str = await sandbox.fs.read(
            f"{request_dir_path}/{PIPELINE_DIR}/{PIPELINE_META}"
        )
        pipeline_meta = json.loads(pipeline_meta_str)
    except Exception:
        logger.exception(f"Failed to read training results - {request_id}")
        return {
            "ok": False,
            "response": "Failed to read training results",
        }

    return {
        "ok": True,
        "response": pipeline_meta,
    }


async def deploy_model(request_id: str) -> dict:
    """Deploy a trained machine learning model to a live server.

    This tool takes the trained model from the training sandbox and deploys it to a
    dedicated server sandbox with a Gradio interface. The model must be trained first
    using train_model(). The deployment creates a web interface for making predictions.

    Args:
        request_id (str): The request ID from generate_request_id()

    Returns:
        dict: Response containing:
            - ok (bool): True if successful, False otherwise
            - response: Preview URL (str) on success, or error message (str) on failure
    """
    try:
        sandbox = await SandboxManager.get_trainer_sandbox()
        if not await _is_valid_request_id(request_id, sandbox):
            return {
                "ok": False,
                "response": "Invalid request ID",
            }
    except Exception:
        logger.exception("Failed to verify request ID")
        return {
            "ok": False,
            "response": "Failed to verify request ID",
        }

    pipeline_dir_path = f"{BLAXEL_BASE_DIR}/{request_id}/{PIPELINE_DIR}"

    try:
        pipeline_dir = await sandbox.fs.ls(pipeline_dir_path)
        file_names = [file.name for file in pipeline_dir.files]
        if PIPELINE_META not in file_names or PIPELINE_MODEL not in file_names:
            return {
                "ok": False,
                "response": "Model not found—please train the model first",
            }
    except Exception:
        logger.exception("Failed to access request dir")
        return {
            "ok": False,
            "response": "Failed to access request dir",
        }

    try:
        (
            server_sandbox,
            preview_url,
        ) = await SandboxManager.create_server_sandbox_with_preview(request_id)
    except Exception:
        logger.exception("Failed to create server sandbox")
        return {
            "ok": False,
            "response": "Failed to create server sandbox",
        }

    with contextlib.suppress(Exception):
        await server_sandbox.process.kill(GRADIO_SERVER_PROCESS)

    try:
        for file in pipeline_dir.files:
            data = await sandbox.fs.read_binary(f"{pipeline_dir_path}/{file.name}")
            await server_sandbox.fs.write_binary(f"{BLAXEL_BASE_DIR}/{file.name}", data)
    except Exception:
        logger.exception("Failed to copy model files")
        return {
            "ok": False,
            "response": "Failed to copy model files",
        }

    try:
        process = await server_sandbox.process.exec(
            {
                "name": GRADIO_SERVER_PROCESS,
                "command": (
                    f"python {BLAXEL_BASE_DIR}/{MODEL_SERVER_SCRIPT}"
                    f" --model_dir {BLAXEL_BASE_DIR}"
                ),
                "wait_for_ports": [SandboxManager.MODEL_SERVER_SANDBOX_PORT],
            }
        )
        if process.status != "running":
            return {
                "ok": False,
                "response": "Failed to start server",
            }
    except Exception:
        logger.exception("Failed to start server")
        return {
            "ok": False,
            "response": "Failed to start server",
        }

    return {
        "ok": True,
        "response": preview_url,
    }


async def _is_valid_request_id(request_id: str, sandbox: SandboxInstance) -> bool:
    try:
        await sandbox.fs.ls(f"{BLAXEL_BASE_DIR}/{request_id}")
        return True
    except Exception as e:
        if hasattr(e, "args") and "404" in e.args[0]:
            return False
        raise
