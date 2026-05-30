<img
  src="https://res.cloudinary.com/djeszvhyx/image/upload/v1772734058/logo_lrd2ts.png"
  width="200px"
/>

# BlaxML MCP

Autonomous Machine Learning MCP server powered by [Blaxel](https://blaxel.ai)

<p>
  <img
    src="https://img.shields.io/badge/license-MIT-FF8B3D?style=for-the-badge&logoColor=white"
  />
  &nbsp;
  <img
    src="https://img.shields.io/badge/Python-3.11+-FF8B3D?style=for-the-badge&logo=python&logoColor=white"
  />
</p>

## 1. Demo Video

[YouTube](https://youtu.be/6oU_Xs1p5xM)

## 2. What is 🏀 BlaxML MCP Server?

`BlaxML MCP Server` = `Blaxel` + `Autonomous Machine Learning` + `MCP Server`

The BlaxML MCP Server is a fully autonomous machine learning system built on top of Blaxel's sandboxing infrastructure, enabling AI agents to transform raw CSV datasets into completely trained and deployed machine learning models. And it doesn’t stop there; every deployed model comes with a built-in interactive Gradio GUI and a REST API endpoint, making inference accessible for both humans and applications.

## 3. Key Features

1. **Automated ML Workflow**: BlaxML handles everything except the agent’s reasoning. Once the agent decides what needs to be done, BlaxML takes over the entire workflow from profiling the CSV, training the model, managing sandboxes, and deploying the model automatically. Each deployed model runs inside its own isolated Blaxel sandbox, ensuring strong security and clean separation.

2. **Powerful Dataset Profiler**: BlaxML includes a dedicated CSV profiler that provides rich statistical insights, distributions, missing-value reports, outlier detection reports, and schema summaries. These insights give the agent the precise context it needs to choose the most effective training strategy.

3. **Flexible Model Trainer**: BlaxML includes a dedicated model trainer that supports Random Forest, SVM, and linear models for both regression and classification tasks. Agents define the task type, target column, feature set, and additional configuration options related to preprocessing, dataset splitting, and model-specific settings.

4. **Automatic Model Deployment**: BlaxML doesn’t just train models; it deploys them on demand. When the agent requests a deployment, BlaxML provisions a fresh, isolated Blaxel sandbox, transfers all model artifacts into it, launches an inference server, and creates a preview URL that is returned to the user.

5. **Powerful Gradio Inference UI**: BlaxML’s inference UI is fully type-aware and schema-driven and is exposed through both an interactive Gradio UI and a REST API. When a model is trained, BlaxML saves metadata for every feature's data types, valid categorical values, numeric ranges, and smart imputed defaults, which allows the deployed Gradio interface to automatically adapt; for example, if your CSV had a gender column with values male and female, the UI automatically renders a dropdown instead of a text box. Numeric fields display their observed training ranges, ideal for production use. The result is a self-adapting, intelligent inference UI with no manual setup required.

6. **Optimized for Speed**: You might think creating sandboxes, transferring files, profiling a CSV, training a model, and spinning up an inference server would take a lot of time, but it doesn’t. Blaxel sandboxes are lightweight, Docker-based environments that start up extremely fast, and both the profiler and trainer are built to run quickly even in non-GPU environments. The whole BlaxML workflow feels surprisingly snappy from start to finish.

## 4. Architecture

BlaxML uses a two-sandbox architecture built on Blaxel.

### Components

BlaxML is built around three core components:

1. **CSV Profiler**: Profiles the dataset and generates rich statistical insights.

2. **Model Trainer**: Trains machine learning models based on agent-provided configuration options.

3. **Inference Server**: Serves the trained model with an intelligent, schema-aware UI and REST inference endpoint.

### Sandbox Design

BlaxML runs using two sandbox images:

1. **BlaxML Model Trainer Image**: This image contains both the CSV profiler and model trainer. It runs inside a shared training sandbox reused across workflows. This sandbox handles:
   - CSV profiling
   - Model training
   - Metadata generation
   - Transfer of artifacts for deployment

2. **BlaxML Model Server Image**: This image contains the inference server. The model server runs inside a new sandbox created per deployment. Each deployed model runs in its own isolated server environment with:
   - Trained model artifacts
   - Dynamic Gradio UI

### Workflow

1. **Agent Requests a New Workflow**: The agent calls `generate_request_id`, and BlaxML prepares a fresh workspace.

2. **Agent Uploads the CSV**: The agent calls `upload_and_profile_csv`, sending the dataset to the training sandbox. BlaxML validates the file and stores it under `/blaxel/<request_id>/dataset.csv`.

3. **BlaxML Profiles the Dataset**: The CSV profiler runs and produces `csv_profile.json`, containing column summaries, data types, distributions, and missing-value insights.

4. **Agent Decides Training Parameters**: Using the profiler’s output, the agent selects the configuration (task type, features, target, preprocessing options, etc.) and calls `train_model`.

5. **BlaxML Trains the Model**: The model trainer loads the dataset, applies preprocessing, trains the model, saves artifacts, and generates metadata for the inference UI.

6. **Agent Requests Deployment**: The agent calls `deploy_model`, and BlaxML provisions a new model server sandbox for deployment.

7. **BlaxML Deploys the Model**: The trained model and metadata are transferred to the server sandbox, the Gradio inference server launches, and a public preview URL is returned.

8. **User Runs Inference**: The deployed model is accessible through a schema-aware Gradio web UI and a REST inference endpoint, both powered by the isolated model server sandbox.

> Agent thinks. BlaxML does the work. Blaxel runs the work safely and fast.

## 5. Quick Start

### Install Blaxel CLI

Follow the official guide:

https://docs.blaxel.ai/cli-reference/introduction

### Login to Blaxel

```bash
bl login
```

### Clone the Repository

```bash
git clone https://github.com/aniketppanchal/blax-ml-mcp
cd blax-ml-mcp
```

### Deploy Sandbox Images

```bash
bl deploy -d sandboxes/blax-ml-model-trainer
bl deploy -d sandboxes/blax-ml-model-server
```

### Setup Environment Variables

```bash
export BL_API_KEY=your-api-key
export BL_WORKSPACE=your-workspace-name
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Run the MCP Server

```bash
python -m blax_ml_mcp.main
```

Your MCP server will now be available at:

```
http://<your-public-mcp-url>/gradio_api/mcp/
```

## 6. Configuration

BlaxML can be configured using environment variables.

| Environment Variable                         | Default                        | Description                                                                                       |
| -------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| `BL_API_KEY`                                 | required                       | Blaxel API key.                                                                                   |
| `BL_WORKSPACE`                               | required                       | Blaxel workspace name.                                                                            |
| `BL_REGION`                                  | `us-pdx-1`                     | Blaxel region. Allowed values: `us-pdx-1`, `us-was-1`, `eu-lon-1`, `eu-fra-1`.                    |
| `GRADIO_SERVER_NAME`                         | `127.0.0.1`                    | Host name used by the MCP server. Set to `0.0.0.0` to allow external access.                      |
| `GRADIO_SERVER_PORT`                         | `7860`                         | Port used by the MCP server.                                                                      |
| `GRADIO_SHARE`                               | `False`                        | Whether to generate a public shareable URL for the MCP server.                                    |
| `BLAX_ML_MCP_MODEL_TRAINER_SANDBOX_NAME`     | `blax-ml-model-trainer`        | Name of the model trainer sandbox.                                                                |
| `BLAX_ML_MCP_MODEL_TRAINER_SANDBOX_IMAGE`    | `blax-ml-model-trainer:latest` | Container image used to create the model trainer sandbox.                                         |
| `BLAX_ML_MCP_MODEL_TRAINER_SANDBOX_MEMORY`   | `4096`                         | Memory allocation (MB) for the model trainer sandbox. Allowed values: `2048`, `4096`, `8192`.     |
| `BLAX_ML_MCP_MODEL_SERVER_SANDBOX_IMAGE`     | `blax-ml-model-server:latest`  | Container image used to create the model server sandbox.                                          |
| `BLAX_ML_MCP_MODEL_SERVER_SANDBOX_MEMORY`    | `4096`                         | Memory allocation (MB) for the model server sandbox. Allowed values: `2048`, `4096`, `8192`.      |
| `BLAX_ML_MCP_MODEL_SERVER_SANDBOX_TTL`       | `1d`                           | Lifetime of the model server sandboxes before automatic shutdown. Supported units: `h`, `d`, `w`. |
| `BLAX_ML_MCP_CSV_DOWNLOAD_TIMEOUT_SECONDS`   | `300`                          | Maximum time (seconds) allowed for downloading a CSV dataset.                                     |
| `BLAX_ML_MCP_CSV_PROFILE_TIMEOUT_SECONDS`    | `60000`                        | Maximum time (milliseconds) allowed for the dataset profiling.                                    |
| `BLAX_ML_MCP_MODEL_TRAINING_TIMEOUT_SECONDS` | `300000`                       | Maximum time (milliseconds) allowed for the model training.                                       |

## 7. Integrations

### Claude Desktop (Requires npx)

Add/update your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "py-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://<your-public-mcp-url>/gradio_api/mcp/"
      ]
    }
  }
}
```

### LangChain

```bash
pip install langchain langchain-mcp-adapters langchain-openai
```

```python
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "blax-ml-mcp": {
                "transport": "http",
                "url": "https://<your-public-mcp-url>/gradio_api/mcp/",
            },
        }
    )

    tools = await client.get_tools()
    agent = create_agent(
        model="openai:gpt-5.1",
        tools=tools,
    )

    query = "Train and deploy a model using this CSV: https://<your-dataset-url>.csv"
    response = await agent.ainvoke({"messages": query})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

## 8. License

- [MIT License](LICENSE)
