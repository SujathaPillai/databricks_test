# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Demo - Building and Logging a Retrieval Agent
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC In this demo, we'll explore how to build and log a production-ready retrieval agent using Databricks Mosaic AI. A retrieval agent combines the power of large language models with your organization's knowledge base to provide accurate, context-aware responses. We'll walk through testing vector search in the AI Playground, building an agent with LangChain, implementing MLflow tracing for observability, and registering the agent as a model for deployment.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - **Test** vector search functionality using the AI Playground UI for rapid prototyping.
# MAGIC - **Build** a retrieval agent using LangChain with vector search index.
# MAGIC - **Implement** MLflow tracing to monitor and debug agent interactions.
# MAGIC - **Register** the agent as a model in Model Registry.
# MAGIC
# MAGIC ## Requirements:
# MAGIC - A pre-created **vector search endpoint**. This is pre-created for you.
# MAGIC - **Serverless Compute (environment version 4)**. Follow the instructions [here](https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version) to select the appropriate environment version.
# MAGIC - Basic familiarity with LangChain and retrieval-augmented generation (RAG) concepts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC Run the code below to install required libraries and configure your classroom environment. This step ensures all dependencies are available and your workspace is ready for the demo.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04 $section="demo"

# COMMAND ----------

pip install databricks-langchain

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Testing Vector Search in AI Playground
# MAGIC
# MAGIC Before building our retrieval agent programmatically, we'll test the vector search index using the **AI Playground**. The AI Playground provides a **user-friendly interface for experimenting** with your vector search index, allowing you to quickly validate that your knowledge base is working correctly and understand how the retrieval system responds to different queries.
# MAGIC
# MAGIC This interactive testing phase is valuable for understanding retrieval quality, identifying potential issues with your embeddings or chunking strategy, and refining your approach before investing time in code development.
# MAGIC
# MAGIC **Dataset Information:** The vector search index contains data from a fictitious robot manufacturer for a robot named Orion. Documents include context-grounded answers sourced from internal design manuals, compliance documentation, and maintenance guides, enabling accurate retrieval of technical and regulatory information.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Configure Vector Search in Playground via Catalog Explorer
# MAGIC
# MAGIC Databricks now streamlines the process of testing your vector search index in the AI Playground.
# MAGIC
# MAGIC **Follow these steps to launch Playground with your vector search index:**
# MAGIC
# MAGIC 1. In your Databricks workspace, open **Catalog Explorer** and navigate to your vector search index.
# MAGIC    - Example: `{{{catalog_name}}}.{{{schema_name}}}.docs_chunked_index`
# MAGIC
# MAGIC 1. On the top right of the index details page, click the **Try in Playground** button.
# MAGIC
# MAGIC 1. The AI Playground will open automatically with your vector search index pre-configured as a retrieval tool.
# MAGIC
# MAGIC 1. Select your preferred large language model (LLM) in the Playground interface. We recommend using `GPT OSS 120B` model.
# MAGIC
# MAGIC    * Click on **Use Endpoint**.
# MAGIC
# MAGIC 1. Start entering queries to test retrieval and response quality.
# MAGIC
# MAGIC **Tip:** You could manually add your retrieval tool in the Playground too. But this method saves time as vector search is pre-configured, so you can select your LLM and start experimenting right away.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Test Queries and Experiment 
# MAGIC
# MAGIC Now that your vector search is configured, we'll test it with various queries to evaluate retrieval quality and response accuracy.
# MAGIC
# MAGIC **Follow these steps to test your retrieval system:**
# MAGIC
# MAGIC 1. In the chat interface, enter a question related to your knowledge base.
# MAGIC    - Example: *"How does the Orion motion controller maintain stability during high-speed movement?"*
# MAGIC    - Example: *How does Orion verify compliance with ISO 13849-1?"*
# MAGIC
# MAGIC 1. Submit the query and observe the response from the language model.
# MAGIC
# MAGIC
# MAGIC **Experimenting Best Practices:**
# MAGIC
# MAGIC 1. **Examine the retrieved context:** Check which documents or chunks were retrieved and confirm their relevance to your question.
# MAGIC
# MAGIC 1. **Evaluate the response quality:** Ensure the answer accurately reflects your knowledge base and is grounded in the retrieved documents.
# MAGIC
# MAGIC 1. **Test edge cases:** Ask questions outside your knowledge base and try different phrasings to assess robustness.
# MAGIC
# MAGIC 1. **Iterate and refine:** Adjust retrieval parameters if results are poor and note patterns in what works well.
# MAGIC
# MAGIC
# MAGIC Once you're satisfied with the retrieval quality in the Playground, you're ready to build the agent programmatically in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Building a Retrieval Agent with LangChain
# MAGIC
# MAGIC In this section, we'll build a retrieval agent using **LangChain**, which is the framework in scope for this demo. LangChain offers robust tools for connecting language models to your organization's data, enabling context-aware responses and flexible agent workflows. **While we use LangChain here to demonstrate best practices, you can apply similar retrieval agent patterns with other frameworks or libraries that support large language models and vector search.** The concepts and architecture are transferable—choose the technology that best fits your production requirements.
# MAGIC
# MAGIC We'll follow an **"agent as code"** approach by writing our agent implementation to a Python file (`agent.py`). This is the recommended method when logging the models with `mlflow`. The agent will use Unity Catalog's vector search as a tool, allowing it to dynamically retrieve relevant context when answering user questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Enable MLflow Tracing
# MAGIC
# MAGIC Before we start building the agent, **let's enable MLflow tracing for LangChain** so we can observe agent inputs, tool usage, and outputs in detail.
# MAGIC
# MAGIC MLflow offers robust tracing and observability for GenAI workflows, including LangChain, and supports many other frameworks and flavors. This broad integration allows you to monitor, debug, and analyze generative AI applications across diverse environments—all within a unified MLflow interface.
# MAGIC
# MAGIC **💡 Note:** MLflow tracing (`autolog()`) is enabled by default in classic compute, but on serverless compute, you must enable it manually.
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Create a LangChain Agent

# COMMAND ----------

llm_endpoint_name = "databricks-gpt-oss-120b"

# COMMAND ----------

from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langgraph.checkpoint.memory import InMemorySaver


def build_agent(llm_endpoint:str, index_name: str, num_results: int = 3):
    model = ChatDatabricks(
        endpoint=llm_endpoint,
        max_tokens=500,
    )

    vs_tool = VectorSearchRetrieverTool(
        name="orion_knowledge_search",
        index_name=index_name,
        description="Search Orion knowledge base for relevant information",
        num_results=num_results,
    )

    # Optional: use a in memory saver to save the agent's state
    checkpointer = InMemorySaver()

    system_prompt = """You are the Orion Knowledge Assistant (OKA). Respond in a clear, professional, and factual tone appropriate for engineers and technical staff. Use only verified information from Orion's internal documents, and include source references when available. If the answer cannot be found, clearly state that and suggest related sections or next steps. Do not speculate, make assumptions, or provide information outside the provided context."""

    agent = create_agent(
        model=model, 
        tools=[vs_tool], 
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        )
    return agent

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "databricks-demo-4"}}

# Quick smoke test
agent = build_agent(llm_endpoint_name, vs_index_name, 3)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is Orion?"}]},
    config=config
)
print(response['messages'][-1].content)


# COMMAND ----------

# MAGIC %md
# MAGIC ### B3. Review MLflow Tracing UI
# MAGIC
# MAGIC The MLflow Tracing UI provides a comprehensive view of your agent's execution and tool usage. The output above shows tracing UI. Alternatively, if you want to review traces of an experiment that you run before, you can view in the **Experiments** page.
# MAGIC To access traces of an experiment, select an experiment and navigate to the **Traces** tab within your experiment.
# MAGIC
# MAGIC - The **Summary** tab displays high-level information for each trace, including inputs, outputs, and trace metadata.
# MAGIC - The **Details & Timeline** tab offers a breakdown of every step in the trace, showing all LLM invocations, tools called, results returned from tools, and the final generated output. This helps you understand the agent's reasoning and data flow.
# MAGIC - On the left side, clicking on the timeline icon, you can enable the execution of **timeline view** to visualize the duration of each step, making it easy to identify bottlenecks or performance issues.
# MAGIC - Selecting an individual trace reveals additional details on the right panel. If any errors occurred, you can inspect them in the **Events** tab, which lists error messages and relevant context.
# MAGIC
# MAGIC
# MAGIC Reviewing these tabs helps you validate agent behavior, debug issues, and optimize your agent development workflows with clear, actionable insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Log the Agent to Model registry
# MAGIC
# MAGIC In this section we will show to log a model to model registry. First, we need to abstract the agent code from this notebook by creating a file with all agent code. Also, as the agent code can be run in any environment, we will create a `.yaml` config file. This will be logged along with the agent code.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Create `agent-config` 

# COMMAND ----------

import yaml

def create_config(llm_endpoint_name: str, index_name: str, num_results: int = 3):
    """Create a minimal YAML config for the agent."""
    config = {
        "llm_endpoint_name": llm_endpoint_name,
        "vector_search": {
            "index_name": index_name,
            "num_results": num_results
        }
    }
    return config


# Create config file
llm_endpoint_name = "databricks-gpt-oss-120b"

agent_config = create_config(llm_endpoint_name, vs_index_name)

# Write YAML file (for agent.py to read later)
with open("agent-config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(agent_config, f, sort_keys=False)

print("✅ Config file written: agent-conf.yaml")
print(yaml.safe_dump(agent_config, sort_keys=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Write Agent Code to the File
# MAGIC
# MAGIC We will create an `agent.py` file to be used when logging the agent. This file includes;
# MAGIC - Loading config file 
# MAGIC - LangChain code that we created in the previous step
# MAGIC - Predict and Response format based on mlflow API
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import os
# MAGIC from uuid import uuid4
# MAGIC from typing import Any, Dict, List
# MAGIC
# MAGIC import yaml
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC from langchain.agents import create_agent
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC from langgraph.checkpoint.memory import InMemorySaver
# MAGIC
# MAGIC # Load agent configuration from YAML file
# MAGIC def _load_config(path: str = "agent-config.yaml") -> Dict[str, Any]:
# MAGIC     if not os.path.exists(path):
# MAGIC         raise FileNotFoundError(f"Config file not found at '{path}'")
# MAGIC     with open(path, "r", encoding="utf-8") as f:
# MAGIC         cfg = yaml.safe_load(f) or {}
# MAGIC     llm_endpoint = cfg.get("llm_endpoint_name")
# MAGIC     vs = cfg.get("vector_search", {}) or {}
# MAGIC     index_name = vs.get("index_name")
# MAGIC     num_results = int(vs.get("num_results", 3))
# MAGIC     if not llm_endpoint or not index_name:
# MAGIC         raise ValueError("Missing 'llm_endpoint_name' or 'vector_search.index_name' in agent-config.yaml")
# MAGIC     return {
# MAGIC         "llm_endpoint_name": llm_endpoint,
# MAGIC         "vs_index_name": index_name,
# MAGIC         "vs_num_results": num_results,
# MAGIC     }
# MAGIC
# MAGIC # Build LangChain agent with LLM and vector search tool
# MAGIC def build_agent(llm_endpoint: str, index_name: str, num_results: int = 3):
# MAGIC     model = ChatDatabricks(endpoint=llm_endpoint, max_tokens=500)
# MAGIC     vs_tool = VectorSearchRetrieverTool(
# MAGIC         name="orion_knowledge_search",
# MAGIC         index_name=index_name,
# MAGIC         description="Search Orion knowledge base for relevant information",
# MAGIC         num_results=num_results,
# MAGIC     )
# MAGIC     checkpointer = InMemorySaver()
# MAGIC     system_prompt = (
# MAGIC         "You are the Orion Knowledge Assistant (OKA). Respond in a clear, professional, and factual tone "
# MAGIC         "appropriate for engineers and technical staff. Use only verified information from Orion's internal "
# MAGIC         "documents, and include source references when available. If the answer cannot be found, clearly state "
# MAGIC         "that and suggest related sections or next steps. Do not speculate, make assumptions, or provide "
# MAGIC         "information outside the provided context."
# MAGIC     )
# MAGIC     agent = create_agent(
# MAGIC         model=model,
# MAGIC         tools=[vs_tool],
# MAGIC         system_prompt=system_prompt,
# MAGIC         checkpointer=checkpointer,
# MAGIC     )
# MAGIC     return agent
# MAGIC
# MAGIC # Extract last user message from conversation
# MAGIC def _last_user_text(messages: List[Dict[str, Any]]) -> str:
# MAGIC     user_msgs = [m for m in messages if (m.get("role") == "user")]
# MAGIC     return str(user_msgs[-1].get("content", "")) if user_msgs else str(messages[-1].get("content", ""))
# MAGIC
# MAGIC # MLflow ResponsesAgent implementation for LangChain agent
# MAGIC class LangChainResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         cfg = _load_config()
# MAGIC         self._cfg = cfg
# MAGIC         self._agent = build_agent(
# MAGIC             llm_endpoint=cfg["llm_endpoint_name"],
# MAGIC             index_name=cfg["vs_index_name"],
# MAGIC             num_results=cfg["vs_num_results"],
# MAGIC         )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         msgs = [m.model_dump() for m in request.input]  # [{'role': 'user'|'assistant', 'content': '...'}, ...]
# MAGIC         _ = _last_user_text(msgs) if msgs else ""
# MAGIC
# MAGIC         # Generate a unique thread ID for each prediction
# MAGIC         thread_id = f"oka-{uuid4()}"
# MAGIC
# MAGIC         result = self._agent.invoke(
# MAGIC             {"messages": msgs},
# MAGIC             config={"configurable": {"thread_id": thread_id}},
# MAGIC         )
# MAGIC         # Extract agent response text
# MAGIC         try:
# MAGIC             text = result["messages"][-1].content
# MAGIC         except Exception:
# MAGIC             text = str(result)
# MAGIC
# MAGIC         return ResponsesAgentResponse(
# MAGIC             output=[self.create_text_output_item(text, str(uuid4()))],
# MAGIC             custom_outputs=request.custom_inputs,
# MAGIC         )
# MAGIC
# MAGIC # Set the model for mlflow. This is needed when using agent-as-code approach
# MAGIC AGENT = LangChainResponsesAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C3. Import the Agent from File

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from agent import AGENT as agent

mlflow.langchain.autolog()

response  = agent.predict(
    {"input": [{"role": "user", "content": "What is Orion?"}]}
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Logging and Registering the Model to Model registry
# MAGIC
# MAGIC After loading and testing our agent we can go ahead and log the agent to model registry. This allows us to version the model, give aliases, tag and manage permissions. We can deploy the models from model registry using model serving. Note that the deployment of agents is not in the scope of this module.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Define Model Resources
# MAGIC
# MAGIC Before logging the agent to MLflow, we need to define the **resources** that the agent depends on. Resources represent external dependencies such as vector search indexes, serving endpoints, tables, or functions that the agent uses during inference.
# MAGIC
# MAGIC By explicitly declaring these resources, MLflow can:
# MAGIC * **Track dependencies** for reproducibility and lineage.
# MAGIC * **Validate availability** of required resources before deployment.
# MAGIC * **Enable proper permissions** and access control in production environments.
# MAGIC
# MAGIC For our Orion Knowledge Assistant, we'll define two key resources:
# MAGIC 1. **DatabricksVectorSearchIndex**: The vector search index containing Orion's knowledge base.
# MAGIC 1. **DatabricksServingEndpoint**: The LLM endpoint used for generating responses.
# MAGIC
# MAGIC These resources ensure that when the model is deployed, it has access to the necessary infrastructure components.
# MAGIC
# MAGIC **🚨 Important:** Before proceeding, ensure you have imported the classroom setup config file. This step is required because the Python kernel was restarted in the previous section, and you need to reload the catalog and schema names for correct resource configuration.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-Common

# COMMAND ----------

# DBTITLE 1,Define Agent Resources
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

# Redefine variables after Python restart
vs_index_name = f"{catalog}.{schema}.docs_chunked_index"
llm_endpoint_name = "databricks-gpt-oss-120b"

# Define the resources that the agent depends on
resources = [
    DatabricksVectorSearchIndex(index_name=vs_index_name),
    DatabricksServingEndpoint(endpoint_name=llm_endpoint_name)
]

print("Resources defined:")
for resource in resources:
    print(f"  - {resource}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Log the Agent Model with MLflow
# MAGIC
# MAGIC Now that we've defined our agent's resources, we'll **log the agent as a model** using MLflow. Logging captures the agent code, configuration, dependencies, and metadata in a structured format that can be versioned, tracked, and deployed.
# MAGIC
# MAGIC When we log a model with MLflow, we create a **run** that records:
# MAGIC * **Model artifacts**: The agent code (`agent.py`) and configuration (`agent-config.yaml`).
# MAGIC * **Dependencies**: Python packages required to run the agent.
# MAGIC * **Resources**: External dependencies like vector search indexes and serving endpoints.
# MAGIC * **Input/output examples**: Sample data demonstrating the expected model interface.
# MAGIC * **Metadata and tags**: Information about the model version, purpose, and lineage.
# MAGIC
# MAGIC This logged model becomes a **reproducible artifact** that can be loaded, tested, and deployed in any environment with the same dependencies and resources available.

# COMMAND ----------

# DBTITLE 1,Log Agent Model to MLflow
import mlflow
from importlib.metadata import version as get_version

# Define model name and tags
model_name = "orion_knowledge_assistant"
tags_to_register = {
    "model_type": "retrieval_agent",
    "framework": "langchain",
    "use_case": "orion_knowledge_base"
}

# Create an input example for the model signature
input_example = {
    "input": [
        {"role": "user", "content": "What is Orion?"}
    ]
}

# Start an MLflow run and log the model
with mlflow.start_run():
    mlflow.set_tags(tags_to_register)
    
    logged_agent_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model="agent.py",
        code_paths=["agent-config.yaml"],
        input_example=input_example,
        pip_requirements=[
            f"databricks-vectorsearch=={get_version('databricks-vectorsearch')}",
            f"databricks-langchain=={get_version('databricks-langchain')}",
            f"langchain=={get_version('langchain')}",
            f"mlflow=={get_version('mlflow')}",
        ],
        resources=resources
    )
    
    # Save the model URI for later use
    model_uri = logged_agent_info.model_uri
    
print(f"✅ Model logged successfully!")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Register the Model to Unity Catalog
# MAGIC
# MAGIC Now that we've logged our agent model, we'll **register it to Unity Catalog's Model Registry**. While logging and registering may seem similar, they serve distinct purposes in the MLOps lifecycle.
# MAGIC
# MAGIC **Understanding the Difference:**
# MAGIC
# MAGIC * **Logging** creates a versioned artifact within an MLflow experiment run. It captures the model code, dependencies, and metadata at a specific point in time. Logged models are tied to individual runs and are primarily used for experimentation and development.
# MAGIC
# MAGIC * **Registering** promotes a logged model to the Model Registry, making it a **managed, governed asset** with a unique name in Unity Catalog. Registered models support:
# MAGIC   - **Version management**: Track multiple versions of the same model.
# MAGIC   - **Aliases**: Assign labels like `Champion` or `Challenger` to specific versions.
# MAGIC   - **Governance**: Apply permissions, tags, and lineage tracking.
# MAGIC   - **Deployment**: Serve models directly from the registry to production endpoints.
# MAGIC
# MAGIC By registering our agent to Unity Catalog, we transform it from an experimental artifact into a production-ready asset that can be discovered, governed, and deployed across the organization.

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
# Set the registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define the fully qualified model name in Unity Catalog
UC_MODEL_NAME = f"{catalog}.{schema}.orion_knowledge_assistant"

# Register the model to Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=model_uri, 
    name=UC_MODEL_NAME
)

print(f"✅ Model registered successfully to Unity Catalog!")
print(f"Model Name: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D4. Test Model Inference
# MAGIC
# MAGIC With our agent registered in Unity Catalog, we can now **load and test it** to verify that it works correctly. Loading a model from the registry ensures that we're using the exact version that was logged, with all its dependencies and configurations intact.
# MAGIC
# MAGIC We'll demonstrate two approaches:
# MAGIC 1. **Loading from the model URI**: Using the URI from the logging step.
# MAGIC 1. **Loading from Unity Catalog**: Using the fully qualified model name.
# MAGIC
# MAGIC Both methods allow us to invoke the agent with test inputs and validate that it retrieves relevant context from the vector search index and generates appropriate responses.

# COMMAND ----------

# DBTITLE 1,Load and Test the Model
# Load the model from the model URI (pyfunc flavor)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# Use the input example that was logged with the model
input_data = pyfunc_model.input_example

print("Input data:")
print(input_data)
print("\n" + "="*50 + "\n")

# Make a prediction using the loaded model
result = mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)

print("Agent Response:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D5. Explore the Model Registry UI
# MAGIC
# MAGIC With our agent successfully registered in Unity Catalog, we can now **explore the Model Registry UI** to understand how to manage, monitor, and govern our model. The Model Registry provides a centralized interface for tracking model versions, lineage, artifacts, and performance.
# MAGIC
# MAGIC **Follow these steps to explore your registered model:**
# MAGIC
# MAGIC 1. Navigate to your model in Catalog Explorer:
# MAGIC    - Go to **Catalog Explorer** in your Databricks workspace.
# MAGIC    - Navigate to `{{{catalog_name}}}.{{{schema_name}}}.orion_knowledge_assistant`.
# MAGIC    - Click on the most recent version of the model.
# MAGIC
# MAGIC 1. Explore the four main tabs:
# MAGIC
# MAGIC    * **Overview**: This tab displays key information about your model version:
# MAGIC      - **Metrics**: Any metrics logged with the model during training or evaluation.
# MAGIC      - **Activity Log**: A chronological record of changes, updates, and deployments.
# MAGIC      - **Model Signature**: The input/output schema defining the expected data format.
# MAGIC      - **Version Information**: Details about this specific version, including creation date and creator.
# MAGIC      - **Active Endpoints**: Any serving endpoints currently using this model version.
# MAGIC      - **Tags**: Custom metadata tags for organization and discovery.
# MAGIC
# MAGIC    * **Lineage**: Shows upstream sources and downstream consumers for model version tracking.
# MAGIC
# MAGIC    * **Artifacts**: Lists all files and dependencies registered with the model.
# MAGIC
# MAGIC    * **Traces**: Displays detailed execution traces for agent invocations and debugging.
# MAGIC
# MAGIC The Model Registry UI provides a comprehensive view of your model's lifecycle, making it easy to manage versions, track lineage, and ensure governance across your organization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Summary
# MAGIC
# MAGIC In this demo, we explored the complete workflow for building, logging, and registering a production-ready **retrieval agent** using Databricks Mosaic AI. We started by **testing vector search in the AI Playground**, then built a LangChain-based agent that combines LLMs with a knowledge base stored in Vector Search for context-aware responses. We utilized **MLflow tracing** for observability, adopted an **"agent as code"** approach with Python files and YAML configuration, and logged our agent to MLflow with explicit **resource dependencies**. Finally, we registered the agent to Unity Catalog's Model Registry, transforming it into a **governed, versioned asset ready for production deployment**.
# MAGIC
# MAGIC **Key Takeaways:**
# MAGIC
# MAGIC * **Test vector search indexes** in the **AI Playground** before writing code to validate retrieval quality and response accuracy.
# MAGIC * **Enable MLflow tracing** to monitor agent behavior, including tool usage, LLM calls, and response generation.
# MAGIC * **Adopt an "agent as code" approach** by abstracting logic into **Python files** with **YAML configuration** for portability and maintainability.
# MAGIC * **Define explicit resource dependencies** (vector search indexes, serving endpoints) when logging models to **MLflow**.
# MAGIC * **Register models to Unity Catalog's Model Registry** to enable version management, governance, lineage tracking, and production deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
