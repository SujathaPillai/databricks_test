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
# MAGIC # Lab - Building and Registering a Retrieval Agent
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC In this lab, you will build and register a production-ready retrieval agent using Databricks Mosaic AI. You will create a LangChain-based agent that combines large language models with a vector search index, implement MLflow tracing for observability, and register the agent to Unity Catalog's Model Registry with proper versioning and aliases.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC 1. Enable MLflow tracing for LangChain to monitor agent behavior.
# MAGIC 1. Build a retrieval agent using LangChain with vector search integration.
# MAGIC 1. Analyze agent execution traces to identify performance characteristics.
# MAGIC 1. Write agent code to a Python file following the "agent as code" pattern.
# MAGIC 1. Register the agent model to Unity Catalog with an alias.
# MAGIC 1. Test the registered model to validate functionality.
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC - A pre-created **vector search endpoint**. This is pre-created for you.
# MAGIC - **Serverless Compute (environment version 4)**. Follow the instructions [here](https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version) to select the appropriate environment version.
# MAGIC
# MAGIC
# MAGIC **📌 Your Task: In this lab, your task is to replace `<FILL_IN>` sections with appropriate code.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC Run the code below to install required libraries and configure your classroom environment. This step ensures all dependencies are available and your workspace is ready for the demo.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04 $section="lab"

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Building a Retrieval Agent with LangChain
# MAGIC
# MAGIC In this section, you will build a retrieval agent using **LangChain**. The agent will use Unity Catalog's vector search as a tool, allowing it to dynamically retrieve relevant context when answering user questions.
# MAGIC
# MAGIC You will follow an **"agent as code"** approach by writing your agent implementation to a Python file (`agent.py`). This is the recommended method when logging models with MLflow.
# MAGIC
# MAGIC **Dataset Information:** The vector search index contains data from a fictitious robot manufacturer for a robot named Orion. Documents include context-grounded answers sourced from internal design manuals, compliance documentation, and maintenance guides.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Task 1 - Enable MLflow Tracing
# MAGIC
# MAGIC Before building the agent, you need to **enable MLflow tracing for LangChain** so you can observe agent inputs, tool usage, and outputs in detail.
# MAGIC
# MAGIC **Your Task:**
# MAGIC
# MAGIC 1. Enable MLflow autologging for LangChain using the appropriate method.

# COMMAND ----------

# Import mlflow and enable autologging for LangChain

<FILL_IN>

# COMMAND ----------

# DBTITLE 1,Task 1 - ANSWER
# MAGIC %skip
# MAGIC import mlflow
# MAGIC mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Task 2 - Create a LangChain Agent
# MAGIC
# MAGIC Now you will create a LangChain agent that uses a vector search retriever tool to access the Orion knowledge base.
# MAGIC
# MAGIC **Your Task:**
# MAGIC
# MAGIC 1. Define the LLM endpoint name as `"databricks-gpt-oss-20b"`.
# MAGIC 1. Complete the `build_agent` function by:
# MAGIC    * Creating a `ChatDatabricks` model with the provided endpoint and `max_tokens=300`.
# MAGIC    * Creating a `VectorSearchRetrieverTool` with:
# MAGIC      - `name="orion_knowledge_search_lab"`
# MAGIC      - `index_name` from the parameter
# MAGIC      - `description="Search Orion knowledge base for relevant information"`
# MAGIC      - `num_results` from the parameter (5 results)
# MAGIC    * Using the provided system prompt.
# MAGIC    * Creating an agent using `create_agent` with the model, tools list, system prompt, and checkpointer.
# MAGIC 1. Test the agent with the question "What is Orion?"

# COMMAND ----------

from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool

llm_endpoint_name = <FILL_IN>

def build_agent(llm_endpoint:str, index_name: str, num_results: int = 3):
    model = <FILL_IN>

    vs_tool = <FILL_IN>

    # Optional: use an in memory saver to save the agent's state
    checkpointer = <FILL_IN>

    system_prompt = """You are the Orion Knowledge Assistant (OKA). Respond in a clear, professional, and factual tone appropriate for engineers and technical staff. Use only verified information from Orion's internal documents, and include source references when available. If the answer cannot be found, clearly state that and suggest related sections or next steps. Do not speculate, make assumptions, or provide information outside the provided context."""

    agent = <FILL_IN>
    return agent

# Quick smoke test
agent = build_agent(llm_endpoint_name, index_name, 3)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is Orion?"}]}
)
print(response['messages'][-1].content)

# COMMAND ----------

# DBTITLE 1,Task 2 - ANSWER
# MAGIC %skip
# MAGIC from langchain.agents import create_agent
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC
# MAGIC llm_endpoint_name = "databricks-gpt-oss-20b"
# MAGIC
# MAGIC def build_agent(llm_endpoint:str, index_name: str, num_results: int = 5):
# MAGIC     model = ChatDatabricks(
# MAGIC         endpoint=llm_endpoint,
# MAGIC         max_tokens=300,
# MAGIC     )
# MAGIC
# MAGIC     vs_tool = VectorSearchRetrieverTool(
# MAGIC         name="orion_knowledge_search_lab",
# MAGIC         index_name=index_name,
# MAGIC         description="Search Orion knowledge base for relevant information",
# MAGIC         num_results=num_results,
# MAGIC     )
# MAGIC
# MAGIC     system_prompt = """You are the Orion Knowledge Assistant (OKA). Respond in a clear, professional, and factual tone appropriate for engineers and technical staff. Use only verified information from Orion's internal documents, and include source references when available. If the answer cannot be found, clearly state that and suggest related sections or next steps. Do not speculate, make assumptions, or provide information outside the provided context."""
# MAGIC
# MAGIC     agent = create_agent(
# MAGIC         model=model, 
# MAGIC         tools=[vs_tool], 
# MAGIC         system_prompt=system_prompt,
# MAGIC     )
# MAGIC     return agent
# MAGIC
# MAGIC # Quick smoke test
# MAGIC agent = build_agent(llm_endpoint_name, vs_index_name, 3)
# MAGIC
# MAGIC response = agent.invoke(
# MAGIC     {"messages": [{"role": "user", "content": "What is Orion?"}]}
# MAGIC )
# MAGIC print(response['messages'][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Task 3 - Review MLflow Tracing UI
# MAGIC
# MAGIC The MLflow Tracing UI provides a comprehensive view of your agent's execution and tool usage. The output above shows the tracing UI.
# MAGIC
# MAGIC **Your Task:**
# MAGIC
# MAGIC 1. Find the name of the longest step in the execution timeline.
# MAGIC 1. Identify which tool or model was responsible for that step.
# MAGIC 1. Check the total number of tokens used in the agent run (input, output, and total tokens).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Logging and Registering the Agent to Model Registry
# MAGIC
# MAGIC In this section, you will prepare the agent for production by writing it to a Python file and registering it to Unity Catalog's Model Registry. You will use the configuration file from the demo notebook instead of creating a new one.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Write Agent Code and Configuration Files
# MAGIC
# MAGIC The following steps create the required files for logging and registering your agent:
# MAGIC
# MAGIC - **`agent-lab.py`**: This file wraps the agent logic using MLflow `pyfunc`, enabling support for `ResponseAgent` requests.
# MAGIC - **`agent-config-lab.yaml`**: This file contains the agent configuration.
# MAGIC
# MAGIC No action is needed in this section. Simply run the code to generate the necessary files.
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent-lab.py
# MAGIC import os
# MAGIC from typing import Any, Dict, List
# MAGIC
# MAGIC import yaml
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC from langchain.agents import create_agent
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC
# MAGIC # Load agent configuration from YAML file
# MAGIC def _load_config(path: str = "agent-config-lab.yaml") -> Dict[str, Any]:
# MAGIC     if not os.path.exists(path):
# MAGIC         raise FileNotFoundError(f"Config file not found at '{path}'")
# MAGIC     with open(path, "r", encoding="utf-8") as f:
# MAGIC         cfg = yaml.safe_load(f) or {}
# MAGIC     llm_endpoint = cfg.get("llm_endpoint_name")
# MAGIC     vs = cfg.get("vector_search", {}) or {}
# MAGIC     index_name = vs.get("index_name")
# MAGIC     num_results = int(vs.get("num_results", 5))
# MAGIC     if not llm_endpoint or not index_name:
# MAGIC         raise ValueError("Missing 'llm_endpoint_name' or 'vector_search.index_name' in agent-config-lab.yaml")
# MAGIC     return {
# MAGIC         "llm_endpoint_name": llm_endpoint,
# MAGIC         "vs_index_name": index_name,
# MAGIC         "vs_num_results": num_results,
# MAGIC     }
# MAGIC
# MAGIC # Build LangChain agent with LLM and vector search tool
# MAGIC def build_agent(llm_endpoint: str, index_name: str, num_results: int = 5):
# MAGIC     model = ChatDatabricks(endpoint=llm_endpoint, max_tokens=300)
# MAGIC     vs_tool = VectorSearchRetrieverTool(
# MAGIC         name="orion_knowledge_search",
# MAGIC         index_name=index_name,
# MAGIC         description="Search Orion knowledge base for relevant information",
# MAGIC         num_results=num_results,
# MAGIC     )
# MAGIC
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
# MAGIC         system_prompt=system_prompt
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
# MAGIC         result = self._agent.invoke(
# MAGIC             {"messages": msgs}
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
llm_endpoint_name = "databricks-gpt-oss-20b"

agent_config = create_config(llm_endpoint_name, vs_index_name)

# Write YAML file (for agent.py to read later)
with open("agent-config-lab.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(agent_config, f, sort_keys=False)

print("✅ Config file written: agent-config-lab.yaml")
print(yaml.safe_dump(agent_config, sort_keys=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Task 4 - Register the Agent Model to Unity Catalog with Alias
# MAGIC
# MAGIC Now you will register the agent model to Unity Catalog's Model Registry. This combines logging and registering into a single workflow. You will also add an **alias** to the registered model version.
# MAGIC
# MAGIC **Your Task:**
# MAGIC
# MAGIC 1. Define the model resources (vector search index and serving endpoint).
# MAGIC 1. Log the agent model using `mlflow.pyfunc.log_model()` with:
# MAGIC    * Model name: **`"orion_knowledge_assistant_lab"`**
# MAGIC    * Python model: **`"agent-lab.py"`**
# MAGIC    * Code paths: **`["agent-config-lab.yaml"]`**
# MAGIC    * **Model name** to register it to UC model registry.
# MAGIC    * Input example
# MAGIC    * Required pip packages
# MAGIC    * Resources
# MAGIC 1. Set an alias for the registered model version.
# MAGIC
# MAGIC **Hint:** To learn how to set an alias, refer to the [MLflow Model Registry documentation](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/#use-model-aliases). 

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from importlib.metadata import version as get_version
import mlflow

# Step 1: Define the resources
resources = <FILL_IN>

print("Resources defined:")
for resource in resources:
    print(f"  - {resource}")

# Step 2: Define model configuration
model_name = "orion_knowledge_assistant"
tags_to_register = {
    "model_type": "retrieval_agent",
    "framework": "langchain",
    "use_case": "orion_knowledge_base"
}

input_example = {
    "input": [
        {"role": "user", "content": "What is Orion?"}
    ]
}

# Step 3: Log the model
with mlflow.start_run():
    mlflow.set_tags(tags_to_register)
    
    logged_agent_info = <FILL_IN>
    
    model_uri = logged_agent_info.model_uri
    
print(f"✅ Model logged successfully!")
print(f"Model URI: {model_uri}")

# Step 4: Register the model to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{catalog}.{schema}.orion_knowledge_assistant_lab"

uc_registered_model_info = <FILL_IN>

print(f"✅ Model registered successfully to Unity Catalog!")
print(f"Model Name: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")

# Step 5: Set an alias for the registered model version
# Refer to the documentation to learn how to set an alias
client = <FILL_IN>
<FILL_IN>

print(f"✅ Alias 'Champion' set for version {uc_registered_model_info.version}")

# COMMAND ----------

# DBTITLE 1,Task 4 - ANSWER
# MAGIC %skip
# MAGIC from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
# MAGIC from importlib.metadata import version as get_version
# MAGIC import mlflow
# MAGIC
# MAGIC # Step 1: Define the resources
# MAGIC resources = [
# MAGIC     DatabricksVectorSearchIndex(index_name=vs_index_name),
# MAGIC     DatabricksServingEndpoint(endpoint_name=llm_endpoint_name)
# MAGIC ]
# MAGIC
# MAGIC print("Resources defined:")
# MAGIC for resource in resources:
# MAGIC     print(f"  - {resource}")
# MAGIC
# MAGIC # Step 2: Define model configuration
# MAGIC model_name = "orion_knowledge_assistant_lab"
# MAGIC UC_MODEL_NAME = f"{catalog}.{schema}.orion_knowledge_assistant_lab"
# MAGIC
# MAGIC input_example = {
# MAGIC     "input": [
# MAGIC         {"role": "user", "content": "What is Orion?"}
# MAGIC     ]
# MAGIC }
# MAGIC
# MAGIC # Step 3: Log and register the model in one step
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC with mlflow.start_run():
# MAGIC     logged_agent_info = mlflow.pyfunc.log_model(
# MAGIC         name=model_name,
# MAGIC         python_model="agent-lab.py",
# MAGIC         code_paths=["agent-config-lab.yaml"],
# MAGIC         input_example=input_example,
# MAGIC         pip_requirements=[
# MAGIC             f"databricks-vectorsearch=={get_version('databricks-vectorsearch')}",
# MAGIC             f"databricks-langchain=={get_version('databricks-langchain')}",
# MAGIC             f"langchain=={get_version('langchain')}",
# MAGIC             f"mlflow=={get_version('mlflow')}",
# MAGIC         ],
# MAGIC         resources=resources,
# MAGIC         registered_model_name=UC_MODEL_NAME
# MAGIC     )
# MAGIC     model_uri = logged_agent_info.model_uri
# MAGIC     model_version = logged_agent_info.registered_model_version
# MAGIC
# MAGIC print(f"✅ Model logged and registered successfully!")
# MAGIC print(f"Model URI: {model_uri}")
# MAGIC print(f"Model Name: {UC_MODEL_NAME}")
# MAGIC print(f"Version: {model_version}")
# MAGIC
# MAGIC # Step 4: Set an alias for the registered model version
# MAGIC client = mlflow.MlflowClient()
# MAGIC client.set_registered_model_alias(
# MAGIC     name=UC_MODEL_NAME,
# MAGIC     alias="Champion",
# MAGIC     version=model_version
# MAGIC )
# MAGIC
# MAGIC print(f"✅ Alias 'Champion' set for version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Testing the Registered Model
# MAGIC
# MAGIC In this final section, you will test the registered model to validate that it works correctly. You will load the model from Unity Catalog and make predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Task 5 - Test the Registered Model
# MAGIC
# MAGIC Now that your agent is registered in Unity Catalog, you will test it to verify functionality.
# MAGIC
# MAGIC **Your Task:**
# MAGIC
# MAGIC 1. Use the model URI from the previous step to make a prediction.
# MAGIC 1. Use the input example that was logged with the model.
# MAGIC 1. Print the agent's response.
# MAGIC

# COMMAND ----------

# Test the registered model by making a prediction

query = <FILL_IN>

result = <FILL_IN>

print("Agent Response:")
print(result)

# COMMAND ----------

# DBTITLE 1,Task 5 - ANSWER
# MAGIC %skip
# MAGIC import mlflow
# MAGIC
# MAGIC query = {
# MAGIC     "input": [
# MAGIC         {"role": "user", "content": "What are the safety procedures for the Orion?"}
# MAGIC     ]
# MAGIC }
# MAGIC
# MAGIC # Make a prediction using the model URI
# MAGIC result = mlflow.models.predict(
# MAGIC     model_uri=model_uri,
# MAGIC     input_data=query,
# MAGIC     env_manager="uv",
# MAGIC )
# MAGIC
# MAGIC print("Agent Response:")
# MAGIC print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Task 6 - Explore the Model Registry UI
# MAGIC
# MAGIC With your agent successfully registered in Unity Catalog, you can now explore the Model Registry UI to understand how to manage, monitor, and govern your model.
# MAGIC
# MAGIC **Your Tasks:**
# MAGIC
# MAGIC - Identify the four main tabs in the Model Registry UI and describe their purpose.
# MAGIC - Locate the **model requirements file** that was logged as an artifact with your model.
# MAGIC - Find the **alias** you set for your registered model version.
# MAGIC - Review the **execution traces** for agent invocations.
# MAGIC - Summarize how the Model Registry UI supports **model lifecycle management** and **governance**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Summary
# MAGIC
# MAGIC You have successfully built, logged, and registered a production-ready retrieval agent using Databricks Mosaic AI.
# MAGIC
# MAGIC In this lab, you:
# MAGIC
# MAGIC * **Enabled MLflow tracing** for LangChain to monitor agent behavior and execution.
# MAGIC * **Built a retrieval agent** using LangChain with vector search integration.
# MAGIC * **Analyzed agent traces** to identify performance characteristics and understand execution flow.
# MAGIC * **Wrote agent code** to a Python file following the "agent as code" pattern.
# MAGIC * **Registered the agent** to Unity Catalog's Model Registry with proper versioning and aliases.
# MAGIC * **Tested the registered model** to validate functionality.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
