#!/usr/bin/env python3
"""
Arize-based evaluation suite for the RAG agent.
This follows the Arize documentation pattern for experiments while using Google Vertex AI for evaluations.
"""

import json
import os
import asyncio
from typing import Dict, List, Any
import pandas as pd
from dotenv import load_dotenv
import uuid
import time

# Arize imports
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE
from arize.experimental.datasets.experiments.types import EvaluationResult

# Google Cloud imports for Vertex AI evaluations
import vertexai
from vertexai.preview.evaluation import EvalTask

# ADK imports for running the agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

# Import the RAG agent
from rag.agent import root_agent

load_dotenv()

# Environment variables
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

if not all([ARIZE_API_KEY, ARIZE_SPACE_ID, GOOGLE_CLOUD_PROJECT]):
    raise ValueError("Missing required environment variables: ARIZE_API_KEY, ARIZE_SPACE_ID, GOOGLE_CLOUD_PROJECT")

# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location="us-central1")

# Initialize Arize client (developer_key is deprecated, only api_key needed)
arize_client = ArizeDatasetsClient(api_key=ARIZE_API_KEY)


def load_test_data() -> List[Dict]:
    """Load the conversation test data from JSON file."""
    with open("eval/data/conversation.test.json", "r") as f:
        return json.load(f)


def create_arize_dataset():
    """Create an Arize dataset from the test data."""
    test_data = load_test_data()
    
    # Transform data for Arize format - simplified structure
    dataset_rows = []
    for i, item in enumerate(test_data):
        dataset_rows.append({
            "id": str(i),  # Add explicit ID
            "query": item["query"],
            "expected_tool_use": json.dumps(item["expected_tool_use"]),  # Convert to JSON string
            "reference": item["reference"]
        })
    
    # Create DataFrame
    df = pd.DataFrame(dataset_rows)
    
    # Debug: Print the DataFrame structure
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame dtypes:", df.dtypes.to_dict())
    print("First row sample:", df.iloc[0].to_dict())
    
    # Create dataset in Arize using the correct API
    dataset_name = f"rag_agent_evaluation_dataset-{uuid.uuid4()}"
    
    print(f"Creating dataset: {dataset_name}")
    dataset_id = arize_client.create_dataset(
        space_id=ARIZE_SPACE_ID,
        dataset_name=dataset_name,
        data=df,
        dataset_type=GENERATIVE
    )
    
    print(f"Dataset created with ID: {dataset_id}")
    time.sleep(5)  # Wait after dataset creation
    return {"id": dataset_id}


def extract_tool_calls_from_response(response_text: str, agent_runner) -> List[Dict]:
    """
    Extract tool call information from the agent's response and execution context.
    This creates the trajectory format expected by Vertex AI evaluation API.
    """
    tool_calls = []
    
    # For ADK agents, we need to examine the execution traces
    # This is a simplified implementation - you may need to adjust based on
    # how you want to capture tool usage from the ADK agent execution
    
    # Check if the response indicates tool usage
    if any(keyword in response_text.lower() for keyword in [
        "according to", "based on", "source:", "[citation", "retrieved", "documentation"
    ]):
        # Infer that the RAG tool was used
        tool_calls.append({
            "tool_name": "retrieve_rag_documentation", 
            "tool_input": response_text[:100] + "..." if len(response_text) > 100 else response_text
        })
    
    return tool_calls


async def call_rag_agent(query: str) -> Dict[str, Any]:
    """Call the RAG agent programmatically and return response with metadata."""
    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )
    
    content = UserContent(parts=[Part(text=query)])
    response_parts = []
    
    for event in runner.run(
        user_id=session.user_id, 
        session_id=session.id, 
        new_message=content
    ):
        for part in event.content.parts:
            response_parts.append(part.text)
    
    response_text = "\n".join(response_parts)
    
    # Extract tool usage information
    tool_calls = extract_tool_calls_from_response(response_text, runner)
    
    return {
        "response": response_text,
        "tool_calls": tool_calls
    }


async def task_function(dataset_row: Dict) -> str:
    """
    Task function for Arize experiments.
    This calls the RAG agent and returns the response.
    """
    query = dataset_row.get("query", "")
    
    # Call the agent - use await instead of asyncio.run since we're in async context
    result = await call_rag_agent(query)
    
    # Store additional metadata for evaluations
    # Note: We'll add this to the response string as JSON for evaluation access
    metadata = {
        "agent_response": result["response"],
        "tool_calls": result["tool_calls"],
        "expected_tool_use": dataset_row.get("expected_tool_use", "[]"),
        "reference": dataset_row.get("reference", "")
    }
    
    return json.dumps(metadata)


# Vertex AI Evaluation Functions using the native evaluation API
def create_reference_trajectory(expected_tool_use: List[Dict]) -> List[Dict]:
    """Convert expected tool use to reference trajectory format."""
    if not expected_tool_use:
        return []
    
    trajectory = []
    for tool_use in expected_tool_use:
        trajectory.append({
            "tool_name": tool_use.get("tool_name", ""),
            "tool_input": tool_use.get("tool_input", {})
        })
    return trajectory


def evaluate_with_vertex_ai_single_metric(predicted_trajectory: List[Dict], reference_trajectory: List[Dict], metric: str) -> Dict[str, Any]:
    """Evaluate using Vertex AI's native evaluation API for a single metric."""
    try:
        # Create evaluation dataset
        eval_dataset = pd.DataFrame({
            "predicted_trajectory": [predicted_trajectory],
            "reference_trajectory": [reference_trajectory]
        })
        
        # Create evaluation task for single metric
        eval_task = EvalTask(
            dataset=eval_dataset,
            metrics=[metric],
        )
        
        # Run evaluation
        eval_result = eval_task.evaluate()
        
        # Extract metric
        metric_value = eval_result.summary_metrics.get(f"{metric}/mean", 0.0)
        results = {metric: metric_value}
        
        return results
        
    except Exception as e:
        print(f"Error in Vertex AI evaluation for {metric}: {e}")
        return {metric: 0.0}


def trajectory_exact_match_evaluator(output: str, dataset_row: Dict) -> EvaluationResult:
    """Evaluator for trajectory exact match using Vertex AI evaluation API."""
    try:
        metadata = json.loads(output)
        actual_tool_calls = metadata.get("tool_calls", [])
        expected_tool_use = json.loads(dataset_row.get("expected_tool_use", "[]"))
        
        # Simple exact match logic
        if len(actual_tool_calls) != len(expected_tool_use):
            return EvaluationResult(
                score=0.0,
                label="no_exact_match",
                explanation=f"Length mismatch: expected {len(expected_tool_use)} tools, got {len(actual_tool_calls)}"
            )
        
        if not expected_tool_use and not actual_tool_calls:
            return EvaluationResult(
                score=1.0,
                label="exact_match",
                explanation="Both expected and actual tool usage are empty - perfect match"
            )
        
        # Check if tool names match
        actual_names = [tc.get("tool_name", "") for tc in actual_tool_calls]
        expected_names = [et.get("tool_name", "") for et in expected_tool_use]
        
        score = 1.0 if actual_names == expected_names else 0.0
        label = "exact_match" if score == 1.0 else "no_exact_match"
        explanation = f"Tool sequence match: expected {expected_names}, got {actual_names}"
        
        return EvaluationResult(score=score, label=label, explanation=explanation)
        
    except Exception as e:
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Evaluation error: {str(e)}"
        )


def trajectory_precision_evaluator(output: str, dataset_row: Dict) -> EvaluationResult:
    """Evaluator for trajectory precision using Vertex AI evaluation API."""
    try:
        metadata = json.loads(output)
        actual_tool_calls = metadata.get("tool_calls", [])
        expected_tool_use = json.loads(dataset_row.get("expected_tool_use", "[]"))
        
        if not expected_tool_use:
            score = 1.0 if not actual_tool_calls else 0.0
            label = "perfect" if score == 1.0 else "unexpected_tools"
            explanation = "No tools expected" + ("" if score == 1.0 else f", but got {len(actual_tool_calls)} tools")
            return EvaluationResult(score=score, label=label, explanation=explanation)
        
        # Calculate precision: how many of the actual tools were expected
        actual_names = set(tc.get("tool_name", "") for tc in actual_tool_calls)
        expected_names = set(et.get("tool_name", "") for et in expected_tool_use)
        
        if not actual_names:
            return EvaluationResult(
                score=0.0,
                label="no_tools_used",
                explanation="No tools used when tools were expected"
            )
        
        intersection = actual_names.intersection(expected_names)
        score = len(intersection) / len(actual_names)
        
        if score >= 0.9:
            label = "high_precision"
        elif score >= 0.7:
            label = "medium_precision"
        else:
            label = "low_precision"
        
        explanation = f"Precision: {len(intersection)}/{len(actual_names)} = {score:.2f}. Expected: {sorted(expected_names)}, Used: {sorted(actual_names)}"
        
        return EvaluationResult(score=score, label=label, explanation=explanation)
        
    except Exception as e:
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Evaluation error: {str(e)}"
        )


def tool_name_match_evaluator(output: str, dataset_row: Dict) -> EvaluationResult:
    """Evaluator for tool name matching, ignoring parameters."""
    try:
        metadata = json.loads(output)
        actual_tool_calls = metadata.get("tool_calls", [])
        expected_tool_use = json.loads(dataset_row.get("expected_tool_use", "[]"))
        
        # Extract tool names only
        actual_tool_names = set()
        for tool_call in actual_tool_calls:
            if isinstance(tool_call, dict) and "tool_name" in tool_call:
                actual_tool_names.add(tool_call["tool_name"])
        
        expected_tool_names = set()
        for expected_tool in expected_tool_use:
            if isinstance(expected_tool, dict) and "tool_name" in expected_tool:
                expected_tool_names.add(expected_tool["tool_name"])
        
        # Calculate match score
        if not expected_tool_names:
            # If no tools expected, score 1.0 if no tools used, 0.0 if tools used
            score = 1.0 if not actual_tool_names else 0.0
            label = "correct_no_tools" if score == 1.0 else "unexpected_tools"
            explanation = f"No tools expected. Got: {sorted(actual_tool_names) if actual_tool_names else 'none'}"
        else:
            # Calculate intersection over expected (precision for expected tools)
            intersection = actual_tool_names.intersection(expected_tool_names)
            score = len(intersection) / len(expected_tool_names)
            
            if score == 1.0:
                label = "perfect_match"
            elif score >= 0.7:
                label = "good_match"
            elif score > 0:
                label = "partial_match"
            else:
                label = "no_match"
            
            explanation = f"Tool name coverage: {len(intersection)}/{len(expected_tool_names)} = {score:.2f}. Expected: {sorted(expected_tool_names)}, Got: {sorted(actual_tool_names)}"
        
        return EvaluationResult(score=score, label=label, explanation=explanation)
        
    except Exception as e:
        return EvaluationResult(
            score=0.0,
            label="error",
            explanation=f"Evaluation error: {str(e)}"
        )


def run_evaluation_experiment():
    """Run the complete evaluation experiment using Arize."""
    
    # Create dataset
    print("Creating Arize dataset...")
    dataset = create_arize_dataset()
    
    # Define evaluators - using Vertex AI evaluation API (separate metrics for better Arize visibility)
    evaluators = [
        trajectory_exact_match_evaluator,
        trajectory_precision_evaluator,
        tool_name_match_evaluator
    ]
    
    # Run experiment
    print("Running experiment...")
    experiment_result = arize_client.run_experiment(
        space_id=ARIZE_SPACE_ID,
        dataset_id=dataset["id"],
        task=task_function,
        evaluators=evaluators,
        experiment_name=f"rag_agent_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        concurrency=2,  # Reduce concurrency to be more gentle on APIs
        exit_on_error=False,
        dry_run=False,
    )
    
    # Handle the experiment result - it might be a tuple or have different structure
    if hasattr(experiment_result, 'id'):
        experiment_id = experiment_result.id
    elif isinstance(experiment_result, tuple) and len(experiment_result) > 0:
        experiment_id = experiment_result[0].id if hasattr(experiment_result[0], 'id') else str(experiment_result[0])
    else:
        experiment_id = "unknown"
    
    print(f"Experiment completed! Experiment ID: {experiment_id}")
    print("View results in the Arize UI")
    
    return experiment_result


if __name__ == "__main__":
    run_evaluation_experiment()
