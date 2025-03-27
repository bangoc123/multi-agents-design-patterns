from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Enum, Field, Any
from pydantic import BaseModel

from agents import Agent, AgentHooks, RunContextWrapper, Runner, Tool, function_tool
from dotenv import load_dotenv

load_dotenv()
class TaskType(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    ANALYSIS = "analysis"
    INTEGRATION = "integration"


class Task(BaseModel):
    """Represents a task that needs to be performed"""
    task_id: str
    description: str
    type: TaskType
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"
    context: Dict[str, Any] = Field(default_factory=dict)  # Additional context for workers
    file_paths: List[str] = Field(default_factory=list)    # Files that need modification


class TaskResult(BaseModel):
    """Result of a task execution"""
    task_id: str
    success: bool
    output: str
    error: Optional[str] = None


class ProjectAnalysis(BaseModel):
    """Analysis of the project and required tasks"""
    tasks: List[Task]
    execution_order: List[str]


class OrchestratorResult(BaseModel):
    """Result of the orchestration process"""
    all_tasks_completed: bool
    results: List[TaskResult]
    summary: str


@function_tool
async def analyze_project_requirements(description: str) -> ProjectAnalysis:
    """
    Use LLM to dynamically analyze project requirements and break them down into tasks.
    The LLM determines necessary files to modify and creates appropriate subtasks.
    """
    # This would be an actual LLM call to analyze the project
    prompt = f"""
    Analyze the following project requirements and break them down into specific tasks:
    {description}
    
    For each task:
    1. Identify the type of work needed (backend/frontend/analysis/integration)
    2. Determine which files need to be modified
    3. Specify any dependencies between tasks
    4. Provide detailed context for workers
    """
    
    # Mock LLM response - in reality, this would be dynamic
    tasks = [
        Task(
            task_id="analyze_auth_flow",
            description="Analyze current authentication flow and identify necessary changes",
            type=TaskType.ANALYSIS,
            file_paths=["auth/*", "models/user.py"],
            context={"focus_areas": ["security", "user flow"]}
        ),
        Task(
            task_id="update_api",
            description="Update API endpoints for new authentication flow",
            type=TaskType.BACKEND,
            dependencies=["analyze_auth_flow"],
            file_paths=["api/auth.py", "api/endpoints.py"],
            context={"api_version": "v2"}
        ),
        # ... more tasks ...
    ]
    return ProjectAnalysis(tasks=tasks, execution_order=["analyze_auth_flow", "update_api"])


@function_tool
async def execute_backend_task(task: Dict[str, Any]) -> TaskResult:
    """Execute a backend-related task"""
    try:
        return TaskResult(
            task_id=task["description"].split(":")[0],  # Extract task_id from description
            success=True,
            output=f"Completed backend task: {task['description']}",
            error=None
        )
    except Exception as e:
        return TaskResult(
            task_id=task.get("task_id", "unknown"),
            success=False,
            output="",
            error=str(e)
        )


@function_tool
async def execute_frontend_task(task: Dict[str, Any]) -> TaskResult:
    """Execute a frontend-related task"""
    try:
        return TaskResult(
            task_id=task["description"].split(":")[0],  # Extract task_id from description
            success=True,
            output=f"Completed frontend task: {task['description']}",
            error=None
        )
    except Exception as e:
        return TaskResult(
            task_id=task.get("task_id", "unknown"),
            success=False,
            output="",
            error=str(e)
        )


@function_tool
async def synthesize_results(results: List[TaskResult]) -> str:
    """
    Use LLM to analyze and synthesize results from multiple workers into a coherent summary
    """
    # This would be an actual LLM call to synthesize results
    prompt = f"""
    Review the following task results and provide a coherent summary:
    {results}
    
    Include:
    1. Overall success/failure analysis
    2. Key changes made
    3. Integration points to verify
    4. Next steps or recommendations
    """
    return "Synthesized summary of all work completed..."


class Orchestrator:
    def __init__(self):
        self.workers = {
            TaskType.BACKEND: backend_worker,
            TaskType.FRONTEND: frontend_worker,
            TaskType.ANALYSIS: analysis_worker,
            TaskType.INTEGRATION: integration_worker
        }

    async def adjust_tasks(self, analysis: ProjectAnalysis, result: TaskResult) -> ProjectAnalysis:
        """
        Dynamically adjust remaining tasks based on a task result
        """
        prompt = f"""
        Based on the task result:
        {result}
        
        Analyze the current project state and determine if any tasks need to be:
        1. Added
        2. Modified
        3. Removed
        
        Current tasks:
        {analysis.tasks}
        """
        
        # This would be an actual LLM call
        # For now, return the original analysis
        return analysis

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task using the appropriate worker"""
        worker = self.workers.get(task.type)
        if not worker:
            raise ValueError(f"No worker found for task type: {task.type}")
        
        try:
            result = await Runner.run(
                worker,
                input={
                    "description": task.description,
                    "context": task.context,
                    "file_paths": task.file_paths
                }
            )
            
            # Add requires_task_adjustment field if missing
            if not hasattr(result, 'requires_task_adjustment'):
                result.requires_task_adjustment = False
                
            return result
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                output="",
                error=str(e)
            )

    async def orchestrate(self, project_description: str) -> OrchestratorResult:
        """Main orchestration logic"""
        # Analyze project and break down into tasks
        analysis = await analyze_project_requirements(project_description)
        
        # Execute tasks in order, respecting dependencies
        results = []
        completed_tasks = set()
        
        for task_id in analysis.execution_order:
            task = next(t for t in analysis.tasks if t.task_id == task_id)
            
            # Check dependencies
            if not all(dep in completed_tasks for dep in task.dependencies):
                continue
                
            # Execute task
            result = await self.execute_task(task)
            results.append(result)
            completed_tasks.add(task_id)
            
            # Dynamically adjust remaining tasks based on result if needed
            if result.success and result.requires_task_adjustment:
                new_analysis = await self.adjust_tasks(analysis, result)
                analysis = new_analysis

        # Synthesize final results
        summary = await synthesize_results(results)
        
        return OrchestratorResult(
            all_tasks_completed=len(completed_tasks) == len(analysis.tasks),
            results=results,
            summary=summary
        )


# Create worker agents for different types of tasks
backend_worker = Agent(
    name="Backend Worker",
    instructions="Execute backend-related tasks in the codebase",
    tools=[execute_backend_task],
    output_type=TaskResult
)

frontend_worker = Agent(
    name="Frontend Worker",
    instructions="Execute frontend-related tasks in the codebase",
    tools=[execute_frontend_task],
    output_type=TaskResult
)

# Create additional worker agents
analysis_worker = Agent(
    name="Analysis Worker",
    instructions="Execute analysis-related tasks in the codebase",
    tools=[],  # Add appropriate tools
    output_type=TaskResult
)

integration_worker = Agent(
    name="Integration Worker",
    instructions="Execute integration-related tasks in the codebase",
    tools=[],  # Add appropriate tools
    output_type=TaskResult
)

# Create orchestrator agent
orchestrator = Agent(
    name="Orchestrator",
    instructions="""
    Analyze the project requirements and coordinate task execution:
    1. Break down the project into tasks
    2. Determine task dependencies and execution order
    3. Delegate tasks to appropriate worker agents
    4. Monitor task completion and handle failures
    5. Provide a final summary of all completed work
    """,
    tools=[analyze_project_requirements],
    output_type=OrchestratorResult,
    handoffs=[backend_worker, frontend_worker]
)


async def execute_tasks(tasks: List[Task], execution_order: List[str]) -> List[TaskResult]:
    """Execute tasks in the specified order"""
    results = []
    completed_tasks = set()

    for task_id in execution_order:
        task = next(task for task in tasks if task.task_id == task_id)
        
        # Check if dependencies are met
        if task.dependencies:
            if not all(dep in completed_tasks for dep in task.dependencies):
                continue

        # Select appropriate worker based on task type
        worker = backend_worker if task.type == "backend" else frontend_worker
        
        # Execute task
        result = await Runner.run(
            worker,
            input=f"Execute task: {task.description}"
        )
        
        results.append(result)
        completed_tasks.add(task_id)

    return results


async def main():
    try:
        project_description = """
        Update the user authentication system:
        - Modify API endpoints to support new authentication flow
        - Update data models to include new user properties
        - Update UI components to show new user information
        """

        orchestrator = Orchestrator()
        result = await orchestrator.orchestrate(project_description)

        print("\nProject Execution Summary:")
        print("-" * 50)
        print(f"All tasks completed: {result.all_tasks_completed}")
        print("\nTask Results:")
        for task_result in result.results:
            print(f"\nTask: {task_result.task_id}")
            print(f"Success: {task_result.success}")
            print(f"Output: {task_result.output}")
            if task_result.error:
                print(f"Error: {task_result.error}")
        
        print("\nFinal Summary:")
        print(result.summary)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())