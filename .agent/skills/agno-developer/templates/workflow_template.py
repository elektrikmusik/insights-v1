from agno.workflow import Workflow
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from src.config import settings
from typing import Iterator

class TemplateWorkflow(Workflow):
    """
    Template for creating a workflow using Agno.
    Workflows orchestrate multiple steps (agents or functions) to achieve a complex goal.
    """

    # Example: Define agents used in the workflow
    researcher: Agent = Agent(
        model=Gemini(id="gemini-1.5-flash", api_key=settings.gemini_api_key),
        description="Researcher",
        instructions="Find information on...",
    )
    
    writer: Agent = Agent(
        model=Gemini(id="gemini-1.5-flash", api_key=settings.gemini_api_key),
        description="Writer", 
        instructions="Write a summary based on research..."
    )

    def run(self, topic: str) -> Iterator[RunResponse]:
        """
        Execute the workflow steps.
        This method yields RunResponse objects for each step if streaming is desired,
        or returns the final result.
        """
        
        # Step 1: Research
        research_result = self.researcher.run(f"Research about {topic}")
        yield RunResponse(content=f"Research completed: {research_result.content[:50]}...", event="research_done")
        
        # Step 2: Write
        write_result = self.writer.run(f"Write a comprehensive report based on this research:\n{research_result.content}")
        yield RunResponse(content=write_result.content, event="workflow_complete")
