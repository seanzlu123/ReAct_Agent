from EM_client.trajectory import Trajectory
from EM_client.request import SummarizerRequest, ContextGeneratorRequest
from loguru import logger

def summarize_experience(client, instruction, history, experience_filename, w_id: str) -> None:
    """
    Uses the summarizer service to create and store an experience JSON
    summarizing this agent run for later training or analysis.
    """
    request = SummarizerRequest(
        trajectories=[
            Trajectory(query=instruction, steps=history, answer=history[-1]['content'], done=True)],
        workspace_id=w_id
    )
    try:
        response = client.call_summarizer(request)
        with open(experience_filename, "w") as f:
            for experience in response.experiences:
                f.write(experience.model_dump_json() + "\n")
                logger.info(f"Experience ID: {experience.experience_id}")
    except Exception as e:
        logger.error(f"Failed to create experience: {e}")
        raise

def generate_context(client, query, w_id: str) -> str:
    """
    Retrieves additional context for a user query from the context generator service.
    """
    request = ContextGeneratorRequest(trajectory=Trajectory(query=query), retrieve_top_k=1, workspace_id=w_id)

    try:
        response = client.call_context_generator(request)
        new_query = f"{response.context_msg.content}\n\nUser Question\n{query}"
        return new_query
    except Exception as e:
        logger.error(f"Failed to generate context: {e}")
        raise