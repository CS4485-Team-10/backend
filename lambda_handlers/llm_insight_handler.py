import json
import logging
from pipelines.llm_insight_generation import run_llm_insight_generation_pipeline

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Lambda handler for LLM insight generation."""
    try:
        logger.info("Starting LLM insight generation pipeline")
        
        # Pass lambda context for time management
        result = run_llm_insight_generation_pipeline(
            lambda_context=context,
        )

        logger.info("LLM insight generation pipeline completed successfully")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        logger.exception(f"Unexpected error in LLM insight handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }
