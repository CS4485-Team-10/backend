import json
import logging
from pipelines.yt_data_ingestion import run_youtube_data_ingestion_pipeline

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Lambda handler for YouTube data ingestion."""
    try:
        # Handle both direct invocation and API Gateway events
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event

        max_search_pages = body.get("max_search_pages", 1)
        verbose = body.get("verbose", True)

        logger.info(
            f"Starting YouTube ingestion with max_search_pages={max_search_pages}"
        )
        result = run_youtube_data_ingestion_pipeline(
            max_search_pages=max_search_pages,
            verbose=verbose,
        )

        logger.info("YouTube ingestion pipeline completed successfully")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        logger.exception(f"Unexpected error in YouTube ingestion handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }
