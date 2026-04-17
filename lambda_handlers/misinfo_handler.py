import json
import logging
from dataclasses import asdict, is_dataclass
from pipelines.misinfo_checker import check_video

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Lambda handler for misinformation checking."""
    try:
        # Handle both direct invocation and API Gateway events
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event

        video_id = body.get("video_id")
        if not video_id:
            logger.error("Missing video_id in request")
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "video_id is required"}),
            }

        logger.info(f"Processing misinfo check for video: {video_id}")
        result = check_video(video_id)

        # Convert dataclass to dict if needed
        if is_dataclass(result):
            result = asdict(result)

        logger.info(f"Misinfo check completed for video: {video_id}")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        logger.exception(f"Unexpected error in misinfo handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }
