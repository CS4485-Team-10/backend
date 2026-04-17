import json
import logging
from pipelines.sentiment_analysis import analyze_video_sentiment

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """Lambda handler for sentiment analysis."""
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

        logger.info(f"Processing sentiment analysis for video: {video_id}")
        result = analyze_video_sentiment(video_id)

        if "error" in result:
            logger.error(f"Sentiment analysis failed: {result['error']}")
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result),
            }

        logger.info(f"Sentiment analysis completed for video: {video_id}")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        logger.exception(f"Unexpected error in sentiment handler: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }
