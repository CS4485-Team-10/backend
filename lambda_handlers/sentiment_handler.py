from pipelines.sentiment_analysis import analyze_video_sentiment


def lambda_handler(event, context):
    video_id = event.get("video_id")
    if not video_id:
        return {
            "statusCode": 400,
            "body": {"error": "video_id is required"},
        }

    result = analyze_video_sentiment(video_id)
    return {
        "statusCode": 200,
        "body": result,
    }
