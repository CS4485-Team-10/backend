from dataclasses import asdict, is_dataclass
from pipelines.misinfo_checker import check_video

def lambda_handler(event, context):
    video_id = event.get("video_id")
    if not video_id:
        return {
            "statusCode": 400,
            "body": {"error": "video_id is required"},
        }

    result = check_video(video_id)

    if is_dataclass(result):
        result = asdict(result)

    return {
        "statusCode": 200,
        "body": result,
    }