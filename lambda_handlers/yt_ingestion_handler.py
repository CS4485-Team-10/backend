from pipelines.yt_data_ingestion import run_youtube_data_ingestion_pipeline


def lambda_handler(event, context):
    result = run_youtube_data_ingestion_pipeline(
        max_search_pages=event.get("max_search_pages", 1),
        verbose=event.get("verbose", True),
    )
    return {
        "statusCode": 200,
        "body": result,
    }
