from pipelines.llm_insight_generation import run_llm_insight_generation_pipeline


def lambda_handler(event, context):
    result = run_llm_insight_generation_pipeline(
        verbose=event.get("verbose", True),
    )
    return {
        "statusCode": 200,
        "body": result,
    }
