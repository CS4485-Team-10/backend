# YouTube Intelligence Platform - Backend

## Current Progress/Workflow
Built the data ingestion pipeline that pulls YouTube video metadata and transcripts. The flow is:

1. **Fetch metadata** - Query YouTube Data API to search for videos
2. **Extract transcripts** - Pull the transcript for each video using the YouTube Transcript API
3. **Clean transcripts** - Process raw transcripts to remove noise (speaker tags, sound effects, filler words) and normalize formatting
4. **Store locally** - Save both raw and cleaned versions to the `data/` folder for now

Eventually this data moves to Supabase, but we're keeping it local for development.

## Getting started
1. Go to Google Console >> Get an API key for YouTube Data API
2. Initialize a `.env` file with `YOUTUBE_DATA_API_KEY` set up. 
3. Run the notebook (`yt-data-ingestion.ipynb`) to see the full pipeline in action.

## Setup
1. Create a designated virtual env (either via Python natively or Anaconda) and activate it. 
2. Install the following packages/libraries using the following command:
    ```bash
    pip install google-api-python-client youtube-transcript-api python-dotenv
    ```

## LLM Insight Generation (Claims Extraction)

The `llm_insight_generation.ipynb` notebook extracts claims from cleaned transcripts using Ollama.

### Ollama Setup

1. **Install Ollama**: Download from [ollama.com](https://ollama.com) or run `brew install ollama` (macOS).
2. **Start Ollama**: Open the Ollama app (macOS) or run `ollama serve` in a terminal.
3. **Pull a model** (required before running the notebook):
   ```bash
   ollama pull llama3
   ```
   Or use another model (e.g. `llama3.2`, `mistral`).
4. **Optional – use a different model**: Add `LLM_MODEL=llama3.2` to your `.env` (or export it). Default is `llama3`.

### Running the Claims Pipeline

1. Ensure Ollama is running and at least one model is pulled (`ollama list`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run all cells in `pipelines/llm_insight_generation.ipynb` from the top.
4. The test cell extracts claims from `data/transcripts/cleaned/gpzDxm7qflY.txt`.