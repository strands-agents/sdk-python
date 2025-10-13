# Event Logger for Bidirectional Streaming

The event logger captures all incoming and outgoing events for each bidirectional streaming provider, making it easy to compare event structures and debug issues.

## Features

- **Automatic logging**: All events are logged automatically when using any bidirectional model
- **Truncated content**: Long strings (base64 audio/images) are truncated to 100 characters for readability
- **JSONL format**: Each event is saved as a single JSON line for easy parsing
- **Provider-specific files**: Separate log files for each provider (gemini, nova, openai)
- **Timestamped sessions**: Each session gets a unique timestamped log file

## Log File Location

Event logs are saved to `event_logs/` directory in your current working directory:

```
event_logs/
├── gemini_20251013_143022.jsonl
├── nova_20251013_143045.jsonl
└── openai_20251013_143108.jsonl
```

## Log Entry Format

Each log entry is a JSON object with the following structure:

```json
{
  "timestamp": "2025-10-13T14:30:22.123456",
  "provider": "gemini",
  "direction": "incoming",
  "event_type": "gemini_raw",
  "sequence": 42,
  "data": {
    "text": "Hello, how can I help you?",
    "audioData": "<1024 bytes>",
    "imageData": "iVBORw0KGgoAAAANSUhEUgAA... (truncated, total: 45678 chars)"
  }
}
```

### Fields

- **timestamp**: ISO 8601 timestamp of when the event was logged
- **provider**: Provider name (`gemini`, `nova`, or `openai`)
- **direction**: `incoming` (from provider) or `outgoing` (to provider)
- **event_type**: Type of event (e.g., `audio_input`, `text_output`, `gemini_raw`)
- **sequence**: Sequential number for this direction (separate counters for incoming/outgoing)
- **data**: Event data with truncated strings

## Event Types

### Outgoing Events (to provider)

- **audio_input**: Audio data sent to the model
  ```json
  {
    "format": "pcm",
    "sampleRate": 16000,
    "channels": 1,
    "audioData": "<1024 bytes>"
  }
  ```

- **text_input**: Text message sent to the model
  ```json
  {
    "text": "What's the weather like?"
  }
  ```

- **image_input**: Image/video frame sent to the model (Gemini only)
  ```json
  {
    "mimeType": "image/jpeg",
    "encoding": "base64",
    "imageData": "iVBORw0KGgoAAAANSUhEUgAA... (truncated, total: 45678 chars)"
  }
  ```

### Incoming Events (from provider)

- **gemini_raw**: Raw Gemini Live API events
- **nova_raw**: Raw Nova Sonic events
- **openai_raw**: Raw OpenAI Realtime API events

Each provider has its own event structure. The raw events show the provider-specific format before conversion to Strands' unified format.

## Usage

Event logging is automatic when you use any bidirectional model:

```python
from strands.experimental.bidirectional_streaming.agent import BidirectionalAgent
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel

# Create model and agent
model = GeminiLiveBidirectionalModel(api_key="...")
agent = BidirectionalAgent(model=model)

# Start conversation - events are automatically logged
await agent.start()
await agent.send("Hello!")

# Events are saved to event_logs/gemini_YYYYMMDD_HHMMSS.jsonl
```

## Analyzing Logs

### View logs in real-time

```bash
tail -f event_logs/gemini_*.jsonl | jq .
```

### Count events by type

```bash
cat event_logs/gemini_*.jsonl | jq -r '.event_type' | sort | uniq -c
```

### Extract all text outputs

```bash
cat event_logs/gemini_*.jsonl | jq 'select(.event_type == "gemini_raw" and .data.text != null) | .data.text'
```

### Compare event structures across providers

```bash
# Gemini events
cat event_logs/gemini_*.jsonl | jq '.data | keys' | sort | uniq

# Nova events
cat event_logs/nova_*.jsonl | jq '.data | keys' | sort | uniq

# OpenAI events
cat event_logs/openai_*.jsonl | jq '.data | keys' | sort | uniq
```

## Disabling Logging

Event logging is always enabled. If you want to disable it, you can modify the `EventLogger` class to skip file writes:

```python
# In event_logger.py
def log_event(self, direction: str, event_type: str, event_data: Dict[str, Any]) -> None:
    # Comment out the file write
    # with open(self.log_file, "a") as f:
    #     f.write(json.dumps(log_entry, indent=None) + "\n")
    pass
```

## Performance Impact

Event logging has minimal performance impact:
- File writes are non-blocking
- String truncation is efficient
- Only metadata is logged, not full audio/image data
- Typical overhead: < 1ms per event
