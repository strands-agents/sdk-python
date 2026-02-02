## Description

When using A2A (Agent-to-Agent) protocol, agents frequently need to exchange rich content beyond plain text—images for vision tasks, documents for analysis, and videos for multimedia workflows. The current A2A converters only support text content, forcing developers to work around this limitation or lose content fidelity when communicating between agents.

This PR extends the A2A converters to handle image, document, and video content types, enabling seamless multimodal communication between Strands agents and any A2A-compatible agent.

Resolves: #1504

## Public API Changes

No public API changes. The existing `convert_content_blocks_to_parts` and `convert_response_to_agent_result` functions now automatically handle additional content types.

```python
# Before: only text content was converted, other types were silently dropped
content_blocks = [
    {"text": "Analyze this image:"},
    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
]
parts = convert_content_blocks_to_parts(content_blocks)
# Result: only 1 part (text), image was lost

# After: all content types are preserved
content_blocks = [
    {"text": "Analyze this image:"},
    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
]
parts = convert_content_blocks_to_parts(content_blocks)
# Result: 2 parts - TextPart and FilePart with image/png MIME type
```

The conversion is bidirectional—A2A FileParts received from remote agents are correctly converted back to Strands ImageContent, DocumentContent, or VideoContent based on MIME type.

## Related Issues

#1504

## Documentation PR

N/A - Internal converter changes with no user-facing API modifications.

## Type of Change

New feature

## Testing

How have you tested the change?  Verify that the changes do not break functionality or introduce warnings in consuming repositories: agents-docs, agents-tools, agents-cli

- [x] I ran `hatch run prepare`

Added 31 new unit tests covering:
- Image conversion (all formats: png, jpeg, gif, webp) with both inline bytes and S3 URIs
- Document conversion (all formats: pdf, csv, doc, docx, xls, xlsx, html, txt, md)
- Video conversion (all formats: flv, mkv, mov, mpeg, mpg, mp4, three_gp, webm, wmv)
- Mixed content scenarios and edge cases (unknown MIME types, missing MIME types)
- Full round-trip conversion through response handling

All 136 A2A module tests pass.

## Checklist
- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [x] I have updated the documentation accordingly
- [x] I have added an appropriate example to the documentation to outline the feature, or no new docs are needed
- [x] My changes generate no new warnings
- [x] Any dependent changes have been merged and published

----

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
