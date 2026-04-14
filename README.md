# Video Editing AI MCP Server
**By MEOK AI Labs** | [meok.ai](https://meok.ai)

AI-powered video editing toolkit: scene splitting, subtitle generation, thumbnail planning, color grading, and aspect ratio conversion.

## Tools

| Tool | Description |
|------|-------------|
| `split_scenes` | Detect and split video into scenes with timestamps and frame numbers |
| `generate_subtitles` | Generate timed SRT subtitles from a transcript |
| `thumbnail_data` | Recommend thumbnail timestamps, composition, and text overlays |
| `recommend_color_grading` | Color grading settings based on genre, mood, and lighting |
| `convert_aspect_ratio` | Plan aspect ratio conversion with crop/pad calculations |

## Installation

```bash
pip install mcp
```

## Usage

### Run the server

```bash
python server.py
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "video-editing": {
      "command": "python",
      "args": ["/path/to/video-editing-ai-mcp/server.py"]
    }
  }
}
```

## Pricing

| Tier | Limit | Price |
|------|-------|-------|
| Free | 30 calls/day | $0 |
| Pro | Unlimited + premium features | $9/mo |
| Enterprise | Custom + SLA + support | Contact us |

## License

MIT
