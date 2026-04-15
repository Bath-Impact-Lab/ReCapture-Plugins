# ReCapture Plugins

Plugin ecosystem for extending **ReCapture** with custom processing, analytics, and integrations.

## Overview

ReCapture Plugins provide a modular way to extend the core ReCapture pipeline. They enable:

- Custom data processing (e.g. gait metrics, biomechanical features)
- Integration with external systems (APIs, storage, analytics)
- Experimentation without modifying the core engine

The architecture is designed for **low coupling** and **rapid iteration**.

---

## Architecture

- **Core ReCapture Engine** → Produces motion capture data
- **Plugin Layer** → Consumes and processes data
- **Outputs** → Metrics, visualisations, or external integrations

Plugins operate as independent units with a defined interface.

---

## Plugin Structure

Each plugin follows a standard layout:

```text
plugin-name/
├── plugin.py        # Entry point
├── config.json      # Plugin configuration
├── requirements.txt # Optional dependencies
└── README.md        #```text Plugin-specific docs
```

---

## Plugin Interface

A plugin must expose a standard interface:

```python
class Plugin:
    def __init__(self, config):
        pass

    def process(self, data):
        """Process incoming frame or sequence data"""
        return data
```

### Inputs
- Pose/keypoint data
- Temporal sequences (optional)
- Metadata (session, subject, calibration)

### Outputs
- Transformed data
- Derived metrics
- Events or annotations

---

## Installation

```bash
git clone https://github.com/Bath-Impact-Lab/ReCapture-Plugins.git
cd ReCapture-Plugins
```

Install dependencies (if required):

```bash
pip install -r requirements.txt
```

---

## Usage

1. Add plugin to your ReCapture configuration
2. Ensure it is discoverable (path or registry)
3. Run ReCapture as normal

Example:

```json
{
  "plugins": [
    "plugins/gait_analysis",
    "plugins/stride_detection"
  ]
}
```

---

## Creating a Plugin

1. Create a new directory under `plugins/`
2. Implement the `Plugin` interface
3. Add configuration if required
4. Register it in your pipeline

Minimal example:

```python
class Plugin:
    def __init__(self, config):
        self.threshold = config.get("threshold", 0.5)

    def process(self, data):
        return {
            k: v for k, v in data.items()
            if v["confidence"] > self.threshold
        }
```

---

## Design Principles

- Modularity
- Determinism
- Performance-aware
- Extensible

---

## Use Cases

- Gait analysis
- Rehabilitation metrics
- Sports performance tracking
- Real-time feedback systems
- Data export pipelines

---

## Contributing

- Keep plugins focused and single-purpose
- Document inputs/outputs clearly
- Include example configs where possible

---

## License

See `LICENSE` for details.
