# Contributing to Local Video & Audio Transformer

Thank you for your interest in improving this project! As a local-first ML project, we prioritize reliability, performance, and high-quality results.

## 🛠 Development Setup

1. **Prerequisites**:
   - Python 3.12+
   - NVIDIA GPU with 12GB+ VRAM
   - CUDA 12.1+

2. **Environment**:
   ```bash
   make setup
   source .venv/bin/activate
   ```

3. **Sample Assets**:
   ```bash
   ./samples/download_samples.sh
   ```

## 🧪 Testing

We use `pytest` for unit testing. All new features must include corresponding tests in the `tests/` directory.

```bash
make test
```

## 📐 Standards

- **Architecture**: Keep the pipelines decoupled. Use the `AudioPipeline`, `VisualPipeline`, and `LipsyncPipeline` classes for logic.
- **Configuration**: Avoid hardcoding paths. Add new settings to `config.yaml`.
- **Logging**: Use the shared logger from `logger_utils.py` instead of `print()`.
- **Performance**: Test large changes with `make benchmark` to ensure generation speed is maintained.

## 🚀 Pull Request Process

1. Create a new branch for your feature or bugfix.
2. Ensure all tests pass.
3. Update the `README.md` if you add new CLI flags or features.
4. Submit the PR with a clear description of the changes and a sample transformation result if possible.

## 📄 License
By contributing, you agree that your contributions will be licensed under the project's MIT License.
