# Contributing to PG&E Rate Calculator

We welcome contributions to the PG&E Rate Calculator! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/pge-calc.git
   cd pge-calc
   ```
3. **Install dependencies** using Poetry:
   ```bash
   poetry install
   ```

## Development Setup

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Running the Application
```bash
# Using the runner script
python run_app.py

# Or directly with Streamlit
poetry run streamlit run main.py
```

## Making Changes

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and well-named

### Project Structure
```
pge-calc/
├── src/
│   └── pge_calculator/
│       ├── __init__.py
│       ├── calculator.py      # Core calculation logic
│       └── app.py            # Streamlit interface
├── data/
│   └── pge_rates.csv         # Rate data
├── tests/                    # Test files (if any)
├── main.py                   # Main entry point
├── run_app.py               # Runner script
├── pyproject.toml           # Project configuration
└── README.md
```

### Commit Guidelines
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, etc.)
- Keep the first line under 50 characters
- Add details in the body if needed

Example:
```
Add EV rate plan comparison feature

- Implement EV2-A and EV-B rate calculations
- Add visualization for EV-specific time periods
- Update tests for new rate plans
```

## Testing

If you add new features, please include tests:
```bash
# Run tests (when test suite is available)
poetry run pytest
```

## Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Screenshots if applicable
   - Reference any related issues

## Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Include steps to reproduce
- Provide system information
- Include error messages and screenshots

### Feature Requests
- Describe the feature and its benefits
- Explain the use case
- Consider implementation complexity

### Code Contributions
- New rate plans or calculation methods
- UI/UX improvements
- Performance optimizations
- Documentation improvements

### Data Updates
- Updated PG&E rates
- New territory information
- Baseline allowance changes

## Code Review Process

1. All submissions require review
2. Maintainers will review and provide feedback
3. Address review comments
4. Once approved, changes will be merged

## Questions?

Feel free to open an issue for questions about:
- Development setup
- Implementation details
- Feature suggestions
- Bug reports

## Recognition

Contributors will be acknowledged in the README and release notes.

Thank you for contributing to the PG&E Rate Calculator! 