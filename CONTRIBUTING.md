# Contributing to Predictive Maintenance with ML

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search the [Issues](https://github.com/harshitaphadtare/predictive-maintenance/issues) page to see if the bug has already been reported.

2. **Create a new issue** - If the bug hasn't been reported, create a new issue with:
   - Clear and descriptive title
   - Detailed description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment information (OS, Python version, etc.)

### Suggesting Enhancements

1. **Check existing issues** - Search for similar feature requests
2. **Create a feature request** - Describe the enhancement and its benefits
3. **Provide use cases** - Explain how this would be useful

### Code Contributions

#### Prerequisites

- Python 3.11 or higher
- Git
- Basic knowledge of machine learning concepts

#### Development Setup

1. **Fork the repository**

   ```bash
   git clone https://github.com/yourusername/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Guidelines

##### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting

```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/
```

##### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Aim for good test coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

##### Documentation

- Add docstrings to new functions and classes
- Update README.md if needed
- Include examples for new features

##### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add support for FD002 dataset"

# Bad
git commit -m "fix stuff"
```

#### Pull Request Process

1. **Ensure your code follows the style guidelines**
2. **Add tests for new functionality**
3. **Update documentation if needed**
4. **Push your changes**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots if UI changes

## 📋 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Be open to different viewpoints
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Any conduct inappropriate in a professional setting

## 🏷️ Issue Labels

We use the following labels to categorize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested

## 📚 Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pytest Testing Framework](https://docs.pytest.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

## 🎯 Areas for Contribution

### High Priority

- **Performance improvements** - Optimize feature engineering or model training
- **Additional algorithms** - Implement new ML models
- **Better evaluation metrics** - Add more comprehensive metrics
- **Documentation** - Improve README, add tutorials

### Medium Priority

- **Data preprocessing** - Add more data cleaning options
- **Visualization** - Add plotting functions for results
- **Configuration** - More flexible configuration options
- **Testing** - Add more unit tests

### Low Priority

- **UI improvements** - Web interface or dashboard
- **Deployment** - Docker containerization
- **CI/CD** - GitHub Actions workflows

## 🙏 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- GitHub contributors page

## 📞 Questions?

If you have questions about contributing:

- Check the [Issues](https://github.com/harshitaphadtare/predictive-maintenance/issues) page
- Create a new issue with the `question` label
- Contact the maintainers directly

Thank you for contributing to Predictive Maintenance with ML! 🚀
