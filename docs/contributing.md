# Contributing Guide

## Getting Started

Thank you for your interest in contributing to the Gen AI Services platform! This guide will help you get started with contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork
   ```bash
   git clone https://github.com/your-username/gen-ai-reusable.git
   cd gen-ai-reusable
   ```
3. Set up development environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Create a development branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable names
- Comment complex logic

### Example

```python
from typing import List, Optional

def process_document(
    document_id: str,
    content: str,
    options: Optional[dict] = None
) -> dict:
    """
    Process a document with the given options.

    Args:
        document_id: Unique identifier for the document
        content: Document content to process
        options: Optional processing parameters

    Returns:
        dict: Processing results

    Raises:
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")

    # Process document
    result = {
        "id": document_id,
        "status": "processed"
    }

    return result
```

## Testing

### Writing Tests

1. Unit Tests
   - Test individual components
   - Use pytest
   - Mock external dependencies
   - Test edge cases

2. Integration Tests
   - Test component interactions
   - Use test databases
   - Test API endpoints
   - Test service communication

3. Performance Tests
   - Test under load
   - Measure response times
   - Test resource usage
   - Test scaling behavior

### Example Test

```python
import pytest
from services.format_conversion_service import convert_document

def test_convert_document():
    """Test document conversion functionality."""
    # Arrange
    document_id = "test123"
    content = "Test content"
    options = {"quality": "high"}

    # Act
    result = convert_document(document_id, content, options)

    # Assert
    assert result["id"] == document_id
    assert result["status"] == "processed"
    assert "converted_content" in result

def test_convert_document_invalid_id():
    """Test document conversion with invalid ID."""
    # Arrange
    document_id = ""
    content = "Test content"

    # Act & Assert
    with pytest.raises(ValueError):
        convert_document(document_id, content)
```

## Documentation

### Code Documentation

1. Docstrings
   - Use Google style
   - Include type hints
   - Document exceptions
   - Provide examples

2. Comments
   - Explain complex logic
   - Document assumptions
   - Reference related code
   - Keep comments up to date

### API Documentation

1. OpenAPI/Swagger
   - Document all endpoints
   - Include request/response examples
   - Document error responses
   - Keep documentation current

2. README Updates
   - Update installation instructions
   - Document new features
   - Update configuration
   - Add usage examples

## Git Workflow

### Branch Strategy

1. Main Branches
   - main: Production code
   - develop: Development code

2. Feature Branches
   - feature/*: New features
   - bugfix/*: Bug fixes
   - hotfix/*: Production fixes

### Commit Messages

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

### Pull Request Process

1. Create Pull Request
   - Use template
   - Link related issues
   - Add reviewers
   - Add labels

2. Code Review
   - Address comments
   - Update documentation
   - Run tests
   - Squash commits

3. Merge
   - Get approval
   - Pass CI/CD
   - Update documentation
   - Delete branch

## Review Process

### Code Review Guidelines

1. What to Look For
   - Code quality
   - Test coverage
   - Documentation
   - Performance
   - Security

2. Review Comments
   - Be constructive
   - Explain reasoning
   - Suggest improvements
   - Reference standards

### Review Checklist

- [ ] Code follows style guide
- [ ] Tests are included
- [ ] Documentation is updated
- [ ] No security issues
- [ ] Performance is considered
- [ ] Error handling is complete
- [ ] Dependencies are managed
- [ ] Configuration is documented

## Release Process

### Versioning

- Follow semantic versioning
- Update CHANGELOG.md
- Tag releases
- Update documentation

### Release Steps

1. Prepare Release
   - Update version
   - Update dependencies
   - Run tests
   - Update docs

2. Create Release
   - Create tag
   - Generate changelog
   - Create release notes
   - Deploy to staging

3. Deploy
   - Deploy to production
   - Monitor deployment
   - Verify functionality
   - Update status

## Getting Help

### Resources

1. Documentation
   - API docs
   - Architecture docs
   - Development guide
   - Contributing guide

2. Communication
   - GitHub issues
   - Team chat
   - Email
   - Meetings

### Asking Questions

1. Before Asking
   - Check documentation
   - Search issues
   - Try debugging
   - Prepare context

2. When Asking
   - Be specific
   - Provide context
   - Share error messages
   - Show code snippets

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 