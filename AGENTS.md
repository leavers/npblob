# npblob - AGENTS.md

## Project Overview

**npblob** is a library for transferring NumPy arrays between Python and JavaScript. It provides a binary serialization format that allows efficient transfer of n-dimensional arrays along with optional metadata between the two languages.

This is a dual-language project with parallel implementations in Python and TypeScript, sharing a common binary protocol.

## Project Structure

```
/home/gaochang/Projects/npblob/
├── python/                  # Python implementation
│   ├── npblob/             # Main package
│   │   ├── __init__.py     # Package entry, exports encode/decode
│   │   └── encode.py       # Core encoding/decoding logic
│   ├── tests/              # Test directory (currently minimal)
│   ├── pyproject.toml      # Python project configuration
│   ├── noxfile.py          # Python-specific nox tasks
│   └── README.md
├── typescript/             # TypeScript/JavaScript implementation
│   ├── src/
│   │   └── npblob.ts       # Main source file
│   ├── test/
│   │   └── npblob.test.ts  # Vitest test suite
│   ├── dist/               # Build output (generated)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts      # Vite build configuration
│   └── biome.json          # Biome linting/formatting config
├── .github/workflows/      # CI/CD workflows
└── AGENTS.md               # This file
```

## Technology Stack

### Python
- **Package Manager**: PDM (Python Dependency Manager)
- **Build Backend**: pdm-backend
- **Task Runner**: nox
- **Testing**: pytest
- **Linting/Formatting**: ruff, autoflake, isort, mypy
- **Dependencies**: numpy (required), orjson/ujson (optional JSON libraries)
- **Supported Versions**: Python 3.8+

### TypeScript/JavaScript
- **Runtime**: Bun (primary), Node.js (compatible)
- **Package Manager**: Bun
- **Build Tool**: Vite with vite-plugin-dts
- **Testing**: Vitest
- **Linting/Formatting**: Biome
- **Output Formats**: ESM, CJS, UMD
- **Type Declarations**: Generated automatically

## Build Commands

### Python
```bash
cd python/

# Install dependencies
pdm install

# Run tests
pdm run pytest

# Format code
pdm run nox -s format

# Type checking
pdm run mypy npblob
```

### TypeScript
```bash
cd typescript

# Install dependencies
bun install

# Build
bun run build

# Run tests
bun run test

# Lint
bun run lint

# Format
bun run format
```

### TypeScript
```bash
cd typescript/

# Install dependencies
bun install

# Build
bun run build

# Run tests
bun run test

# Lint
bun run lint

# Format
bun run format
```

## Code Style Guidelines

### Python
- **Line length**: 88 characters (ruff configured)
- **Target Python**: 3.8+
- **Import style**: Use autoflake + isort for organization
- **Type hints**: Required, checked with mypy
- **Coverage**: branch coverage enabled with pytest-cov

### TypeScript
- **Indentation**: 2 spaces
- **Module system**: ESNext
- **Target**: ESNext
- **Strict mode**: Enabled
- **Formatter**: Biome (replaces Prettier/ESLint)

## Testing Instructions

### Python Tests
- Test files located in `python/tests/`
- Currently minimal test coverage (test files exist but are empty)
- Run with: `pdm run pytest` or `pytest` from python directory
- Coverage reporting configured via `pytest-cov`

### TypeScript Tests
- Test file: `typescript/test/npblob.test.ts`
- Uses Vitest testing framework
- Tests encode/decode roundtrips and streaming functionality
- Run with: `bun run test`
- Coverage with: `bun run test:coverage`

## Binary Protocol

The npblob format encodes:
1. **Data type flag** (1 byte): Signed integer indicating dtype and endianness
2. **Shape info** (1+ bytes): Number of dimensions and shape array
3. **Raw data**: The array buffer content
4. **Optional extra metadata**: JSON or raw bytes with encoding flag

Multiple arrays can be concatenated with null byte (`\x00`) separators.

Supported dtypes:
- uint8, uint16, uint32 (uint64 not supported in JS)
- int8, int16, int32 (int64 not supported in JS)
- float16 (emulated), float32, float64

## Development Workflow

### Adding Features
1. Implement in both Python and TypeScript for parity
2. Update tests in both languages
3. Ensure binary compatibility between implementations

### Version Management
- Python version defined in `python/npblob/__init__.py`
- TypeScript version defined in `typescript/package.json`
- Keep versions in sync for releases

## Deployment Process

### Python Package
- Published to PyPI via GitHub Actions
- Triggered on git tag push (any tag pattern)
- Uses PDM's trusted publishing (OIDC)
- Creates GitHub Release with distribution artifacts

### TypeScript Package
- Published to npm (configuration present, workflow details in package.json)
- Build outputs to `typescript/dist/`
- Files included in package: `dist/` directory only

## Security Considerations

- Binary parsing uses `np.frombuffer()` in Python - ensure input validation
- TypeScript parsing validates chunk boundaries to prevent buffer overruns
- JSON parsing in both languages should be aware of potential large payloads
- No encryption or authentication in the protocol - transport security is the caller's responsibility

## Dependencies to be Aware of

### Python
- **numpy**: Core dependency for array handling
- **orjson/ujson**: Optional faster JSON libraries (fallback to stdlib)

### TypeScript
- **Bun**: Primary runtime for development
- **Vite**: Build tool
- **Biome**: All-in-one linter/formatter

## Notes for AI Agents

- This is a binary serialization library - changes to the format must maintain backward compatibility or be carefully versioned
- Both implementations must stay in sync for interoperability
- The TypeScript implementation includes streaming support (`stream()` function) that doesn't exist in Python
- Float16 is emulated in JavaScript (converted to Float32) due to lack of native support
- The project uses different tooling in each language but aims for functional parity
