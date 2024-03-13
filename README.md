# Project Title: LLM Auto-Eval

This Python application is to evaluate GenAI pipelines.

## Features
- Process set of .pdf files (or .zip file)
- Generate Summaries and Q&A pairs of the docs

## Installation and Setup

### Prerequisites

You need to have Python 3.8 or later to use this application. If you don't have Python installed, you can download it from the official site: https://www.python.org/downloads/

### Steps

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/Markovian99/llm-auto-eval.git
   ```
   
2. Navigate to the cloned project directory.

   ```bash
   cd "Auto-GenAI"
   ```
   
3. Install the necessary packages using pip. (We recommend using a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```
   
4. In the project's root directory, create a `.env` file to store your API keys securely.
   
5. Open the `.env` file using any text editor and enter your API keys as shown below:

   ```bash
   export BARD_API_KEY = "YOUR BARD API KEY"
   export OPENAI_API_KEY = "YOUR OPENAI API KEY"
   ```
   If using Azure for OPEN AI, also include
   ```bash
   export OPENAI_API_BASE = "YOUR OPENAI API BASE"
   export OPENAI_API_TYPE = "YOUR OPENAI API TYPE"
   export OPENAI_API_VERSION = "YOUR OPENAI API VERSION"
   ```

## Usage

To run the LLM Auto-Eval application:

```bash
cd src
streamlit run app.py
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the terms of the Apache License 2.0. For more details, please see the [LICENSE](LICENSE) file.

## Support

For any questions or issues, please contact the maintainers, or raise an issue in the GitHub repository.
