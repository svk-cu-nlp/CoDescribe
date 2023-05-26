# An AI-based Query-Answering Tool for Code Documentation and Summarization


## 🔗 Features
- Code explanation summary generation.
- Customized documentation generation of source code.
- Syntactic error checking and alteration of codes.
- Conversion of code into other programming languages.
- Chat-based conversation regarding the provided source codes.

## Getting Started
### ⚙️ Setup
```bash
pip install -r requirements.txt
```
### 🔌 Setting OpenAI API
- Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
- Add the API Key and Organization Key in [config file](src/config.py)
### 💻 Execution
To Run the normal version of the app where user has to upload source codes
```bash
streamlit run ./src/app.py
```
To run the app with Git support where user has to provide the link of the GitHub link
```bash
streamlit run ./src/app_with_git.py
```
