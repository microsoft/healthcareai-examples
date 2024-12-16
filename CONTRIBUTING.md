# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Goals

This repository contains a toolkit for healthcare AI examples, providing various utilities and models to streamline the use of the examples in the Microsoft HealthcareAI ecosystem. Use the samples in this repository to explore and implement healthcare AI scenarios.

## Code Quality Guidelines for Adding Example Code

When contributing new examples please follow these guidelines:

* Examples must be **clear**, **consistent**, and **modular**.
* Examples should manage **dependencies** carefully and must **run without error**. 
* Please provide thorough **documentation and explanations** including **diagrams** and **visualizations**. 
* Please optimize for **performance** by handling **data** responsibly. 
* Following **security** best practices to maintain high code quality and consistency across the repository.
* **Use repository libraries**: Whenever possible, use the `healthcareai_toolkit` library when possible, particularly for loading and preparing images, instead of writing new code.

Below are more detailed guidelines.

### Guidelines

#### Clarity
- Ensure your code is easy to understand.
- Use clear and descriptive variable and function names.
- Include comments and docstrings explaining complex sections, functions, and classes.

#### Consistency
- Follow the existing coding style and conventions used in the repository.
- Adhere to a consistent code style (e.g., following PEP 8 guidelines for Python).
- Maintain consistent formatting, indentation, and organization.

#### Modularity
- Write modular code by encapsulating repetitive tasks into functions or classes.
- Enhance code reusability and clarity through modular design.
- If code does not contribute to the understanding of the core principles explained in the notebook, consider moving it into a helper file.

#### Executable Code
- Ensure that notebooks run from start to finish without errors.
- You may leave the output of cells in the notebook. However, if you are leaving cell output, make sure it is meaningful and adds to the understanding of the notebook. Do not leave error output or extremely lengthy/repetitive output. Make sure that output does not contain your personal information like user IDs, keys, or local paths.
- Implement error handling to make the code robust against invalid inputs or unexpected situations.
- Use try-except blocks where appropriate to handle exceptions gracefully.

#### Documentation
- Provide a clear explanation of what the example does.
- Include any assumptions, prerequisites, and setup instructions necessary to run the code.

#### Diagrams
- Diagrams are highly encouraged to explain what is happening in the notebook.
- Use diagrams to illustrate workflows, data processing steps, model architectures, etc.
- Ensure diagrams are clear, properly labeled, and enhance the understanding of the content.

#### Visualization Standards
- Use clear and informative visualizations.
- Include titles, axis labels, legends, and captions to make plots self-explanatory.

#### Execution Time
- Ensure that each notebook can be completed within 20-30 minutes.
- Optimize your code for large datasets or time-sensitive operations.
- Use efficient algorithms and data structures appropriate for the task.
- Consider including precomputed results or smaller datasets if necessary to meet this requirement.

#### Resource Management
- Be mindful of memory and computational resource usage, especially with large datasets.
- Optimize code to prevent unnecessary resource consumption.

#### Dependencies
- List any external dependencies.
- Ensure they are necessary and up-to-date.
- Avoid including unnecessary packages.

#### Reproducibility
- Specify versions for critical dependencies and include them in your prerequisites.

#### Data
- No data can be included in the repository.
- Provide a public location or method for obtaining the data.
- Contact us if there are any issues.

#### Security and Privacy
- Ensure that security best practices are followed.
- Do not hard-code credentials, API keys, or personal data.
- Use secure methods to handle sensitive information.
- Use environment variables or configuration files (excluded from version control) for any credentials or API keys.

