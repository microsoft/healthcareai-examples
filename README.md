# HealthcareAI Examples

This repository contains a examples for healthcare AI, providing various examples to streamline the use of the models in the Microsoft HealthcareAI ecosystem. Use the samples in this repository to explore and implement healthcare AI scenarios.

**Disclaimer**: _The Microsoft healthcare AI models, code and examples are intended for research and model development exploration. The models, code and examples are not designed or intended to be deployed in clinical settings as-is nor for use in the diagnosis or treatment of any health or medical condition, and the individual models’ performances for such purposes have not been established. You bear sole responsibility and liability for any use of the healthcare AI models, code and examples, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals._

## Getting Started

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/microsoft/healthcareai-examples.git
    cd healthcareai-examples
    ```

2. **Set up your environment**:

    ### 1. Prerequisites

    To run most examples, you will need to download the data and have an appropriate endpoint deployed.

    #### Download data

    Use the following command to download the dataset with samples into your data folder located at `/home/azureuser/data/healthcare-ai/`:

    ```sh
    azcopy copy --recursive https://azuremlexampledata.blob.core.windows.net/data/healthcare-ai/ /home/azureuser/data/
    ```

    #### Deploy and configure an endpoint

    To run the examples you will need to access to a Azure-deployed endpoints:

    **Model Endpoints**

    You can use the the SDK to programmatically deploy the endpoints:
    * [MedImageInsight deployment](https://aka.ms/healthcare-ai-examples-mi2-deploy) 
    * [MedImageParse deployment](https://aka.ms/healthcare-ai-examples-mip-deploy) 
    * [CXRReportGen deployment](https://aka.ms/healthcare-ai-examples-cxr-deploy)

    #### Set up .env file

    You need to set up your environment variables by creating a `.env` file. An example file named `env.example` is provided in the repository. Copy this file to create your own `.env` file:

    ```sh
    cp env.example .env
    ```

    After copying, open the `.env` file and fill in the values.


    #### Sample Toolkit Installation

    **Prerequisites**

    - Python version: `>=3.9.0,<3.12`
    - pip version: `>=21.3`

    Many examples require the `healthcareai_toolkit` package. Install it by running the following command in the repository root:

    ```sh
    pip install package
    ```

    If you wish to edit the package easily, you can also install it in editable mode using the `-e` flag:
    
    ```sh
    pip install -e package
    ``` 

3. **Examples and Sample Code**:
    Explore the notebooks in the `azureml` directory to see various examples of how to use the healathcare ai models.


## Folder Structure

- **azureml**: Contains Jupyter notebooks and scripts for deploying and using AI models with Azure Machine Learning.
  - **cxrreportgen**: Notebooks for deploying and and examples using CXRReportGen.
  - **medimageinsight**: Notebooks for deploying and examples using the MedImageInsight.
  - **medimageparse**: Notebooks for deploying and and examples using MedImageParse.
- **package**: Contains a helper toolkit and model libraries.
  - **healthcareai_toolkit**: Helper utilities and functions for to run the examples.
  - **model_library**: Useful pre-defined models and related utilities.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
 