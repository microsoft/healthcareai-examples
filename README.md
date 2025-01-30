# Healthcare AI Examples

## Introduction

Welcome to the Healthcare AI Examples repository! This repository is designed to help you get started with Microsoft's healthcare AI models. Whether you are a researcher, data scientist, or developer, you will find a variety of examples and solution templates that showcase how to leverage these powerful models for different healthcare scenarios. From basic deployment and usage patterns to advanced solutions addressing real-world medical problems, this repository aims to provide you with the tools and knowledge to build and implement healthcare AI solutions using Microsoft AI ecosystem effectively.

**Disclaimer**: _The Microsoft healthcare AI models, code and examples are intended for research and model development exploration. The models, code and examples are not designed or intended to be deployed in clinical settings as-is nor for use in the diagnosis or treatment of any health or medical condition, and the individual models’ performances for such purposes have not been established. You bear sole responsibility and liability for any use of the healthcare AI models, code and examples, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals._

## What's in the repository?

In this repository you will find examples and solution templates that will help you get started with multimodal Healthcare AI models available in Microsoft AI Foundry. This is what is available:

### Deployment samples and basic usage examples

These notebooks show how to programmatically deploy some of the models available in the catalog:

* [MedImageInsight](https://aka.ms/healthcare-ai-examples-mi2-deploy)
* [MedImageParse](https://aka.ms/healthcare-ai-examples-mip-deploy)
* [CXRReportGen](https://aka.ms/healthcare-ai-examples-cxr-deploy)

### Basic usage examples and patterns

These notebooks show basic patterns that require very little specialized knowledge about medical data or implementation specifics.

* [MedImageParse call patterns](./azureml/medimageparse/medimageparse_segmentation_demo.ipynb) - a collection of snippets showcasing how to send various image types to MedImageParse and retrieve segmentation masks. See how to read and package xrays, ophthalmology images, CT scans, pathology patches, and more.
* [Zero shot classification with MedImageInsight](./azureml/medimageinsight/zero-shot-classification.ipynb) - learn how to use MedImageInsight to perform zero-shot classification of medical images using its text or image encoding abilities.
* [Training adapters using MedImageInsight](./azureml/medimageinsight/adapter-training.ipynb) - build on top of zero shot pattern and learn how to train simple task adapters for MedImageInsight to create classification models out of this powerful image encoder. For additional thoughts on when you would use this and the zero shot patterns as well as considerations on fine tuning, [read our blog on Microsoft Techcommunity Hub](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/unlocking-the-magic-of-embedding-models-practical-patterns-for-healthcare-ai/4358000).
* [Advanced calling patterns](./azureml/medimageinsight/advanced-call-example.ipynb) - no production implementation is complete without understanding how to deal with concurrent calls, batches, efficient image preprocessing, and deep understanding of parallelism. This notebook contains snippets that will help you write more efficient code to build your cloud-based healthcare AI systems.

### Advanced examples and Solution templates

These examples take a closer look at certain solutions and patterns of usage for the multimodal healthcare AI models to address real world medical problems.

* [Detecting outliers in MedImageInsight](./azureml/medimageinsight/outlier-detection-demo.ipynb) - go beyond encoding single image instances and learn how to use MedImageInsight to encode CT/MR series and studies, and detect outliers in image collections.
* [Exam Parameter Detection](./azureml/medimageinsight/exam-parameter-demo/exam-parameter-detection.ipynb) - dealing with entire MRI imaging series, this notebook explores an approach to a common problem in radiological imaging - normalizing and understanding image acquisition parameters. Surprisingly (or not), in many cases DICOM metadata can not be relied upon to retrieve exam parameters. Take a look inside this notebook to understand how you can build a computationally efficient exam parameter detection system using an embedding model like MedImageInsight.
* [Multimodal image analysis using radiology and pathology imaging](./azureml/advanced_demo/radpath/rad_path_survival_demo.ipynb) - can foundational models be connected together to build systems that understand multiple modalities? This notebook shows a way this can be done using the problem of predicting cancer hazard score via a combination of MRI studies and digital pathology slides. Also [read our blog](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/cancer-survival-with-radiology-pathology-analysis-and-healthcare-ai-models-in-az/4366241) that goes into more depth on this topic.

## Getting Started

To get started with this project, follow these steps:

### 1. Clone the repository

```sh
git clone https://github.com/microsoft/healthcareai-examples.git
cd healthcareai-examples
```

### 2. Set up your environment

#### Prerequisites

To run most examples, you will need to download the data and have an appropriate endpoint deployed.

#### Download data

The sample data used by the examples is located in our Blob Storage account. 

Use the following command to download the dataset with samples into your data folder located at `/home/azureuser/data/healthcare-ai/` (note that you will need to use [azcopy tool](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy)):

```sh
azcopy copy --recursive https://azuremlexampledata.blob.core.windows.net/data/healthcare-ai/ /home/azureuser/data/
```

#### Deploy and configure an endpoint

To run the examples you will need to access to a Azure-deployed endpoints. You can use the the SDK to programmatically deploy the endpoints:

* [MedImageInsight deployment](https://aka.ms/healthcare-ai-examples-mi2-deploy)
* [MedImageParse deployment](https://aka.ms/healthcare-ai-examples-mip-deploy)
* [CXRReportGen deployment](https://aka.ms/healthcare-ai-examples-cxr-deploy)

#### Set up .env file

You need to set up your environment variables by creating a `.env` file. The environment variables define parameters like endpoint paths, keys, etc. You don't need to set them all up upfront, each notebook will describe which values it relies upon. An example file named `env.example` is provided in the repository. Copy this file to create your own `.env` file:

```sh
cp env.example .env
```

After copying, open the `.env` file and fill in the values as you need them.

#### Healthcare AI Toolkit Installation

A lot of useful functions that facilitate working with endpoints, DICOM files, etc, have been organized into a simple package called **healthcareai_toolkit** that goes alongside this repository to make the code inside the notebooks cleaner. In order to install it, follow the steps below:

##### Package Prerequisites

* Python version: `>=3.9.0,<3.12`
* pip version: `>=21.3`

Many examples in this repository require the `healthcareai_toolkit` package. Install it by running the following command in the repository root:

```sh
pip install ./package/
```

If you wish to edit the package easily, you can also install it in editable mode using the `-e` flag:

```sh
pip install -e ./package/
```

### 3. Examples and Sample Code

Now you are ready to explore the notebooks in the `azureml` directory to see various examples of how to use the healthcare ai models!

## Folder Structure

* **azureml**: Contains Jupyter notebooks and scripts for deploying and using AI models with Azure Machine Learning. Inside you will find various folders with sample notebooks such as
  * **cxrreportgen**: Notebooks for deploying and and examples using CXRReportGen.
  * **medimageinsight**: Notebooks for deploying and examples using the MedImageInsight.
  * **medimageparse**: Notebooks for deploying and and examples using MedImageParse.
  * and many more as this repository grows!
* **package**: Contains the helper toolkit and model libraries.
  * **healthcareai_toolkit**: Helper utilities and functions for to run the examples.
  * **model_library**: Useful pre-defined models and related utilities.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Read [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
 