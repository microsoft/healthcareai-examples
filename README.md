# Healthcare AI Examples
Healthcare AI Examples is a comprehensive collection of code samples, templates, and solution patterns that demonstrate how to deploy and use Microsoft's healthcare AI models across diverse medical scenarios—from basic model deployment to advanced multimodal healthcare applications.

This repository contains comprehensive information to help you get started with Microsoft's cutting-edge healthcare AI models.

> [!IMPORTANT]
> Healthcare AI Examples is a code sample collection intended for research and model development exploration only. The models, code and examples are not designed or intended to be deployed in clinical settings as-is nor for use in the diagnosis or treatment of any health or medical condition, and the individual models' performances for such purposes have not been established. By using the Healthcare AI Examples, you are acknowledging that you bear sole responsibility and liability for any use of these models and code, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals.

## Features

- **Model Deployment Patterns**: Programmatic deployment examples for key healthcare AI models including MedImageInsight, MedImageParse, and CXRReportGen
- **Basic Usage Examples**: Zero-shot classification, image segmentation, and foundational calling patterns for medical imaging
- **Advanced Solutions**: Multimodal analysis, outlier detection, exam parameter extraction, and 3D image search capabilities
- **Production-Ready Code**: Concurrent calling patterns, batch processing, and efficient image preprocessing for scalable healthcare AI systems
- **Fine-tuning Templates**: Complete workflows for adapter training and model fine-tuning using AzureML pipelines
- **Real-World Applications**: Cancer survival prediction, radiology-pathology analysis, and clinical decision support scenarios
- **Integrated Toolkit**: Helper utilities and model libraries through the `healthcareai_toolkit` package
- **Azure Integration**: Seamless deployment using Azure Developer CLI (azd) and Azure Machine Learning

## What's Available

### 🚀 Deployment Samples and Basic Usage Examples

These notebooks show how to programmatically deploy some of the models available in the catalog:

* **[MedImageInsight](https://aka.ms/healthcare-ai-examples-mi2-deploy)** [MI2] - Image and text embedding foundation model deployment
* **[MedImageParse](https://aka.ms/healthcare-ai-examples-mip-deploy)** [MIP] - Medical image segmentation model deployment  
* **[CXRReportGen](https://aka.ms/healthcare-ai-examples-cxr-deploy)** [CXR] - Chest X-ray report generation model deployment
* **Providence-GigaPath** [PGP] - Embedding model specifically for histopathology

### 📋 Basic Usage Examples and Patterns

These notebooks show basic patterns that require very little specialized knowledge about medical data or implementation specifics:

* **[MedImageParse call patterns](./azureml/medimageparse/medimageparse_segmentation_demo.ipynb)** [MIP] - a collection of snippets showcasing how to send various image types to MedImageParse and retrieve segmentation masks. See how to read and package xrays, ophthalmology images, CT scans, pathology patches, and more.
* **[Zero shot classification](./azureml/medimageinsight/zero-shot-classification.ipynb)** [MI2] - learn how to use MedImageInsight to perform zero-shot classification of medical images using its text or image encoding abilities.
* **[Training adapters](./azureml/medimageinsight/adapter-training.ipynb)** [MI2] - build on top of zero shot pattern and learn how to train simple task adapters for MedImageInsight to create classification models out of this powerful image encoder. For additional thoughts on when you would use this and the zero shot patterns as well as considerations on fine tuning, [read our blog on Microsoft Techcommunity Hub](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/unlocking-the-magic-of-embedding-models-practical-patterns-for-healthcare-ai/4358000).
* **[Advanced calling patterns](./azureml/medimageinsight/advanced-call-example.ipynb)** [MI2] - no production implementation is complete without understanding how to deal with concurrent calls, batches, efficient image preprocessing, and deep understanding of parallelism. This notebook contains snippets that will help you write more efficient code to build your cloud-based healthcare AI systems.
* **[Fine-tuning MedImageInsight with AzureML Pipelines](./azureml/medimageinsight/finetuning/mi2-finetuning.ipynb)** [MI2] - comprehensive guide through prerequisites, data preprocessing, GPU-accelerated training, model deployment, and performance validation. Read [our blog](https://aka.ms/MedImageFinetuning) for additional insights on fine-tuning strategies.

### 🏥 Advanced Examples and Solution Templates

These examples take a closer look at certain solutions and patterns of usage for the multimodal healthcare AI models to address real world medical problems:

* **[Detecting outliers in MedImageInsight](./azureml/medimageinsight/outlier-detection-demo.ipynb)** [MI2] - go beyond encoding single image instances and learn how to use MedImageInsight to encode CT/MR series and studies, and detect outliers in image collections. Learn more in our [detailed resource guide](https://aka.ms/HLSOutlierDetection).
* **[Exam Parameter Detection](./azureml/medimageinsight/exam-parameter-demo/exam-parameter-detection.ipynb)** [MI2, GPT*] - dealing with entire MRI imaging series, this notebook explores an approach to a common problem in radiological imaging - normalizing and understanding image acquisition parameters. Surprisingly (or not), in many cases DICOM metadata can not be relied upon to retrieve exam parameters. Take a look inside this notebook to understand how you can build a computationally efficient exam parameter detection system using an embedding model like MedImageInsight.
* **[Multimodal image analysis using radiology and pathology imaging](./azureml/advanced_demos/radpath/rad_path_survival_demo.ipynb)** [MI2, PGP] - can foundational models be connected together to build systems that understand multiple modalities? This notebook shows a way this can be done using the problem of predicting cancer hazard score via a combination of MRI studies and digital pathology slides. Also [read our blog](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/cancer-survival-with-radiology-pathology-analysis-and-healthcare-ai-models-in-az/4366241) that goes into more depth on this topic.
* **[Image Search Series Pt 1: Searching for similar XRay images](./azureml/advanced_demos/image_search/2d_image_search.ipynb)** [MI2] - an opener in the series on image-based search. How do you use foundation models to build an efficient system to look up similar Xrays? Read [our blog](https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/image-search-series-part-1-chest-x-ray-lookup-with-medimageinsight/4372736) for more details.
* **[Image Search Series Pt 2: 3D Image Search with MedImageInsight](./azureml/advanced_demos/image_search/3d_image_search.ipynb)** [MI2] - expanding on the image-based search topics we look at 3D images. How do you use foundation models to build a system to search the archive of CT scans for those with similar lesions in the pancreas? Read [our blog](https://aka.ms/3DImageSearch) for more details.

### 🤖 Agentic AI Examples

These examples demonstrate how to build intelligent conversational agents that integrate healthcare AI models with natural language understanding:

* **[Medical Image Classification Agent](./azureml/medimageinsight/agent-classification-example.ipynb)** [MI2, GPT] - build a conversational AI agent that classifies medical images through natural language interactions. Learn practical patterns for coordinating image data with LLM function calls, managing conversation state, and routing image analysis tasks to MedImageInsight embeddings.

## Getting Started

To get started with using our healthcare AI models and examples, follow the instructions below to set up your environment and run the sample applications.

### Prerequisites

> [!IMPORTANT]
> Follow the steps in order. Each step builds on the previous ones, and jumping ahead may require restarting deployments that can take significant time to complete. Detailed documentation is linked for each step if you need additional context.

- **Azure Subscription** with access to:
  - Azure Machine Learning workspace _or_ permissions to create one.
    - See [required permissions](#required-permissions) for details and [Step 3](#step-3-deploy-healthcare-ai-models) for deployment options.
  - Models deployed or permissions to deploy them into a subscription or AzureML workspace.
    - See [Step 3](#step-3-deploy-healthcare-ai-models) for deployment options and tips on selecting models.
  - GPU compute resource availablity (quota) for model deployments.
    - See [Step 1](#step-1-verify-prerequisites-quota) for details.
  - **Optional**: Azure OpenAI access for GPT models (limited use in examples).
- **Tools**:
  - **For running examples**:
    - Python `>=3.10.0,<3.12` and pip `>=21.3` (for running locally)
    - [Git LFS](https://git-lfs.github.com/) for cloning the data repository
  - **For deploying models**:
    - [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
    - [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd?tabs=winget-windows%2Cbrew-mac%2Cscript-linux&pivots=os-linux)


### Step 1: Verify Prerequisites (Quota)

Before deploying, verify your Azure subscription has sufficient quota and your account has the necessary permissions to avoid resource limitations during deployment.

**Azure Quota Management Tools**
- [Azure Machine Learning Quota Management](https://ml.azure.com/quota) - For GPU compute quota
- [Azure AI Management Center](https://ai.azure.com/managementCenter/quota) - For Azure OpenAI quota management

**Quota Requirements** 

You need quota for **one or more** of the following:

| Model | VM Family | Instance Type | Instance Count | Cores per Instance | Min Cores Needed |
|-------|-----------|---------------|----------------|-------------------|---------------------|
| **MedImageInsight** | NCasT4_v3 | `Standard_NC4as_T4_v3` | 2 | 4 | **8 cores** |
| **MedImageParse** | NCadsH100_v5 | `Standard_NC40ads_H100_v5` | 1 | 40 | **40 cores** |
| **CXRReportGen** | NCadsH100_v5 | `Standard_NC40ads_H100_v5` | 1 | 40 | **40 cores** |
| **Prov-GigaPath*** | NCv3 | `Standard_NC6s_v3` | 1 | 6 | **6 cores** |
| **GPT-4o or GPT-4.1** (optional) | GlobalStandard | GPT-4o or GPT-4.1 | - | - | **50K-100K TPM** |

*Used in advanced demos only

> [!TIP]
> **Healthcare AI Models**: All healthcare AI models (MedImageInsight, MedImageParse, CXRReportGen, Prov-GigaPath) require GPU compute quota as shown above.
>
> **GPT Models (Optional)**: GPT models are deployed to Azure AI Services with Tokens Per Minute (TPM) capacity instead of compute cores. GPT deployment is completely optional and can be skipped by leaving the `AZURE_GPT_MODEL` environment variable empty.
>
> **Quota Management**: We recommend requesting quota for **all models** and requesting **2-3x+ the minimum cores** shown above. Requesting quota does not incur any charges - you only pay for what you actually use. Having extra quota available prevents deployment delays and allows for scaling when needed.
> 
> **⚠️ ONGOING COSTS**: Online model endpoints bill continuously while deployed, even when not actively processing requests. Monitor your usage in the Azure portal and use cleanup procedures when finished (see deployment-specific instructions in the deploy folders).

#### Required Permissions:

**To run examples only**: If models are already deployed by your admin, you only need access to the deployed model endpoints. Your admin can provide you with revelevant information and authentication credentials.

**To deploy models yourself**: 
- **Fresh deployment**: Requires **Owner** role OR **User Access Administrator + Contributor** roles on the Azure subscription to create resource groups, workspaces, and configure role-based access controls.
- **Existing deployment**: Requires **Contributor** role on the resource group containing your existing Azure ML workspace.

> [!TIP]
> If you lack deployment permissions, ask your IT administrator to either grant you the appropriate access in a resource group or deploy the models for you and provide the endpoint details.

### Step 2: Clone the Repository

```sh
git clone https://github.com/microsoft/healthcareai-examples.git
cd healthcareai-examples
```

### Step 3: Deploy Healthcare AI Models

The examples in this repository require AI model endpoints to be deployed. We provide several deployment methods to accommodate different workflows and preferences.

> [!WARNING]
> **⚠️ COST ALERT**: Deploying these models will create Azure resources that **incur charges**. Online model endpoints **continue billing even when idle**. Review the [quota requirements table](#step-1-verify-prerequisites-quota) to understand compute costs before proceeding. See deployment-specific cleanup instructions in the respective deploy folders when finished.

> [!TIP]
> **Not all models are required:** you can deploy only the subset you need for specific notebooks.
> **For basic examples**: Deploy only the specific model you want to explore:
>  - `cxrreportgen/` notebooks → **CXRReportGen** model
>  - `medimageinsight/` notebooks → **MedImageInsight** model  
>  - `medimageparse/` notebooks → **MedImageParse** model
>
> **For advanced demos**: You'll need **MedImageInsight** + **Prov-GigaPath** (plus others depending on the specific demo).
> See [Model Selection](./docs/deployment-guide.md#model-selection) in the deployment guide to select specific models.

#### Automatic Deployment - Recommended

The Azure Developer CLI provides automated infrastructure provisioning and configuration. See the [Deployment Guide](docs/deployment-guide.md) for more information. You can use the [Quickstart Deployment](docs/deployment-guide.md#quick-start) or choose the option that matches your situation:

> [!TIP]
> **Authentication Issues?** If the standard login commands fail (especially in constrained network environments or when using certain authentication methods), try using device code authentication instead:
> - `az login --use-device-code`
> - `azd auth login --use-device-code`
> 
> This opens a browser-based authentication flow that can work around common login issues.

**Deploy into your existing Azure ML workspace:**

*Recommended if:*
- ✅ You're running from an Azure ML workspace compute instance
- ✅ You have an existing workspace you want to use
- ✅ You have Contributor permissions (no role assignment permissions needed)

  *Quick Start*
  ```bash
  cd deploy/existing
  az login
  azd auth login
  azd env new <envName>
  # Auto-configure environment from current AML compute instance:
  ./setup_azd_env_for_aml.sh
  azd env set AZURE_GPT_LOCATION "southcentralus"
  azd env set AZURE_GPT_MODEL "gpt-4.1;2025-04-14"
  azd up
  ```
  See [Existing Deplpyment Guide](deploy/existing/README.md) for more details.


**Create a Fresh AML Environment**:

Creates a new resource group and Azure ML workspace from scratch.

*Recommended if:*
- ✅ You want to run the examples locally
- ✅ You need a completely new workspace setup
- ✅ You have Owner or User Access Administrator permissions
- ⚠️  Note: May be slower if you don't have a stable connection
  
  *Quick start*
  ```bash
  cd deploy/fresh
  az login
  azd auth login
  azd env new <envName>
  azd env set AZURE_LOCATION <location>
  azd env set AZURE_GPT_LOCATION <gpt_location> # if different from AZURE_LOCATION
  azd env set AZURE_GPT_MODEL "gpt-4.1;2025-04-14"
  azd up
  ```
  See [Fresh Deplpyment Guide](deploy/fresh/README.md) for more details.

> [!TIP]
> **GPT Model Integration**: Both deployment options now support optional GPT model deployment (GPT-4o or GPT-4.1) alongside healthcare AI models. This enables multimodal workflows combining medical imaging AI with language models. See the deployment guides for configuration details.

> [!NOTE]
> **For Admins**: You can deploy resources on behalf of another user by setting `AZURE_PRINCIPAL_ID` to their Azure AD object ID during deployment. This grants the target user access to the deployed resources while you maintain the deployment permissions. This is useful when deploying fresh infrastructure where role assignments are created.

#### Manual Deployment Methods

For users who prefer other deployment approaches, we provide instructions for:

- **[Complete Deployment Guide](docs/deployment-guide.md)** - Comprehensive guide covering all deployment options with troubleshooting.
- **[Manual Deployment](docs/manual-deployment.md)** - Portal and SDK deployment methods.

### Step 4: Setup your local environment

> [!CAUTION]
> If you followed the automatic deployment steps, you might currently be in either the `deploy/fresh/` or `deploy/existing/` directory. You should move back to the repository root level.

Now that you have deployed the models, you need to configure your local environment to use them effectively. This invols three key tasks: verifying your environment configuration, installing the required toolkit, and downloading sample data.

#### Verify Your Environment File

After deployment, verify that your root level `.env` file contains the necessary environment variables for connecting to your deployed models. Each automatic deployment method will configure this file with the appropriate settings for your chosen approach. 

> [!IMPORTANT]
> Check the value of `DATA_ROOT` in your `.env` file to ensure it's appropriate for your setup. The default value is `/home/azureuser/data/healthcare-ai/`, but you may need to modify it based on your environment. **Use an absolute path** (not a relative path like `./data/`) to ensure consistent access across different working directories. If you change the `DATA_ROOT` value, you'll also need to update the destination path in the git clone command in the following step.
>
> **Azure OpenAI Configuration**: If you deployed GPT models, your `.env` file will contain `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. The endpoint supports two formats:
> 1. **Full inference URI** (deployed automatically): `https://{your-service}.cognitiveservices.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}`.
> 2. **Base endpoint** (for manual configuration): `https://{your-service}.cognitiveservices.azure.com/` with separate `AZURE_OPENAI_DEPLOYMENT_NAME` variable.
>
> See `env.example`.

> [!NOTE]
> If you used a manual deployment method you will have to configure this file yourself, see [Manual Deployment](docs/manual-deployment.md) for more information.

#### Download Sample Data

The sample data used by the examples is available in the [healthcareai-examples-data](https://github.com/microsoft/healthcareai-examples-data) GitHub repository. 

> [!IMPORTANT]
> The data repository uses Git LFS (Large File Storage) for medical image files. Make sure you have [Git LFS](https://git-lfs.github.com/) installed before cloning. Without it, you'll only download placeholder files instead of the actual data.

Clone the repository to download the data:

```sh
git clone https://github.com/microsoft/healthcareai-examples-data.git /home/azureuser/data/healthcare-ai
```

> [!TIP]
> This downloads the entire dataset. If you prefer a different location, adjust the target path and update the `DATA_ROOT` value in your `.env` file accordingly. For more information about the data, see the [data repository README](https://github.com/microsoft/healthcareai-examples-data/blob/main/README.md).

#### Install Healthcare AI Toolkit

Install the helper toolkit that facilitates working with endpoints, DICOM files, and medical imaging:

```sh
pip install -e ./package/
```

After installation, you can test your endpoint connectivity:

```sh
# Test all configured endpoints
healthcareai-test

# Test specific model endpoint quietly
healthcareai-test --models cxr,pgp --quiet
```

### Step 5: Explore Examples

Now you're ready to explore the notebooks! Start with one of these paths:

**🎯 Beginners**: Try **[zero-shot classification](./azureml/medimageinsight/zero-shot-classification.ipynb)** and **[adapter training](./azureml/medimageinsight/adapter-training.ipynb)**.

**🔍 Image Segmentation**: Try **[segmentation patterns](./azureml/medimageparse/medimageparse_segmentation_demo.ipynb)**.

**📋 Report Generation**: See example usage in **[CXRReportGen deployment](./azureml/cxrreportgen/cxr-deploy.ipynb)**.

**🤖 Agentic AI**: Learn how to use models within an agentic framework with the **[medical image classification agent](./azureml/medimageinsight/agent-classification-example.ipynb)**.

**🚀 Advanced**: Explore **[image search](./azureml/advanced_demos/image_search/2d_image_search.ipynb)**, **[outlier detection](./azureml/medimageinsight/outlier-detection-demo.ipynb)**, or **[multimodal analysis](./azureml/advanced_demos/radpath/rad_path_survival_demo.ipynb)**.

## Project Structure

```
healthcareai-examples/
├── azureml/                   # Core notebooks and examples
│   ├── cxrreportgen/          # Chest X-ray report generation examples
│   ├── medimageinsight/       # Medical image embedding and analysis
│   ├── medimageparse/         # Medical image segmentation examples
│   └── advanced_demos/        # Advanced multimodal solutions
├── package/                   # Healthcare AI Toolkit
│   ├── healthcareai_toolkit/  # Helper utilities and functions
│   └── model_library/         # Pre-defined models and utilities
├── deploy/                    # Infrastructure as Code (Bicep templates)
├── docs/                      # Additional documentation
└── tests/                     # Test suites and validation notebooks
```

### Key Components

* **azureml**: Contains Jupyter notebooks and scripts for deploying and using AI models with Azure Machine Learning
  * **cxrreportgen**: Notebooks for deploying and examples using CXRReportGen
  * **medimageinsight**: Notebooks for deploying and examples using MedImageInsight  
  * **medimageparse**: Notebooks for deploying and examples using MedImageParse
  * **advanced_demos**: Complex multimodal healthcare applications
* **package**: Contains the helper toolkit and model libraries
  * **healthcareai_toolkit**: Helper utilities and functions to run the examples
  * **model_library**: Useful pre-defined models and related utilities

## See Also

- **[Healthcare Model Studio](https://aka.ms/healthcaremodelstudio)** - AI Foundry Healthcare Model Catalog
- **[CXRReportGen Model Card](https://aka.ms/cxrreportgenmodelcard)** - Model card for CXRReportGen, a chest X-ray report generation model
- **[MedImageParse Model Card](https://aka.ms/medimageparsemodelcard)** - Model card for MedImageParse, a model for medical image segmentation
- **[MedImageInsight Model Card](https://aka.ms/mi2modelcard)** - Model card for MedImageInsight, an image and text embedding foundation model

## Resources

### External Documentation

- [Foundation models for healthcare AI](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/healthcare-ai/healthcare-ai-models)
- [Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/)
- [Azure AI Services](https://learn.microsoft.com/azure/ai-services/)
- [Generative AI For Beginners](https://github.com/microsoft/generative-ai-for-beginners)

## How to Contribute

We welcome contributions to improve this project! Please see our [Contribution Guide](./CONTRIBUTING.md) for information on how to get started with contributing code, documentation, or other improvements. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq) or contact <opencode@microsoft.com> with any additional questions or comments.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Authorized Use

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
