# Portfolio Repository

## Introduction

This repository is a collection of projects showcasing my skills in Machine Learning and Software Development. Each project is structured to provide an overview, problem statement, solution approach, technologies used, and performance metrics.

---

## Table of Contents

1. [po001_mrm](#po001_mrm) ⸺ [🔗 Project Repo](https://github.com/cassidythilton/portfolio/tree/main/po001_mrm)
2. [po002_hou](#po002_hou) ⸺ [🔗 Project Repo](https://github.com/cassidythilton/portfolio/tree/main/po002_hou)

---

## po001_mrm

### Overview

The `po001_mrm` project focuses on automating the creation of model risk management reports in the financial services sector. It leverages the capabilities of ChatGPT-4 for text generation and completion, along with a custom application that performs specific query operations. 

### Problem Statement

Manual generation of model risk management reports can be labor-intensive and error-prone. These reports require a deep understanding of both machine learning models and financial metrics.

### Solution Approach

This project employs a pipeline where text requests from a model risk management report template are processed as queries or function calls. These queries fetch required variables, which are then summarized by ChatGPT-4. The project is comprised of notebook `AutoRiskReport_FinancialServices.ipynb` and helper file `helperMRM.py`. The notebook serves as the front end for the user whereas the helper file serves as the back end. The user will provide an MRM template with requests for specific information to be automated into individual sections (e.g. _"analyze model performance metrics for the current ended quarter (2023-01-01 to 2023-03-31) comparing to the prior ended quarter (2022-10-01 to 2022-12-31)"._ The solution reads and parses each request and then performs the appropriate tasks to accomodate each request. Finally, once complete an "out" file is rendered in the same `src` location as the template files. 

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1P-eHM2_z1wHfrq-WQEb3Z9gTyW2h52c6">
<br>
</div>  

### Technologies Used

- ChatGPT-4
- Python
- HTML
- Custom MLOps platform

### Code Sample: Task Identification and Querying

```python
def chat_with_model_tasks_opAiFuncN(analysisRequest, i, tokens=1000, temperature=0.05, max_retries=3):
    """
    Function to determine the most appropriate task for a given analysis request using OpenAI's GPT-4.
    
    Parameters:
    - analysisRequest: A text request specifying the desired analysis from MRM template.
    - i: Request index.
    - tokens, temperature: GPT-4 configurations.
    - max_retries: Maximum number of retry attempts.
    
    Returns:
    - A dictionary containing task details.
    """
    
    openai.api_key = setOpenAItoken(cluster, token) 
    retry_count = 0
    today = datetime.today().strftime('%Y-%m-%d')
    
    while retry_count <= max_retries:
        try:
            # Sample task list and request
            sysTasks = "Retrieve model performance, Retrieve feature drift."
            textRequest = f'You receive a request to "{analysisRequest}". Which task should be performed?'

            message_objs = [{'role': 'system', 'content': 'You are a data analyst.'}, {'role': 'user', 'content': sysTasks}, {'role': 'user', 'content': textRequest}]
            
            # Call to GPT-4
            response = openai.ChatCompletion.create(model="gpt-4", messages=message_objs, max_tokens=tokens, temperature=temperature)
            task_response = response.choices[0].message['content']
            
            print(f"Identified task: {task_response}")
            specificDates = f"Today's date is {today}."            
            message_objs.append({'role': 'user', 'content': specificDates})

            # ...
            
            if task_response:  # Placeholder condition
                return {"task": task_response}
            
            retry_count += 1
            
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}")
            time.sleep(2)

    print("Max retry attempts reached.")
    return None

```

### Results
The project has shown promising results in reducing the time and complexity of generating comprehensive model risk management reports saving impressive amounts of manual work across multiple FTE's. 

### Conclusion
The po001_mrm project successfully demonstrates how GenerativeAI and Large Language Models can be synergistically combined with traditional data querying mechanisms to optimize the generation of model risk management reports.

---
## po002_hou

### Overview

The `po002_hou` project focuses on predicting home prices using machine learning techniques, combining both image and numerical data. The solution features a Streamlit app that allows users to input parameters such as square footage, number of baths, and an image of the property to predict its value. It leverages two types of neural networks: a Convolutional Neural Network (CNN) for image analysis and a Multilayer Perceptron (MLP) for numerical analysis.

### Problem Statement

Determining an accurate and fair price for a home involves considering multiple factors, which can be both laborious and subjective. Real estate professionals and homeowners alike need an automated, unbiased solution for home valuation.

### Solution Approach

The project employs a two-pronged approach combining image and numerical data to predict home prices. The Streamlit app collects user inputs which are sent to a FastAPI backend for processing as well as MLFlow for artifact logging and monitoring. The neural networks provide their independent analysis, which is then merged into a final layer to output the home's estimated value. Moreover, the system identifies four comparable homes to provide additional context for the user.

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1PMdTgx-37RRg6Kp-cF5EkInkurxPbMKX">
<br>
</div>

### Technologies Used

- Streamlit
- FastAPI
- MLFlow
- Python
- Convolutional Neural Networks (CNN)
- Multilayer Perceptron (MLP)

### Code Sample: FastAPI Prediction Endpoint

```python
@app.post("/predict/")
async def predict(n_citi: float = Form(...), bed: float = Form(...), bath: float = Form(...), sqft: float = Form(...), file: UploadFile = File(...)):

    tabular_data = np.array([[n_citi, bed, bath, sqft]])
    tabular_data = tabular_data / np.array([citiM, bM, bathM, sqftM])  # Normalize like you did during training

    # Read and preprocess the image
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict([tabular_data, image])

    return {"prediction": float(prediction[0]*priceM)}  # Convert prediction to float or JSON serializable format

```


### FastAPI
FastAPI is a modern, high-performance web framework for building APIs with Python. In this home pricing prediction solution, FastAPI serves as the backbone for managing all API calls, specifically for handling prediction requests. When the Streamlit app gathers information about a property, such as square footage, number of bedrooms, and an image, it sends this data to a FastAPI endpoint. The endpoint then invokes the underlying machine learning model, which consists of a Convolutional Neural Network (CNN) for image analysis and a Multilayer Perceptron (MLP) for numerical data processing. The FastAPI application ensures that these operations are executed in a fast, efficient, and secure manner. Its benefits include automatic generation of OpenAPI documentation, validation of incoming requests, and support for concurrent handling of multiple requests, making it an ideal choice for building robust and scalable machine learning APIs.

<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1HUa6tvprIA1NXmuNfCGP4v1e9WCIRQmZ" width="1000">
</div>

### MLFlow
MLflow plays a crucial role in the management and tracking of machine learning models in this solution. It is responsible for logging various aspects of the model such as parameters, metrics, and artifacts. When the home price prediction model is trained using the Jupyter Notebook, metrics like accuracy, loss, and other performance indicators are logged into MLflow. These logs serve as an invaluable resource for understanding model behavior, debugging, and iterative development. MLflow's user interface also allows for the easy comparison of different model versions, thus assisting in model selection. The artifact logging feature is particularly beneficial for keeping track of the trained models, making it straightforward to roll back to a previous model version if needed. Overall, MLflow enhances the traceability and reproducibility of machine learning projects.
<div style="text-align:center">
    <img src="https://drive.google.com/uc?export=view&id=1KhJY0O5iFozWcyB0GAMYReffkohZyJBn" width="1000">
</div>

### Streamlit
Streamlit serves as the front-end interface for this solution, creating an intuitive environment for users to engage with the machine learning model. Users can easily input property-specific details such as square footage and the number of bathrooms, as well as upload images for more accurate predictions. These inputs are seamlessly routed to the FastAPI backend for processing. An additional feature of this application is its ability to not only predict the value of a property but also present data on four comparable homes, offering extra context to the estimated price. While Flask is often considered more robust for large-scale applications requiring complex customization, Streamlit excels in situations where rapid development and ease of use are prioritized. Its straightforward widgetry and compatibility with other technologies like FastAPI make it a highly effective tool for crafting user interfaces in machine learning projects.

<div align="center">
  <a href="https://drive.google.com/file/d/10pAADAGK6zB5AoUJNpA6uW9xLwWK6xdL/view?t=3s"><img src="https://drive.google.com/uc?export=view&id=1QaASEvndCqhkSg1O0qaSh7s_tnfiWs7Y" width="700"></a>
</div>

### Conclusion
The po002_hou project successfully employs machine learning techniques, specifically neural networks, to automate and optimize home price predictions. This approach not only increases efficiency but also adds a layer of objectivity to the valuation process.



