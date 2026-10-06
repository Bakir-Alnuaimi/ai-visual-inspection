# Recipe for the "shipping container" of our API

# 1. Start from a small, official Python image (Linux inside)
FROM python:3.12-slim

# 2. All following commands run inside the folder /app
WORKDIR /app

# 3. Install the libraries FIRST (slow step).
#    Docker remembers this step: if only our code changes, it is not repeated.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 4. Download the pretrained ResNet18 ("the eyes") once, while building
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

# 5. Copy our Python code into the container
COPY *.py ./

# 6. Start the API. 0.0.0.0 = accept requests from outside the container
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
