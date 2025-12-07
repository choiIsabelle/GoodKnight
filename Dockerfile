FROM public.ecr.aws/lambda/python:3.12

# Copy only requirements files first (cached unless dependencies change)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY src/GoodKnightCommon/requirements.txt ${LAMBDA_TASK_ROOT}/GoodKnightCommon-requirements.txt
COPY src/GoodKnightModel/requirements-cpu.txt ${LAMBDA_TASK_ROOT}/GoodKnightModel-requirements.txt

# Install all dependencies in one layer
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r GoodKnightCommon-requirements.txt && \
    pip install --no-cache-dir -r GoodKnightModel-requirements.txt

# Copy application code after (this layer changes more frequently)
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Fix file permissions to ensure Lambda can read all files
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}

CMD ["lambda_handler.handler"]
