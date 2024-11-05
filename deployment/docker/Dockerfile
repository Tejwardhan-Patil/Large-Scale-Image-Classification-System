# Stage 1: Build the C++ components
FROM gcc:11.2 as build-stage
WORKDIR /app

# Copy C++ source files
COPY core_components/data/scripts/cpp_augment.cpp /app/
COPY core_components/data/scripts/cpp_preprocess.cpp /app/
COPY core_components/models/custom_layer.cpp /app/
COPY core_components/deployment/api/cpp_backend.cpp /app/
COPY core_components/utils/cpp_utils.cpp /app/

# Build the C++ components
RUN g++ cpp_augment.cpp -o cpp_augment
RUN g++ cpp_preprocess.cpp -o cpp_preprocess
RUN g++ custom_layer.cpp -o custom_layer
RUN g++ cpp_backend.cpp -o cpp_backend
RUN g++ cpp_utils.cpp -o cpp_utils

# Stage 2: Build the final image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy Python dependencies and install them
COPY core_components/deployment/api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY core_components/ /app/

# Copy built C++ binaries from the build stage
COPY --from=build-stage /app/cpp_augment /usr/local/bin/
COPY --from=build-stage /app/cpp_preprocess /usr/local/bin/
COPY --from=build-stage /app/custom_layer /usr/local/bin/
COPY --from=build-stage /app/cpp_backend /usr/local/bin/
COPY --from=build-stage /app/cpp_utils /usr/local/bin/

# Expose the port for the API
EXPOSE 5000

# Command to run the API server
CMD ["python", "deployment/api/app.py"]