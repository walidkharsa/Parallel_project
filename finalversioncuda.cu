#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <float.h>

// Define constants
#define INPUT_SIZE 224
#define KERNEL_SIZE 3
#define NUM_CHANNELS 3
#define NUM_CLASSES 1000
#define BLOCK_SIZE 16

// Device kernel for shared convolution
__global__ void convolution_3x3_shared(
    float *input, float *kernel, float *output, 
    int input_size, int channels, int filters
) {
    // Shared memory for input tile
    extern __shared__ float shared_input[];

    // Calculate global thread index
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // Thread index within block
    int local_tx = threadIdx.x;
    int local_ty = threadIdx.y;

    // Half kernel size for indexing
    int pad = KERNEL_SIZE / 2;

    // Load input tile into shared memory
    for (int c = 0; c < channels; ++c) {
        int shared_idx = (local_ty + pad) * (blockDim.x + 2 * pad) + (local_tx + pad);
        int input_idx = (c * input_size + ty) * input_size + tx;

        if (ty < input_size && tx < input_size) {
            shared_input[shared_idx] = input[input_idx];
        } else {
            shared_input[shared_idx] = 0.0f;  // Handle boundary conditions
        }

        // Load halo cells for shared memory
        if (local_tx < pad) {
            shared_input[shared_idx - pad] = 
                (tx >= pad) ? input[input_idx - pad] : 0.0f;
            shared_input[shared_idx + blockDim.x] = 
                (tx + blockDim.x < input_size) ? input[input_idx + blockDim.x] : 0.0f;
        }

        if (local_ty < pad) {
            shared_input[shared_idx - pad * (blockDim.x + 2 * pad)] = 
                (ty >= pad) ? input[input_idx - pad * input_size] : 0.0f;
            shared_input[shared_idx + blockDim.y * (blockDim.x + 2 * pad)] = 
                (ty + blockDim.y < input_size) ? input[input_idx + blockDim.y * input_size] : 0.0f;
        }
    }
    __syncthreads();

    // Compute convolution if within valid output region
    if (tx < input_size && ty < input_size) {
        for (int f = 0; f < filters; ++f) {
            float result = 0.0f;

            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < KERNEL_SIZE; ++i) {
                    for (int j = 0; j < KERNEL_SIZE; ++j) {
                        result += shared_input[
                            (local_ty + i) * (blockDim.x + 2 * pad) + (local_tx + j)
                        ] * kernel[(f * channels + c) * KERNEL_SIZE * KERNEL_SIZE + (i * KERNEL_SIZE + j)];
                    }
                }
            }
            output[(f * input_size + ty) * input_size + tx] = result;
        }
    }
}

// Device kernel for convolution
__global__ void convolution_3_x_3_gpu(float *input, float *kernel, float *output, int input_size, int channels, int filters) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < input_size && ty < input_size) {
        float result = 0.0;
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    int x = tx - KERNEL_SIZE / 2 + i;
                    int y = ty - KERNEL_SIZE / 2 + j;
                    if (x >= 0 && y >= 0 && x < input_size && y < input_size) {
                        result += input[(c * input_size + y) * input_size + x] * kernel[(filters * channels + c) * KERNEL_SIZE * KERNEL_SIZE + (i * KERNEL_SIZE + j)];
                    }
                }
            }
        }
        output[(filters * input_size + ty) * input_size + tx] = result;
    }
}

// Device kernel for ReLU activation
__global__ void relu_gpu(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]); // Intristic function to avoid thread divergence
    }
}

// Device kernel for max pooling
__global__ void maxpooling(float *input, float *output, int input_size, int pool_size) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int output_size = input_size / pool_size;

    if (tx < output_size && ty < output_size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int x = tx * pool_size + i;
                int y = ty * pool_size + j;
                max_val = max(max_val, input[y * input_size + x]);
            }
        }
        output[ty * output_size + tx] = max_val;
    }
}

// Device kernel for dense layer
__global__ void dense_gpu(float *input, float *weights, float *output, int input_size, int output_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < output_size) {
        float sum = 0.0;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + tid];
        }
        output[tid] = sum;
    }
}


__global__ void dense_gpu_tiled(float *input, float *weights, float *output, int input_size, int output_size) {
    // Shared memory for tiles
    __shared__ float tile_input[BLOCK_SIZE];
    __shared__ float tile_weights[BLOCK_SIZE][BLOCK_SIZE];

    // Thread index
    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tx; // Output neuron index

    float result = 0.0;

    // Iterate over tiles of the input vector and weights
    for (int tile_start = 0; tile_start < input_size; tile_start += BLOCK_SIZE) {
        // Load input tile into shared memory
        if (tile_start + tx < input_size)
            tile_input[tx] = input[tile_start + tx];
        else
            tile_input[tx] = 0.0;

        // Load weight tile into shared memory
        if (row < output_size && (tile_start + tx) < input_size)
            tile_weights[tx][threadIdx.y] = weights[row * input_size + tile_start + tx];
        else
            tile_weights[tx][threadIdx.y] = 0.0;

        __syncthreads();

        // Compute partial sum for this tile
        for (int i = 0; i < BLOCK_SIZE; i++) {
            result += tile_input[i] * tile_weights[i][tx];
        }

        __syncthreads();
    }

    // Store the result
    if (row < output_size)
        output[row] = result;
}


void initialize_dense_weights(float *weights, float *biases, int input_size, int output_size) {
    // Seed for random number generation
    srand(time(0));
    for (int i = 0; i < input_size * output_size; ++i) {
        weights[i] = ((float)rand() / RAND_MAX) * 0.01f; // Small random values
    }
    for (int i = 0; i < output_size; ++i) {
        biases[i] = 0.0f; // Bias initialized to zero
    }
}


void initialize_data_from_file(const char *filename, float **input, int input_size, int channels) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the input
    *input = (float *)malloc(input_size * input_size * channels * sizeof(float));
    if (*input == NULL) {
        printf("Error: Unable to allocate memory for input\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read data from file
    for (int i = 0; i < input_size * input_size * channels; i++) {
        if (fscanf(file, "%f", &(*input)[i]) != 1) {
            printf("Error: Insufficient data in file %s\n", filename);
            fclose(file);
            free(*input);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
    printf("Data successfully loaded from %s\n", filename);
}

// Function to initialize weights and data
void initialize_data(float **input, float **weights, float **output, int input_size, int channels, int filters, int dense_input, int dense_output) {
    // Allocate and initialize host data
    *input = (float *)malloc(input_size * input_size * channels * sizeof(float));
    *weights = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * channels * filters * sizeof(float));
    *output = (float *)malloc(input_size * input_size * filters * sizeof(float));
    for (int i = 0; i < input_size * input_size * channels; i++) {
        (*input)[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE * channels * filters; i++) {
        (*weights)[i] = rand() / (float)RAND_MAX;
    }
}

__global__ void softmax_gpu(float *data, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;

    // Step 1: Compute the max value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = max(max_val, data[i]);
    }
    shared_mem[tid] = max_val;
    __syncthreads();

    // Step 2: Subtract max and compute exponentials
    float sum = 0.0;
    for (int i = tid; i < size; i += blockDim.x) {
        data[i] = expf(data[i] - shared_mem[0]);
        sum += data[i];
    }
    shared_mem[tid] = sum;
    __syncthreads();

    // Step 3: Normalize to get probabilities
    for (int i = tid; i < size; i += blockDim.x) {
        data[i] /= shared_mem[0];
    }
}

// Main function
int main() {
    // Host and device pointers
    float *d_intermediate;
    float *h_input, *h_weights, *h_output, *d_input, *d_weights, *d_output;


    // VGG16 Architecture Specifications
    int input_size = INPUT_SIZE, channels = NUM_CHANNELS;
    int filters[] = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512};
    int dense_input = 512 * 7 * 7, dense_output = NUM_CLASSES;
    int intermediate_size = input_size * input_size * NUM_CHANNELS;
    
    // Read input image data from a file
    const char *filename = "cat.txt";  // Replace with your actual file name
    initialize_data_from_file(filename, &h_input, input_size, channels);
    
    // Randomly initialize weights
    initialize_data(&h_weights, NULL, NULL, input_size, channels, filters[0], dense_input, dense_output);

    // Allocate device memory
    cudaMalloc(&d_intermediate, intermediate_size * sizeof(float)); // Replace intermediate_size as needed
    cudaMalloc(&d_input, input_size * input_size * channels * sizeof(float));
    cudaMalloc(&d_weights, KERNEL_SIZE * KERNEL_SIZE * channels * filters[0] * sizeof(float));
    cudaMalloc(&d_output, input_size * input_size * filters[0] * sizeof(float));

    // Copy input and weights to device
    cudaMemcpy(d_input, h_input, input_size * input_size * channels * sizeof(float), cudaMemcpyHostToDevice);
    
    // Total execution time start
    clock_t start_time = clock();

    // Timing variables for CUDA events
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Layers execution
    float *current_input = d_input, *current_output = d_output;
    int current_input_size = input_size;

    // Convolutional and pooling layers
    for (int i = 0; i < 13; i++) {
        // Allocate new buffers for output as needed
        cudaMalloc(&current_output, current_input_size * current_input_size * filters[i] * sizeof(float));
        
        // Start timing for convolution
        cudaEventRecord(start, 0);
        
        // Launch convolution kernel
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((current_input_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (current_input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        int shared_mem_size = (BLOCK_SIZE + 2 * (KERNEL_SIZE / 2)) * (BLOCK_SIZE + 2 * (KERNEL_SIZE / 2)) * sizeof(float);
        
        convolution_3x3_shared<<<gridSize, blockSize, shared_mem_size>>>(
            current_input, d_weights, current_output, current_input_size, channels, filters[i]);
        cudaDeviceSynchronize();
        
        // End timing for convolution
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Convolution Layer %d execution time: %.2f ms\n", i + 1, elapsedTime);

        // Max pooling layer after each block
        if ((i + 1) % 2 == 0 || i == 12) {  // Pool after 2 conv layers or last
            float *pooled_output;
            int pool_size = 2;
            int pooled_size = current_input_size / pool_size;
            cudaMalloc(&pooled_output, pooled_size * pooled_size * filters[i] * sizeof(float));
            
            // Start timing for pooling
            cudaEventRecord(start, 0);
                                          
            dim3 poolGridSize((pooled_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              (pooled_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

            maxpooling<<<poolGridSize, blockSize>>>(current_output, pooled_output, current_input_size, pool_size);
            cudaDeviceSynchronize();
            
            // End timing for pooling
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            printf("Max Pooling Layer %d execution time: %.2f ms\n", (i + 1) / 2, elapsedTime);
                                          
            // Free unused memory
            cudaFree(current_output);
            current_output = pooled_output;
            current_input_size = pooled_size;
        }

        // Update input for the next layer
        if (i < 12) { // No need to reallocate input for the last layer
            cudaFree(current_input);
            current_input = current_output;
        }
    }


    // -------- Dense Layers --------
    // Flatten the input (Assume already flattened in GPU memory)
    float *d_fc1_weights, *d_fc2_weights, *d_fc3_weights;
    float *d_fc1_output, *d_fc2_output;

    // Allocate device memory for weights and intermediate outputs
    cudaMalloc(&d_fc1_weights, dense_input * 4096 * sizeof(float));
    cudaMalloc(&d_fc2_weights, 4096 * 4096 * sizeof(float));
    cudaMalloc(&d_fc3_weights, 4096 * dense_output * sizeof(float));

    cudaMalloc(&d_fc1_output, 4096 * sizeof(float));
    cudaMalloc(&d_fc2_output, 4096 * sizeof(float));
    cudaMalloc(&d_output, dense_output * sizeof(float));

    // Initialize dense layer weights (host-side and then copy to device)
    initialize_dense_weights(h_weights, d_fc1_weights, dense_input, 4096);
    initialize_dense_weights(h_weights, d_fc2_weights, 4096, 4096);
    initialize_dense_weights(h_weights, d_fc3_weights, 4096, dense_output);

    // Fully connected layer 1: (512 * 7 * 7 -> 4096)
    cudaEventRecord(start, 0);
    dim3 fcGridSize1((4096 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dense_gpu<<<fcGridSize1, BLOCK_SIZE>>>(d_input, d_fc1_weights, d_fc1_output, dense_input, 4096);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Fully Connected Layer 1 execution time: %.2f ms\n", elapsedTime);

    // Fully connected layer 2: (4096 -> 4096)
    cudaEventRecord(start, 0);
    dim3 fcGridSize2((4096 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dense_gpu<<<fcGridSize2, BLOCK_SIZE>>>(d_fc1_output, d_fc2_weights, d_fc2_output, 4096, 4096);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Fully Connected Layer 2 execution time: %.2f ms\n", elapsedTime);

    // Fully connected layer 3: (4096 -> 1000)
    cudaEventRecord(start, 0);
    dim3 fcGridSize3((NUM_CLASSES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dense_gpu<<<fcGridSize3, BLOCK_SIZE>>>(d_fc2_output, d_fc3_weights, d_output, 4096, NUM_CLASSES);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Fully Connected Layer 3 execution time: %.2f ms\n", elapsedTime);
    
    // Apply softmax
    int num_classes = NUM_CLASSES;
    int threads = 256;
    int blocks = (num_classes + threads - 1) / threads;
    cudaEventRecord(start, 0);
    softmax_gpu<<<blocks, threads, threads * sizeof(float)>>>(d_output, num_classes);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Softmax execution time: %.2f ms\n", elapsedTime);

    // -------- Copy Output and Save --------
    
    memset(h_output, 0, output_size * sizeof(float)); 
    cudaMemcpy(h_output, d_output, dense_output * sizeof(float), cudaMemcpyDeviceToHost);

    // Write to file
    FILE *output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        printf("Error opening file for writing.\n");
        return -1;
    }
    for (int i = 0; i < dense_output; i++) {
        fprintf(output_file, "%f\n", h_output[i]);
    }
    fclose(output_file);
    printf("Results saved to output.txt\n");
    
    // Total execution time end
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total program execution time: %.2f seconds\n", total_time);

    // -------- Free Memory --------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_intermediate);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc3_weights);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_output);

    free(h_input);
    free(h_weights);
    free(h_output);

    return 0;
}
