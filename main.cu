#include <stdio.h>
#include <cuda_runtime.h>


__constant__ unsigned char sBox[256];

__device__ unsigned char gmul(unsigned char a, unsigned char b) {
    unsigned char p = 0;  // the product of the multiplication
    for (int i = 0; i < 8; i++) {
        if (b & 1) {
            p ^= a; // add 'a' to 'p' if the least significant bit of 'b' is set
        }
        bool hi_bit_set = (a & 0x80); // check if the high bit of 'a' is set
        a <<= 1; // left shift 'a' by 1
        if (hi_bit_set) {
            a ^= 0x1B; // XOR 'a' with the irreducible polynomial if the high bit was set
        }
        b >>= 1; // right shift 'b' by 1
    }
    return p;
}

__global__ void addRoundKey(unsigned char *state, unsigned char *roundKey) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    state[idx] ^= roundKey[idx];
}

__global__ void subBytes(unsigned char *state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    state[idx] = sBox[state[idx]];
}

__global__ void shiftRows(unsigned char *state) {
    __shared__ unsigned char temp[16];
    int idx = threadIdx.x;

    // Calculate shifted index
    int shiftedIdx = (idx % 4) * 4 + (idx / 4);
    temp[shiftedIdx] = state[idx];

    __syncthreads();

    if (idx < 16) {
        state[idx] = temp[idx];
    }
}

__global__ void mixColumns(unsigned char *state) {
    __shared__ unsigned char temp[16];
    int idx = threadIdx.x;

    if (idx < 4) { // Each thread handles one column
        int base = idx * 4;
        temp[base] = (gmul(state[base], 2) ^ gmul(state[base+1], 3) ^ state[base+2] ^ state[base+3]);
        temp[base+1] = (state[base] ^ gmul(state[base+1], 2) ^ gmul(state[base+2], 3) ^ state[base+3]);
        temp[base+2] = (state[base] ^ state[base+1] ^ gmul(state[base+2], 2) ^ gmul(state[base+3], 3));
        temp[base+3] = (gmul(state[base], 3) ^ state[base+1] ^ state[base+2] ^ gmul(state[base+3], 2));
    }

    __syncthreads();

    if (idx < 16) {
        state[idx] = temp[idx];
    }
}
// Function to initialize data for simplicity
void initData(unsigned char *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = i % 256;  // Dummy data; in real applications, this would be your plaintext/ciphertext
    }
}

int main() {
    const int dataSize = 16; // Size of the AES block

    unsigned char h_state[dataSize];
    unsigned char h_roundKey[dataSize];
    initData(h_state, dataSize);
    initData(h_roundKey, dataSize);

    unsigned char *d_state, *d_roundKey;

    // Allocate device memory
    cudaMalloc((void **)&d_state, dataSize * sizeof(unsigned char));
    cudaMalloc((void **)&d_roundKey, dataSize * sizeof(unsigned char));

    // Copy data from host to device
    cudaMemcpy(d_state, h_state, dataSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_roundKey, h_roundKey, dataSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch kernels
    addRoundKey<<<1, dataSize>>>(d_state, d_roundKey);
    subBytes<<<1, dataSize>>>(d_state);
    shiftRows<<<1, dataSize>>>(d_state);
    mixColumns<<<1, 4>>>(d_state); // 4 threads, each handles a column

    // Copy result back to host
    cudaMemcpy(h_state, d_state, dataSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_roundKey);

    // Print result
    printf("Processed State:\n");
    for (int i = 0; i < dataSize; i++) {
        printf("%02x ", h_state[i]);
        if ((i + 1) % 4 == 0) printf("\n");
    }

    return 0;
}