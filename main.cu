#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>




#define AES_KEY_SIZE 16  // AES-128
#define AES_ROUND_KEY_SIZE 176 // 16 * (10 + 1)

__constant__ unsigned char sBox[256];

void KeyExpansion(const uint8_t* key, uint8_t* roundKeys) {
    int i, j;
    uint8_t temp[4], k;

    // The first round key is the key itself.
    for (i = 0; i < AES_KEY_SIZE; i++) {
        roundKeys[i] = key[i];
    }

    // All other round keys are found from the previous round keys.
    for (; i < AES_ROUND_KEY_SIZE; i += 4) {
        for (j = 0; j < 4; j++) {
            temp[j] = roundKeys[i - 4 + j];
        }

        if (i % AES_KEY_SIZE == 0) {
            // Rotate the 4-byte word
            k = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = k;

            // Apply SBox
            temp[0] = sBox[temp[0]];
            temp[1] = sBox[temp[1]];
            temp[2] = sBox[temp[2]];
            temp[3] = sBox[temp[3]];

            // XOR with round constant
            temp[0] = temp[0] ^ (0x01 << (i / AES_KEY_SIZE - 1)); // Example for Rcon, needs correct implementation
        }

        roundKeys[i] = roundKeys[i - AES_KEY_SIZE] ^ temp[0];
        roundKeys[i + 1] = roundKeys[i + 1 - AES_KEY_SIZE] ^ temp[1];
        roundKeys[i + 2] = roundKeys[i + 2 - AES_KEY_SIZE] ^ temp[2];
        roundKeys[i + 3] = roundKeys[i + 3 - AES_KEY_SIZE] ^ temp[3];
    }
}

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

static const char *BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void base64_encode(const unsigned char *input, int input_length, char *output) {
    int i, j, k, b;
    for (i = 0, j = 0; i < input_length; i += 3) {
        b = (input[i] & 0xFC) >> 2;
        output[j++] = BASE64_CHARS[b];
        b = (input[i] & 0x03) << 4;
        if (i + 1 < input_length) {
            b |= (input[i + 1] & 0xF0) >> 4;
            output[j++] = BASE64_CHARS[b];
            b = (input[i + 1] & 0x0F) << 2;
            if (i + 2 < input_length) {
                b |= (input[i + 2] & 0xC0) >> 6;
                output[j++] = BASE64_CHARS[b];
                b = input[i + 2] & 0x3F;
                output[j++] = BASE64_CHARS[b];
            } else {
                output[j++] = BASE64_CHARS[b];
                output[j++] = '=';
            }
        } else {
            output[j++] = BASE64_CHARS[b];
            output[j++] = '=';
            output[j++] = '=';
        }
    }
    output[j] = '\0';
}

int calc_base64_encoded_len(int input_length) {
    int n = input_length;
    return ((n + 2) / 3 * 4) + 1;  // Plus one for the null terminator
}

void printData(const char* tag, const unsigned char* data, int size) {
    printf("%s: ", tag);
    for (int i = 0; i < size; i++) {
        printf("%02x", data[i]);
        if ((i + 1) % 16 == 0) printf("\n");
        else printf(" ");
    }
    printf("\n");
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    const char *inputFileName = "input.txt";
    const char *outputFileName = "encrypted_output.txt";

    // 读取输入文件
    FILE *file = fopen(inputFileName, "rb");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    unsigned char *fileData = (unsigned char*)malloc(fileSize);
    if (!fileData) {
        perror("Failed to allocate memory for file data");
        fclose(file);
        return 1;
    }

    fread(fileData, 1, fileSize, file);
    fclose(file);
 //   printData("Initial Data", fileData, fileSize);

    // 密钥和轮密钥
    uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c}; // 示例密钥
    uint8_t roundKeys[176]; // 为AES-128存储轮密钥
    KeyExpansion(key, roundKeys);

    // CUDA内存分配
    unsigned char *d_state, *d_roundKeys;
    cudaMalloc((void **)&d_state, fileSize);
    cudaMalloc((void **)&d_roundKeys, 176);
    cudaMemcpy(d_state, fileData, fileSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roundKeys, roundKeys, 176, cudaMemcpyHostToDevice);

    clock_t AESstart, AESend;
    double AES_time_used;

    AESstart = clock();
    // 执行AES加密
    int numBlocks = fileSize / 16; // 假设文件大小是16的倍数
    for (int round = 0; round <= 10; ++round) {
        addRoundKey<<<numBlocks, 16>>>(d_state, d_roundKeys + round * 16);
        cudaDeviceSynchronize(); // 确保CUDA操作完成
        cudaMemcpy(fileData, d_state, fileSize, cudaMemcpyDeviceToHost);
      //  printData("After addRoundKey", fileData, fileSize);

        if (round < 10) {
            subBytes<<<numBlocks, 16>>>(d_state);
            cudaDeviceSynchronize();
            cudaMemcpy(fileData, d_state, fileSize, cudaMemcpyDeviceToHost);
         //   printData("After subBytes", fileData, fileSize);

            shiftRows<<<numBlocks, 16>>>(d_state);
            cudaDeviceSynchronize();
            cudaMemcpy(fileData, d_state, fileSize, cudaMemcpyDeviceToHost);
           // printData("After shiftRows", fileData, fileSize);

            if (round < 9) {
                mixColumns<<<numBlocks, 4>>>(d_state);
                cudaDeviceSynchronize();
                cudaMemcpy(fileData, d_state, fileSize, cudaMemcpyDeviceToHost);
               // printData("After mixColumns", fileData, fileSize);
            }
        }
    }
    AESend = clock();
    AES_time_used = ((double) (AESend - AESstart)) / CLOCKS_PER_SEC;
    printf("AES took %f seconds to execute \n", AES_time_used);
    // 从设备获取加密数据
    cudaMemcpy(fileData, d_state, fileSize, cudaMemcpyDeviceToHost);
   // printData("Final Encrypted Data", fileData, fileSize);

    // Base64编码
    int encodedSize = calc_base64_encoded_len(fileSize);
    char *base64EncodedData = (char *)malloc(encodedSize);
    if (!base64EncodedData) {
        perror("Failed to allocate memory for Base64 encoded data");
        return 1;
    }

    base64_encode(fileData, fileSize, base64EncodedData);
   // printf("Base64 Encoded Data: %s\n", base64EncodedData);

    // 将Base64编码的数据写入文件
    FILE *outFile = fopen(outputFileName, "w");
    if (!outFile) {
        perror("Failed to open output file");
        free(base64EncodedData);
        return 1;
    }
    fprintf(outFile, "%s", base64EncodedData);
    fclose(outFile);




    // 清理
    free(base64EncodedData);
    free(fileData);
    cudaFree(d_state);
    cudaFree(d_roundKeys);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Encryption took %f seconds to execute \n", cpu_time_used);
    printf("comm took %f seconds to execute \n", cpu_time_used-AES_time_used);

    printf("Encryption complete. Output written to '%s'\n", outputFileName);
    return 0;
}