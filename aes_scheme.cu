#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

const u_int8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};
const u_int8_t inv_sbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

#define AES_BLOCK_SIZE 16

u_int8_t gmul(u_int8_t a, u_int8_t b) {
    u_int8_t p = 0;
    while (a != 0 && b != 0) {
        if (b & 1) /* if the polynomial for b has a constant term, add the corresponding a to p */
            p ^= a; /* addition in GF(2^m) is an XOR of the polynomial coefficients */

        if (a & 0x80) /* GF modulo: if a has a nonzero term x^7, then must be reduced when it becomes x^8 */
            a = (a << 1) ^ 0x11b; /* subtract (XOR) the primitive polynomial x^8 + x^4 + x^3 + x + 1 (0b1_0001_1011) – you can change it but it must be irreducible */
        else
            a <<= 1; /* equivalent to a*x */
        b >>= 1;
    }
    return p;
}

// Rcon table
const u_int8_t Rcon[11] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
};

// RotWord function (rotate 4-byte word left)
void RotWord(u_int8_t *word) {
    const u_int8_t temp = word[0];
    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = temp;
}

// SubWord function (apply S-box to 4-byte word)
void SubWord(u_int8_t *word) {
    for (int i = 0; i < 4; i++) {
        word[i] = sbox[word[i]];
    }
}

void key_expansion(u_int8_t short_key[16], u_int8_t round_key[176]) {
    u_int8_t temp[4];
    int i = 0;

    // Copy the original key to the expanded key
    for (i = 0; i < 16; i++) {
        round_key[i] = short_key[i];
    }

    // Generate the rest of the expanded key
    for (i = 16; i < 176; i += 4) {
        // Copy the previous 4 bytes to the temporary array
        for (int j = 0; j < 4; j++) {
            temp[j] = round_key[i - 4 + j];
        }

        // Perform key schedule core
        if (i % 16 == 0) {
            RotWord(temp);
            SubWord(temp);
            temp[0] ^= Rcon[i / 16 - 1];
        }

        // XOR temp with the 4-byte block n bytes before the new expanded key.
        for (int j = 0; j < 4; j++) {
            round_key[i + j] = round_key[i - 16 + j] ^ temp[j];
        }
    }
}

void AddRoundKey(u_int8_t data[16], u_int8_t round_key[16]) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
        data[i] ^= round_key[i];
}

void SubBytes(u_int8_t data[16]) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++)
        data[i] = sbox[data[i]];
}

void InverseSubBytes(u_int8_t data[16]) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        data[i] = inv_sbox[data[i]];
    }
}

void ShiftRows(u_int8_t data[16]) {
    // Define temporary array to store the shifted rows
    u_int8_t temp[16];

    // Perform the shift operation
    for (int i = 0; i < 16; i++) {
        int row = i / 4;
        int col = i % 4;
        int new_col = (col + row) % 4;
        temp[i] = data[row * 4 + new_col];
    }

    // Copy the shifted values back to the original data array
    for (int i = 0; i < 16; i++) {
        data[i] = temp[i];
    }
}

void InverseShiftRows(u_int8_t data[16]) {
    // Define temporary array to store the shifted rows
    u_int8_t temp[16];

    // Perform the inverse shift operation
    for (int i = 0; i < 16; i++) {
        int row = i / 4;
        int col = i % 4;
        int new_col = (col - row + 4) % 4; // Adjust for negative values
        temp[i] = data[row * 4 + new_col];
    }

    // Copy the shifted values back to the original data array
    for (int i = 0; i < 16; i++) {
        data[i] = temp[i];
    }
}

void MixColumns(u_int8_t data[16]) {
    for (int i = 0; i < 4; i++) {
        u_int8_t a[4];
        for (int j = 0; j < 4; j++) {
            a[j] = data[i + 4 * j]; // Copy the column into a temporary array
        }
        data[i] = gmul(a[0], 0x02) ^ gmul(a[1], 0x03) ^ a[2] ^ a[3];
        data[i + 4] = a[0] ^ gmul(a[1], 0x02) ^ gmul(a[2], 0x03) ^ a[3];
        data[i + 8] = a[0] ^ a[1] ^ gmul(a[2], 0x02) ^ gmul(a[3], 0x03);
        data[i + 12] = gmul(a[0], 0x03) ^ a[1] ^ a[2] ^ gmul(a[3], 0x02);
    }
}

void InverseMixColumns(u_int8_t data[16]) {
    for (int i = 0; i < 4; i++) {
        u_int8_t a[4];
        for (int j = 0; j < 4; j++) {
            a[j] = data[i + 4 * j]; // Copy the column into a temporary array
        }
        data[i] = gmul(a[0], 14) ^ gmul(a[1], 11) ^ gmul(a[2], 13) ^ gmul(a[3], 9);
        data[i + 4] = gmul(a[0], 9) ^ gmul(a[1], 14) ^ gmul(a[2], 11) ^ gmul(a[3], 13);
        data[i + 8] = gmul(a[0], 13) ^ gmul(a[1], 9) ^ gmul(a[2], 14) ^ gmul(a[3], 11);
        data[i + 12] = gmul(a[0], 11) ^ gmul(a[1], 13) ^ gmul(a[2], 9) ^ gmul(a[3], 14);
    }
}

void my_encrption(u_int8_t data[16], u_int8_t round_key[176]) {
    for (int round = 0; round <= 10; round++) {
        u_int8_t this_round_key[16];
        for (int j = 0; j < AES_BLOCK_SIZE; j++)
            this_round_key[j] = round_key[round*16+j];
        if (round==0) {
            AddRoundKey(data, this_round_key);
        }
        else if (round==10) {
            SubBytes(data);
            ShiftRows(data);
            AddRoundKey(data, this_round_key);
        }
        else {
            SubBytes(data);
            ShiftRows(data);
            MixColumns(data);
            AddRoundKey(data, this_round_key);
        }
    }
}

void my_decrption(u_int8_t data[16], u_int8_t round_key[176]) {
    for (int round = 10; round >= 0; round--) {
        u_int8_t this_round_key[16];
        for (int j = 0; j < AES_BLOCK_SIZE; j++)
            this_round_key[j] = round_key[round*16+j];
        if (round==0) {
            AddRoundKey(data, this_round_key);
        }
        else if (round==10) {
            AddRoundKey(data, this_round_key);
            InverseShiftRows(data);
            InverseSubBytes(data);
        }
        else {
            AddRoundKey(data, this_round_key);
            InverseMixColumns(data);
            InverseShiftRows(data);
            InverseSubBytes(data);
        }
    }
}

int main() {
    u_int8_t key[16] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x97, 0x25, 0x4e, 0x85, 0xe9, 0x92
    };
    u_int8_t round_key[176];
    key_expansion(key, round_key);

    u_int8_t data[16] = {
        0x21, 0x9a, 0x33, 0x34, 0x91, 0x47, 0xf3, 0xc1,
        0xd4, 0x11, 0x22, 0xdd, 0x39, 0x93, 0xb4, 0x77
    };

    my_encrption(data, round_key);

    for (int i = 0; i < 16; i++) {
        printf("%x ", data[i]);
    }
    printf("\n");
    my_decrption(data, round_key);

    for (int i = 0; i < 16; i++) {
        printf("%x ", data[i]);
    } // matches the original test data
    printf("\n");
    return 0;
}