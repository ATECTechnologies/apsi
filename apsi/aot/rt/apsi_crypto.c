/* Apsi native runtime: the standard hash functions (SHA-2, SHA-3, BLAKE2, and SHA-1 / MD5 for checksums), HMAC,
 * PBKDF2 and scrypt, Base64, CRC-32 and the operating system's secure random numbers. Each algorithm follows its
 * published specification (FIPS 180-4, FIPS 202, RFC 7693, RFC 2104, RFC 8018, RFC 7914) and is tested against
 * the specification's vectors and Python's hashlib. Included by apsi_tensor.c. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#if defined(__linux__)
#include <sys/random.h>
#endif
#endif

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define ROR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
static inline uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static inline uint64_t be64(const uint8_t *p) { return (uint64_t)be32(p) << 32 | be32(p + 4); }
static inline uint32_t le32(const uint8_t *p) { return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | p[0]; }
static inline uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p + 4) << 32 | le32(p); }
static inline void put_be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static inline void put_be64(uint8_t *p, uint64_t v) { put_be32(p, (uint32_t)(v >> 32)); put_be32(p + 4, (uint32_t)v); }
static inline void put_le32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static inline void put_le64(uint8_t *p, uint64_t v) { put_le32(p, (uint32_t)v); put_le32(p + 4, (uint32_t)(v >> 32)); }

/* ------------------------------------------------------------------ one interface for every hash */
typedef struct HCtx {
    int algo;                /* index in hash_algos */
    union { uint32_t s32[16]; uint64_t s64[25]; } st;
    uint8_t buf[200]; size_t blen; uint64_t total;
    uint64_t t0, t1;         /* BLAKE2 counters */
} HCtx;
typedef struct { const char *name; int out, block; } HAlgo;
enum { H_MD5, H_SHA1, H_SHA256, H_SHA512, H_SHA3_256, H_SHA3_512, H_BLAKE2B, H_BLAKE2S, H_N };
static const HAlgo hash_algos[H_N] = {{"md5", 16, 64}, {"sha1", 20, 64}, {"sha256", 32, 64}, {"sha512", 64, 128},
    {"sha3_256", 32, 136}, {"sha3_512", 64, 72}, {"blake2b", 64, 128}, {"blake2s", 32, 64}};

/* ---- MD5 (RFC 1321) */
static const uint32_t md5_k[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af,
    0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97,
    0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
static const uint8_t md5_r[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
static void md5_block(uint32_t *h, const uint8_t *p) {
    uint32_t m[16], a = h[0], b = h[1], c = h[2], d = h[3];
    for (int i = 0; i < 16; i++) m[i] = le32(p + 4 * i);
    for (int i = 0; i < 64; i++) {
        uint32_t f; int g;
        if (i < 16) { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) & 15; }
        else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) & 15; }
        else { f = c ^ (b | ~d); g = (7 * i) & 15; }
        uint32_t t = d; d = c; c = b; b = b + ROL32(a + f + md5_k[i] + m[g], md5_r[i]); a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
}
/* ---- SHA-1 (FIPS 180-4) */
static void sha1_block(uint32_t *h, const uint8_t *p) {
    uint32_t w[80], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
    for (int i = 16; i < 80; i++) w[i] = ROL32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
        else { f = b ^ c ^ d; k = 0xca62c1d6; }
        uint32_t t = ROL32(a, 5) + f + e + k + w[i]; e = d; d = c; c = ROL32(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}
/* ---- SHA-256 */
static const uint32_t k256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static void sha256_block(uint32_t *h, const uint8_t *p) {
    uint32_t w[64], s[8];
    for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(s, h, sizeof s);
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR32(s[4], 6) ^ ROR32(s[4], 11) ^ ROR32(s[4], 25), ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
        uint32_t t1 = s[7] + S1 + ch + k256[i] + w[i];
        uint32_t S0 = ROR32(s[0], 2) ^ ROR32(s[0], 13) ^ ROR32(s[0], 22), mj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
        uint32_t t2 = S0 + mj;
        s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++) h[i] += s[i];
}
/* ---- SHA-512 */
static const uint64_t k512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL,
    0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
    0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL,
    0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL,
    0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
    0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL,
    0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL,
    0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
    0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};
static void sha512_block(uint64_t *h, const uint8_t *p) {
    uint64_t w[80], s[8];
    for (int i = 0; i < 16; i++) w[i] = be64(p + 8 * i);
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = ROR64(w[i - 15], 1) ^ ROR64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        uint64_t s1 = ROR64(w[i - 2], 19) ^ ROR64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(s, h, sizeof s);
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = ROR64(s[4], 14) ^ ROR64(s[4], 18) ^ ROR64(s[4], 41), ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
        uint64_t t1 = s[7] + S1 + ch + k512[i] + w[i];
        uint64_t S0 = ROR64(s[0], 28) ^ ROR64(s[0], 34) ^ ROR64(s[0], 39), mj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
        uint64_t t2 = S0 + mj;
        s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++) h[i] += s[i];
}
/* ---- SHA-3 (Keccak-f[1600]) */
static const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL, 0x000000000000808bULL,
    0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL, 0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
static const int keccak_rot[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
static void keccak_f(uint64_t *a) {
    for (int r = 0; r < 24; r++) {
        uint64_t c[5], d[5], b[25];
        for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++) d[x] = c[(x + 4) % 5] ^ ROL64(c[(x + 1) % 5], 1);
        for (int i = 0; i < 25; i++) a[i] ^= d[i % 5];
        for (int x = 0; x < 5; x++) for (int y = 0; y < 5; y++) {
            int k = keccak_rot[x + 5 * y]; uint64_t v = a[x + 5 * y];
            b[y + 5 * ((2 * x + 3 * y) % 5)] = k ? ROL64(v, k) : v;
        }
        for (int x = 0; x < 5; x++) for (int y = 0; y < 5; y++) a[x + 5 * y] = b[x + 5 * y] ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
        a[0] ^= keccak_rc[r];
    }
}
static void sha3_block(uint64_t *a, const uint8_t *p, int rate) { for (int i = 0; i < rate / 8; i++) a[i] ^= le64(p + 8 * i); keccak_f(a); }
/* ---- BLAKE2b / BLAKE2s (RFC 7693), unkeyed */
static const uint64_t b2b_iv[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
static const uint32_t b2s_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
static const uint8_t b2_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
static void b2b_compress(uint64_t *h, const uint8_t *p, uint64_t t0, uint64_t t1, int last) {
    uint64_t v[16], m[16];
    for (int i = 0; i < 16; i++) m[i] = le64(p + 8 * i);
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = b2b_iv[i]; }
    v[12] ^= t0; v[13] ^= t1; if (last) v[14] = ~v[14];
#define G64(a, b, c, d, x, y) { v[a] += v[b] + x; v[d] = ROR64(v[d] ^ v[a], 32); v[c] += v[d]; v[b] = ROR64(v[b] ^ v[c], 24); \
    v[a] += v[b] + y; v[d] = ROR64(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = ROR64(v[b] ^ v[c], 63); }
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = b2_sigma[r];
        G64(0, 4, 8, 12, m[s[0]], m[s[1]]) G64(1, 5, 9, 13, m[s[2]], m[s[3]]) G64(2, 6, 10, 14, m[s[4]], m[s[5]]) G64(3, 7, 11, 15, m[s[6]], m[s[7]])
        G64(0, 5, 10, 15, m[s[8]], m[s[9]]) G64(1, 6, 11, 12, m[s[10]], m[s[11]]) G64(2, 7, 8, 13, m[s[12]], m[s[13]]) G64(3, 4, 9, 14, m[s[14]], m[s[15]])
    }
#undef G64
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}
static void b2s_compress(uint32_t *h, const uint8_t *p, uint32_t t0, uint32_t t1, int last) {
    uint32_t v[16], m[16];
    for (int i = 0; i < 16; i++) m[i] = le32(p + 4 * i);
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = b2s_iv[i]; }
    v[12] ^= t0; v[13] ^= t1; if (last) v[14] = ~v[14];
#define G32(a, b, c, d, x, y) { v[a] += v[b] + x; v[d] = ROR32(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = ROR32(v[b] ^ v[c], 12); \
    v[a] += v[b] + y; v[d] = ROR32(v[d] ^ v[a], 8); v[c] += v[d]; v[b] = ROR32(v[b] ^ v[c], 7); }
    for (int r = 0; r < 10; r++) {
        const uint8_t *s = b2_sigma[r];
        G32(0, 4, 8, 12, m[s[0]], m[s[1]]) G32(1, 5, 9, 13, m[s[2]], m[s[3]]) G32(2, 6, 10, 14, m[s[4]], m[s[5]]) G32(3, 7, 11, 15, m[s[6]], m[s[7]])
        G32(0, 5, 10, 15, m[s[8]], m[s[9]]) G32(1, 6, 11, 12, m[s[10]], m[s[11]]) G32(2, 7, 8, 13, m[s[12]], m[s[13]]) G32(3, 4, 9, 14, m[s[14]], m[s[15]])
    }
#undef G32
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

static void h_init(HCtx *c, int algo) {
    memset(c, 0, sizeof *c); c->algo = algo;
    switch (algo) {
    case H_MD5: { static const uint32_t iv[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476}; memcpy(c->st.s32, iv, sizeof iv); break; }
    case H_SHA1: { static const uint32_t iv[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0}; memcpy(c->st.s32, iv, sizeof iv); break; }
    case H_SHA256: { static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}; memcpy(c->st.s32, iv, sizeof iv); break; }
    case H_SHA512: { static const uint64_t iv[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL}; memcpy(c->st.s64, iv, sizeof iv); break; }
    case H_BLAKE2B: memcpy(c->st.s64, b2b_iv, sizeof b2b_iv); c->st.s64[0] ^= 0x01010000ULL ^ 64; break;
    case H_BLAKE2S: memcpy(c->st.s32, b2s_iv, sizeof b2s_iv); c->st.s32[0] ^= 0x01010000U ^ 32; break;
    default: break;   /* SHA-3: a zero state */
    }
}
static void h_block(HCtx *c, const uint8_t *p) {
    switch (c->algo) {
    case H_MD5: md5_block(c->st.s32, p); break;
    case H_SHA1: sha1_block(c->st.s32, p); break;
    case H_SHA256: sha256_block(c->st.s32, p); break;
    case H_SHA512: sha512_block(c->st.s64, p); break;
    case H_SHA3_256: case H_SHA3_512: sha3_block(c->st.s64, p, hash_algos[c->algo].block); break;
    }
}
static void h_update(HCtx *c, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data; size_t B = (size_t)hash_algos[c->algo].block;
    if (c->algo == H_BLAKE2B || c->algo == H_BLAKE2S) {   /* the last block is compressed only at the end */
        while (n) {
            if (c->blen == B) {
                c->t0 += B; if (c->t0 < B) c->t1++;
                if (c->algo == H_BLAKE2B) b2b_compress(c->st.s64, c->buf, c->t0, c->t1, 0); else b2s_compress(c->st.s32, c->buf, (uint32_t)c->t0, (uint32_t)(c->t0 >> 32), 0);
                c->blen = 0;
            }
            size_t k = B - c->blen; if (k > n) k = n;
            memcpy(c->buf + c->blen, p, k); c->blen += k; p += k; n -= k;
        }
        return;
    }
    c->total += n;
    if (c->blen) { size_t k = B - c->blen; if (k > n) k = n; memcpy(c->buf + c->blen, p, k); c->blen += k; p += k; n -= k; if (c->blen == B) { h_block(c, c->buf); c->blen = 0; } }
    while (n >= B) { h_block(c, p); p += B; n -= B; }
    if (n) { memcpy(c->buf, p, n); c->blen = n; }
}
static int h_final(HCtx *c, uint8_t *out) {
    int algo = c->algo, B = hash_algos[algo].block, outn = hash_algos[algo].out;
    if (algo == H_BLAKE2B || algo == H_BLAKE2S) {
        c->t0 += c->blen; if (c->t0 < c->blen) c->t1++;
        memset(c->buf + c->blen, 0, (size_t)B - c->blen);
        if (algo == H_BLAKE2B) { b2b_compress(c->st.s64, c->buf, c->t0, c->t1, 1); for (int i = 0; i < 8; i++) put_le64(out + 8 * i, c->st.s64[i]); }
        else { b2s_compress(c->st.s32, c->buf, (uint32_t)c->t0, (uint32_t)(c->t0 >> 32), 1); for (int i = 0; i < 8; i++) put_le32(out + 4 * i, c->st.s32[i]); }
        return outn;
    }
    if (algo == H_SHA3_256 || algo == H_SHA3_512) {
        memset(c->buf + c->blen, 0, (size_t)B - c->blen);
        c->buf[c->blen] ^= 0x06; c->buf[B - 1] ^= 0x80;
        sha3_block(c->st.s64, c->buf, B);
        for (int i = 0; i < outn / 8; i++) put_le64(out + 8 * i, c->st.s64[i]);
        return outn;
    }
    uint64_t bits = c->total * 8; int lenb = algo == H_SHA512 ? 16 : 8;
    c->buf[c->blen++] = 0x80;
    if (c->blen > (size_t)(B - lenb)) { memset(c->buf + c->blen, 0, (size_t)B - c->blen); h_block(c, c->buf); c->blen = 0; }
    memset(c->buf + c->blen, 0, (size_t)B - c->blen);
    if (algo == H_MD5) put_le64(c->buf + B - 8, bits); else put_be64(c->buf + B - 8, bits);
    h_block(c, c->buf);
    if (algo == H_MD5) for (int i = 0; i < 4; i++) put_le32(out + 4 * i, c->st.s32[i]);
    else if (algo == H_SHA512) for (int i = 0; i < 8; i++) put_be64(out + 8 * i, c->st.s64[i]);
    else for (int i = 0; i < outn / 4; i++) put_be32(out + 4 * i, c->st.s32[i]);
    return outn;
}
static int h_algo(const char *name) {
    for (int i = 0; i < H_N; i++) if (!strcmp(hash_algos[i].name, name)) return i;
    if (!strcmp(name, "sha-256") || !strcmp(name, "SHA256")) return H_SHA256;
    return -1;
}
static int hash_of(int algo, const void *p, size_t n, uint8_t *out) { HCtx c; h_init(&c, algo); h_update(&c, p, n); return h_final(&c, out); }
/* HMAC (RFC 2104) */
static int hmac_of(int algo, const uint8_t *key, size_t kn, const uint8_t *msg, size_t mn, uint8_t *out) {
    int B = hash_algos[algo].block; uint8_t k[200] = {0}, pad[200]; HCtx c;
    if (kn > (size_t)B) hash_of(algo, key, kn, k); else memcpy(k, key, kn);
    for (int i = 0; i < B; i++) pad[i] = k[i] ^ 0x36;
    h_init(&c, algo); h_update(&c, pad, (size_t)B); h_update(&c, msg, mn); uint8_t inner[64]; int hn = h_final(&c, inner);
    for (int i = 0; i < B; i++) pad[i] = k[i] ^ 0x5c;
    h_init(&c, algo); h_update(&c, pad, (size_t)B); h_update(&c, inner, (size_t)hn); return h_final(&c, out);
}
/* PBKDF2-HMAC-SHA256 (RFC 8018), the HMAC keys absorbed once */
static void pbkdf2_sha256(const uint8_t *pw, size_t pn, const uint8_t *salt, size_t sn, uint64_t iters, uint8_t *out, size_t dk) {
    uint8_t k[64] = {0}, ipad[64], opad[64];
    if (pn > 64) hash_of(H_SHA256, pw, pn, k); else memcpy(k, pw, pn);
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    HCtx ci, co; h_init(&ci, H_SHA256); h_update(&ci, ipad, 64); h_init(&co, H_SHA256); h_update(&co, opad, 64);
    for (uint32_t blk = 1; dk; blk++) {
        uint8_t u[32], t[32], be[4]; put_be32(be, blk); HCtx c = ci;
        h_update(&c, salt, sn); h_update(&c, be, 4); h_final(&c, u);
        c = co; h_update(&c, u, 32); h_final(&c, u); memcpy(t, u, 32);
        for (uint64_t r = 1; r < iters; r++) {
            c = ci; h_update(&c, u, 32); h_final(&c, u); c = co; h_update(&c, u, 32); h_final(&c, u);
            for (int i = 0; i < 32; i++) t[i] ^= u[i];
        }
        size_t k2 = dk < 32 ? dk : 32; memcpy(out, t, k2); out += k2; dk -= k2;
    }
}
/* scrypt (RFC 7914) */
static void salsa20_8(uint32_t *B) {
    uint32_t x[16]; memcpy(x, B, 64);
    for (int i = 0; i < 8; i += 2) {
#define R(a, b) ROL32(a, b)
        x[4] ^= R(x[0] + x[12], 7); x[8] ^= R(x[4] + x[0], 9); x[12] ^= R(x[8] + x[4], 13); x[0] ^= R(x[12] + x[8], 18);
        x[9] ^= R(x[5] + x[1], 7); x[13] ^= R(x[9] + x[5], 9); x[1] ^= R(x[13] + x[9], 13); x[5] ^= R(x[1] + x[13], 18);
        x[14] ^= R(x[10] + x[6], 7); x[2] ^= R(x[14] + x[10], 9); x[6] ^= R(x[2] + x[14], 13); x[10] ^= R(x[6] + x[2], 18);
        x[3] ^= R(x[15] + x[11], 7); x[7] ^= R(x[3] + x[15], 9); x[11] ^= R(x[7] + x[3], 13); x[15] ^= R(x[11] + x[7], 18);
        x[1] ^= R(x[0] + x[3], 7); x[2] ^= R(x[1] + x[0], 9); x[3] ^= R(x[2] + x[1], 13); x[0] ^= R(x[3] + x[2], 18);
        x[6] ^= R(x[5] + x[4], 7); x[7] ^= R(x[6] + x[5], 9); x[4] ^= R(x[7] + x[6], 13); x[5] ^= R(x[4] + x[7], 18);
        x[11] ^= R(x[10] + x[9], 7); x[8] ^= R(x[11] + x[10], 9); x[9] ^= R(x[8] + x[11], 13); x[10] ^= R(x[9] + x[8], 18);
        x[12] ^= R(x[15] + x[14], 7); x[13] ^= R(x[12] + x[15], 9); x[14] ^= R(x[13] + x[12], 13); x[15] ^= R(x[14] + x[13], 18);
#undef R
    }
    for (int i = 0; i < 16; i++) B[i] += x[i];
}
static void blockmix(uint32_t *B, uint32_t *Y, int r) {
    uint32_t X[16]; memcpy(X, B + (2 * r - 1) * 16, 64);
    for (int i = 0; i < 2 * r; i++) { for (int j = 0; j < 16; j++) X[j] ^= B[i * 16 + j]; salsa20_8(X); memcpy(Y + i * 16, X, 64); }
    for (int i = 0; i < r; i++) memcpy(B + i * 16, Y + (2 * i) * 16, 64);
    for (int i = 0; i < r; i++) memcpy(B + (i + r) * 16, Y + (2 * i + 1) * 16, 64);
}
static int scrypt_kdf(const uint8_t *pw, size_t pn, const uint8_t *salt, size_t sn, uint64_t N, int r, int p, uint8_t *out, size_t dk) {
    if (N < 2 || (N & (N - 1)) || r < 1 || p < 1 || N > ((uint64_t)1 << 24) || (uint64_t)r * p >= (1u << 30)) return 0;
    size_t bl = (size_t)128 * (size_t)r;
    uint8_t *B = (uint8_t *)malloc(bl * (size_t)p); uint32_t *V = (uint32_t *)malloc(bl * (size_t)N), *X = (uint32_t *)malloc(bl), *Y = (uint32_t *)malloc(bl);
    if (!B || !V || !X || !Y) { free(B); free(V); free(X); free(Y); return 0; }
    pbkdf2_sha256(pw, pn, salt, sn, 1, B, bl * (size_t)p);
    size_t w = bl / 4;
    for (int i = 0; i < p; i++) {
        uint8_t *Bi = B + (size_t)i * bl;
        for (size_t k = 0; k < w; k++) X[k] = le32(Bi + 4 * k);
        for (uint64_t j = 0; j < N; j++) { memcpy(V + j * w, X, bl); blockmix(X, Y, r); }
        for (uint64_t j = 0; j < N; j++) {
            uint64_t idx = X[(2 * r - 1) * 16] & (N - 1);
            for (size_t k = 0; k < w; k++) X[k] ^= V[idx * w + k];
            blockmix(X, Y, r);
        }
        for (size_t k = 0; k < w; k++) put_le32(Bi + 4 * k, X[k]);
    }
    pbkdf2_sha256(pw, pn, B, bl * (size_t)p, 1, out, dk);
    memset(V, 0, bl * (size_t)N); free(B); free(V); free(X); free(Y);
    return 1;
}
/* the operating system's random numbers (getrandom / BCryptGenRandom) */
static void os_random(uint8_t *out, size_t n) {
#ifdef _WIN32
    if (BCryptGenRandom(NULL, out, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) a_error("OSError", "no secure random numbers from the system");
#else
    size_t got = 0;
#if defined(__linux__)
    while (got < n) { ssize_t k = getrandom(out + got, n - got, 0); if (k < 0) { if (errno == EINTR) continue; break; } got += (size_t)k; }
#endif
    if (got < n) {
        int fd = open("/dev/urandom", O_RDONLY);
        if (fd < 0) a_error("OSError", "no secure random numbers from the system");
        while (got < n) { ssize_t k = read(fd, out + got, n - got); if (k <= 0) { if (k < 0 && errno == EINTR) continue; close(fd); a_error("OSError", "reading /dev/urandom failed"); } got += (size_t)k; }
        close(fd);
    }
#endif
}
/* Base64 (RFC 4648): url = the URL-safe alphabet without padding */
static AV b64_enc(const uint8_t *p, size_t n, int url) {
    const char *al = url ? "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *o = (char *)malloc(n / 3 * 4 + 5); size_t k = 0, i = 0;
    for (; i + 2 < n; i += 3) { uint32_t v = (uint32_t)p[i] << 16 | (uint32_t)p[i + 1] << 8 | p[i + 2]; o[k++] = al[v >> 18]; o[k++] = al[(v >> 12) & 63]; o[k++] = al[(v >> 6) & 63]; o[k++] = al[v & 63]; }
    if (n - i == 1) { uint32_t v = (uint32_t)p[i] << 16; o[k++] = al[v >> 18]; o[k++] = al[(v >> 12) & 63]; if (!url) { o[k++] = '='; o[k++] = '='; } }
    else if (n - i == 2) { uint32_t v = (uint32_t)p[i] << 16 | (uint32_t)p[i + 1] << 8; o[k++] = al[v >> 18]; o[k++] = al[(v >> 12) & 63]; o[k++] = al[(v >> 6) & 63]; if (!url) o[k++] = '='; }
    AV r = a_strn(o, (int64_t)k); free(o); return r;
}
/* both alphabets, padding optional, whitespace skipped; -1: not Base64 */
static int64_t b64_dec(const char *s, size_t n, uint8_t *out) {
    uint32_t acc = 0; int bits = 0; int64_t k = 0; size_t pad = 0;
    for (size_t i = 0; i < n; i++) {
        int c = (unsigned char)s[i], v;
        if (c == '=') { pad++; continue; }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        if (pad) return -1;
        if (c >= 'A' && c <= 'Z') v = c - 'A'; else if (c >= 'a' && c <= 'z') v = c - 'a' + 26; else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+' || c == '-') v = 62; else if (c == '/' || c == '_') v = 63; else return -1;
        acc = acc << 6 | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out[k++] = (uint8_t)(acc >> bits); }
    }
    if (bits >= 6) return -1;
    return k;
}
static uint32_t crc32_of(const uint8_t *p, size_t n) {
    static uint32_t T[256]; static int ready = 0;
    if (!ready) { for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320U ^ (c >> 1) : c >> 1; T[i] = c; } ready = 1; }
    uint32_t c = 0xffffffffU; for (size_t i = 0; i < n; i++) c = T[(c ^ p[i]) & 255] ^ (c >> 8); return c ^ 0xffffffffU;
}
static int ct_equal(const uint8_t *a, size_t an, const uint8_t *b, size_t bn) {
    uint8_t d = (uint8_t)(an != bn); size_t n = an < bn ? an : bn;
    for (size_t i = 0; i < n; i++) d |= a[i] ^ b[i];
    return d == 0;
}
static AV hex_of(const uint8_t *p, size_t n) { char o[260]; static const char *h = "0123456789abcdef"; for (size_t i = 0; i < n; i++) { o[2 * i] = h[p[i] >> 4]; o[2 * i + 1] = h[p[i] & 15]; } return a_strn(o, (int64_t)(2 * n)); }

/* ------------------------------------------------------------------ the built-ins (the crypto module wraps them) */
#ifndef STR
#define STR(v) ((AStr *)(v).o)
#endif
static AStr *lib_str(AV v, const char *what) { if (!a_is(v, O_STR) && !a_is(v, O_BYTES)) a_error("TypeError", "%s takes text (Str)", what); return (AStr *)v.o; }
static int lib_algo(AV v) {
    const char *name = lib_str(v, "the hash name")->s; int a = h_algo(name);
    if (a < 0) a_error("ValueError", "unknown hash '%s'; use one of blake2b, blake2s, md5, sha1, sha256, sha3_256, sha3_512, sha512", name);
    return a;
}
BI(lb_hash) { (void)self; (void)kw; AStr *s = lib_str(a[0], "hash"); uint8_t d[64]; int k = hash_of(n > 1 ? lib_algo(a[1]) : H_SHA256, s->s, (size_t)s->len, d); return hex_of(d, (size_t)k); }
/* __hmac(key, message, algo, form): form "hex" | "b64u" | "b64" */
BI(lb_hmac) {
    (void)self; (void)kw; AStr *k = lib_str(a[0], "hmac"), *m = lib_str(a[1], "hmac"); uint8_t d[64];
    int len = hmac_of(lib_algo(a[2]), (const uint8_t *)k->s, (size_t)k->len, (const uint8_t *)m->s, (size_t)m->len, d);
    const char *form = n > 3 ? lib_str(a[3], "hmac")->s : "hex";
    if (!strcmp(form, "b64u")) return b64_enc(d, (size_t)len, 1);
    if (!strcmp(form, "b64")) return b64_enc(d, (size_t)len, 0);
    return hex_of(d, (size_t)len);
}
BI(lb_b64enc) { (void)self; (void)kw; AStr *s = lib_str(a[0], "b64encode"); return b64_enc((const uint8_t *)s->s, (size_t)s->len, n > 1 && a_truthy(a[1])); }
BI(lb_b64dec) {
    (void)self; (void)kw; (void)n; AStr *s = lib_str(a[0], "b64decode"); uint8_t *o = (uint8_t *)malloc((size_t)s->len + 3);
    int64_t k = b64_dec(s->s, (size_t)s->len, o);
    if (k < 0) { free(o); a_error("ValueError", "b64decode: not Base64 text"); }
    AV r = a_strn((const char *)o, k); free(o); return r;
}
BI(lb_crc32) { (void)self; (void)kw; (void)n; AStr *s = lib_str(a[0], "crc32"); return a_int(crc32_of((const uint8_t *)s->s, (size_t)s->len)); }
BI(lb_secure_eq) { (void)self; (void)kw; (void)n; AStr *x = lib_str(a[0], "secure_equal"), *y = lib_str(a[1], "secure_equal"); return a_bool(ct_equal((const uint8_t *)x->s, (size_t)x->len, (const uint8_t *)y->s, (size_t)y->len)); }
/* __random_bytes(n, form): "hex" | "b64u" */
BI(lb_random_bytes) {
    (void)self; (void)kw; int64_t k = n ? a[0].i : 32; if (k < 0 || k > 1 << 20) a_error("ValueError", "random bytes: 0 to 1048576");
    uint8_t *b = (uint8_t *)malloc((size_t)k + 1); os_random(b, (size_t)k);
    AV r = n > 1 && !strcmp(lib_str(a[1], "random")->s, "b64u") ? b64_enc(b, (size_t)k, 1) : k <= 128 ? hex_of(b, (size_t)k) : A_NONE;
    if (r.t == T_NONE) { char *h = (char *)malloc((size_t)k * 2); static const char *hx = "0123456789abcdef"; for (int64_t i = 0; i < k; i++) { h[2 * i] = hx[b[i] >> 4]; h[2 * i + 1] = hx[b[i] & 15]; } r = a_strn(h, 2 * k); free(h); }
    free(b); return r;
}
BI(lb_secure_randint) {
    (void)self; (void)kw; (void)n; int64_t lo = a[0].i, hi = a[1].i; if (hi <= lo) a_error("ValueError", "secure_randint(lo, hi) needs lo < hi");
    uint64_t span = (uint64_t)(hi - lo), lim = UINT64_MAX - UINT64_MAX % span, x;
    do { os_random((uint8_t *)&x, sizeof x); } while (x >= lim);       /* no modulo bias */
    return a_int(lo + (int64_t)(x % span));
}
/* __password_hash(password, n, r, p): "scrypt$n$r$p$salt$hash", as apsi run writes it */
BI(lb_password_hash) {
    (void)self; (void)kw; AStr *pw = lib_str(a[0], "password_hash");
    int64_t N = n > 1 ? a[1].i : 16384, r = n > 2 ? a[2].i : 8, p = n > 3 ? a[3].i : 1;
    uint8_t salt[16], dk[32]; os_random(salt, 16);
    if (!scrypt_kdf((const uint8_t *)pw->s, (size_t)pw->len, salt, 16, (uint64_t)N, (int)r, (int)p, dk, 32)) a_error("ValueError", "password_hash: n must be a power of 2 (up to 2^24), r and p at least 1");
    AV s1 = b64_enc(salt, 16, 0), s2 = b64_enc(dk, 32, 0); char head[96]; snprintf(head, sizeof head, "scrypt$%lld$%lld$%lld$", (long long)N, (long long)r, (long long)p);
    AV parts[4] = {a_str(head), s1, a_str("$"), s2}; return a_concat_strs(4, parts);
}
BI(lb_password_verify) {
    (void)self; (void)kw; (void)n; AStr *pw = lib_str(a[0], "password_verify"); if (!a_is(a[1], O_STR)) return a_bool(0);
    AStr *st = (AStr *)a[1].o; char *s = (char *)malloc((size_t)st->len + 1); memcpy(s, st->s, (size_t)st->len + 1);
    char *f[7]; int nf = 0; for (char *q = s, *t; nf < 7; q = t + 1) { f[nf++] = q; t = strchr(q, '$'); if (!t) break; *t = 0; }
    uint8_t salt[256], want[256], got[256]; int ok = 0;
    if (nf == 6 && !strcmp(f[0], "scrypt")) {
        int64_t sn = (int64_t)strlen(f[4]) < 340 ? b64_dec(f[4], strlen(f[4]), salt) : -1, dn = strlen(f[5]) < 340 ? b64_dec(f[5], strlen(f[5]), want) : -1;
        if (sn >= 0 && dn > 0 && dn <= 256) ok = scrypt_kdf((const uint8_t *)pw->s, (size_t)pw->len, salt, (size_t)sn, strtoull(f[1], NULL, 10), atoi(f[2]), atoi(f[3]), got, (size_t)dn) && ct_equal(got, (size_t)dn, want, (size_t)dn);
    } else if (nf == 4 && !strcmp(f[0], "pbkdf2_sha256")) {
        int64_t sn = strlen(f[2]) < 340 ? b64_dec(f[2], strlen(f[2]), salt) : -1, dn = strlen(f[3]) < 340 ? b64_dec(f[3], strlen(f[3]), want) : -1;
        uint64_t it = strtoull(f[1], NULL, 10);
        if (sn >= 0 && dn > 0 && dn <= 256 && it >= 1 && it <= 100000000ULL) { pbkdf2_sha256((const uint8_t *)pw->s, (size_t)pw->len, salt, (size_t)sn, it, got, (size_t)dn); ok = ct_equal(got, (size_t)dn, want, (size_t)dn); }
    }
    free(s); return a_bool(ok);
}
