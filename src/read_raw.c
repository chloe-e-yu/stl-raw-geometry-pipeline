/*
 * read_raw.c
 * A program to actually verify the point made in item 6 that "C's
 * fread makes it easy to read contiguous bytes."
 *
 * Compile: gcc -o read_raw read_raw.c
 * Run:     ./read_raw <raw_file> <nx> <ny> <nz>
 * Example: ./read_raw capitol_fine.raw 114 265 72
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <raw_path> <nx> <ny> <nz>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    long nx = atol(argv[2]);
    long ny = atol(argv[3]);
    long nz = atol(argv[4]);
    long expected_size = nx * ny * nz;

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        printf("Could not open file: %s\n", path);
        return 1;
    }

    unsigned char *data = malloc(expected_size);
    if (data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return 1;
    }

    size_t read_count = fread(data, 1, expected_size, f);
    fclose(f);

    printf("expected bytes : %ld\n", expected_size);
    printf("actual bytes   : %zu\n", read_count);

    if ((long)read_count != expected_size) {
        printf("File size does not match!\n");
        free(data);
        return 1;
    }

    long solid_count = 0;
    int only_binary = 1;
    for (long i = 0; i < expected_size; i++) {
        if (data[i] == 1) {
            solid_count++;
        } else if (data[i] != 0) {
            only_binary = 0;
        }
    }

    printf("values are 0/1 only : %s\n", only_binary ? "OK" : "FAIL");
    printf("solid voxels        : %ld / %ld (%.2f%%)\n",
           solid_count, expected_size,
           100.0 * solid_count / expected_size);

    printf("data[0] (x=0,y=0,z=0) = %d\n", data[0]);
    printf("data[1] (x=0,y=0,z=1) = %d\n", data[nz > 1 ? 1 : 0]);

    free(data);
    return 0;
}