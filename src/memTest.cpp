#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>

#define FILEPATH "/local/scratch/mmapped.bin"

int main(int argc, char *argv[])
{
    size_t mallocSize = 180000000000 / sizeof(size_t);
    size_t mmapSize = 60000000000 / sizeof(size_t);
    off_t fd;
    off_t result;

    size_t * map; 
    size_t * testMal = (size_t *)_mm_malloc(mallocSize * sizeof(size_t), 64);

    for (size_t i = 1; i <=mallocSize; ++i) {
        testMal[i] = i;
    }

    std::cout << "ram filled: " << mallocSize * sizeof(size_t) << "\n";

    fd = open(FILEPATH, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
	perror("Error opening file for writing");
	exit(EXIT_FAILURE);
    }

   
    result = lseek64(fd, (mmapSize * sizeof(size_t)) -1, SEEK_SET);
    if (result == -1) {
	close(fd);
	perror("Error calling lseek() to 'stretch' the file");
	exit(EXIT_FAILURE);
    }
    
    result = write(fd, "", 1);
    if (result != 1) {
	close(fd);
	perror("Error writing last byte of the file");
	exit(EXIT_FAILURE);
    }

    
    map = (size_t *)mmap(0, mmapSize * sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
	close(fd);
	perror("Error mmapping the file");
	exit(EXIT_FAILURE);
    }

    for (size_t i = 1; i <=mmapSize; ++i) {
	map[i] = i; 
    }

    std::cout << "mmap filled: " << mmapSize * sizeof(size_t) << "\n";
     
    if (munmap(map, mmapSize * sizeof(size_t)) == -1) {
	perror("Error un-mmapping the file");
    }

    close(fd);

    _mm_free(testMal);
    return 0;
}
