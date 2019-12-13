#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#define MASTER 0
// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(int rank,int size,MPI_Status *status,const int ncols, const int ny, const int height,float* loc_image, float* loc_tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
int calc_ncols_from_rank(int rank, int size,int nx);

int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  int rank;               /* 'rank' of process among it's cohort */ 
  int size;               /* size of cohort, i.e. num processes started */
  int flag;               /* for checking whether MPI_Init() has been called */
  MPI_Status status;
  float *sendbuf;
  float *recvbuf;
  char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */

  /* initialise processes */
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
   float* image = malloc(sizeof(float) * width * height);
   float* tmp_image = malloc(sizeof(float) * width * height);
   
  if(rank == MASTER){
    // Set the input image
    init_image(nx, ny, width, height, image, tmp_image);
  }
  //initialise values for process
  int ncols = calc_ncols_from_rank(rank,size,nx); //nx to ignore padding
  //array to store ncol numbers
  int *col_numbers = NULL;
  int *displ = malloc(sizeof(int)*size);
  displ[0] = 0;
  if(rank == MASTER){
    col_numbers = malloc(sizeof(int) * size);
  }
  MPI_Gather(&ncols,1,MPI_INT,col_numbers,1,MPI_INT,MASTER,MPI_COMM_WORLD);
  int loc_width = ncols + 2;
  float* loc_image = malloc(sizeof(float) * loc_width * height); 
  float* loc_tmp_image = malloc(sizeof(float) * loc_width * height);
  int displacement = 0;
 if(rank == MASTER){
   for(int dest = 1; dest < size; dest++){
     displacement += col_numbers[dest-1];
     displ[dest] = displacement * height;
  }
 }
  
  //calculate send counts
  if(rank == MASTER){
    for(int i = 0; i < size; i++){
      col_numbers[i] = col_numbers[i] * height;
    }
   }
 
  
  MPI_Scatterv(&image[height],col_numbers,displ,MPI_FLOAT,&loc_image[height],ncols*height,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
 
  for(int a = 0; a < height; a++){
    loc_image[a] = 0;
    loc_image[a + (ncols + 1) * height] = 0;
  }
  // Call the stencil kernel
 //printf("rank %d about to compute stencil function ncols %d ny %d \n",rank,ncols,ny);
MPI_Barrier(MPI_COMM_WORLD); 
double tic = wtime();
for (int t = 0; t < niters; ++t) {
   stencil(rank,size,&status,ncols, ny, height, loc_image, loc_tmp_image);
   stencil(rank,size,&status,ncols, ny, height, loc_tmp_image, loc_image);
 } 
  MPI_Barrier(MPI_COMM_WORLD); 
  double toc = wtime();
  
  MPI_Gatherv(&loc_image[height], ncols * height,MPI_FLOAT,
              &image[height],col_numbers,displ,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
  if(rank == MASTER){
   // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");
    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }
  

  free(image);
  free(tmp_image);
  free(loc_image);
  free(loc_tmp_image);
  free(displ);
  free(col_numbers);
  MPI_Finalize();
  return 0;
}

void stencil(int rank,int size,MPI_Status *status,const int ncols, const int ny, const int height,float* loc_image, float* loc_tmp_image)
{
 
 int leftNeighbour = (rank == MASTER) ? (MPI_PROC_NULL) : (rank - 1);
 int rightNeighbour = (rank == size - 1) ? MPI_PROC_NULL : (rank + 1) % size;

 //do message passing here

 //send left col to left neighbour
 //recv right col from right neighbour
 MPI_Send(&loc_image[height],height,MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD);

 MPI_Recv(&loc_image[(ncols + 1) * height],height,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD,status);

 //send right col to right neighbour
MPI_Send(&loc_image[ncols *height],height,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD);
 
 //recv left col from left neighbour
MPI_Recv(&loc_image[0],height,MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD,status);
    
  for (int col = 1; col < ncols + 1; ++col) {
    for (int row = 1; row < ny + 1; ++row) {
      float a = 0.6;
      float b = 0.1;
      //top and bottom are applied here to address out of range issues
      int cell = row + col * height ;
      loc_tmp_image[cell] = a * loc_image[cell] + b * (loc_image[cell + height] + loc_image[cell - height]+loc_image[cell + 1] + loc_image[cell - 1]);
     }
  }
}


// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
int calc_ncols_from_rank(int rank, int size,int nx)
{
  int ncols;

  ncols = nx / size;       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += nx % size;  /* add remainder to last rank */
  }
  
  return ncols;
}
