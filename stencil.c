#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(int rank,int size, MPI_Status *status,const int ncols, const int ny, const int width, const int height, float* loc_image, float* loc_tmp_image);
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
  int strlen;             /* length of a character array */
  MPI_Status status;
 // enum bool {FALSE,TRUE}; /* enumerated type: false = 0, true = 1 */  
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
  int ncols = calc_ncols_from_rank(rank,size,nx);
  int loc_width = ncols + 2;
  float* loc_image = malloc(sizeof(float) * ncols * height);
  float* loc_tmp_image = malloc(sizeof(float) * ncols * height);
  MPI_Scatter(image,sizeof(float)*(width * height),MPI_FLOAT,loc_image,sizeof(float)*(ncols * height),MPI_FLOAT,MASTER,MPI_COMM_WORLD);
 MPI_Scatter(tmp_image,sizeof(float)*(width * height),MPI_FLOAT,loc_tmp_image,sizeof(float)*(ncols * height),MPI_FLOAT,MASTER,MPI_COMM_WORLD);
  //initialise both loc_image and tmp loc_image
 /* for(int i = 0; i < ny+2; i++){
    for(int j = 0; j < ncols; j++){
      loc_image[j + i * height] = image[(j + i * height) + ncols * rank];
      loc_tmp_image[j + i * height] = tmp_image[(j + i * height) + ncols * rank];
    }
  }*/  


  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(rank,size,&status,nx, ny, width, height, loc_image, loc_tmp_image);
    stencil(rank,size,&status,nx, ny, width, height, loc_tmp_image, loc_image);
  }
  double toc = wtime();
  MPI_Gather(loc_image,(ncols * height),MPI_FLOAT,
             image, (ncols*height), MPI_FLOAT, MASTER,MPI_COMM_WORLD);
  if(rank == MASTER){
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");
    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }
  free(image);
  free(tmp_image);
}

void stencil(int rank,int size,MPI_Status *status,const int ncols, const int ny, const int width, const int height,float* loc_image, float* loc_tmp_image)

{
 int leftNeighbour = (rank == MASTER) ? (rank + size - 1 % size) : (rank - 1);
 int rightNeighbour = (rank + 1) % size; 
 for (int i = 1; i < ny + 1; ++i) {
    for (int j = 0; j < ncols; ++j) {
      float a = 0.6;
      float b = 0.1;
      int val_1 = i * height;
     // tmp_image[j + val_1] =  image[j + val_1] * a +b* (image[j + (i - 1) * height] + image[j + (i + 1) * height] + image[j - 1 + val_1]  + image[j + 1 + val_1] );
      float fromLeft; //toRight
      float fromRight; //toLeft
      if(j == 0 || j == ncols - 1){
      loc_tmp_image[j + val_1] =  loc_image[j + val_1] * a +b* (loc_image[j + (i - 1) * height] + loc_image[j + (i + 1) * height]);// + loc_image[j - 1 + val_1]  + loc_image[j + 1 + val_1] );
        if(j == 0){
	       if(rank % 2 == 0){
		 fromLeft = loc_image[(ncols - 1) + val_1];
		 MPI_Recv(&fromRight,1,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD,status);
		 MPI_Send(&fromLeft,1,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD);
		 if(rank == MASTER) fromRight = 0;
		 loc_tmp_image[j+val_1] += fromRight;
	       }else{
		 fromRight = loc_image[(ncols - 1) + val_1];
		 MPI_Sendrecv(&fromRight,1, MPI_FLOAT,leftNeighbour,0,
			      &fromLeft,1,MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD,status);
		 loc_tmp_image[j+val_1] += fromLeft;
	       }
	      }
	      if(j == ncols - 1){
		if(rank % 2 == 0){
		  fromLeft = loc_image[val_1];          
		  MPI_Recv(&fromRight,1,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD,status);
		  MPI_Send(&fromLeft,1,MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD);
		  if(rank == size - 1) fromRight = 0;
		  loc_tmp_image[j+val_1] += fromRight;
		}
		else{
		 fromRight = loc_image[(ncols - 1) + val_1];
		 MPI_Sendrecv(&fromRight,1, MPI_FLOAT,leftNeighbour,0,
			      &fromLeft,1,MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD,status);
		 if(rank == size - 1) fromLeft = 0;
		 loc_tmp_image[j+val_1] += fromLeft;
		}
	      }
       }else{ 
          loc_tmp_image[j + val_1] =  loc_image[j + val_1] * a +b* (loc_image[j + (i - 1) * height] + loc_image[j + (i + 1) * height] + loc_image[j - 1 + val_1]  + loc_image[j + 1 + val_1] );
       }

         
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
