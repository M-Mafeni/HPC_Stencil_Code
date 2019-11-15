#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

#define MASTER 0
// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(int rank,int size, MPI_Status *status,const int ncols, const int ny, const int width, const int height, float* loc_image, float* loc_tmp_image,float* sendbuf, float* recvbuf);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);
int calc_ncols_from_rank(int rank, int size,int nx);
void checkLeftAndRight(int rank,int i,int j,int height,int ncols,int leftNeighbour, int rightNeighbour,float* loc_image, float* loc_tmp_image,MPI_Status *status);
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
  float *sendbuf;
  float *recvbuf;
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
 // printf("about to calc ncols rank %d \n",rank);
  int ncols = calc_ncols_from_rank(rank,size,nx); //nx to ignore padding
  //array to store ncol numbers
  int *col_numbers = NULL;
  int *displ = malloc(sizeof(int)*size);
  displ[0] = 0;
  if(rank == MASTER){
    col_numbers = malloc(sizeof(int) * size);
  }
  MPI_Gather(&ncols,1,MPI_INT,col_numbers,1,MPI_INT,MASTER,MPI_COMM_WORLD);
  //if(rank == MASTER) printf("col__numbers: %d %d %d %d \n",col_numbers[0],col_numbers[1],col_numbers[2],col_numbers[3]);
  int loc_width = ncols + 2;
  float* loc_image = malloc(sizeof(float) * ncols * ny); //ny to ignore padding
  float* loc_tmp_image = malloc(sizeof(float) * ncols * ny);
  if(rank == MASTER){
     printf("splitting grid in MASTER\n");
     for(int dest = 1; dest < size; dest++){
       int displacement = col_numbers[dest-1];
       printf("sending part to rank %d, displacement %d \n",dest,displacement);
       MPI_Send(&image[1 + (displacement+1) * height],displacement * ny,MPI_FLOAT,dest,0,MPI_COMM_WORLD);
       MPI_Send(&tmp_image[1 + (displacement+1) * height],displacement * ny,MPI_FLOAT,dest,0,MPI_COMM_WORLD);
       printf("message sent to rank %d\n",dest);
     }
   }else{
    printf("about to receive from master, rank %d, ncols %d \n",rank,ncols);
     MPI_Recv(loc_image,ncols * ny,MPI_FLOAT,MASTER,0,MPI_COMM_WORLD,&status);
     MPI_Recv(loc_tmp_image,ncols * ny,MPI_FLOAT,MASTER,0,MPI_COMM_WORLD,&status);
     printf("received from master, rank %d \n",rank);
   }
    


  // Call the stencil kernel
 double tic = wtime();
 //printf("testing from rank %d \n",rank);
 sendbuf = malloc(sizeof(float) * height); // you are sending 1 column
 recvbuf = malloc(sizeof(float) * height); // you are receiving one column
printf("rank %d about to compute stencil function ncols %d ny %d \n",rank,ncols,ny);
for (int t = 0; t < niters; ++t) {
   stencil(rank,size,&status,ncols, ny, width, ny, loc_image, loc_tmp_image,sendbuf,recvbuf);
   stencil(rank,size,&status,ncols, ny, width, ny, loc_tmp_image, loc_image,sendbuf,recvbuf);
  }
  double toc = wtime();
  //MPI_Gather(loc_image,(ncols * height),MPI_FLOAT,
  //           image, (ncols*height), MPI_FLOAT, MASTER,MPI_COMM_WORLD);
  if(rank == MASTER){
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc - tic);
    printf("------------------------------------\n");
    //output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }
  printf("rank %d has finished \n",rank);

  free(image);
  free(tmp_image);
  free(loc_image);
  free(loc_tmp_image);
  MPI_Finalize();
}
void checkLeftAndRight(int rank,int i,int j,int height,int ncols,int leftNeighbour, int rightNeighbour,float* loc_image, float* loc_tmp_image,MPI_Status *status){
    float a = 0.6;
    float b = 0.1;
    int cell = i + j * ncols;
    int left = j - 1;
    int right = j + 1;
    //loc_tmp_image[j + val_1] =  loc_image[j + val_1] * a +b* (loc_image[j + (i - 1) * height] + loc_image[j + (i + 1) * height] + loc_image[j - 1 + val_1]  + loc_image[j + 1 + val_1] );
   /* if(left < 0){
          float fromLeft;
          //leftNeighbour's rightmost column is needed
          if(rank % 2 == 0){
            //send last col to right neighbour
            MPI_Send(&loc_image[(ncols - 1)+i*height],sizeof(float),MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD);
            //receive last col from left neighbour
            MPI_Recv(&fromLeft,sizeof(float),MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD,status);
          }else{
           //receive last col from left neighbour
           MPI_Recv(&fromLeft,sizeof(float),MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD,status);
           //send last col to right neighbour
           MPI_Send(&loc_image[(ncols - 1)+i*height],sizeof(float),MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD);
          }
          loc_tmp_image[j+val_1] += fromLeft;
        }
    else if(right > ncols){
         //rightNeighbour's leftmost column is needed
         float fromRight;
         loc_tmp_image[j + val_1] =  loc_image[j + val_1] * a +b* (loc_image[j + (i + 1) * height] + loc_image[j + 1 + val_1] );
         if(rank % 2 == 0){
            //send first col to left neighbour
            MPI_Send(&loc_image[i*height],sizeof(float),MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD);
            //receive last col from right neighbour
            MPI_Recv(&fromRight,sizeof(float),MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD,status);
     }else{
            //receive last col from right neighbour
            MPI_Recv(&fromRight,sizeof(float),MPI_FLOAT,rightNeighbour,0,MPI_COMM_WORLD,status);
            //send first col to left neighbour
            MPI_Send(&loc_image[i*height],sizeof(float),MPI_FLOAT,leftNeighbour,0,MPI_COMM_WORLD);
         }
         loc_tmp_image[j+val_1] += fromRight;
        }*/
      //loc_tmp_image[cell] += b * (loc_image[cell + 1] + loc_image[cell - 1] + loc_image[cell - ncols] + loc_image[cell + ncols] 
    if(left < 0){
      //add right value
      loc_tmp_image[cell] += b *loc_image[cell + ncols];
    }else if(right > ncols){
       loc_tmp_image[cell] += b *loc_image[cell - ncols];
    }else{
       loc_tmp_image[cell] += b * (loc_image[cell + ncols] + loc_image[cell - ncols]);
    }
}
void stencil(int rank,int size,MPI_Status *status,const int ncols, const int ny, const int width, const int height,float* loc_image, float* loc_tmp_image,float* sendbuf, float* recvbuf)

{
 int leftNeighbour = (rank == MASTER) ? (rank + size - 1 % size) : (rank - 1);
 int rightNeighbour = (rank + 1) % size;
 //printf("entering stencil for rank %d \n", rank);
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < ncols; ++j) {
   //   printf("rank %d row %d col %d \n",rank,i,j);
      float a = 0.6;
      float b = 0.1;
      int val_1 = i * height;
      //top and bottom are applied here to address out of range issues
      int top = i - 1;
      int bottom = i + 1;
      int left = j - 1;
      int right = j + 1;
      int cell = i + j * ncols;
      loc_tmp_image[cell] =  a * loc_image[cell];
      //loc_tmp_image[cell] += b * (loc_image[cell + 1] + loc_image[cell - 1] + loc_image[cell - ncols] + loc_image[cell + ncols] 
      if(top < 0){
       // printf("for rank %d, no top value adding val from index %d \n",rank,j + (i+1)*height);  
         loc_tmp_image[cell] += b* (loc_image[cell + 1]); // only value you're sure of is the one below you
      }else if(bottom > ny){
       // printf("for rank %d, no bottom value adding val from index %d \n",rank,j + (i-1)*height); 
         loc_tmp_image[cell] += b* (loc_image[cell - 1] );
      }else{
       // printf("for rank %d, both values accounted from indexes %d and %d \n",rank,j + (i-1) * height,j + (i+1)*height);  
         loc_tmp_image[cell] += b* (loc_image[cell + 1] + loc_image[cell - 1] );
      }
     //check left and right
//  void checkLeftAndRight(int rank,int i,int j,int height,int ncols,int leftNeighbour, int rightNeighbour,float* loc_image, float* loc_tmp_image,MPI_Status *status){
     checkLeftAndRight(rank,i,j,height,ncols,leftNeighbour,rightNeighbour,loc_image,loc_tmp_image,status);
     //loc_tmp_image[j + val_1] =  loc_image[j + val_1] * a +b* (loc_image[j + (i - 1) * height] + loc_image[j + (i + 1) * height] + loc_image[j - 1 + val_1]  + loc_image[j + 1 + val_1] );
     // tmp_image[j + val_1] =  image[j + val_1] * a +b* (image[j + (i - 1) * height] + image[j + (i + 1) * height] + image[j - 1 + val_1]  + image[j + 1 + val_1] );
      float fromLeft; //toRight
      float fromRight; //toLeft
               
     }
  } 
// printf("leaving stencil for rank %d \n", rank);
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
