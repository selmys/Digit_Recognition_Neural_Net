#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <png.h>
#include <errno.h>
#include <signal.h>

/*
 * Input   Hidden	Output
 *  784		 32	      10
 * =======================
 *       Wh       Wo 
 * X[0]		H[0]	 O[0]
 * X[1]		H[1]	 O[1]
 *  ..		 ..		  ..	
 *  ..		 ..		  ..	
 *  ..		 ..		 O[9]
 *  ..		 ..	
 *  ..		H[31]
 *  ..
 *  ..
 * X[783]
 * 
 */
 
#define EPOCH 10000

double X[EPOCH][784], Wh[784][32], dWh[784][32], Bh[32], dBh[32];
double H[EPOCH][32],  Wo[32][10],  dWo[32][10],  Bo[10], dBo[10];
double O[EPOCH][10],  Zh[EPOCH][32], Zo[EPOCH][10];
double E[EPOCH][10];
double C[EPOCH];

int number;
double alpha;
int current_e;
FILE *testing_file;

int reverse_int(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int print_Zh() {
	printf("\nZh follows:\n");
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<32;j++) {
			printf("Zh[%d][%d] = %g\n",i,j,Zh[i][j]);
		}
	}
	return 0;
}

int print_Zo() {
	printf("\nZo follows:\n");
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<10;j++) {
			printf("Zo[%d][%d] = %g\n",i,j,Zo[i][j]);
		}
	}
	return 0;
}

int print_dWh() {
	printf("\ndWh follows:\n");
	for (int i=0;i<784;i++) {
		printf("%4d ",i);
		for (int j=0;j<32;j++) {
			printf("%g ",dWh[i][j]);
		}
		printf("\n");
	}
	return 0;
}

int print_dWo() {
	printf("\ndWo follows:\n");
	for (int i=0;i<32;i++) {
		printf("%3d ",i);
		for (int j=0;j<10;j++) {
			printf("%g ",dWo[i][j]);
		}
		printf("\n");
	}
	return 0;
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double dSigmoid(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

int initializeWeights() {
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			Wh[i][j] = rand()%5/5.0;
			if (rand()%10 > 5)
				Wh[i][j] = -Wh[i][j];
		}
	}
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			Wo[i][j] = rand()%5/5.0;
			if (rand()%10 > 5)
				Wo[i][j] = -Wo[i][j];
		}
	} 
	return 0;
}

int print_Wh() {
	printf("\nWh follows:\n");
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			printf("Wh[%d][%d] = %g ", i,j,Wh[i][j]);
		}
		printf("\n");
	}
	return 0;
}

int print_Wo() {
	printf("\nWo follows:\n");
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			printf("Wh[%d][%d] = %g ", i,j,Wh[i][j]);
		}
		printf("\n");
	}
	return 0;
}

int initializeBiases() {
	// set biases to 1
	for (int i=0;i<32;i++) {
		Bh[i] = 1.0;
	}
	for (int i=0;i<10;i++) {
		Bo[i] = 1.0;
	}
	return 0;
}

int initialize_ALL() {
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<32;j++) {
			H[i][j] = 0.0;
		}
	}
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<10;j++) {
			O[i][j] = 0.0;
		}
	}
	for (int i=0;i<EPOCH;i++) {
		C[i] = 0.0;
	}
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<32;j++) {
			Zh[i][j] = 0.0;
		}
	}
	for (int i=0;i<EPOCH;i++) {
		for (int j=0;j<10;j++) {
			Zo[i][j] = 0.0;
		}
	}
	return 0;
}

int compute_H(int b) {
	// X[b][784] x Wh[784][32] = H[b][32]
	for (int j=0;j<32;j++) {
		for (int k=0;k<784;k++) {
			H[b][j] -= X[b][k]*Wh[k][j];
		}
		H[b][j] -= Bh[j];
		H[b][j] = sigmoid(H[b][j]);
	}
	return 0;
}

int compute_O(int b) {
	// H[b][32] x Wo[32][10] = O[b][10]
	for (int j=0;j<10;j++) {
		for (int k=0;k<32;k++) {
			O[b][j] -= H[b][k]*Wo[k][j];
		}
		O[b][j] -= Bo[j];
		O[b][j] = sigmoid(O[b][j]);
	}
	return 0;
}

int error_too_big(int b, double err) {
	double x;
	for (int i=0;i<10;i++) {
		//print_E(b);
		if (E[b][i] < 0.0) x = -E[b][i];
		else x = E[b][i];
		if ((i != number) && (x > err)) return 1;
	}
	if (E[b][number] < 0.0) x = -E[b][number];
		else x = E[b][number];
	if (E[b][number] > err) return 1;
	return 0;
}

int compute_error(int b) {
	// desired - actual
	for (int i=0;i<10;i++) {
		if (i != number) {
			E[b][i] = 0.0 - O[b][i];
		}
	}
	E[b][number] = 1.0 - O[b][number];
	return 0;
}

double compute_cost(int b) {
	// cost is sum of mean squared errors (mse)
	C[b] = 0.0;
	for (int j=0;j<10;j++) {
		C[b] += E[b][j]*E[b][j];
	}
	C[b] = C[b]/10.0;
	return 0;
}

int compute_dBo(int b) {
	// dBo[10] = E[b][10] * f'(H[b][32] x Wo[b][10] + Bo[10])
	
	// zero out Zo[b][10]
	for (int i=0;i<10;i++) {
		Zo[b][i] = 0.0;
	}

	// compute H x Wo + Bo
	for (int j=0;j<10;j++) {
		for (int k=0;k<32;k++) {
			Zo[b][j] += H[b][k]*Wo[k][j];
		}
		Zo[b][j] += Bo[j];
	}
	
	// compute error * dSigmoid(Zo)
	for (int j=0;j<10;j++) {
		dBo[j] = E[b][j]*dSigmoid(Zo[b][j]);
	}
	return 0;
}

int compute_dWo(int b) {
	// Ht x dBo
	// H[b][32] transpose x dBo[10]
	
	double Ht[32][1];
	
	//first transpose H
	for (int j=0;j<32;j++) {
		Ht[j][0] = H[b][j];
	}
	
	// zero out dWo
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			dWo[i][j] = 0.0;
		}
	}
	
	//next do matrix multiplication
	// [32][1] x [10]
	for (int i=0;i<1;i++) {
		for (int j=0;j<32;j++) {
			for (int k=0;k<10;k++) {
				dWo[j][k] += Ht[j][i]*dBo[k];
			}
		}
	}
	return 0;
}

int compute_dBh(int b) {
	// dBh[32] = (dBo[1][10] x Wot[10][32]) * 
	//               f'(X[1][784] x Wh[784][32] + Bh[32])
	
	// compute X x Wh + Bh = Zh
	// [1][784] x [784][32] + [32] = [1][32]
	
	// zero out Zh[b][32]
	for (int i=0;i<32;i++) {
		Zh[b][i] = 0.0;
	}
	
	// multiply X x Wh = Zh
	for (int j=0;j<32;j++) {
		for (int k=0;k<784;k++) {
			Zh[b][j] += X[b][k]*Wh[k][j];
		}
	}
	
	// add Bh to Zh
	for (int i=0;i<32;i++) {
		Zh[b][i] += Bh[i];
	}
	
	// take dSigmoid of Zh
	for (int i=0;i<32;i++) {
		Zh[b][i] = dSigmoid(Zh[b][i]);
	}
	
	// now transpose Wo
	double Wot[10][32];
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			Wot[j][i] = Wo[i][j];
		}
	}
	
	// multiply dBo x Wot
	// [1][10]x[10][32]
	for (int j=0;j<10;j++) {
		for (int k=0;k<32;k++) {
			dBh[k] += dBo[j]*Wot[j][k];
		}
	}
	
	// finally do straight multiplication
	// dBh = dBh times Zh
	for (int j=0;j<32;j++) {
		dBh[j] = dBh[j] * Zh[b][j];
	}
	return 0;
}

int compute_dWh(int b) {
	// Xt x dBh
	// X[b][784] transpose x dBh[1][32]
	
	double Xt[784][1];
	//first transpose X to get Xt
	for (int j=0;j<784;j++) {
		Xt[j][0] = X[b][j];
	}
	
	// zero out dWh
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			dWh[i][j] = 0.0;
		}
	}
	
	//next do matrix multiplication
	// [1][784] x [784][32]
	for (int i=0;i<1;i++) {
		for (int j=0;j<32;j++) {
			for (int k=0;k<784;k++) {
				dWh[k][j] += Xt[k][i]*dBh[j];
			}
		}
	}
	return 0;
}

int update_Wh() {
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			Wh[i][j] = Wh[i][j] - alpha * dWh[i][j];
		}
	}
	return 0;
}

int update_Wo() {
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			Wo[i][j] = Wo[i][j] - alpha * dWo[i][j];
		}
	}
	return 0;
}

int update_Bh() {
	for (int i=0;i<32;i++) {
		Bh[i] = Bh[i] - alpha * dBh[i];
	}
	return 0;
}

int update_Bo() {
	for (int i=0;i<10;i++) {
		Bo[i] = Bo[i] - alpha * dBo[i];
	}
	return 0;
}

int print_O(int b) {
	printf("Number is %d and epoch is %d\n",number,b);
	for (int i=0;i<10;i++) {
		printf("O[%d][%d] = %g\n",b,i,O[b][i]);
	}
	return 0;
}

int print_H() {
	for (int i=0;i<32;i++) {
		printf("H[0][%d] = %g\n",i,H[0][i]);
	}
	return 0;
}

int print_dBo() {
	printf("\ndBo follows:\n");
	for (int i=0;i<10;i++) {
		printf("dBo[%d] = %g\n",i,dBo[i]);
	}
	return 0;
}

int print_dBh() {
	printf("\ndBh follows:\n");
	for (int i=0;i<32;i++) {
		printf("dBh[%d] = %g\n",i,dBh[i]);
	}
	return 0;
}

int get_png_image(int e) {
	const char * png_file = "number-28x28.png";
    png_structp	png_ptr;
    png_infop info_ptr;
    FILE * fp;
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
    png_bytepp rows;
    fp = fopen (png_file, "rb");
    number = 3;
    png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); 
    info_ptr = png_create_info_struct (png_ptr);
    png_init_io (png_ptr, fp);
    png_read_png (png_ptr, info_ptr, 0, 0);
    png_get_IHDR (png_ptr, info_ptr, & width, & height, & bit_depth,
		  & color_type, & interlace_method, & compression_method,
		  & filter_method);
    rows = png_get_rows (png_ptr, info_ptr);
    printf ("Width is %u, height is %u\n", width, height);
    int rowbytes;
    rowbytes = png_get_rowbytes (png_ptr, info_ptr);
    //printf ("Row bytes = %d\n", rowbytes);
    int k=0;
    for (int j = 0; j < height; j++) {
		png_bytep row;
		row = rows[j];
		for (int i = 0; i < rowbytes; i++) {
		    png_byte pixel;
		    pixel = row[i];
		    /* fill X[784] with new pixel data */
		    X[e][k] = (255-pixel)/255.0;
		    //X[e][k] = pixel/255.0;
		    k++;
		    //printf("%c ",pixel==255 ? 32 : 42);
		}
		//printf("\n");
    }
    fclose(fp);
	return 0;
}

int saveBo(FILE *fp) {
	for (int i=0;i<10;i++) {
		fprintf(fp,"%g ",Bo[i]);
	}
	fprintf(fp,"\n");
	return 0;
}

int saveBh(FILE *fp) {
	for (int i=0;i<32;i++) {
		fprintf(fp,"%g ",Bh[i]);
	}
	fprintf(fp,"\n");
	return 0;
}

int saveWo(FILE *fp) {
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			fprintf(fp,"%g ",Wo[i][j]);
		}
		fprintf(fp,"\n");
	}
	return 0;
}

int saveWh(FILE *fp) {
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			fprintf(fp,"%g ",Wh[i][j]);
		}
		fprintf(fp,"\n");
	}
	return 0;
}

int saveWeights() {
	FILE *fp;
	fp = fopen("weights.txt","w");
	saveWo(fp);
	saveWh(fp);
	saveBo(fp);
	saveBh(fp);
	fclose(fp);
	return 0;
}

int restoreBo(FILE *fp) {
	char c;
	for (int i=0;i<10;i++) {
		fscanf(fp,"%lf",&Bo[i]);
	}
	fscanf(fp,"%c",&c);
	return 0;
}

int restoreBh(FILE *fp) {
	char c;
	for (int i=0;i<32;i++) {
		fscanf(fp,"%lf",&Bh[i]);
	}
	fscanf(fp,"%c",&c);
	return 0;
}

int restoreWo(FILE *fp) {
	char c;
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			fscanf(fp,"%lf",&Wo[i][j]);
		}
		fscanf(fp,"%c",&c);
	}
	return 0;
}

int restoreWh(FILE *fp) {
	char c;
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			fscanf(fp,"%lf",&Wh[i][j]);
		}
		fscanf(fp,"%c",&c);
	}
	return 0;
}

int restoreWeights() {
	FILE *fp;
	fp = fopen("weights.txt","r");
	restoreWo(fp);
	restoreWh(fp);
	restoreBo(fp);
	restoreBh(fp);
	fclose(fp);
	return 0;
}

int print_X(int e) {
	for (int i=0;i<784;i++) {
		printf("%g ",X[e][i]);
	}
	printf("\n");
	return 0;
}

int smallest_O(int e) {
	double s=100.0;
	int j=10;
	for (int i=0;i<10;i++) {
		if (O[e][i] < s) {
			s = O[e][i];
			j=i;
		}
	}
	return j;
}

void signalHandler(int sig_num) {
	print_O(current_e);
	fflush(stdout);
	exit(0);
}

int main() {
	int e,pixel,correct=0;
	char c;
	signal(SIGINT, signalHandler);
	restoreWeights();
	initialize_ALL();
	testing_file = fopen("testdata.txt","r");
	for (e=0;e<EPOCH;e++) {
		fscanf(testing_file,"%d",&number);
		printf("Number is %d\n", number);
		for (int i=0;i<784;i++) {
			fscanf(testing_file,"%d",&pixel);
			//printf("pixel is %d\n", pixel);
			//fflush(stdout);
			X[e][i] = pixel/255.0;
		}
		fscanf(testing_file,"%c",&c);
		//printf("Starting feed forward\n");
		fflush(stdout);
		/// feed forward
		compute_H(e); 
		compute_O(e); 
		if (smallest_O(e) == number) {
			printf("%d is correct\n", number);
			correct++;
		}
		//print_O(0);
	}
	printf("Number correct is %d and percent correct is %f\n",correct,
				correct*100.0/e);
	fclose(testing_file);
	get_png_image(0);
	compute_H(0); 
	compute_O(0); 
	if (smallest_O(0) == number) {
		printf("%d is correct\n", number);
		correct++;
	}
	print_O(0);
	return 0;
}
