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
 
#define EPOCH 60001

double X[1][784], Wh[784][32], dWh[784][32], Bh[32], dBh[32];
double H[1][32],  Wo[32][10],  dWo[32][10],  Bo[10], dBo[10];
double O[1][10],  Zh[1][32], Zo[1][10];
double E[1][10];
double total_cost=0;
double alpha = 0.05; 	// training rate
int number;				// label
int current_e;			// loop control variable

FILE *training_file;

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
	for (int i=0;i<1;i++) {
		for (int j=0;j<32;j++) {
			printf("Zh[%d][%d] = %g\n",i,j,Zh[i][j]);
		}
	}
	return 0;
}

int print_Zo() {
	printf("\nZo follows:\n");
	for (int i=0;i<1;i++) {
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
	return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
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

int initialize_deltas() {
	for (int i=0;i<1;i++) {
		for (int j=0;j<32;j++) {
			Zh[i][j] = 0.0;
		}
	}
	for (int i=0;i<1;i++) {
		for (int j=0;j<10;j++) {
			Zo[i][j] = 0.0;
		}
	}
	return 0;
}

int compute_H() {
	// X[b][784] x Wh[784][32] = H[b][32]
	for (int j=0;j<32;j++) {
		H[0][j] = 0.0;
		for (int k=0;k<784;k++) {
			H[0][j] += X[0][k]*Wh[k][j];
		}
		H[0][j] += Bh[j];
		H[0][j] = sigmoid(H[0][j]);
	}
	return 0;
}

int compute_O() {
	// H[b][32] x Wo[32][10] = O[b][10]
	for (int j=0;j<10;j++) {
		O[0][j] = 0.0;
		for (int k=0;k<32;k++) {
			O[0][j] += H[0][k]*Wo[k][j];
		}
		O[0][j] += Bo[j];
		// we want O[b][number] to be 1 and others to be 0
		// so we subtract sigmoid from 1.0
		O[0][j] = 1.0 - sigmoid(O[0][j]);
	}
	return 0;
}

int error_too_big(double err) {
	// ideally err should be zero
	// but we'll settle for something small
	for (int i=0;i<10;i++) {
		if ((i != number) && (E[0][i] < -err)) return 1;
	}
	if ((E[0][number] - 1.0 ) > err) return 1;
	return 0;
}

int compute_error() {
	/// maybe switch to actual - desired
	// desired - actual
	// actuals should all be positive because of sigmoid function
	// so all errors should be negative except for number
	for (int i=0;i<10;i++) {
		if (i != number) {
			E[0][i] = 0.0 - O[0][i];
		}
	}
	E[0][number] = 1.0 - O[0][number];
	return 0;
}

int print_E() {
	printf("Number is %d\n", number);
	for (int i=0;i<10;i++) {
		printf("E[0][%d] = %g\n",i,E[0][i]);
	}
	return 0;
}

double compute_cost() {
	// cost is sum of mean squared errors (MSE)
	double c = 0.0;
	for (int j=0;j<10;j++) {
		c += E[0][j]*E[0][j];
	}
	c = c/10.0;
	return c;
}

int compute_dBo() {
	// dBo[10] = E[b][10] * f'(H[b][32] x Wo[b][10] + Bo[10])
	
	// zero out Zo[b][10]
	for (int i=0;i<10;i++) {
		Zo[0][i] = 0.0;
	}

	// compute H x Wo + Bo
	for (int j=0;j<10;j++) {
		for (int k=0;k<32;k++) {
			Zo[0][j] += H[0][k]*Wo[k][j];
		}
		Zo[0][j] += Bo[j];
	}
	
	// compute error * dSigmoid(Zo)
	for (int j=0;j<10;j++) {
		dBo[j] = E[0][j]*dSigmoid(Zo[0][j]);
	}
	return 0;
}

int compute_dWo() {
	// Ht x dBo
	// H[b][32] transpose x dBo[10]
	
	double Ht[32][1];
	
	//first transpose H
	for (int j=0;j<32;j++) {
		Ht[j][0] = H[0][j];
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

int compute_dBh() {
	// dBh[32] = (dBo[1][10] x Wot[10][32]) * 
	//               f'(X[1][784] x Wh[784][32] + Bh[32])
	
	// compute X x Wh + Bh = Zh
	// [1][784] x [784][32] + [32] = [1][32]
	
	// zero out Zh[0][32]
	for (int i=0;i<32;i++) {
		Zh[0][i] = 0.0;
	}
	
	// multiply X x Wh = Zh
	for (int j=0;j<32;j++) {
		for (int k=0;k<784;k++) {
			Zh[0][j] += X[0][k]*Wh[k][j];
		}
	}
	
	// add Bh to Zh
	for (int i=0;i<32;i++) {
		Zh[0][i] += Bh[i];
	}
	
	// take dSigmoid of Zh
	for (int i=0;i<32;i++) {
		Zh[0][i] = dSigmoid(Zh[0][i]);
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
		dBh[j] = dBh[j] * Zh[0][j];
	}
	return 0;
}

int compute_dWh() {
	// Xt x dBh
	// X[b][784] transpose x dBh[1][32]
	
	double Xt[784][1];
	//first transpose X to get Xt
	for (int j=0;j<784;j++) {
		Xt[j][0] = X[0][j];
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

int print_O() {
	printf("Number is %d\n",number);
	for (int i=0;i<10;i++) {
		printf("O[0][%d] = %g\n",i,O[0][i]);
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

int get_png_image() {
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
		    X[0][k] = (255-pixel)/255.0;
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
		fprintf(fp,"%.17g ",Bo[i]);
	}
	fprintf(fp,"\n");
	return 0;
}

int saveBh(FILE *fp) {
	for (int i=0;i<32;i++) {
		fprintf(fp,"%.17g ",Bh[i]);
	}
	fprintf(fp,"\n");
	return 0;
}

int saveWo(FILE *fp) {
	for (int i=0;i<32;i++) {
		for (int j=0;j<10;j++) {
			fprintf(fp,"%.17g ",Wo[i][j]);
		}
		fprintf(fp,"\n");
	}
	return 0;
}

int saveWh(FILE *fp) {
	for (int i=0;i<784;i++) {
		for (int j=0;j<32;j++) {
			fprintf(fp,"%.17g ",Wh[i][j]);
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

int print_X() {
	for (int i=0;i<784;i++) {
		printf("%g ",X[0][i]);
	}
	printf("\n");
	return 0;
}

void signalHandler(int sig_num) {
	print_O(current_e);
	fflush(stdout);
	exit(0);
}

int main() {
	int e;
	char c;
	int pixel;
	signal(SIGINT, signalHandler);
	initializeWeights();
	initializeBiases();
	for (int s=0;s<12;s++) {
		initialize_deltas();
		training_file = fopen("traindata.txt","r");
		total_cost=0.0;
		for (e=0;e<EPOCH-1;e++) {
			current_e=e;
			fscanf(training_file,"%d",&number);
			for (int i=0;i<784;i++) {
				fscanf(training_file,"%d",&pixel);
				//X[e][i] = 0.99*pixel/255.0 + 0.01;
				X[0][i] = pixel/255.0;
			}
			fscanf(training_file,"%c",&c);
			do {
				/// feed forward
				compute_H(); 
				compute_O(); 
				compute_error();
				/// print_E(e);
				
				/// backpropagation
				compute_dBo();	
				compute_dWo();
				compute_dBh();
				compute_dWh();	
				
				/// update weights and biases		
				update_Wh();
				update_Wo();
				update_Bh();
				update_Bo();
							
			} 
			while (error_too_big(0.15));
			total_cost+=compute_cost();
			//printf("Completed epoch %d\n",e);
		}
		fclose(training_file);
		printf("Completed epoch %d\n",s);
		printf("Network cost over %d images is %g\n",e,total_cost/e);
		printf("Saving new weights\n");
		/// save the weights
		saveWeights();
		/// now test on my png image
		//get_png_image(e);
		//compute_H(e); 
		//compute_O(e); 
		//print_O(e);
		/// shuffle the training file
		system("shuf testdata.txt > testdata1.txt");
		system("mv testdata1.txt testdata.txt");
	}
	return 0;
}
