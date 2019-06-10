#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
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
 
double X[1][784], Wh[784][32], Bh[32];
double H[1][32],  Wo[32][10],  Bo[10];
double O[1][10];

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

int initialize_HO() {
	for (int i=0;i<32;i++) {
		H[0][i] = 0.0;
	}
	for (int i=0;i<10;i++) {
		O[0][i] = 0.0;
	}
	return 0;
}

int compute_H() {
	// X[b][784] x Wh[784][32] = H[b][32]
	for (int j=0;j<32;j++) {
		for (int k=0;k<784;k++) {
			H[0][j] -= X[0][k]*Wh[k][j];
		}
		H[0][j] -= Bh[j];
		H[0][j] = sigmoid(H[0][j]);
	}
	return 0;
}

int compute_O() {
	// H[b][32] x Wo[32][10] = O[b][10]
	for (int j=0;j<10;j++) {
		for (int k=0;k<32;k++) {
			O[0][j] -= H[0][k]*Wo[k][j];
		}
		O[0][j] -= Bo[j];
		O[0][j] = sigmoid(O[0][j]);
	}
	return 0;
}

int print_O() {
	for (int i=0;i<10;i++) {
		printf("O[0][%d] = %g\n",i,O[0][i]);
	}
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

int smallest_O() {
	double s=100.0;
	int j=10;
	for (int i=0;i<10;i++) {
		if (O[0][i] < s) {
			s = O[0][i];
			j=i;
		}
	}
	return j;
}

int main() {
	int number, pixel;
	FILE * png_file;
	restoreWeights();
	initialize_HO();
	png_file = fopen("pngdata.txt","r");
	fscanf(png_file,"%d",&number);
	for (int i=0;i<784;i++) {
		fscanf(png_file,"%d",&pixel);
		X[0][i] = pixel/255.0;
	}
	compute_H(); 
	compute_O(); 
	if (smallest_O() == number) 
		printf("%d is correct\n", number);
	else
		printf("%d is incorrect\n", number);
	print_O();
	fclose(png_file);
	return 0;
}
