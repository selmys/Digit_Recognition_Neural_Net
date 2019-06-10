#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>

 
#define EPOCH 10000

FILE *image_label;
FILE *image_data;

int reverse_int(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int open_training_files() {
	int magic_number, number_of_images;
	int rows, columns;
	image_label = fopen("t10k-labels-idx1-ubyte", "rb");
	fread(&magic_number, sizeof(int), 1, image_label);
	fread(&number_of_images, sizeof(int), 1, image_label);
	magic_number = reverse_int(magic_number);
	number_of_images = reverse_int(number_of_images);
	//printf("Magic number is %d\n", magic_number);
	//printf("Number of images is %d\n",number_of_images);

	image_data = fopen("t10k-images-idx3-ubyte", "rb");
	fread(&magic_number, sizeof(int), 1, image_data);
	fread(&number_of_images, sizeof(int), 1, image_data);
	fread(&rows, sizeof(int), 1, image_data);
	fread(&columns, sizeof(int), 1, image_data);
	magic_number = reverse_int(magic_number);
	number_of_images = reverse_int(number_of_images);
	rows = reverse_int(rows);
	columns = reverse_int(columns);

	//printf("Magic number is %d\n", magic_number);
	//printf("Number of images is %d\n",number_of_images);
	//printf("Number of rows is %d\n",rows);
	//printf("Number of columns is %d\n",columns);
	return 0;
}


int get_next_label() {
	unsigned char label;
	fread(&label, 1, 1, image_label);
	return label;
}

int get_next_pixel() {
	unsigned char pixel;
	fread(&pixel, 1, 1, image_data);
	return pixel;
}

int main() {	
	FILE *fp;
	int p,l;
	fp = fopen("testdata.txt", "w");
	open_training_files();	
	for (int e=0;e<EPOCH;e++) {	
		l=get_next_label();
		fprintf(fp,"%d ",l);
		for (int i=0;i<784;i++) {
			p=get_next_pixel();
			fprintf(fp," %d",p);
		}
		fprintf(fp,"\n");
	}
	fclose(image_data);
	fclose(image_label);
	fclose(fp);
	return 0;
}
