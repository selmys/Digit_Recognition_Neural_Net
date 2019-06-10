#include <stdio.h>
#include <stdlib.h>
#include <png.h>

int get_png_image(int number, char * png_file) {
    png_structp	png_ptr;
    png_infop info_ptr;
    png_uint_32 width;
    png_uint_32 height;
    int bit_depth;
    int color_type;
    int interlace_method;
    int compression_method;
    int filter_method;
    png_bytepp rows;
    FILE * fp_in;
    FILE * fp_out;
    fp_in = fopen (png_file, "rb");
    fp_out = fopen ("pngdata.txt", "w");
    png_ptr = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); 
    info_ptr = png_create_info_struct (png_ptr);
    png_init_io (png_ptr, fp_in);
    png_read_png (png_ptr, info_ptr, 0, 0);
    png_get_IHDR (png_ptr, info_ptr, & width, & height, & bit_depth,
		  & color_type, & interlace_method, & compression_method,
		  & filter_method);
    rows = png_get_rows (png_ptr, info_ptr);
    printf ("Width is %lu, height is %lu\n", width, height);
    int rowbytes;
    rowbytes = png_get_rowbytes (png_ptr, info_ptr);
    fprintf(fp_out,"%d ",number);
    for (int j = 0; j < height; j++) {
		png_bytep row;
		row = rows[j];
		for (int i = 0; i < width; i++) {
		    png_byte pixel;
		    pixel = row[i];
		    fprintf(fp_out," %d",255-pixel);
		}
    }
    fclose(fp_in);
    fclose(fp_out);
	return 0;
}

int main() {
	get_png_image(5, "five.png");
	return 0;
}
