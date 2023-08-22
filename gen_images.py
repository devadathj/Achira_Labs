import os
import argparse
from PIL import Image
import random
import cv2

def get_parser():
    parser = argparse.ArgumentParser(description='Generating images with distributed shapes')
    parser.add_argument('--input-folder', help='Path to the folder containing the image', required=True)
    parser.add_argument('--output-folder', help='Path to the folder where generated images have to be saved', required=True)
    parser.add_argument('--annotation-folder', help='Path to the folder where annotations of generated images have to be saved', required=True)
    parser.add_argument('--nout', help='Number of output images to be generated', type=int, required=True)
    parser.add_argument('--out-dims', help='Size of the output images', type=int, required=True)
    parser.add_argument('--shape_percent', help='Relative area of the image that needs to be covered by shapes', type=int, default=16)

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    annotation_folder = args.annotation_folder
    nout = args.nout
    out_dims = args.out_dims
    shape_percent = args.shape_percent

    # #If output image is not a square, we can take the input arguement in 'M1_M2' format and convert it to an array.
    # out_dims = out_dims.split('_')
    # out_dims = [int(item) for item in out_dims]

    # Reading shapes
    '''
    Reading .tiff files using Pillow library corrupts the image. Hence, they are read through OpenCV and converted to Pillow.
    Pillow is better at handling image augmentation than OpenCV. Hence Pillow is used to perform further operations.
    '''
    shapes_list = list()
    total_shape_area = 0
    for shape in os.listdir(input_folder):
        shapes_list.append(Image.fromarray(cv2.imread(os.path.join(input_folder, shape))))

        # The shapes are of same size in this case. The step below is performed for the situation where the shapes could be of different sizes
        total_shape_area += shapes_list[-1].width * shapes_list[-1].height

    shape_count = int((out_dims ** 2) * (shape_percent / 100) / total_shape_area)

    # Creating images
    for iter_image in range(nout):

        with open(os.path.join(annotation_folder, 'image_' + str(iter_image) + '.txt'), 'w') as f:  # This is to generate annotations for training models.

            # Creating background
            background = Image.new('RGB', (out_dims, out_dims), color='black')

            location_tracker = list()
            for iter_each_shape in range(shape_count):
                for iter_unique_shape in range(len(shapes_list)):

                    overlay = shapes_list[iter_unique_shape]

                    # Overlaying the shape onto background
                    ''' Calculate a random position to overlay the image within the background. 
                    Subtracting the overlay's dimensions ensures that the shape will never exceed the image's boundary.
                    To prevent overlap, every location is stored and checked before overlaying'''
                    while True:
                        x = random.randint(0, background.width - overlay.width)
                        y = random.randint(0, background.height - overlay.height)

                        switch = False
                        for iter_location in location_tracker:
                            if iter_location[0][0] <= x <= iter_location[0][1] or iter_location[0][0] <= x + overlay.width <= iter_location[0][1]:
                                if iter_location[1][0] <= y <= iter_location[1][1] or iter_location[1][0] <= y + overlay.height <= iter_location[1][1]:
                                    switch = True
                                    break

                        if not switch:
                            break

                    location_tracker.append([(x, x + overlay.width), (y, y + overlay.height)])

                    '''
                    Scaling and rotating is done after fixing the location to prevent the new shape sizes from breaking the overlap logic. 
                    This would not bug out as the images are only scaling down. A better logic should be implemented if the images need to be scaled up.
                    '''
                    # Scaling the shape
                    scale_factor = random.uniform(0.75, 1)

                    new_width = int(overlay.width * scale_factor)
                    new_height = int(overlay.height * scale_factor)

                    overlay = overlay.resize((new_width, new_height))

                    # Rotating the shape
                    rotation_factor = random.randint(0, 90)
                    overlay = overlay.rotate(rotation_factor, expand=False)

                    # Overlaying the shapes
                    background.paste(overlay, (x, y))

                    f.write(str(iter_unique_shape) + ' ' + str((x + overlay.width / 2) / background.width) + ' ' + str((y + overlay.height / 2) / background.height) + ' ' + str(overlay.width / background.width) + ' ' + str(overlay.height / background.height) + '\n')

            background.save(os.path.join(output_folder, 'image_' + str(iter_image) + '.png'))

        f.close()
