import argparse
import cv2
from fusion import Fusion

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # fill parser with information about program arguments
    parser.add_argument('-i', '--images', nargs='+', type=str,
                        default=['images/MRI-SPECT/mr.png',
                                 'images/MRI-SPECT/tc.png'],
                        help='define path to images')
    parser.add_argument('-o', '--output', default='./results/test.png',
                        help='define output path of fused image')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()

if __name__ == '__main__':
    # Parse arguments
    args = make_args_parser()
    print_args(args)
    # Read images
    input_images = []
    for image in args.images:
        input_images.append(cv2.imread(image))
    # Compute fusion image
    FU = Fusion(input_images)
    fusion_img = FU.fuse()
    # Write fusion image
    cv2.imwrite(args.output, fusion_img)
