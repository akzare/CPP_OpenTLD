/***********************************************************************
  $FILENAME    : Constants.h

  $TITLE       : Constants header file

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Includes required constants for the entire tracker

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifndef CONSTANTS_H_
#define CONSTANTS_H_


#ifndef NAN
#define NAN 0x7fc00000
#endif

//#define NAN 0/0
#ifndef NANN
#define NANN -99999
#endif

#ifndef M_PI
#define M_PI 3.14159265358979L
#endif

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define MAX_IMAGE_WIDTH 1440
#define MAX_IMAGE_HEIGHT 960
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define INITIAL_WINDOW_WIDTH (IMAGE_WIDTH>>4)+1
#define INITIAL_WINDOW_HEIGHT (IMAGE_WIDTH>>1)+1

#define MIN_NUM_SAMPLES 10

// minimal size of the object's bounding box in the scanning grid, it may significantly influence speed of TLD, set it to minimal size of the object
#define SINGLETRACKEROPT_MIN_WIN 24
// size of normalized patch in the object detector, larger sizes increase discriminability, must be square
#define SINGLETRACKEROPT_PATCHSIZE_X 15
// size of normalized patch in the object detector, larger sizes increase discriminability, must be square
#define SINGLETRACKEROPT_PATCHSIZE_Y 15
// if set to one, the model automatically learns mirrored versions of the object
#define SINGLETRACKEROPT_FLIPLR 0
// fraction of evaluated bounding boxes in every frame, maxbox = 0 means detector is turned off, if you don't care about speed set it to 1
#define SINGLETRACKEROPT_MAXBBOX 1
// online learning on/off, of 0 detector is trained only in the first frame and then remains fixed
#define SINGLETRACKEROPT_UPDATE_DETECTOR 1
// Detector(fern) maximum number of bounding boxes
#define DT_MAX_BB 100
// Gaussian Filter Blur Size 25
#define BLURKSIZE 47
// Lucas Kanade window size
#define LUCASKANADE_WINSIZE 61

#endif /* CONSTANTS_H_ */
