/***********************************************************************
  $FILENAME    : tldTrackerOCL.cpp

  $TITLE       : TLD tracker class implementation for OpenCL

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Defines the TLD tracker class for running on GPU

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#ifdef USE_OCL_

#include <stdexcept>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "Constants.h"
#include "BoundingBox.hpp"
#include "Distance.hpp"
#include "Statistics.hpp"
#include "Linkage.hpp"
#include "Image.hpp"
#include "tldTracker.hpp"

using namespace std;
using namespace cv;

#define PROFILINING_START_TIME( )                    \
                start = clock()

#define PROFILINING_STOP_TIME( string )               \
                stop = clock();                       \
                elapsedTime = (float)(stop - start) / \
                (float)CLOCKS_PER_SEC * 1000.0f;      \
                nowis = time(0);                      \
                localtm = localtime(&nowis);          \
                strftime (timeBuffer, 256, "%b/%e/%Y,%H:%M:%S", localtm); \
                profilingLogFile << timeBuffer << string << elapsedTime << "\n"


tld::ocl::tracker::tracker( trackerApp* pApp )
{
  // print a welcome message
  cout << "Standalone tracker based on TLD (Tracking, learning, detection (aka Predator)) algorithm.\n"
      << "TLD has been developed by Zdenek Kalal.\n"
      << "TLD has been converted to C++ by Armin Zare Zadeh.\n";

  cout << "Current instance: OpenCL(OCL) : NOT IMPLEMENTED!\n";

}

tld::ocl::tracker::~tracker()
{
}

#endif
