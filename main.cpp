/***********************************************************************
  $FILENAME    : main.cpp

  $TITLE       : main application class implementation

  $DATE        : 7 Nov 2017

  $VERSION     : 1.0.0

  $DESCRIPTION : Implements the main application class of the tracker

  $AUTHOR     : Armin Zare Zadeh (ali.a.zarezadeh @ gmail.com)

************************************************************************/

#include <iostream>
#include <stdexcept>
#ifdef USE_OCL_
#include <oclUtils.h>
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef USE_OCL_
#include "opencv2/ocl/ocl.hpp"
#endif

#include "Constants.h"
#include "BoundingBox.hpp"
#include "tldTracker.hpp"
#include "main.hpp"


using namespace std;

bool help_showed = false;

#ifdef USE_OCL_
const char *integralimg_ocl_kernel;
const char *fern_ocl_kernel;
#endif


trackerArgs::trackerArgs()
{
  source.src_is_video = false;
  source.src_is_camera = false;
  source.src_is_image = false;
  source.camera_id = 0;

  source.write_video = false;
  source.dst_video_fps = 1.;

  // --image _input/motorcyclist/00001.png --write_video true --dst_video OpenTLD.avi --dst_video_fps 8
  // for motorcyclist
  source.bbox.x = 287;
  source.bbox.y = 35;
  source.bbox.width = 25;
  source.bbox.height = 42;

  model.min_win = SINGLETRACKEROPT_MIN_WIN;
  model.patchsize[0] = SINGLETRACKEROPT_PATCHSIZE_X;
  model.patchsize[1] = SINGLETRACKEROPT_PATCHSIZE_Y;
  model.fliplr = SINGLETRACKEROPT_FLIPLR;
  model.ncc_thesame = 0.95;
  model.valid = 0.5;
  model.num_trees = 10;
  model.num_features = 13;
  model.thr_fern = 0.5;
  model.thr_nn = 0.65;
  model.thr_nn_valid = 0.7;

  // synthesis of positive examples during initialization
  p_par_init.num_closest = 10;
  p_par_init.num_warps = 20;
  p_par_init.noise = 5;
  p_par_init.angle = 20;
  p_par_init.shift = 0.02;
  p_par_init.scale = 0.02;

  // synthesis of positive examples during update
  p_par_update.num_closest = 10;
  p_par_update.num_warps = 10;
  p_par_update.noise = 5;
  p_par_update.angle = 10;
  p_par_update.shift = 0.02;
  p_par_update.scale = 0.02;

  // negative examples initialization/update
  n_par.overlap = 0.2;
  n_par.num_patches = 100;

  tracker.occlusion = 10;

  control.maxbbox = SINGLETRACKEROPT_MAXBBOX;
  control.update_detector = SINGLETRACKEROPT_UPDATE_DETECTOR;
  control.drop_img = 1;
  control.repeat = 1;

}


trackerArgs trackerArgs::ReadArgs(int argc, char** argv)
{
  trackerArgs tldargs;
  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == "--write_video") tldargs.source.write_video = (string(argv[++i]) == "true");
    else if (string(argv[i]) == "--dst_video") tldargs.source.dst_video = argv[++i];
    else if (string(argv[i]) == "--dst_video_fps") tldargs.source.dst_video_fps = atof(argv[++i]);
    else if (string(argv[i]) == "--help") PrintHelp();
    else if (string(argv[i]) == "--video") { tldargs.source.src = argv[++i]; tldargs.source.src_is_video = true; }
    else if (string(argv[i]) == "--camera") { tldargs.source.camera_id = atoi(argv[++i]); tldargs.source.src_is_camera = true; }
    else if (string(argv[i]) == "--image") { tldargs.source.src = argv[++i]; tldargs.source.src_is_image = true; }
    else if (tldargs.source.src.empty()) tldargs.source.src = argv[i];
//    else throw runtime_error((string("unknown key: ") + argv[i]));
  }
  return tldargs;
}

trackerApp::trackerApp(const trackerArgs& s)
{
  tldargs = s;
  cout << "\nControls:\n"
      << "\tESC - exit\n";

  image_idx = 1; // for motorcyclist

}

trackerApp::~trackerApp()
{
  // Free the display image
  curRGBImg.release();
  curGrayImg.release();
  curBlurImg.release();
  prvGrayImg.release();
  prvBlurImg.release();
#ifdef GRAYBLUR_SHOW_ON_
  targetPatch.release();
#endif
#ifdef FERN_OPENCV_ON_
  correspondImg.release();
#endif

}


void trackerApp::RunTLD()
{
  char text[256];
  running = true;

#ifdef USE_OCL_
  std::vector<cv::ocl::Info> oclinfo;
  devNums = cv::ocl::getDevice(oclinfo);
  if(devNums<1){
    cout << "No GPU device found!\n";
    return;
  }else{
    cout << devNums << " GPU device found.\n";

    size_t opentld_ocl_kernekLength;
    integralimg_ocl_kernel = oclLoadProgSource("./src/integralImg.cl", "// integralImageKernel:OpenCV\n", &opentld_ocl_kernekLength);
    shrCheckError(integralimg_ocl_kernel != NULL, shrTRUE);

    fern_ocl_kernel = oclLoadProgSource("./src/fern.cl", "// fernKernel:OpenCV\n", &opentld_ocl_kernekLength);
    shrCheckError(fern_ocl_kernel != NULL, shrTRUE);
  }

  tld::ocl::tracker* m_ptracker;
  m_ptracker = new tld::ocl::tracker(this);
#else
  tld::cpu::tracker* m_ptracker;
  m_ptracker = new tld::cpu::tracker(this);
#endif

  while (running)
  {
    VideoCapture videoCap;

    if (tldargs.source.src_is_video) {
      videoCap.open(tldargs.source.src.c_str());
      if (!videoCap.isOpened())
        throw runtime_error(string("can't open video file: " + tldargs.source.src));
      videoCap >> curRGBImg;
    }
    else if (tldargs.source.src_is_camera) {
      videoCap.open(tldargs.source.camera_id);
      if (!videoCap.isOpened()) {
        stringstream msg;
        msg << "can't open camera: " << tldargs.source.camera_id;
        throw runtime_error(msg.str());
      }
      videoCap >> curRGBImg;
    }
    else if (tldargs.source.src_is_image) {
      sprintf
      (image_name, "./_input/motorcyclist/%05d.png", image_idx++
      );
      curRGBImg = imread(image_name);
      if (curRGBImg.empty())
        throw runtime_error(string("can't open image file: ./_input/motorcyclist/00001.png"));
    }
    else {
      curRGBImg = imread(tldargs.source.src);
      if (curRGBImg.empty())
        throw runtime_error(string("can't open image file: " + tldargs.source.src));
    }

    cv::VideoWriter video_writer(tldargs.source.dst_video, CV_FOURCC('x','v','i','d'), tldargs.source.dst_video_fps,
        curRGBImg.size(), true);

#ifdef USE_OCL_
    cv::ocl::oclMat gpuCurGrayImg;
    cv::ocl::oclMat gpuCurBlurImg;
#endif

    // Iterate over all frames
    while (running && !curRGBImg.empty()) {

      WorkBegin();

      // Change format of image to grayscale
      cvtColor(curRGBImg, curGrayImg, CV_BGR2GRAY);

      // 1 -> 31
      GaussianBlur( curGrayImg, curBlurImg, Size( BLURKSIZE, BLURKSIZE ), 0, 0 );

#ifdef USE_OCL_
      gpuCurGrayImg.upload(curGrayImg);
      gpuCurBlurImg.upload(curBlurImg);
#endif
      // Perform TLD tracker
      TLDWorkBegin();
#ifndef USE_OCL_
      m_ptracker->process();
#else
      m_ptracker->process(gpuCurGrayImg, gpuCurBlurImg);
#endif
      TLDWorkEnd();

      // Target (display only)
      curGrayImg.copyTo(prvGrayImg);
      curBlurImg.copyTo(prvBlurImg);
      //swap(prvGrayImg, curGrayImg);

      if (m_ptracker->TRDT.prvValid == 1){
        bbox2Rect<float>(m_ptracker->TRDT.prvBBox, m_ptracker->TRDT.bbox_rect);
        rectangle(curRGBImg, m_ptracker->TRDT.bbox_rect.tl(), m_ptracker->TRDT.bbox_rect.br(), CV_RGB(255, 0, 0), 1);// red
      }
      bbox2Rect<float>(m_ptracker->TRDT.curBBox, m_ptracker->TRDT.bbox_rect);
      sprintf( text, "BBox: %d,%d", m_ptracker->TRDT.bbox_rect.x, m_ptracker->TRDT.bbox_rect.y );
      putText(curRGBImg, text, Point(5, 195), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(255, 100, 0), 2);
      if (m_ptracker->TRDT.curValid == 1){
        rectangle(curRGBImg, m_ptracker->TRDT.bbox_rect.tl(), m_ptracker->TRDT.bbox_rect.br(), CV_RGB(0, 255, 0), 2);// green
        sprintf( text, "Valid:1" );
      }else{
        sprintf( text, "Valid:0" );
      }
      putText(curRGBImg, text, Point(5, 215), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(255, 100, 0), 2);
      if (m_ptracker->TRDT.TR == 1){
        sprintf( text, "TR:1" );
      } // is tracker defined?
      else{
        sprintf( text, "TR:0" );
      }
      putText(curRGBImg, text, Point(95, 215), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(255, 100, 0), 2);
      if (m_ptracker->TRDT.DT == 1){
        sprintf( text, "DT:1(%d)", m_ptracker->TRDT.DTNum );
      } // is detector defined?
      else{
        sprintf( text, "DT:0(%d)", m_ptracker->TRDT.DTNum );
      }
      putText(curRGBImg, text, Point(155, 215), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(255, 100, 0), 2);
      if (m_ptracker->TRDT.DTNum>0){
        for (int i = 0; i<m_ptracker->detector.dt.num_dt; i++){
          m_ptracker->TRDT.bbox_rect.x = (int)(m_ptracker->detector.dt.bb.at<float>(0,i));
          m_ptracker->TRDT.bbox_rect.y = (int)(m_ptracker->detector.dt.bb.at<float>(1,i));
          m_ptracker->TRDT.bbox_rect.width = (int)(m_ptracker->detector.dt.bb.at<float>(2,i) - m_ptracker->TRDT.bbox_rect.x);
          m_ptracker->TRDT.bbox_rect.height = (int)(m_ptracker->detector.dt.bb.at<float>(3,i) - m_ptracker->TRDT.bbox_rect.y);
          rectangle(curRGBImg, m_ptracker->TRDT.bbox_rect.tl(), m_ptracker->TRDT.bbox_rect.br(), CV_RGB(0, 0, 255), 1);// green
        }
      }
      sprintf( text, "Conf: %f", m_ptracker->TRDT.curConf );
      putText(curRGBImg, text, Point(5, 235), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(255, 100, 0), 2);
      // plot TRDT.xFJ points
    #define draw_cross( center, color, d )                       \
       line( curRGBImg, Point( center.x - d, center.y - d ),     \
       Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
       line( curRGBImg, Point( center.x + d, center.y - d ),     \
       Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )
    #define draw_circle( center, color, r )                      \
      circle( curRGBImg, center, r, color, -1, 8);

      Point predict_pt;
      if (m_ptracker->TRDT.numMN > 0){
        for (int i=0; i<m_ptracker->TRDT.numMN; i++){
          predict_pt.x = cvRound(m_ptracker->TRDT.xFI.at<float>(0,i));
          predict_pt.y = cvRound(m_ptracker->TRDT.xFI.at<float>(1,i));
          draw_circle( predict_pt, Scalar(0,0,255), 1 ) // (red circle)
        }
        for (int i=0; i<m_ptracker->TRDT.numMN; i++){
          predict_pt.x = cvRound(m_ptracker->TRDT.xFJ.at<float>(0,i));
          predict_pt.y = cvRound(m_ptracker->TRDT.xFJ.at<float>(1,i));
          draw_cross( predict_pt, CV_RGB(0,255,0), 2 );   // (green cross)
        }
      }

#ifdef GRAYBLUR_SHOW_ON_
      imgPatch<float>(curGrayImg, targetPatch, m_ptracker->TRDT.curBBox, 0);
      uchar* src = (uchar *) (targetPatch.data);
      uchar* dest = (uchar *) (curGrayImg.data);
      int sStep = targetPatch.cols * sizeof(uchar);
      int dStep = curGrayImg.cols * sizeof(uchar);
      for(int y=0; y<targetPatch.rows; y++) {// row : height
        memcpy ( dest, src, sStep );
        src += sStep;
        dest += dStep;
      }
#endif

#ifdef FERN_OPENCV_ON_
      if (correspondImg.empty())
        correspondImg.create( m_ptracker->TRDT.object.rows + curGrayImg.rows, std::max(m_ptracker->TRDT.object.cols, curGrayImg.cols), CV_8UC3);
      correspondImg = Scalar(0.);
      Mat part(correspondImg, Rect(0, 0, m_ptracker->TRDT.object.cols, m_ptracker->TRDT.object.rows));
      cvtColor(m_ptracker->TRDT.object, part, CV_GRAY2BGR);
      part = Mat(correspondImg, Rect(0, m_ptracker->TRDT.object.rows, curGrayImg.cols, curGrayImg.rows));
      cvtColor(curGrayImg, part, CV_GRAY2BGR);

      if( m_ptracker->TRDT.DTFound ) {
        for( int i = 0; i < 4; i++ ) {
          Point r1 = m_ptracker->TRDT.dst_corners[i%4];
          Point r2 = m_ptracker->TRDT.dst_corners[(i+1)%4];
          line( correspondImg, Point(r1.x, r1.y+m_ptracker->TRDT.object.rows),
              Point(r2.x, r2.y+m_ptracker->TRDT.object.rows), Scalar(0,0,255) );
        }
      }

      for( int i = 0; i < (int)m_ptracker->TRDT.pairs.size(); i += 2 ) {
        line( correspondImg, m_ptracker->TRDT.objKeypoints[m_ptracker->TRDT.pairs[i]].pt,
            m_ptracker->TRDT.imgKeypoints[m_ptracker->TRDT.pairs[i+1]].pt + Point2f(0,(float)m_ptracker->TRDT.object.rows),
            Scalar(0,255,0) );
      }

//      Mat objectColor;
//      cvtColor(object, objectColor, CV_GRAY2BGR);
//      for( i = 0; i < (int)m_ptracker->TRDT.objKeypoints.size(); i++ )
//      {
//        circle( objectColor, m_ptracker->TRDT.objKeypoints[i].pt, 2, Scalar(0,0,255), -1 );
//        circle( objectColor, m_ptracker->TRDT.objKeypoints[i].pt, (1 << m_ptracker->TRDT.objKeypoints[i].octave)*15, Scalar(0,255,0), 1 );
//      }

      for( int i = 0; i < (int)m_ptracker->TRDT.imgKeypoints.size(); i++ ) {
        circle( curRGBImg, m_ptracker->TRDT.imgKeypoints[i].pt, 2, Scalar(0,0,255), -1 );
        circle( curRGBImg, m_ptracker->TRDT.imgKeypoints[i].pt, (1 << m_ptracker->TRDT.imgKeypoints[i].octave)*15, Scalar(0,255,0), 1 );
      }
      m_ptracker->TRDT.pairs.clear();
      m_ptracker->TRDT.dst_corners.clear();
      m_ptracker->TRDT.objKeypoints.clear();
      m_ptracker->TRDT.imgKeypoints.clear();
#endif

#ifdef USE_OCL_
      putText(curRGBImg, "Mode: OCL", Point(5, 255), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(25, 0, 255), 2);
#else
      putText(curRGBImg, "Mode: CPU", Point(5, 255), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(25, 0, 255), 2);
#endif

      putText(curRGBImg, "FPS (TLD only): " + TLDWorkFps(), Point(5, 275), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(25, 0, 250), 2);
      putText(curRGBImg, "FPS (total): " + WorkFPS(), Point(5, 295), FONT_HERSHEY_COMPLEX_SMALL, 1., Scalar(25, 0, 250), 2);
      imshow("OCL_TLD Color", curRGBImg);
//      moveWindow("OCL_TLD Color", 0, 0);
#ifdef GRAYBLUR_SHOW_ON_
      imshow("OCL_TLD Gray", curGrayImg);
      moveWindow("OCL_TLD Gray", 470, 0);
      imshow("OCL_TLD Blur", curBlurImg);
      moveWindow("OCL_TLD Blur", 940, 0);
#endif
#ifdef FERN_OPENCV_ON_
      imshow("OCL_TLD Correspond", correspondImg);
      moveWindow("OCL_TLD Correspond", 470, 0);
#endif

      if (tldargs.source.write_video) {
        if (!video_writer.isOpened()) {
          video_writer.open(tldargs.source.dst_video, CV_FOURCC('x','v','i','d'), tld_work_fps,
              curRGBImg.size(), true);
          if (!video_writer.isOpened())
            throw std::runtime_error("can't create video writer");
        }

        video_writer << curRGBImg;
      }
      if (tldargs.source.src_is_video || tldargs.source.src_is_camera) {
        videoCap >> curRGBImg;
      }else if (tldargs.source.src_is_image) {
        sprintf
        (image_name, "./_output/motorcyclist/%05d.jpg", image_idx
        );
#ifdef FERN_OPENCV_ON_
        imwrite(image_name, correspondImg);
#else
        imwrite(image_name, curRGBImg);
#endif
        sprintf
        (image_name, "./_input/motorcyclist/%05d.png", image_idx++
        );
        if (image_idx == 99)
          image_idx = 1;
        curRGBImg = imread(image_name);
      }

      WorkEnd();

      HandleKey((char)waitKey(3));
    }
  }
  delete m_ptracker;
  m_ptracker = NULL;

}

void trackerApp::HandleKey(char key)
{
  switch (key)
  {
  case 27:
    running = false;
    break;
  case 'j':
    if (tldargs.source.bbox.x > 0)
      tldargs.source.bbox.x--;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'i':
    if (tldargs.source.bbox.y > 0)
      tldargs.source.bbox.y--;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'l':
    tldargs.source.bbox.x++;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'm':
    tldargs.source.bbox.y++;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 's': // 37 (left arrow)
    if (tldargs.source.bbox.width > 0)
      tldargs.source.bbox.width--;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'e': // 38 (up arrow)
    tldargs.source.bbox.height++;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'd': // 39 (right arrow)
    tldargs.source.bbox.width++;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'x': // 40 (down arrow)
    if (tldargs.source.bbox.height > 0)
      tldargs.source.bbox.height--;
    cout << "BBox.(x:y) = (" << tldargs.source.bbox.x << ":" << tldargs.source.bbox.y << ")\n";
    cout << "BBox.(width:height) = (" << tldargs.source.bbox.width << ":" << tldargs.source.bbox.height << ")\n";
    break;
  case 'h':
  case 'H':
    PrintHelp();
    break;
  }
}

inline void trackerApp::TLDWorkBegin() { tld_work_begin = getTickCount(); }

inline void trackerApp::TLDWorkEnd()
{
  int64 delta = getTickCount() - tld_work_begin;
  double freq = getTickFrequency();
  tld_work_fps = freq / delta;
}

inline string trackerApp::TLDWorkFps() const
{
  stringstream ss;
  ss << tld_work_fps;
  return ss.str();
}

inline void trackerApp::WorkBegin() { work_begin = getTickCount(); }

inline void trackerApp::WorkEnd()
{
  int64 delta = getTickCount() - work_begin;
  double freq = getTickFrequency();
  work_fps = freq / delta;
}

inline string trackerApp::WorkFPS() const
{
  stringstream ss;
  ss << work_fps;
  return ss.str();
}

void PrintHelp()
{
  cout << "Standalone tracker based on TLD (Tracking, learning, detection) algorithm.\n"
      << "\nUsage: \n"
      << "  (<image>|--video <vide>|--camera <camera_id>) # frames source\n"
      << "  [--write_video <bool>] # write video or not\n"
      << "  [--dst_video <path>] # output video path\n"
      << "  [--dst_video_fps <double>] # output video fps\n";

  help_showed = true;
}

int main(int argc, char** argv)
{
  try
  {
    if (argc < 2)
      PrintHelp();
    trackerArgs tldargs = trackerArgs::ReadArgs(argc, argv);
    if (help_showed)
      return -1;
    trackerApp tldapp(tldargs);
    tldapp.RunTLD();

  }
  catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
  catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
  catch(...) { return cout << "unknown exception" << endl, 1; }
  return 0;
}
