#include "nativeCode.h"
#include "ripcurrents.h"

#include <android/log.h>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#define DEBUG_TAG "gDebug"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, DEBUG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, DEBUG_TAG, __VA_ARGS__)

JNIEXPORT void JNICALL Java_edu_borgorucsc_ripcurrents_MainActivity_callOpenCV(JNIEnv* env, jobject){


   LOGD("Native successfully called");

   cv::VideoCapture cap("/storage/emulated/0/rip_video.mp4");

   cv::VideoWriter video_out("/storage/emulated/0/video_out.avi",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);

   if(!cap.isOpened()){
      LOGE("Error Opening Input Video");
   }
   if(!video_out.isOpened()){
          LOGE("Error Opening Output Video");
   }
   if(!(cap.isOpened()&&video_out.isOpened())){
      LOGE("Error Opening Video");
   }else{

      cv::Mat frame;

      cap >> frame; //this is of type 8UC3

      LOGD("Video resolution is %d   by %d " , frame.cols, frame.rows);
      LOGD("Video format is depth %d and %d channels, in other words type %d", frame.depth(), frame.channels(), frame.type());
      wheel();
      rip_main(cap, video_out);
   }

}