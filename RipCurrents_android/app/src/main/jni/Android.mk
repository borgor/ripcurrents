LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

ANDROID_SDK_JNI := sdk/native/jni

OPENCV_ANDROID_SDK := ~/OpenCV_android_3_1_0

OPENCV_THIS_DIR := ${OPENCV_ANDROID_SDK}/sdk

OPENCV_INSTALL_MODULES:=on

OPENCV_CAMERA_MODULES:=on

OPENCV_LIB_TYPE:=STATIC

include ${OPENCV_ANDROID_SDK}/${ANDROID_SDK_JNI}/OpenCV.mk

LOCAL_MODULE := myNative

LOCAL_SRC_FILES := nativeCode.cpp ripcurrents.cpp


LOCAL_C_INCLUDES += ${OPENCV_ANDROID_SDK}/${ANDROID_SDK_JNI}/include

#add and compile ffmpeg

LOCAL_STATIC_LIBRARIES += libavformat_static libavcodec_static libavutil_static libswresample_static  libswscale_static

include $(BUILD_SHARED_LIBRARY)

$(call import-add-path,~)
$(call import-module,ffmpeg-3.0.3/android/arm)
