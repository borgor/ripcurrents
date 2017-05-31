APP_MODULES := myNative

APP_ABI := armeabi-v7a

APP_STL :=  gnustl_static

APP_PLATFORM = android-19

#ensure to use std=c++11 flag, some parts of the code takes advantage of c++11 compiler
APP_CPPFLAGS := -frtti -fexceptions  -std=c++11
