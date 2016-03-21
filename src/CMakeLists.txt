# The MIT License (MIT)
#
# Copyright (c) 2015 Yu Jing <yu@argcv.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
ADD_EXECUTABLE(santander-customer-satisfaction-cc main.cc knn.cc)

ADD_DEPENDENCIES(santander-customer-satisfaction-cc argcv)


# http://www.gnu.org/software/libc/manual/html_node/Feature-Test-Macros.html#index-_005fPOSIX_005fSOURCE 
#ADD_DEFINITIONS( " -D_POSIX_SOURCE ") 
#INCLUDE(ExternalProject) # IF NEED EXTEND PROJECT
#INCLUDE(ExternalLeveldb)
#MESSAGE(STATUS "$$$$$$$$$$ INCLUDE LEVELDB DIR " ${LEVELDB_INCLUDE_DIR})
#INCLUDE_DIRECTORIES(${LEVELDB_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_ir_index_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_string_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_sys_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_wrapper_leveldb_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_wrapper_ini_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_storage_static)
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_wrapper_svm_light_static)

#TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc argcv_wrapper_curl_static)

TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc pthread)

TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc ${CRFPP_LIBRARY_STATIC})
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc ${LEVELDB_LIBRARY_STATIC})
TARGET_LINK_LIBRARIES(santander-customer-satisfaction-cc ${MONGO_CXX_LIBRARY_STATIC})



#  executable properties
SET_TARGET_PROPERTIES(santander-customer-satisfaction-cc PROPERTIES OUTPUT_NAME "santander-customer-satisfaction-cc")




