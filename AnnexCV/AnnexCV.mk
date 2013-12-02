
USER_LOCAL_PATH:=$(LOCAL_PATH)
LOCAL_PATH:=$(subst ?,,$(firstword ?$(subst \, ,$(subst /, ,$(call my-dir)))))

ifeq ($(ANNEXCV_LIB_TYPE),)
    ANNEXCV_LIB_TYPE:=STATIC
endif

ifeq ($(ANNEXCV_LIB_TYPE),SHARED)
	ANNEXCV_LIBS_DIR:=lib
	ANNEXCV_LIB_SUFFIX:=so
else
	ifeq ($(ANNEXCV_LIB_TYPE),STATIC)
		ANNEXCV_LIBS_DIR:=staticlib
		ANNEXCV_LIB_SUFFIX:=a
	else
		$(error wrong linkage type specified [STATIC|SHARED])
	endif
endif

LOCAL_C_INCLUDES		+= $(LOCAL_PATH)/AnnexCV/include 
LOCAL_STATIC_LIBRARIES	+= $(LOCAL_PATH)/../$(ANNEXCV_LIB_DIR)/$(TARGET_ARCH_ABI)/libannexcv.$(ANNEXCV_LIB_SUFFIX)

LOCAL_PATH:=USER_LOCAL_PATH