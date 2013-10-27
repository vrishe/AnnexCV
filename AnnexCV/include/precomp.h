#pragma once

#ifndef _ANNEX_PRECOMP_H_
#define _ANNEX_PRECOMP_H_


#if defined ANNEX_MACROS_EXTENDED
#	if defined WIN32_LEAN_AND_MEAN
#		include <Windows.h>
#	else
#		include <iostream>
#	endif
#endif

#include <vector>
#include <string>
#include <cmath>

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\features2d\features2d.hpp"


#endif // #ifndef _ANNEX_PRECOMP_H_
