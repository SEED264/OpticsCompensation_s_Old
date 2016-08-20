#ifndef _AUCUDA_CUH_
#define _AUCUDA_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "windows.h"
#include "lua.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

/*    画像の高さ等のデータ用構造体    */
struct Idata{
	//Width
	int w;
	//Height
	int h;
	//Half Width
	float hw;
	//Half Height
	float hh;
	//コンストラクタ
	Idata(int width, int height){
		w = width;
		h = height;
		hw = w / 2.f;
		hh = h / 2.f;
	}
};

/*    たまに使う円周率    */
#define PI 3.141592653589793

/*    テクスチャリファレンス    */
texture<float, 2, cudaReadModeElementType> rtex;
texture<float, 2, cudaReadModeElementType> gtex;
texture<float, 2, cudaReadModeElementType> btex;
texture<float, 2, cudaReadModeElementType> atex;

#endif