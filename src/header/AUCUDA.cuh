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

/*    �摜�̍������̃f�[�^�p�\����    */
struct Idata{
	//Width
	int w;
	//Height
	int h;
	//Half Width
	float hw;
	//Half Height
	float hh;
	//�R���X�g���N�^
	Idata(int width, int height){
		w = width;
		h = height;
		hw = w / 2.f;
		hh = h / 2.f;
	}
};

/*    ���܂Ɏg���~����    */
#define PI 3.141592653589793

/*    �e�N�X�`�����t�@�����X    */
texture<float, 2, cudaReadModeElementType> rtex;
texture<float, 2, cudaReadModeElementType> gtex;
texture<float, 2, cudaReadModeElementType> btex;
texture<float, 2, cudaReadModeElementType> atex;

#endif