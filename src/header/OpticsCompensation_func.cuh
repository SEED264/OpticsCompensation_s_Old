#ifndef _OPTICSCOMPENSATION_GPU_CUH_
#define _OPTICSCOMPENSATION_GPU_CUH_

#include "AUCUDA.cuh"

/*    �����Y�␳�ɓn���p�����[�^�[�p�\����    */
struct OpticsCompensation_Data{
	//�ω���
	float amount;
	//�œ_����
	float fd;
	//1/�œ_����
	float ofd;
	//�R���X�g���N�^
	OpticsCompensation_Data(float a){
		amount = a / 100.f;
		fd = 450 / tan(0.5 * amount * PI);
		ofd = 1.f / fd;
	}
};

/*    �l��͈͓��Ɋۂ߂�֐�    */
__device__ float clamp(float x, float a, float b);

/*    �؂�グ�p�֐�    */
__host__ int RoundUp(float value, int radix);

/*    �󂯎�����f�[�^��4�Ffloat�ɕ���    */
__global__ void Separate(unsigned long *data, float *rt, float *gt, float *bt, float *at, int w, int h);

/*    �M�^�����Y�␳    */
__global__ void Reverse(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag);
/*    �������^�����Y�␳    */
__global__ void Normal(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag);
/*    ���ۂɏ�������Core�֐�    */
__host__ int OpticsCompensation_Core(lua_State *L);

/*    ���ۂɏ�������Core�֐�(Direct)    */
__host__ int OpticsCompensation_Direct_Core(lua_State *L);
#endif