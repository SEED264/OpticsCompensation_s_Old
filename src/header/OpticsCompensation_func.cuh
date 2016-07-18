#ifndef _OPTICSCOMPENSATION_GPU_CUH_
#define _OPTICSCOMPENSATION_GPU_CUH_

#include "AUCUDA.cuh"

/*    レンズ補正に渡すパラメーター用構造体    */
struct OpticsCompensation_Data{
	//変化量
	float amount;
	//焦点距離
	float fd;
	//1/焦点距離
	float ofd;
	//コンストラクタ
	OpticsCompensation_Data(float a){
		amount = a / 100.f;
		fd = 450 / tan(0.5 * amount * PI);
		ofd = 1.f / fd;
	}
};

/*    値を範囲内に丸める関数    */
__device__ float clamp(float x, float a, float b);

/*    切り上げ用関数    */
__host__ int RoundUp(float value, int radix);

/*    受け取ったデータを4色floatに分解    */
__global__ void Separate(unsigned long *data, float *rt, float *gt, float *bt, float *at, int w, int h);

/*    樽型レンズ補正    */
__global__ void Reverse(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag);
/*    糸巻き型レンズ補正    */
__global__ void Normal(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag);
/*    実際に処理するCore関数    */
__host__ int OpticsCompensation_Core(lua_State *L);

/*    実際に処理するCore関数(Direct)    */
__host__ int OpticsCompensation_Direct_Core(lua_State *L);
#endif