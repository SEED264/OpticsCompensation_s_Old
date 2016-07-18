#include "AUCUDA.cuh"
#include "OpticsCompensation_func.cuh"
#define HPI PI/2.f

/*    �l��͈͓��Ɋۂ߂�֐�    */
__device__ float clamp(float x, float a, float b){
	return min(max(x, a), b);
}

/*    �؂�グ�p�֐�    */
__host__ int RoundUp(float value, int radix) {
	return (value + radix - 1) / radix;
}

/*    �󂯎�����f�[�^��4�Ffloat�ɕ���    */
__global__ void Separate(unsigned long *data, float *rt, float *gt, float *bt, float *at, int w, int h){
	long ox = blockIdx.x;
	long oy = blockIdx.y;
	long ox2 = blockIdx.x;
	long oy2 = blockIdx.y;
	unsigned long datap;
	if (ox<w && oy<h){
		unsigned long i0 = ox + gridDim.x * blockDim.x * oy;
		unsigned long i = ox2 + w * (oy2);
		datap = data[i0];
		rt[i] = (((datap >> 16) & 0xff));
		gt[i] = (((datap >> 8) & 0xff));
		bt[i] = ((datap & 0xff));
		at[i] = (float)((datap >> 24) & 0xff);
	}
}

/*    �M�^�����Y�␳    */
__global__ void Reverse(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	unsigned long r = 0, g = 0, b = 0, a = 0;
	float p1, p2;
	float rt = 0, gt = 0, bt = 0, at = 0;
	float hw = id.hw + pos2.x, hh = id.hh + pos2.y;
	float2 pos = make_float2((ox - hw), (oy - hh));
	float d = sqrt((float)(pos.x*pos.x + pos.y*pos.y))*mag;
	if (d <= 0.001f){
		p1 = (ox + 0.5) / (float)id.w;
		p2 = (oy + 0.5) / (float)id.h;
	}
	else {
		p1 = ((pos.x) / d * od.fd*atan(d*od.ofd) + hw + 0.5) / (float)id.w;
		p2 = ((pos.y) / d * od.fd*atan(d*od.ofd) + hh + 0.5) / (float)id.h;
	}
	rt = tex2D(rtex, p1, p2);
	gt = tex2D(gtex, p1, p2);
	bt = tex2D(btex, p1, p2);
	at = tex2D(atex, p1, p2);

	if (ox < id.w && oy < id.h){
		r = floor((float)(rt + 0.5f));
		g = floor((float)(gt + 0.5f));
		b = floor((float)(bt + 0.5f));
		a = floor((float)(at + 0.5f));
		unsigned long off = (double)ox + id.w * oy;
		data[off] = (r << 16) | (g << 8) | b | (a << 24);
	}
}

/*    �������^�����Y�␳    */
__global__ void Normal(unsigned long *data, Idata id, OpticsCompensation_Data od, float2 pos2, float mag){
	float ox = (threadIdx.x + blockIdx.x * blockDim.x);
	float oy = (threadIdx.y + blockIdx.y * blockDim.y);
	unsigned long r = 0, g = 0, b = 0, a = 0;
	float p1, p2;
	float rt = 0, gt = 0, bt = 0, at = 0;
	float hw = id.hw + pos2.x, hh = id.hh + pos2.y;
	float2 pos = make_float2(ox - hw, oy - hh);
	float d = sqrt((float)(pos.x*pos.x + pos.y*pos.y))*mag;
	float hpi = PI*0.5;
	if (d <= 0.001f){
		p1 = (ox + 0.5) / (float)id.w;
		p2 = (oy + 0.5) / (float)id.h;
	}
	else {
		p1 = ((pos.x) / d * od.fd*tan(clamp(d*od.ofd, -hpi, hpi)) + hw + 0.5) / (float)id.w;
		p2 = ((pos.y) / d * od.fd*tan(clamp(d*od.ofd, -hpi, hpi)) + hh + 0.5) / (float)id.h;

	}
	rt = tex2D(rtex, p1, p2);
	gt = tex2D(gtex, p1, p2);
	bt = tex2D(btex, p1, p2);
	at = tex2D(atex, p1, p2);

	if (od.amount == 1.f) rt = gt = bt = at = 0;

	if (ox < id.w && oy < id.h){
		r = floor((float)(rt + 0.5f));
		g = floor((float)(gt + 0.5f));
		b = floor((float)(bt + 0.5f));
		a = floor((float)(at + 0.5f));
		unsigned long off = (double)ox + id.w * oy;
		data[off] = (r << 16) | (g << 8) | b | (a << 24);
	}
}

/*    ���ۂɏ�������Core�֐�    */
__host__ int OpticsCompensation_Core(lua_State *L){
	// �摜�f�[�^�A���A�������̃p�����[�^���擾
	unsigned long *data = (unsigned long*)lua_touserdata(L, 1);
	int w = (int)lua_tonumber(L, 2);
	int h = (int)lua_tonumber(L, 3);
	float l = (float)lua_tonumber(L, 4);
	int argnum = lua_gettop(L);
	float x = (argnum >= 5) ? (float)lua_tonumber(L, 5) : 0;
	float y = (argnum >= 6) ? (float)lua_tonumber(L, 6) : 0;

	float hdd = sqrt((float)1280 * 1280 + 720 * 720);
	float imd = sqrt((float)w*w + h*h);
	float mag = hdd / imd;
	float2 pos = make_float2(x, y);
	int nw = w, nh = h;

	Idata id(nw, nh);
	OpticsCompensation_Data od(l);
	OpticsCompensation_Data rod(-l);

	if (l != 0){
		dim3 blockn(RoundUp(nw, 32), RoundUp(nh, 32));
		dim3 thread(32, 32);
		dim3 tm(w, h);
		unsigned long *datat, *data2;
		float *rt, *gt, *bt, *at;
		cudaArray *r_array, *g_array, *b_array, *a_array;
		cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
		cudaMalloc((void**)&datat, sizeof(unsigned long)*nw*nh);
		cudaMalloc((void**)&data2, sizeof(unsigned long)*w*h);
		cudaMalloc((void**)&rt, sizeof(float)*w*h);
		cudaMalloc((void**)&gt, sizeof(float)*w*h);
		cudaMalloc((void**)&bt, sizeof(float)*w*h);
		cudaMalloc((void**)&at, sizeof(float)*w*h);
		cudaMallocArray(&r_array, &cd, w, h);
		cudaMallocArray(&g_array, &cd, w, h);
		cudaMallocArray(&b_array, &cd, w, h);
		cudaMallocArray(&a_array, &cd, w, h);
		cudaTextureAddressMode add = cudaAddressModeMirror;
		cudaTextureAddressMode aadd = cudaAddressModeBorder;
		cudaTextureFilterMode fil = cudaFilterModeLinear;
		int nor = 1;
		rtex.addressMode[0] = add;
		rtex.addressMode[1] = add;
		rtex.filterMode = fil;
		rtex.normalized = nor;
		gtex.addressMode[0] = add;
		gtex.addressMode[1] = add;
		gtex.filterMode = fil;
		gtex.normalized = nor;
		btex.addressMode[0] = add;
		btex.addressMode[1] = add;
		btex.filterMode = fil;
		btex.normalized = nor;
		atex.addressMode[0] = aadd;
		atex.addressMode[1] = aadd;
		atex.filterMode = fil;
		atex.normalized = nor;
		unsigned long *datan = (unsigned long*)lua_newuserdata(L, sizeof(unsigned long)*nw*nh);
		cudaMemcpy(data2, data, sizeof(unsigned long)*w*h, cudaMemcpyHostToDevice);

		Separate <<< tm, 1 >>> (data2, rt, gt, bt, at, w, h);

		cudaMemcpyToArray(r_array, 0, 0, rt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g_array, 0, 0, gt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b_array, 0, 0, bt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a_array, 0, 0, at, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(rtex, r_array, cd);
		cudaBindTextureToArray(gtex, g_array, cd);
		cudaBindTextureToArray(btex, b_array, cd);
		cudaBindTextureToArray(atex, a_array, cd);
		if (l>0.f){
			Normal <<< blockn, thread >>> (datat, id, od, pos, mag);
		}
		else {
			Reverse <<< blockn, thread >>> (datat, id, rod, pos, mag);
		}
		cudaMemcpy(datan, datat, sizeof(unsigned long)*nw*nh, cudaMemcpyDeviceToHost);

		cudaUnbindTexture(rtex);
		cudaUnbindTexture(gtex);
		cudaUnbindTexture(btex);
		cudaUnbindTexture(atex);
		cudaFreeArray(r_array);
		cudaFreeArray(g_array);
		cudaFreeArray(b_array);
		cudaFreeArray(a_array);
		cudaFree(datat);
		cudaFree(data2);
		cudaFree(rt);
		cudaFree(gt);
		cudaFree(bt);
		cudaFree(at);
	}
	lua_pushinteger(L, nw);
	lua_pushinteger(L, nh);
	return 3;
	// Lua ���ł̖߂�l�̌���Ԃ�
}

/*    ���ۂɏ�������Core�֐�(Direct)    */
__host__ int OpticsCompensation_Direct_Core(lua_State *L){
	// �摜�f�[�^�A���A�������̃p�����[�^���擾
	float l = (float)lua_tonumber(L, 1);
	int argnum = lua_gettop(L);
	float x = (argnum >= 2) ? (float)lua_tonumber(L, 2) : 0;
	float y = (argnum >= 3) ? (float)lua_tonumber(L, 3) : 0;

	/*    AviUtl��obj.getpixeldata()�Ăяo��    */
	lua_getglobal(L, "obj");
	lua_getfield(L, -1, "getpixeldata");
	lua_call(L, 0, 3);
	int h = lua_tointeger(L, -1);
	lua_pop(L, 1);
	int w = lua_tointeger(L, -1);
	lua_pop(L, 1);
	unsigned long *data = (unsigned long*)lua_touserdata(L, -1);
	lua_pop(L, 1);

	float hdd = sqrt((float)1280 * 1280 + 720 * 720);
	float imd = sqrt((float)w*w + h*h);
	float mag = hdd / imd;
	float2 pos = make_float2(x, y);
	int nw = w, nh = h;

	Idata id(nw, nh);
	OpticsCompensation_Data od(l);
	OpticsCompensation_Data rod(-l);

	if (l != 0){
		dim3 blockn(RoundUp(nw, 32), RoundUp(nh, 32));
		dim3 thread(32, 32);
		dim3 tm(w, h);
		unsigned long *datat, *data2;
		float *rt, *gt, *bt, *at;
		cudaArray *r_array, *g_array, *b_array, *a_array;
		cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
		cudaMalloc((void**)&datat, sizeof(unsigned long)*nw*nh);
		cudaMalloc((void**)&data2, sizeof(unsigned long)*w*h);
		cudaMalloc((void**)&rt, sizeof(float)*w*h);
		cudaMalloc((void**)&gt, sizeof(float)*w*h);
		cudaMalloc((void**)&bt, sizeof(float)*w*h);
		cudaMalloc((void**)&at, sizeof(float)*w*h);
		cudaMallocArray(&r_array, &cd, w, h);
		cudaMallocArray(&g_array, &cd, w, h);
		cudaMallocArray(&b_array, &cd, w, h);
		cudaMallocArray(&a_array, &cd, w, h);
		cudaTextureAddressMode add = cudaAddressModeMirror;
		cudaTextureAddressMode aadd = cudaAddressModeBorder;
		cudaTextureFilterMode fil = cudaFilterModeLinear;
		int nor = 1;
		rtex.addressMode[0] = add;
		rtex.addressMode[1] = add;
		rtex.filterMode = fil;
		rtex.normalized = nor;
		gtex.addressMode[0] = add;
		gtex.addressMode[1] = add;
		gtex.filterMode = fil;
		gtex.normalized = nor;
		btex.addressMode[0] = add;
		btex.addressMode[1] = add;
		btex.filterMode = fil;
		btex.normalized = nor;
		atex.addressMode[0] = aadd;
		atex.addressMode[1] = aadd;
		atex.filterMode = fil;
		atex.normalized = nor;
		unsigned long *datan = (unsigned long*)lua_newuserdata(L, sizeof(unsigned long)*nw*nh);

		cudaMemcpy(data2, data, sizeof(unsigned long)*w*h, cudaMemcpyHostToDevice);

		Separate <<< tm, 1 >>> (data2, rt, gt, bt, at, w, h);

		cudaMemcpyToArray(r_array, 0, 0, rt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(g_array, 0, 0, gt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(b_array, 0, 0, bt, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray(a_array, 0, 0, at, sizeof(float)*w*h, cudaMemcpyDeviceToDevice);

		cudaBindTextureToArray(rtex, r_array, cd);
		cudaBindTextureToArray(gtex, g_array, cd);
		cudaBindTextureToArray(btex, b_array, cd);
		cudaBindTextureToArray(atex, a_array, cd);
		if (l>0.f){
			Normal <<< blockn, thread >>> (datat, id, od, pos, mag);
		}
		else {
			Reverse <<< blockn, thread >>> (datat, id, rod, pos, mag);
		}

		cudaMemcpy(datan, datat, sizeof(unsigned long)*nw*nh, cudaMemcpyDeviceToHost);

		/*    AviUtl��obj.putpixeldata()�Ăяo��    */
		lua_getglobal(L, "obj");
		lua_getfield(L, -1, "putpixeldata");
		lua_pushlightuserdata(L, datan);
		lua_call(L, 1, 0);

		cudaUnbindTexture(rtex);
		cudaUnbindTexture(gtex);
		cudaUnbindTexture(btex);
		cudaUnbindTexture(atex);
		cudaFreeArray(r_array);
		cudaFreeArray(g_array);
		cudaFreeArray(b_array);
		cudaFreeArray(a_array);
		cudaFree(datat);
		cudaFree(data2);
		cudaFree(rt);
		cudaFree(gt);
		cudaFree(bt);
		cudaFree(at);
	}
	return 0;
	// Lua ���ł̖߂�l�̌���Ԃ�
}