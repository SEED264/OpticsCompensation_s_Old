#include "AUCUDA.cuh"
#include "OpticsCompensation_func.cuh"

/*    Lua����f�[�^���󂯎���Ă��̂܂�Core�֐����Ăяo��    */
int OpticsCompensation(lua_State *L){
	int r = OpticsCompensation_Core(L);
	return r;
	// Lua ���ł̖߂�l�̌���Ԃ�
}

/*    Lua����f�[�^���󂯎���Ă��̂܂�Core�֐�(Direct)���Ăяo��    */
int OpticsCompensation_Direct(lua_State *L){
	int r = OpticsCompensation_Direct_Core(L);
	return r;
	// Lua ���ł̖߂�l�̌���Ԃ�
}

static luaL_Reg OpticsCompensation_s[] = {
	{ "OpticsCompensation", OpticsCompensation },
	{ "OpticsCompensation_Direct", OpticsCompensation_Direct },
	{ NULL, NULL }
};

/*
������dll���`���܂�
�ʂ̂��̂����ꍇ��
OpticsCompensation_s
�̕�����V�������O�ɕς��Ă�������
*/
extern "C"{
	__declspec(dllexport) int luaopen_OpticsCompensation_s(lua_State *L) {
		luaL_register(L, "OpticsCompensation_s", OpticsCompensation_s);
	return 1;
}
}
