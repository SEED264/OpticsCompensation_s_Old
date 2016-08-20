#include "AUCUDA.cuh"
#include "OpticsCompensation_func.cuh"

/*    Luaからデータを受け取ってそのままCore関数を呼び出す    */
int OpticsCompensation(lua_State *L){
	int r = OpticsCompensation_Core(L);
	return r;
	// Lua 側での戻り値の個数を返す
}

/*    Luaからデータを受け取ってそのままCore関数(Direct)を呼び出す    */
int OpticsCompensation_Direct(lua_State *L){
	int r = OpticsCompensation_Direct_Core(L);
	return r;
	// Lua 側での戻り値の個数を返す
}

static luaL_Reg OpticsCompensation_s[] = {
	{"info", info},
	{ "OpticsCompensation", OpticsCompensation },
	{ "OpticsCompensation_Direct", OpticsCompensation_Direct },
	{ NULL, NULL }
};

/*
ここでdllを定義します
別のものを作る場合は
OpticsCompensation_s
の部分を新しい名前に変えてください
*/
extern "C"{
	__declspec(dllexport) int luaopen_OpticsCompensation_s(lua_State *L) {
		luaL_register(L, "OpticsCompensation_s", OpticsCompensation_s);
	return 1;
}
}
