/* Initialization */
#include "Radiator_model.h"
#include "Radiator_11mix.h"
#include "Radiator_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void Radiator_functionInitialEquations_0(DATA *data, threadData_t *threadData);

/*
equation index: 1
type: SIMPLE_ASSIGN
Energy = $START.Energy
*/
void Radiator_eqFunction_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1};
  data->localData[0]->realVars[0] /* Energy STATE(1,Power) */ = data->modelData->realVarsData[0].attribute /* Energy STATE(1,Power) */.start;
  TRACE_POP
}

/*
equation index: 2
type: SIMPLE_ASSIGN
flow_sink.ports[2].m_flow = 0.0
*/
void Radiator_eqFunction_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2};
  data->localData[0]->realVars[144] /* flow_sink.ports[2].m_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 3
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.X[1] = 1.0
*/
void Radiator_eqFunction_3(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3};
  data->localData[0]->realVars[100] /* Radiator.vol[1].dynBal.medium.X[1] variable */ = 1.0;
  TRACE_POP
}

/*
equation index: 4
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.mWat_flow_internal = 0.0
*/
void Radiator_eqFunction_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4};
  data->localData[0]->realVars[70] /* Radiator.vol[1].dynBal.mWat_flow_internal variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 5
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.X[1] = 1.0
*/
void Radiator_eqFunction_5(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5};
  data->localData[0]->realVars[101] /* Radiator.vol[2].dynBal.medium.X[1] variable */ = 1.0;
  TRACE_POP
}

/*
equation index: 6
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.mWat_flow_internal = 0.0
*/
void Radiator_eqFunction_6(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6};
  data->localData[0]->realVars[71] /* Radiator.vol[2].dynBal.mWat_flow_internal variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 7
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.X[1] = 1.0
*/
void Radiator_eqFunction_7(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7};
  data->localData[0]->realVars[102] /* Radiator.vol[3].dynBal.medium.X[1] variable */ = 1.0;
  TRACE_POP
}

/*
equation index: 8
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.mWat_flow_internal = 0.0
*/
void Radiator_eqFunction_8(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,8};
  data->localData[0]->realVars[72] /* Radiator.vol[3].dynBal.mWat_flow_internal variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 9
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.X[1] = 1.0
*/
void Radiator_eqFunction_9(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,9};
  data->localData[0]->realVars[103] /* Radiator.vol[4].dynBal.medium.X[1] variable */ = 1.0;
  TRACE_POP
}

/*
equation index: 10
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.mWat_flow_internal = 0.0
*/
void Radiator_eqFunction_10(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,10};
  data->localData[0]->realVars[73] /* Radiator.vol[4].dynBal.mWat_flow_internal variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 11
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.X[1] = 1.0
*/
void Radiator_eqFunction_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,11};
  data->localData[0]->realVars[104] /* Radiator.vol[5].dynBal.medium.X[1] variable */ = 1.0;
  TRACE_POP
}

/*
equation index: 12
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.mWat_flow_internal = 0.0
*/
void Radiator_eqFunction_12(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,12};
  data->localData[0]->realVars[74] /* Radiator.vol[5].dynBal.mWat_flow_internal variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 13
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.R_s = 0.0
*/
void Radiator_eqFunction_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13};
  data->localData[0]->realVars[85] /* Radiator.vol[1].dynBal.medium.R_s variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 14
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.MM = 0.018015268
*/
void Radiator_eqFunction_14(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14};
  data->localData[0]->realVars[80] /* Radiator.vol[1].dynBal.medium.MM variable */ = 0.018015268;
  TRACE_POP
}

/*
equation index: 15
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.R_s = 0.0
*/
void Radiator_eqFunction_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,15};
  data->localData[0]->realVars[86] /* Radiator.vol[2].dynBal.medium.R_s variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 16
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.MM = 0.018015268
*/
void Radiator_eqFunction_16(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,16};
  data->localData[0]->realVars[81] /* Radiator.vol[2].dynBal.medium.MM variable */ = 0.018015268;
  TRACE_POP
}

/*
equation index: 17
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.R_s = 0.0
*/
void Radiator_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  data->localData[0]->realVars[87] /* Radiator.vol[3].dynBal.medium.R_s variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 18
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.MM = 0.018015268
*/
void Radiator_eqFunction_18(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,18};
  data->localData[0]->realVars[82] /* Radiator.vol[3].dynBal.medium.MM variable */ = 0.018015268;
  TRACE_POP
}

/*
equation index: 19
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.R_s = 0.0
*/
void Radiator_eqFunction_19(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,19};
  data->localData[0]->realVars[88] /* Radiator.vol[4].dynBal.medium.R_s variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 20
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.MM = 0.018015268
*/
void Radiator_eqFunction_20(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,20};
  data->localData[0]->realVars[83] /* Radiator.vol[4].dynBal.medium.MM variable */ = 0.018015268;
  TRACE_POP
}

/*
equation index: 21
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.R_s = 0.0
*/
void Radiator_eqFunction_21(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,21};
  data->localData[0]->realVars[89] /* Radiator.vol[5].dynBal.medium.R_s variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 22
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.MM = 0.018015268
*/
void Radiator_eqFunction_22(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,22};
  data->localData[0]->realVars[84] /* Radiator.vol[5].dynBal.medium.MM variable */ = 0.018015268;
  TRACE_POP
}

/*
equation index: 23
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.mb_flow = 0.0
*/
void Radiator_eqFunction_23(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,23};
  data->localData[0]->realVars[75] /* Radiator.vol[1].dynBal.mb_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 24
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.mb_flow = 0.0
*/
void Radiator_eqFunction_24(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,24};
  data->localData[0]->realVars[76] /* Radiator.vol[2].dynBal.mb_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 25
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.mb_flow = 0.0
*/
void Radiator_eqFunction_25(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,25};
  data->localData[0]->realVars[77] /* Radiator.vol[3].dynBal.mb_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 26
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.mb_flow = 0.0
*/
void Radiator_eqFunction_26(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,26};
  data->localData[0]->realVars[78] /* Radiator.vol[4].dynBal.mb_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 27
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.mb_flow = 0.0
*/
void Radiator_eqFunction_27(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,27};
  data->localData[0]->realVars[79] /* Radiator.vol[5].dynBal.mb_flow variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 28
type: SIMPLE_ASSIGN
Radiator.dp = 0.0
*/
void Radiator_eqFunction_28(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,28};
  data->localData[0]->realVars[40] /* Radiator.dp variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 29
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.p_bar = 1e-05 * flow_sink.p
*/
void Radiator_eqFunction_29(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,29};
  data->localData[0]->realVars[114] /* Radiator.vol[5].dynBal.medium.p_bar variable */ = (1e-05) * (data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */);
  TRACE_POP
}

/*
equation index: 30
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.p_bar = 1e-05 * flow_sink.p
*/
void Radiator_eqFunction_30(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,30};
  data->localData[0]->realVars[113] /* Radiator.vol[4].dynBal.medium.p_bar variable */ = (1e-05) * (data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */);
  TRACE_POP
}

/*
equation index: 31
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.p_bar = 1e-05 * flow_sink.p
*/
void Radiator_eqFunction_31(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,31};
  data->localData[0]->realVars[112] /* Radiator.vol[3].dynBal.medium.p_bar variable */ = (1e-05) * (data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */);
  TRACE_POP
}

/*
equation index: 32
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.p_bar = 1e-05 * flow_sink.p
*/
void Radiator_eqFunction_32(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,32};
  data->localData[0]->realVars[111] /* Radiator.vol[2].dynBal.medium.p_bar variable */ = (1e-05) * (data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */);
  TRACE_POP
}

/*
equation index: 33
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.p_bar = 1e-05 * flow_sink.p
*/
void Radiator_eqFunction_33(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,33};
  data->localData[0]->realVars[110] /* Radiator.vol[1].dynBal.medium.p_bar variable */ = (1e-05) * (data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */);
  TRACE_POP
}

/*
equation index: 34
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.m = 995.586 * Radiator.vol[5].dynBal.fluidVolume
*/
void Radiator_eqFunction_34(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,34};
  data->localData[0]->realVars[69] /* Radiator.vol[5].dynBal.m DUMMY_STATE */ = (995.586) * (data->simulationInfo->realParameter[136] /* Radiator.vol[5].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 35
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.m = 995.586 * Radiator.vol[4].dynBal.fluidVolume
*/
void Radiator_eqFunction_35(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,35};
  data->localData[0]->realVars[68] /* Radiator.vol[4].dynBal.m DUMMY_STATE */ = (995.586) * (data->simulationInfo->realParameter[135] /* Radiator.vol[4].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 36
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.m = 995.586 * Radiator.vol[3].dynBal.fluidVolume
*/
void Radiator_eqFunction_36(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,36};
  data->localData[0]->realVars[67] /* Radiator.vol[3].dynBal.m DUMMY_STATE */ = (995.586) * (data->simulationInfo->realParameter[134] /* Radiator.vol[3].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 37
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.m = 995.586 * Radiator.vol[2].dynBal.fluidVolume
*/
void Radiator_eqFunction_37(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,37};
  data->localData[0]->realVars[66] /* Radiator.vol[2].dynBal.m DUMMY_STATE */ = (995.586) * (data->simulationInfo->realParameter[133] /* Radiator.vol[2].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 38
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.m = 995.586 * Radiator.vol[1].dynBal.fluidVolume
*/
void Radiator_eqFunction_38(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,38};
  data->localData[0]->realVars[65] /* Radiator.vol[1].dynBal.m DUMMY_STATE */ = (995.586) * (data->simulationInfo->realParameter[132] /* Radiator.vol[1].dynBal.fluidVolume PARAM */);
  TRACE_POP
}
extern void Radiator_eqFunction_325(DATA *data, threadData_t *threadData);


/*
equation index: 40
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[1].m_flow = Radiator.vol[2].ports[1].m_flow
*/
void Radiator_eqFunction_40(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,40};
  data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */ = data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */;
  TRACE_POP
}

/*
equation index: 41
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[1].m_flow = Radiator.vol[3].ports[1].m_flow
*/
void Radiator_eqFunction_41(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,41};
  data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */ = data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */;
  TRACE_POP
}

/*
equation index: 42
type: SIMPLE_ASSIGN
Radiator.vol[5].ports[1].m_flow = Radiator.vol[4].ports[1].m_flow
*/
void Radiator_eqFunction_42(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,42};
  data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */ = data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */;
  TRACE_POP
}

/*
equation index: 43
type: SIMPLE_ASSIGN
flow_sink.ports[1].m_flow = Radiator.vol[5].ports[1].m_flow
*/
void Radiator_eqFunction_43(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,43};
  data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */ = data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */;
  TRACE_POP
}
extern void Radiator_eqFunction_382(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_388(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_326(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_327(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_328(DATA *data, threadData_t *threadData);


/*
equation index: 49
type: SIMPLE_ASSIGN
flow_source.ports[1].h_outflow = Radiator.flow_source.Medium.specificEnthalpy(Radiator.flow_source.Medium.setState_pTX(flow_sink.p, flow_source.T_in, flow_source.X_in_internal))
*/
void Radiator_eqFunction_49(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,49};
  real_array tmp0;
  real_array_create(&tmp0, ((modelica_real*)&((&data->localData[0]->realVars[146] /* flow_source.X_in_internal[1] variable */)[calc_base_index_dims_subs(1, (_index_t)1, ((modelica_integer) 1))])), 1, (_index_t)1);
  data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */ = omc_Radiator_flow__source_Medium_specificEnthalpy(threadData, omc_Radiator_flow__source_Medium_setState__pTX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[145] /* flow_source.T_in variable */, tmp0));
  TRACE_POP
}

void Radiator_eqFunction_50(DATA*, threadData_t*);
void Radiator_eqFunction_51(DATA*, threadData_t*);
void Radiator_eqFunction_52(DATA*, threadData_t*);
void Radiator_eqFunction_53(DATA*, threadData_t*);
void Radiator_eqFunction_54(DATA*, threadData_t*);
void Radiator_eqFunction_55(DATA*, threadData_t*);
void Radiator_eqFunction_56(DATA*, threadData_t*);
void Radiator_eqFunction_60(DATA*, threadData_t*);
void Radiator_eqFunction_59(DATA*, threadData_t*);
void Radiator_eqFunction_58(DATA*, threadData_t*);
void Radiator_eqFunction_57(DATA*, threadData_t*);
/*
equation index: 61
indexNonlinear: 0
type: NONLINEAR

vars: {Radiator.QEle_flow_nominal[4], Radiator.TWat_nominal[3], Radiator.TWat_nominal[1], Radiator.UAEle}
eqns: {50, 51, 52, 53, 54, 55, 56, 60, 59, 58, 57}
*/
void Radiator_eqFunction_61(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,61};
  int retValue;
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving nonlinear system 61 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  /* get old value */
  data->simulationInfo->nonlinearSystemData[0].nlsxOld[0] = data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */;
  data->simulationInfo->nonlinearSystemData[0].nlsxOld[1] = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
  data->simulationInfo->nonlinearSystemData[0].nlsxOld[2] = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
  data->simulationInfo->nonlinearSystemData[0].nlsxOld[3] = data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */;
  retValue = solve_nonlinear_system(data, threadData, 0);
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,61};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving non-linear system 61 failed at time=%.15g.\nFor more information please use -lv LOG_NLS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */ = data->simulationInfo->nonlinearSystemData[0].nlsx[0];
  data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ = data->simulationInfo->nonlinearSystemData[0].nlsx[1];
  data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ = data->simulationInfo->nonlinearSystemData[0].nlsx[2];
  data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */ = data->simulationInfo->nonlinearSystemData[0].nlsx[3];
  TRACE_POP
}

/*
equation index: 62
type: SIMPLE_ASSIGN
Radiator.dTRad_nominal[1] = Radiator.TWat_nominal[1] - Radiator.TRad_nominal
*/
void Radiator_eqFunction_62(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,62};
  data->simulationInfo->realParameter[28] /* Radiator.dTRad_nominal[1] PARAM */ = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 63
type: SIMPLE_ASSIGN
Radiator.dTCon_nominal[1] = -293.15 + Radiator.TWat_nominal[1]
*/
void Radiator_eqFunction_63(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,63};
  data->simulationInfo->realParameter[23] /* Radiator.dTCon_nominal[1] PARAM */ = -293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
  TRACE_POP
}

/*
equation index: 64
type: SIMPLE_ASSIGN
Radiator.dTRad_nominal[2] = Radiator.TWat_nominal[2] - Radiator.TRad_nominal
*/
void Radiator_eqFunction_64(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,64};
  data->simulationInfo->realParameter[29] /* Radiator.dTRad_nominal[2] PARAM */ = data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 65
type: SIMPLE_ASSIGN
Radiator.dTCon_nominal[2] = -293.15 + Radiator.TWat_nominal[2]
*/
void Radiator_eqFunction_65(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,65};
  data->simulationInfo->realParameter[24] /* Radiator.dTCon_nominal[2] PARAM */ = -293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */;
  TRACE_POP
}

/*
equation index: 66
type: SIMPLE_ASSIGN
Radiator.dTRad_nominal[5] = Radiator.TWat_nominal[5] - Radiator.TRad_nominal
*/
void Radiator_eqFunction_66(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,66};
  data->simulationInfo->realParameter[32] /* Radiator.dTRad_nominal[5] PARAM */ = data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 67
type: SIMPLE_ASSIGN
Radiator.dTCon_nominal[5] = -293.15 + Radiator.TWat_nominal[5]
*/
void Radiator_eqFunction_67(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,67};
  data->simulationInfo->realParameter[27] /* Radiator.dTCon_nominal[5] PARAM */ = -293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */;
  TRACE_POP
}

/*
equation index: 68
type: SIMPLE_ASSIGN
Radiator.dTRad_nominal[4] = Radiator.TWat_nominal[4] - Radiator.TRad_nominal
*/
void Radiator_eqFunction_68(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,68};
  data->simulationInfo->realParameter[31] /* Radiator.dTRad_nominal[4] PARAM */ = data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 69
type: SIMPLE_ASSIGN
Radiator.dTCon_nominal[4] = -293.15 + Radiator.TWat_nominal[4]
*/
void Radiator_eqFunction_69(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,69};
  data->simulationInfo->realParameter[26] /* Radiator.dTCon_nominal[4] PARAM */ = -293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */;
  TRACE_POP
}

/*
equation index: 70
type: SIMPLE_ASSIGN
Radiator.dTRad_nominal[3] = Radiator.TWat_nominal[3] - Radiator.TRad_nominal
*/
void Radiator_eqFunction_70(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,70};
  data->simulationInfo->realParameter[30] /* Radiator.dTRad_nominal[3] PARAM */ = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 71
type: SIMPLE_ASSIGN
Radiator.dTCon_nominal[3] = -293.15 + Radiator.TWat_nominal[3]
*/
void Radiator_eqFunction_71(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,71};
  data->simulationInfo->realParameter[25] /* Radiator.dTCon_nominal[3] PARAM */ = -293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
  TRACE_POP
}

/*
equation index: 72
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.T = 293.15
*/
void Radiator_eqFunction_72(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,72};
  data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */ = 293.15;
  TRACE_POP
}

/*
equation index: 73
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.T_degC = -273.15 + Radiator.vol[5].dynBal.medium.T
*/
void Radiator_eqFunction_73(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,73};
  data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */ = -273.15 + data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */;
  TRACE_POP
}

/*
equation index: 74
type: SIMPLE_ASSIGN
Radiator.vol[5].ports[2].h_outflow = 4184.0 * Radiator.vol[5].dynBal.medium.T_degC
*/
void Radiator_eqFunction_74(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,74};
  data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */ = (4184.0) * (data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_372(DATA *data, threadData_t *threadData);


/*
equation index: 76
type: SIMPLE_ASSIGN
flow_sink.ports[2].h_outflow = Radiator.flow_sink.Medium.specificEnthalpy(Radiator.flow_sink.Medium.setState_pTX(flow_sink.p_in_internal, Radiator.vol[5].T, flow_sink.X_in_internal))
*/
void Radiator_eqFunction_76(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,76};
  real_array tmp0;
  real_array_create(&tmp0, ((modelica_real*)&((&data->localData[0]->realVars[140] /* flow_sink.X_in_internal[1] variable */)[calc_base_index_dims_subs(1, (_index_t)1, ((modelica_integer) 1))])), 1, (_index_t)1);
  data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */ = omc_Radiator_flow__sink_Medium_specificEnthalpy(threadData, omc_Radiator_flow__sink_Medium_setState__pTX(threadData, data->localData[0]->realVars[141] /* flow_sink.p_in_internal variable */, data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */, tmp0));
  TRACE_POP
}
extern void Radiator_eqFunction_373(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_389(DATA *data, threadData_t *threadData);


/*
equation index: 79
type: SIMPLE_ASSIGN
Radiator.preRad[5].Q_flow = homotopy(Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[5] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[5], Radiator.n - 1.0, 0.05), Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[5]) ^ (Radiator.n - 1.0) * Radiator.dTRad[5])
*/
void Radiator_eqFunction_79(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,79};
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  tmp1 = fabs(data->simulationInfo->realParameter[32] /* Radiator.dTRad_nominal[5] PARAM */);
  tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp1 < 0.0 && tmp2 != 0.0)
  {
    tmp4 = modf(tmp2, &tmp5);
    
    if(tmp4 > 0.5)
    {
      tmp4 -= 1.0;
      tmp5 += 1.0;
    }
    else if(tmp4 < -0.5)
    {
      tmp4 += 1.0;
      tmp5 -= 1.0;
    }
    
    if(fabs(tmp4) < 1e-10)
      tmp3 = pow(tmp1, tmp5);
    else
    {
      tmp7 = modf(1.0/tmp2, &tmp6);
      if(tmp7 > 0.5)
      {
        tmp7 -= 1.0;
        tmp6 += 1.0;
      }
      else if(tmp7 < -0.5)
      {
        tmp7 += 1.0;
        tmp6 -= 1.0;
      }
      if(fabs(tmp7) < 1e-10 && ((unsigned long)tmp6 & 1))
      {
        tmp3 = -pow(-tmp1, tmp4)*pow(tmp1, tmp5);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1, tmp2);
      }
    }
  }
  else
  {
    tmp3 = pow(tmp1, tmp2);
  }
  if(isnan(tmp3) || isinf(tmp3))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp1, tmp2);
  }
  data->localData[0]->realVars[51] /* Radiator.preRad[5].Q_flow variable */ = homotopy((((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp3)) * (data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_396(DATA *data, threadData_t *threadData);


/*
equation index: 81
type: SIMPLE_ASSIGN
Radiator.preCon[5].Q_flow = homotopy((1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTCon[5] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTCon[5], Radiator.n - 1.0, 0.05), (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[5]) ^ (Radiator.n - 1.0) * Radiator.dTCon[5])
*/
void Radiator_eqFunction_81(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,81};
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  tmp8 = fabs(data->simulationInfo->realParameter[27] /* Radiator.dTCon_nominal[5] PARAM */);
  tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp8 < 0.0 && tmp9 != 0.0)
  {
    tmp11 = modf(tmp9, &tmp12);
    
    if(tmp11 > 0.5)
    {
      tmp11 -= 1.0;
      tmp12 += 1.0;
    }
    else if(tmp11 < -0.5)
    {
      tmp11 += 1.0;
      tmp12 -= 1.0;
    }
    
    if(fabs(tmp11) < 1e-10)
      tmp10 = pow(tmp8, tmp12);
    else
    {
      tmp14 = modf(1.0/tmp9, &tmp13);
      if(tmp14 > 0.5)
      {
        tmp14 -= 1.0;
        tmp13 += 1.0;
      }
      else if(tmp14 < -0.5)
      {
        tmp14 += 1.0;
        tmp13 -= 1.0;
      }
      if(fabs(tmp14) < 1e-10 && ((unsigned long)tmp13 & 1))
      {
        tmp10 = -pow(-tmp8, tmp11)*pow(tmp8, tmp12);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp8, tmp9);
      }
    }
  }
  else
  {
    tmp10 = pow(tmp8, tmp9);
  }
  if(isnan(tmp10) || isinf(tmp10))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp8, tmp9);
  }
  data->localData[0]->realVars[46] /* Radiator.preCon[5].Q_flow variable */ = homotopy((((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[34] /* Radiator.dTCon[5] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[34] /* Radiator.dTCon[5] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp10)) * (data->localData[0]->realVars[34] /* Radiator.dTCon[5] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_394(DATA *data, threadData_t *threadData);


/*
equation index: 83
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports_H_flow[2] = semiLinear(-flow_sink.ports[1].m_flow, flow_sink.ports[2].h_outflow, Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_83(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,83};
  data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */ = semiLinear((-data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */), data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */);
  TRACE_POP
}

/*
equation index: 84
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.U = Radiator.vol[5].dynBal.m * Radiator.vol[5].ports[2].h_outflow + Radiator.vol[5].dynBal.CSen * Radiator.vol[5].dynBal.medium.T_degC
*/
void Radiator_eqFunction_84(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,84};
  data->localData[0]->realVars[5] /* Radiator.vol[5].dynBal.U STATE(1) */ = (data->localData[0]->realVars[69] /* Radiator.vol[5].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */) + (data->simulationInfo->realParameter[116] /* Radiator.vol[5].dynBal.CSen PARAM */) * (data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */);
  TRACE_POP
}

/*
equation index: 85
type: SIMPLE_ASSIGN
Radiator.sta_b.T = 273.15 + 0.0002390057361376673 * (if noEvent((-flow_sink.ports[1].m_flow) > 0.0) then flow_sink.ports[2].h_outflow else Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_85(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,85};
  modelica_boolean tmp15;
  tmp15 = Greater((-data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */),0.0);
  data->localData[0]->realVars[54] /* Radiator.sta_b.T variable */ = 273.15 + (0.0002390057361376673) * ((tmp15?data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */:data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */));
  TRACE_POP
}

/*
equation index: 86
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.T = 293.15
*/
void Radiator_eqFunction_86(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,86};
  data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */ = 293.15;
  TRACE_POP
}

/*
equation index: 87
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.T_degC = -273.15 + Radiator.vol[4].dynBal.medium.T
*/
void Radiator_eqFunction_87(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,87};
  data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */ = -273.15 + data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */;
  TRACE_POP
}

/*
equation index: 88
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[2].h_outflow = 4184.0 * Radiator.vol[4].dynBal.medium.T_degC
*/
void Radiator_eqFunction_88(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,88};
  data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */ = (4184.0) * (data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_361(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_397(DATA *data, threadData_t *threadData);


/*
equation index: 91
type: SIMPLE_ASSIGN
Radiator.preRad[4].Q_flow = homotopy(Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[4] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[4], Radiator.n - 1.0, 0.05), Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[4]) ^ (Radiator.n - 1.0) * Radiator.dTRad[4])
*/
void Radiator_eqFunction_91(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,91};
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  tmp16 = fabs(data->simulationInfo->realParameter[31] /* Radiator.dTRad_nominal[4] PARAM */);
  tmp17 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp16 < 0.0 && tmp17 != 0.0)
  {
    tmp19 = modf(tmp17, &tmp20);
    
    if(tmp19 > 0.5)
    {
      tmp19 -= 1.0;
      tmp20 += 1.0;
    }
    else if(tmp19 < -0.5)
    {
      tmp19 += 1.0;
      tmp20 -= 1.0;
    }
    
    if(fabs(tmp19) < 1e-10)
      tmp18 = pow(tmp16, tmp20);
    else
    {
      tmp22 = modf(1.0/tmp17, &tmp21);
      if(tmp22 > 0.5)
      {
        tmp22 -= 1.0;
        tmp21 += 1.0;
      }
      else if(tmp22 < -0.5)
      {
        tmp22 += 1.0;
        tmp21 -= 1.0;
      }
      if(fabs(tmp22) < 1e-10 && ((unsigned long)tmp21 & 1))
      {
        tmp18 = -pow(-tmp16, tmp19)*pow(tmp16, tmp20);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp16, tmp17);
      }
    }
  }
  else
  {
    tmp18 = pow(tmp16, tmp17);
  }
  if(isnan(tmp18) || isinf(tmp18))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp16, tmp17);
  }
  data->localData[0]->realVars[50] /* Radiator.preRad[4].Q_flow variable */ = homotopy((((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp18)) * (data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_404(DATA *data, threadData_t *threadData);


/*
equation index: 93
type: SIMPLE_ASSIGN
Radiator.preCon[4].Q_flow = homotopy((1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTCon[4] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTCon[4], Radiator.n - 1.0, 0.05), (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[4]) ^ (Radiator.n - 1.0) * Radiator.dTCon[4])
*/
void Radiator_eqFunction_93(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,93};
  modelica_real tmp23;
  modelica_real tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  tmp23 = fabs(data->simulationInfo->realParameter[26] /* Radiator.dTCon_nominal[4] PARAM */);
  tmp24 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp23 < 0.0 && tmp24 != 0.0)
  {
    tmp26 = modf(tmp24, &tmp27);
    
    if(tmp26 > 0.5)
    {
      tmp26 -= 1.0;
      tmp27 += 1.0;
    }
    else if(tmp26 < -0.5)
    {
      tmp26 += 1.0;
      tmp27 -= 1.0;
    }
    
    if(fabs(tmp26) < 1e-10)
      tmp25 = pow(tmp23, tmp27);
    else
    {
      tmp29 = modf(1.0/tmp24, &tmp28);
      if(tmp29 > 0.5)
      {
        tmp29 -= 1.0;
        tmp28 += 1.0;
      }
      else if(tmp29 < -0.5)
      {
        tmp29 += 1.0;
        tmp28 -= 1.0;
      }
      if(fabs(tmp29) < 1e-10 && ((unsigned long)tmp28 & 1))
      {
        tmp25 = -pow(-tmp23, tmp26)*pow(tmp23, tmp27);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp23, tmp24);
      }
    }
  }
  else
  {
    tmp25 = pow(tmp23, tmp24);
  }
  if(isnan(tmp25) || isinf(tmp25))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp23, tmp24);
  }
  data->localData[0]->realVars[45] /* Radiator.preCon[4].Q_flow variable */ = homotopy((((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[33] /* Radiator.dTCon[4] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[33] /* Radiator.dTCon[4] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp25)) * (data->localData[0]->realVars[33] /* Radiator.dTCon[4] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_402(DATA *data, threadData_t *threadData);


/*
equation index: 95
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports_H_flow[1] = semiLinear(Radiator.vol[5].ports[1].m_flow, Radiator.vol[4].ports[2].h_outflow, Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_95(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,95};
  data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_380(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_395(DATA *data, threadData_t *threadData);


/*
equation index: 98
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports_H_flow[2] = semiLinear(-Radiator.vol[5].ports[1].m_flow, Radiator.vol[5].ports[2].h_outflow, Radiator.vol[4].ports[2].h_outflow)
*/
void Radiator_eqFunction_98(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,98};
  data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */ = semiLinear((-data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */), data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */);
  TRACE_POP
}

/*
equation index: 99
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.U = Radiator.vol[4].dynBal.m * Radiator.vol[4].ports[2].h_outflow + Radiator.vol[4].dynBal.CSen * Radiator.vol[4].dynBal.medium.T_degC
*/
void Radiator_eqFunction_99(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,99};
  data->localData[0]->realVars[4] /* Radiator.vol[4].dynBal.U STATE(1) */ = (data->localData[0]->realVars[68] /* Radiator.vol[4].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */) + (data->simulationInfo->realParameter[115] /* Radiator.vol[4].dynBal.CSen PARAM */) * (data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */);
  TRACE_POP
}

/*
equation index: 100
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.T = 293.15
*/
void Radiator_eqFunction_100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,100};
  data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */ = 293.15;
  TRACE_POP
}

/*
equation index: 101
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.T_degC = -273.15 + Radiator.vol[3].dynBal.medium.T
*/
void Radiator_eqFunction_101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,101};
  data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */ = -273.15 + data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */;
  TRACE_POP
}

/*
equation index: 102
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[2].h_outflow = 4184.0 * Radiator.vol[3].dynBal.medium.T_degC
*/
void Radiator_eqFunction_102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,102};
  data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */ = (4184.0) * (data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_351(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_405(DATA *data, threadData_t *threadData);


/*
equation index: 105
type: SIMPLE_ASSIGN
Radiator.preRad[3].Q_flow = homotopy(Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[3] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[3], Radiator.n - 1.0, 0.05), Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[3]) ^ (Radiator.n - 1.0) * Radiator.dTRad[3])
*/
void Radiator_eqFunction_105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,105};
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_real tmp34;
  modelica_real tmp35;
  modelica_real tmp36;
  tmp30 = fabs(data->simulationInfo->realParameter[30] /* Radiator.dTRad_nominal[3] PARAM */);
  tmp31 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp30 < 0.0 && tmp31 != 0.0)
  {
    tmp33 = modf(tmp31, &tmp34);
    
    if(tmp33 > 0.5)
    {
      tmp33 -= 1.0;
      tmp34 += 1.0;
    }
    else if(tmp33 < -0.5)
    {
      tmp33 += 1.0;
      tmp34 -= 1.0;
    }
    
    if(fabs(tmp33) < 1e-10)
      tmp32 = pow(tmp30, tmp34);
    else
    {
      tmp36 = modf(1.0/tmp31, &tmp35);
      if(tmp36 > 0.5)
      {
        tmp36 -= 1.0;
        tmp35 += 1.0;
      }
      else if(tmp36 < -0.5)
      {
        tmp36 += 1.0;
        tmp35 -= 1.0;
      }
      if(fabs(tmp36) < 1e-10 && ((unsigned long)tmp35 & 1))
      {
        tmp32 = -pow(-tmp30, tmp33)*pow(tmp30, tmp34);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp30, tmp31);
      }
    }
  }
  else
  {
    tmp32 = pow(tmp30, tmp31);
  }
  if(isnan(tmp32) || isinf(tmp32))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp30, tmp31);
  }
  data->localData[0]->realVars[49] /* Radiator.preRad[3].Q_flow variable */ = homotopy((((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp32)) * (data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_412(DATA *data, threadData_t *threadData);


/*
equation index: 107
type: SIMPLE_ASSIGN
Radiator.preCon[3].Q_flow = homotopy((1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTCon[3] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTCon[3], Radiator.n - 1.0, 0.05), (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[3]) ^ (Radiator.n - 1.0) * Radiator.dTCon[3])
*/
void Radiator_eqFunction_107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,107};
  modelica_real tmp37;
  modelica_real tmp38;
  modelica_real tmp39;
  modelica_real tmp40;
  modelica_real tmp41;
  modelica_real tmp42;
  modelica_real tmp43;
  tmp37 = fabs(data->simulationInfo->realParameter[25] /* Radiator.dTCon_nominal[3] PARAM */);
  tmp38 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp37 < 0.0 && tmp38 != 0.0)
  {
    tmp40 = modf(tmp38, &tmp41);
    
    if(tmp40 > 0.5)
    {
      tmp40 -= 1.0;
      tmp41 += 1.0;
    }
    else if(tmp40 < -0.5)
    {
      tmp40 += 1.0;
      tmp41 -= 1.0;
    }
    
    if(fabs(tmp40) < 1e-10)
      tmp39 = pow(tmp37, tmp41);
    else
    {
      tmp43 = modf(1.0/tmp38, &tmp42);
      if(tmp43 > 0.5)
      {
        tmp43 -= 1.0;
        tmp42 += 1.0;
      }
      else if(tmp43 < -0.5)
      {
        tmp43 += 1.0;
        tmp42 -= 1.0;
      }
      if(fabs(tmp43) < 1e-10 && ((unsigned long)tmp42 & 1))
      {
        tmp39 = -pow(-tmp37, tmp40)*pow(tmp37, tmp41);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp37, tmp38);
      }
    }
  }
  else
  {
    tmp39 = pow(tmp37, tmp38);
  }
  if(isnan(tmp39) || isinf(tmp39))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp37, tmp38);
  }
  data->localData[0]->realVars[44] /* Radiator.preCon[3].Q_flow variable */ = homotopy((((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[32] /* Radiator.dTCon[3] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[32] /* Radiator.dTCon[3] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp39)) * (data->localData[0]->realVars[32] /* Radiator.dTCon[3] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_410(DATA *data, threadData_t *threadData);


/*
equation index: 109
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports_H_flow[1] = semiLinear(Radiator.vol[4].ports[1].m_flow, Radiator.vol[3].ports[2].h_outflow, Radiator.vol[4].ports[2].h_outflow)
*/
void Radiator_eqFunction_109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,109};
  data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_377(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_403(DATA *data, threadData_t *threadData);


/*
equation index: 112
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports_H_flow[2] = semiLinear(-Radiator.vol[4].ports[1].m_flow, Radiator.vol[4].ports[2].h_outflow, Radiator.vol[3].ports[2].h_outflow)
*/
void Radiator_eqFunction_112(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,112};
  data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */ = semiLinear((-data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */), data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */);
  TRACE_POP
}

/*
equation index: 113
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.U = Radiator.vol[3].dynBal.m * Radiator.vol[3].ports[2].h_outflow + Radiator.vol[3].dynBal.CSen * Radiator.vol[3].dynBal.medium.T_degC
*/
void Radiator_eqFunction_113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,113};
  data->localData[0]->realVars[3] /* Radiator.vol[3].dynBal.U STATE(1) */ = (data->localData[0]->realVars[67] /* Radiator.vol[3].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */) + (data->simulationInfo->realParameter[114] /* Radiator.vol[3].dynBal.CSen PARAM */) * (data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */);
  TRACE_POP
}

/*
equation index: 114
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.T = 293.15
*/
void Radiator_eqFunction_114(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,114};
  data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */ = 293.15;
  TRACE_POP
}

/*
equation index: 115
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.T_degC = -273.15 + Radiator.vol[2].dynBal.medium.T
*/
void Radiator_eqFunction_115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,115};
  data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */ = -273.15 + data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */;
  TRACE_POP
}

/*
equation index: 116
type: SIMPLE_ASSIGN
Radiator.vol[2].ports[2].h_outflow = 4184.0 * Radiator.vol[2].dynBal.medium.T_degC
*/
void Radiator_eqFunction_116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,116};
  data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */ = (4184.0) * (data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_342(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_413(DATA *data, threadData_t *threadData);


/*
equation index: 119
type: SIMPLE_ASSIGN
Radiator.preRad[2].Q_flow = homotopy(Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[2] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[2], Radiator.n - 1.0, 0.05), Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[2]) ^ (Radiator.n - 1.0) * Radiator.dTRad[2])
*/
void Radiator_eqFunction_119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,119};
  modelica_real tmp44;
  modelica_real tmp45;
  modelica_real tmp46;
  modelica_real tmp47;
  modelica_real tmp48;
  modelica_real tmp49;
  modelica_real tmp50;
  tmp44 = fabs(data->simulationInfo->realParameter[29] /* Radiator.dTRad_nominal[2] PARAM */);
  tmp45 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp44 < 0.0 && tmp45 != 0.0)
  {
    tmp47 = modf(tmp45, &tmp48);
    
    if(tmp47 > 0.5)
    {
      tmp47 -= 1.0;
      tmp48 += 1.0;
    }
    else if(tmp47 < -0.5)
    {
      tmp47 += 1.0;
      tmp48 -= 1.0;
    }
    
    if(fabs(tmp47) < 1e-10)
      tmp46 = pow(tmp44, tmp48);
    else
    {
      tmp50 = modf(1.0/tmp45, &tmp49);
      if(tmp50 > 0.5)
      {
        tmp50 -= 1.0;
        tmp49 += 1.0;
      }
      else if(tmp50 < -0.5)
      {
        tmp50 += 1.0;
        tmp49 -= 1.0;
      }
      if(fabs(tmp50) < 1e-10 && ((unsigned long)tmp49 & 1))
      {
        tmp46 = -pow(-tmp44, tmp47)*pow(tmp44, tmp48);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp44, tmp45);
      }
    }
  }
  else
  {
    tmp46 = pow(tmp44, tmp45);
  }
  if(isnan(tmp46) || isinf(tmp46))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp44, tmp45);
  }
  data->localData[0]->realVars[48] /* Radiator.preRad[2].Q_flow variable */ = homotopy((((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp46)) * (data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_420(DATA *data, threadData_t *threadData);


/*
equation index: 121
type: SIMPLE_ASSIGN
Radiator.preCon[2].Q_flow = homotopy((1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTCon[2] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTCon[2], Radiator.n - 1.0, 0.05), (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[2]) ^ (Radiator.n - 1.0) * Radiator.dTCon[2])
*/
void Radiator_eqFunction_121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,121};
  modelica_real tmp51;
  modelica_real tmp52;
  modelica_real tmp53;
  modelica_real tmp54;
  modelica_real tmp55;
  modelica_real tmp56;
  modelica_real tmp57;
  tmp51 = fabs(data->simulationInfo->realParameter[24] /* Radiator.dTCon_nominal[2] PARAM */);
  tmp52 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp51 < 0.0 && tmp52 != 0.0)
  {
    tmp54 = modf(tmp52, &tmp55);
    
    if(tmp54 > 0.5)
    {
      tmp54 -= 1.0;
      tmp55 += 1.0;
    }
    else if(tmp54 < -0.5)
    {
      tmp54 += 1.0;
      tmp55 -= 1.0;
    }
    
    if(fabs(tmp54) < 1e-10)
      tmp53 = pow(tmp51, tmp55);
    else
    {
      tmp57 = modf(1.0/tmp52, &tmp56);
      if(tmp57 > 0.5)
      {
        tmp57 -= 1.0;
        tmp56 += 1.0;
      }
      else if(tmp57 < -0.5)
      {
        tmp57 += 1.0;
        tmp56 -= 1.0;
      }
      if(fabs(tmp57) < 1e-10 && ((unsigned long)tmp56 & 1))
      {
        tmp53 = -pow(-tmp51, tmp54)*pow(tmp51, tmp55);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp51, tmp52);
      }
    }
  }
  else
  {
    tmp53 = pow(tmp51, tmp52);
  }
  if(isnan(tmp53) || isinf(tmp53))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp51, tmp52);
  }
  data->localData[0]->realVars[43] /* Radiator.preCon[2].Q_flow variable */ = homotopy((((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[31] /* Radiator.dTCon[2] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[31] /* Radiator.dTCon[2] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp53)) * (data->localData[0]->realVars[31] /* Radiator.dTCon[2] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_418(DATA *data, threadData_t *threadData);


/*
equation index: 123
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports_H_flow[1] = semiLinear(Radiator.vol[3].ports[1].m_flow, Radiator.vol[2].ports[2].h_outflow, Radiator.vol[3].ports[2].h_outflow)
*/
void Radiator_eqFunction_123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,123};
  data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_363(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_411(DATA *data, threadData_t *threadData);


/*
equation index: 126
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports_H_flow[2] = semiLinear(-Radiator.vol[3].ports[1].m_flow, Radiator.vol[3].ports[2].h_outflow, Radiator.vol[2].ports[2].h_outflow)
*/
void Radiator_eqFunction_126(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,126};
  data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */ = semiLinear((-data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */), data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */);
  TRACE_POP
}

/*
equation index: 127
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.U = Radiator.vol[2].dynBal.m * Radiator.vol[2].ports[2].h_outflow + Radiator.vol[2].dynBal.CSen * Radiator.vol[2].dynBal.medium.T_degC
*/
void Radiator_eqFunction_127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,127};
  data->localData[0]->realVars[2] /* Radiator.vol[2].dynBal.U STATE(1) */ = (data->localData[0]->realVars[66] /* Radiator.vol[2].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */) + (data->simulationInfo->realParameter[113] /* Radiator.vol[2].dynBal.CSen PARAM */) * (data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */);
  TRACE_POP
}

/*
equation index: 128
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.T = Radiator.vol[1].dynBal.T_start
*/
void Radiator_eqFunction_128(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,128};
  data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */ = data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */;
  TRACE_POP
}

/*
equation index: 129
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.T_degC = -273.15 + Radiator.vol[1].dynBal.medium.T
*/
void Radiator_eqFunction_129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,129};
  data->localData[0]->realVars[95] /* Radiator.vol[1].dynBal.medium.T_degC variable */ = -273.15 + data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */;
  TRACE_POP
}

/*
equation index: 130
type: SIMPLE_ASSIGN
Radiator.port_a.h_outflow = -1142859.6 - (-4184.0) * Radiator.vol[1].dynBal.medium.T
*/
void Radiator_eqFunction_130(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,130};
  data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */ = -1142859.6 - ((-4184.0) * (data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_336(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_421(DATA *data, threadData_t *threadData);


/*
equation index: 133
type: SIMPLE_ASSIGN
Radiator.preRad[1].Q_flow = homotopy(Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[1] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[1], Radiator.n - 1.0, 0.05), Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[1]) ^ (Radiator.n - 1.0) * Radiator.dTRad[1])
*/
void Radiator_eqFunction_133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,133};
  modelica_real tmp58;
  modelica_real tmp59;
  modelica_real tmp60;
  modelica_real tmp61;
  modelica_real tmp62;
  modelica_real tmp63;
  modelica_real tmp64;
  tmp58 = fabs(data->simulationInfo->realParameter[28] /* Radiator.dTRad_nominal[1] PARAM */);
  tmp59 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp58 < 0.0 && tmp59 != 0.0)
  {
    tmp61 = modf(tmp59, &tmp62);
    
    if(tmp61 > 0.5)
    {
      tmp61 -= 1.0;
      tmp62 += 1.0;
    }
    else if(tmp61 < -0.5)
    {
      tmp61 += 1.0;
      tmp62 -= 1.0;
    }
    
    if(fabs(tmp61) < 1e-10)
      tmp60 = pow(tmp58, tmp62);
    else
    {
      tmp64 = modf(1.0/tmp59, &tmp63);
      if(tmp64 > 0.5)
      {
        tmp64 -= 1.0;
        tmp63 += 1.0;
      }
      else if(tmp64 < -0.5)
      {
        tmp64 += 1.0;
        tmp63 -= 1.0;
      }
      if(fabs(tmp64) < 1e-10 && ((unsigned long)tmp63 & 1))
      {
        tmp60 = -pow(-tmp58, tmp61)*pow(tmp58, tmp62);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp58, tmp59);
      }
    }
  }
  else
  {
    tmp60 = pow(tmp58, tmp59);
  }
  if(isnan(tmp60) || isinf(tmp60))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp58, tmp59);
  }
  data->localData[0]->realVars[47] /* Radiator.preRad[1].Q_flow variable */ = homotopy((((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp60)) * (data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_427(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_434(DATA *data, threadData_t *threadData);


/*
equation index: 136
type: SIMPLE_ASSIGN
Radiator.preCon[1].Q_flow = homotopy((1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTCon[1] * Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTCon[1], Radiator.n - 1.0, 0.05), (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[1]) ^ (Radiator.n - 1.0) * Radiator.dTCon[1])
*/
void Radiator_eqFunction_136(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,136};
  modelica_real tmp65;
  modelica_real tmp66;
  modelica_real tmp67;
  modelica_real tmp68;
  modelica_real tmp69;
  modelica_real tmp70;
  modelica_real tmp71;
  tmp65 = fabs(data->simulationInfo->realParameter[23] /* Radiator.dTCon_nominal[1] PARAM */);
  tmp66 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp65 < 0.0 && tmp66 != 0.0)
  {
    tmp68 = modf(tmp66, &tmp69);
    
    if(tmp68 > 0.5)
    {
      tmp68 -= 1.0;
      tmp69 += 1.0;
    }
    else if(tmp68 < -0.5)
    {
      tmp68 += 1.0;
      tmp69 -= 1.0;
    }
    
    if(fabs(tmp68) < 1e-10)
      tmp67 = pow(tmp65, tmp69);
    else
    {
      tmp71 = modf(1.0/tmp66, &tmp70);
      if(tmp71 > 0.5)
      {
        tmp71 -= 1.0;
        tmp70 += 1.0;
      }
      else if(tmp71 < -0.5)
      {
        tmp71 += 1.0;
        tmp70 -= 1.0;
      }
      if(fabs(tmp71) < 1e-10 && ((unsigned long)tmp70 & 1))
      {
        tmp67 = -pow(-tmp65, tmp68)*pow(tmp65, tmp69);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp65, tmp66);
      }
    }
  }
  else
  {
    tmp67 = pow(tmp65, tmp66);
  }
  if(isnan(tmp67) || isinf(tmp67))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp65, tmp66);
  }
  data->localData[0]->realVars[42] /* Radiator.preCon[1].Q_flow variable */ = homotopy((((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[30] /* Radiator.dTCon[1] variable */)) * (omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[30] /* Radiator.dTCon[1] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05)), (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp67)) * (data->localData[0]->realVars[30] /* Radiator.dTCon[1] variable */));
  TRACE_POP
}
extern void Radiator_eqFunction_424(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_428(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_429(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_430(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_431(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_432(DATA *data, threadData_t *threadData);


/*
equation index: 143
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports_H_flow[1] = semiLinear(Radiator.vol[2].ports[1].m_flow, Radiator.port_a.h_outflow, Radiator.vol[2].ports[2].h_outflow)
*/
void Radiator_eqFunction_143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,143};
  data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_353(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_419(DATA *data, threadData_t *threadData);


/*
equation index: 146
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.ports_H_flow[2] = semiLinear(-Radiator.vol[2].ports[1].m_flow, Radiator.vol[2].ports[2].h_outflow, Radiator.port_a.h_outflow)
*/
void Radiator_eqFunction_146(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,146};
  data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */ = semiLinear((-data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */), data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_385(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_386(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_433(DATA *data, threadData_t *threadData);


/*
equation index: 150
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.U = Radiator.vol[1].dynBal.m * Radiator.port_a.h_outflow + Radiator.vol[1].dynBal.CSen * Radiator.vol[1].dynBal.medium.T_degC
*/
void Radiator_eqFunction_150(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,150};
  data->localData[0]->realVars[1] /* Radiator.vol[1].dynBal.U STATE(1) */ = (data->localData[0]->realVars[65] /* Radiator.vol[1].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */) + (data->simulationInfo->realParameter[112] /* Radiator.vol[1].dynBal.CSen PARAM */) * (data->localData[0]->realVars[95] /* Radiator.vol[1].dynBal.medium.T_degC variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_387(DATA *data, threadData_t *threadData);


/*
equation index: 152
type: SIMPLE_ASSIGN
Radiator.res.dp = 0.0
*/
void Radiator_eqFunction_152(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,152};
  data->localData[0]->realVars[52] /* Radiator.res.dp variable */ = 0.0;
  TRACE_POP
}

/*
equation index: 153
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.d = 995.586
*/
void Radiator_eqFunction_153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,153};
  data->localData[0]->realVars[105] /* Radiator.vol[1].dynBal.medium.d variable */ = 995.586;
  TRACE_POP
}

/*
equation index: 154
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.d = 995.586
*/
void Radiator_eqFunction_154(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,154};
  data->localData[0]->realVars[106] /* Radiator.vol[2].dynBal.medium.d variable */ = 995.586;
  TRACE_POP
}

/*
equation index: 155
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.d = 995.586
*/
void Radiator_eqFunction_155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,155};
  data->localData[0]->realVars[107] /* Radiator.vol[3].dynBal.medium.d variable */ = 995.586;
  TRACE_POP
}

/*
equation index: 156
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.d = 995.586
*/
void Radiator_eqFunction_156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,156};
  data->localData[0]->realVars[108] /* Radiator.vol[4].dynBal.medium.d variable */ = 995.586;
  TRACE_POP
}

/*
equation index: 157
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.d = 995.586
*/
void Radiator_eqFunction_157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,157};
  data->localData[0]->realVars[109] /* Radiator.vol[5].dynBal.medium.d variable */ = 995.586;
  TRACE_POP
}

/*
equation index: 163
type: ALGORITHM

  assert(Radiator.res.m_flow_nominal_pos > 0.0, "m_flow_nominal_pos must be non-zero. Check parameters.");
*/
void Radiator_eqFunction_163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,163};
  modelica_boolean tmp72;
  static const MMC_DEFSTRINGLIT(tmp73,54,"m_flow_nominal_pos must be non-zero. Check parameters.");
  static int tmp74 = 0;
  {
    tmp72 = Greater(data->simulationInfo->realParameter[78] /* Radiator.res.m_flow_nominal_pos PARAM */,0.0);
    if(!tmp72)
    {
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/FixedResistances/PressureDrop.mo",30,2,30,90,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.m_flow_nominal_pos > 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(MMC_REFSTRINGLIT(tmp73)));
      }
    }
  }
  TRACE_POP
}

/*
equation index: 162
type: ALGORITHM

  assert(Radiator.T_a_nominal > 303.15, "In RadiatorEN442_2, T_a_nominal must be higher than T_b_nominal.");
*/
void Radiator_eqFunction_162(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,162};
  modelica_boolean tmp75;
  static const MMC_DEFSTRINGLIT(tmp76,64,"In RadiatorEN442_2, T_a_nominal must be higher than T_b_nominal.");
  static int tmp77 = 0;
  {
    tmp75 = Greater(data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */,303.15);
    if(!tmp75)
    {
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",191,6,192,75,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.T_a_nominal > 303.15", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(MMC_REFSTRINGLIT(tmp76)));
      }
    }
  }
  TRACE_POP
}

/*
equation index: 161
type: ALGORITHM

  assert(Radiator.Q_flow_nominal > 0.0, "In RadiatorEN442_2, nominal power must be bigger than zero if T_b_nominal > TAir_nominal.");
*/
void Radiator_eqFunction_161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,161};
  modelica_boolean tmp78;
  static const MMC_DEFSTRINGLIT(tmp79,89,"In RadiatorEN442_2, nominal power must be bigger than zero if T_b_nominal > TAir_nominal.");
  static int tmp80 = 0;
  {
    tmp78 = Greater(data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */,0.0);
    if(!tmp78)
    {
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",193,6,194,100,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.Q_flow_nominal > 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(MMC_REFSTRINGLIT(tmp79)));
      }
    }
  }
  TRACE_POP
}

/*
equation index: 160
type: ALGORITHM

  Modelica.Fluid.Utilities.checkBoundary("SimpleLiquidWater", {"SimpleLiquidWater"}, true, true, flow_sink.X_in_internal, "Boundary_pT");
*/
void Radiator_eqFunction_160(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,160};
  static const MMC_DEFSTRINGLIT(tmp81,17,"SimpleLiquidWater");
  string_array tmp82;
  static const MMC_DEFSTRINGLIT(tmp83,17,"SimpleLiquidWater");
  real_array tmp84;
  static const MMC_DEFSTRINGLIT(tmp85,11,"Boundary_pT");
  array_alloc_scalar_string_array(&tmp82, 1, (modelica_string)MMC_REFSTRINGLIT(tmp83));
  real_array_create(&tmp84, ((modelica_real*)&((&data->localData[0]->realVars[140] /* flow_sink.X_in_internal[1] variable */)[calc_base_index_dims_subs(1, (_index_t)1, ((modelica_integer) 1))])), 1, (_index_t)1);
  omc_Modelica_Fluid_Utilities_checkBoundary(threadData, MMC_REFSTRINGLIT(tmp81), tmp82, 1, 1, tmp84, MMC_REFSTRINGLIT(tmp85));
  TRACE_POP
}

/*
equation index: 159
type: ALGORITHM

  assert(flow_sink.p_in_internal > 10000.0, "In Radiator.flow_sink: The parameter value p=" + String(flow_sink.p_in_internal, 6, 0, true) + " is low for water. This is likely an error.");
*/
void Radiator_eqFunction_159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,159};
  modelica_boolean tmp86;
  static const MMC_DEFSTRINGLIT(tmp87,45,"In Radiator.flow_sink: The parameter value p=");
  modelica_string tmp88;
  static const MMC_DEFSTRINGLIT(tmp89,43," is low for water. This is likely an error.");
  static int tmp90 = 0;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  {
    tmp86 = Greater(data->localData[0]->realVars[141] /* flow_sink.p_in_internal variable */,10000.0);
    if(!tmp86)
    {
      tmp88 = modelica_real_to_modelica_string(data->localData[0]->realVars[141] /* flow_sink.p_in_internal variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp87),tmp88);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp89));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/Boundary_pT.mo",46,7,47,104,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.p_in_internal > 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[1]));
      }
    }
  }
  TRACE_POP
}

/*
equation index: 158
type: ALGORITHM

  Modelica.Fluid.Utilities.checkBoundary("SimpleLiquidWater", {"SimpleLiquidWater"}, true, true, flow_source.X_in_internal, "Boundary_pT");
*/
void Radiator_eqFunction_158(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,158};
  static const MMC_DEFSTRINGLIT(tmp91,17,"SimpleLiquidWater");
  string_array tmp92;
  static const MMC_DEFSTRINGLIT(tmp93,17,"SimpleLiquidWater");
  real_array tmp94;
  static const MMC_DEFSTRINGLIT(tmp95,11,"Boundary_pT");
  array_alloc_scalar_string_array(&tmp92, 1, (modelica_string)MMC_REFSTRINGLIT(tmp93));
  real_array_create(&tmp94, ((modelica_real*)&((&data->localData[0]->realVars[146] /* flow_source.X_in_internal[1] variable */)[calc_base_index_dims_subs(1, (_index_t)1, ((modelica_integer) 1))])), 1, (_index_t)1);
  omc_Modelica_Fluid_Utilities_checkBoundary(threadData, MMC_REFSTRINGLIT(tmp91), tmp92, 1, 1, tmp94, MMC_REFSTRINGLIT(tmp95));
  TRACE_POP
}
OMC_DISABLE_OPT
void Radiator_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  Radiator_eqFunction_1(data, threadData);
  Radiator_eqFunction_2(data, threadData);
  Radiator_eqFunction_3(data, threadData);
  Radiator_eqFunction_4(data, threadData);
  Radiator_eqFunction_5(data, threadData);
  Radiator_eqFunction_6(data, threadData);
  Radiator_eqFunction_7(data, threadData);
  Radiator_eqFunction_8(data, threadData);
  Radiator_eqFunction_9(data, threadData);
  Radiator_eqFunction_10(data, threadData);
  Radiator_eqFunction_11(data, threadData);
  Radiator_eqFunction_12(data, threadData);
  Radiator_eqFunction_13(data, threadData);
  Radiator_eqFunction_14(data, threadData);
  Radiator_eqFunction_15(data, threadData);
  Radiator_eqFunction_16(data, threadData);
  Radiator_eqFunction_17(data, threadData);
  Radiator_eqFunction_18(data, threadData);
  Radiator_eqFunction_19(data, threadData);
  Radiator_eqFunction_20(data, threadData);
  Radiator_eqFunction_21(data, threadData);
  Radiator_eqFunction_22(data, threadData);
  Radiator_eqFunction_23(data, threadData);
  Radiator_eqFunction_24(data, threadData);
  Radiator_eqFunction_25(data, threadData);
  Radiator_eqFunction_26(data, threadData);
  Radiator_eqFunction_27(data, threadData);
  Radiator_eqFunction_28(data, threadData);
  Radiator_eqFunction_29(data, threadData);
  Radiator_eqFunction_30(data, threadData);
  Radiator_eqFunction_31(data, threadData);
  Radiator_eqFunction_32(data, threadData);
  Radiator_eqFunction_33(data, threadData);
  Radiator_eqFunction_34(data, threadData);
  Radiator_eqFunction_35(data, threadData);
  Radiator_eqFunction_36(data, threadData);
  Radiator_eqFunction_37(data, threadData);
  Radiator_eqFunction_38(data, threadData);
  Radiator_eqFunction_325(data, threadData);
  Radiator_eqFunction_40(data, threadData);
  Radiator_eqFunction_41(data, threadData);
  Radiator_eqFunction_42(data, threadData);
  Radiator_eqFunction_43(data, threadData);
  Radiator_eqFunction_382(data, threadData);
  Radiator_eqFunction_388(data, threadData);
  Radiator_eqFunction_326(data, threadData);
  Radiator_eqFunction_327(data, threadData);
  Radiator_eqFunction_328(data, threadData);
  Radiator_eqFunction_49(data, threadData);
  Radiator_eqFunction_61(data, threadData);
  Radiator_eqFunction_62(data, threadData);
  Radiator_eqFunction_63(data, threadData);
  Radiator_eqFunction_64(data, threadData);
  Radiator_eqFunction_65(data, threadData);
  Radiator_eqFunction_66(data, threadData);
  Radiator_eqFunction_67(data, threadData);
  Radiator_eqFunction_68(data, threadData);
  Radiator_eqFunction_69(data, threadData);
  Radiator_eqFunction_70(data, threadData);
  Radiator_eqFunction_71(data, threadData);
  Radiator_eqFunction_72(data, threadData);
  Radiator_eqFunction_73(data, threadData);
  Radiator_eqFunction_74(data, threadData);
  Radiator_eqFunction_372(data, threadData);
  Radiator_eqFunction_76(data, threadData);
  Radiator_eqFunction_373(data, threadData);
  Radiator_eqFunction_389(data, threadData);
  Radiator_eqFunction_79(data, threadData);
  Radiator_eqFunction_396(data, threadData);
  Radiator_eqFunction_81(data, threadData);
  Radiator_eqFunction_394(data, threadData);
  Radiator_eqFunction_83(data, threadData);
  Radiator_eqFunction_84(data, threadData);
  Radiator_eqFunction_85(data, threadData);
  Radiator_eqFunction_86(data, threadData);
  Radiator_eqFunction_87(data, threadData);
  Radiator_eqFunction_88(data, threadData);
  Radiator_eqFunction_361(data, threadData);
  Radiator_eqFunction_397(data, threadData);
  Radiator_eqFunction_91(data, threadData);
  Radiator_eqFunction_404(data, threadData);
  Radiator_eqFunction_93(data, threadData);
  Radiator_eqFunction_402(data, threadData);
  Radiator_eqFunction_95(data, threadData);
  Radiator_eqFunction_380(data, threadData);
  Radiator_eqFunction_395(data, threadData);
  Radiator_eqFunction_98(data, threadData);
  Radiator_eqFunction_99(data, threadData);
  Radiator_eqFunction_100(data, threadData);
  Radiator_eqFunction_101(data, threadData);
  Radiator_eqFunction_102(data, threadData);
  Radiator_eqFunction_351(data, threadData);
  Radiator_eqFunction_405(data, threadData);
  Radiator_eqFunction_105(data, threadData);
  Radiator_eqFunction_412(data, threadData);
  Radiator_eqFunction_107(data, threadData);
  Radiator_eqFunction_410(data, threadData);
  Radiator_eqFunction_109(data, threadData);
  Radiator_eqFunction_377(data, threadData);
  Radiator_eqFunction_403(data, threadData);
  Radiator_eqFunction_112(data, threadData);
  Radiator_eqFunction_113(data, threadData);
  Radiator_eqFunction_114(data, threadData);
  Radiator_eqFunction_115(data, threadData);
  Radiator_eqFunction_116(data, threadData);
  Radiator_eqFunction_342(data, threadData);
  Radiator_eqFunction_413(data, threadData);
  Radiator_eqFunction_119(data, threadData);
  Radiator_eqFunction_420(data, threadData);
  Radiator_eqFunction_121(data, threadData);
  Radiator_eqFunction_418(data, threadData);
  Radiator_eqFunction_123(data, threadData);
  Radiator_eqFunction_363(data, threadData);
  Radiator_eqFunction_411(data, threadData);
  Radiator_eqFunction_126(data, threadData);
  Radiator_eqFunction_127(data, threadData);
  Radiator_eqFunction_128(data, threadData);
  Radiator_eqFunction_129(data, threadData);
  Radiator_eqFunction_130(data, threadData);
  Radiator_eqFunction_336(data, threadData);
  Radiator_eqFunction_421(data, threadData);
  Radiator_eqFunction_133(data, threadData);
  Radiator_eqFunction_427(data, threadData);
  Radiator_eqFunction_434(data, threadData);
  Radiator_eqFunction_136(data, threadData);
  Radiator_eqFunction_424(data, threadData);
  Radiator_eqFunction_428(data, threadData);
  Radiator_eqFunction_429(data, threadData);
  Radiator_eqFunction_430(data, threadData);
  Radiator_eqFunction_431(data, threadData);
  Radiator_eqFunction_432(data, threadData);
  Radiator_eqFunction_143(data, threadData);
  Radiator_eqFunction_353(data, threadData);
  Radiator_eqFunction_419(data, threadData);
  Radiator_eqFunction_146(data, threadData);
  Radiator_eqFunction_385(data, threadData);
  Radiator_eqFunction_386(data, threadData);
  Radiator_eqFunction_433(data, threadData);
  Radiator_eqFunction_150(data, threadData);
  Radiator_eqFunction_387(data, threadData);
  Radiator_eqFunction_152(data, threadData);
  Radiator_eqFunction_153(data, threadData);
  Radiator_eqFunction_154(data, threadData);
  Radiator_eqFunction_155(data, threadData);
  Radiator_eqFunction_156(data, threadData);
  Radiator_eqFunction_157(data, threadData);
  Radiator_eqFunction_163(data, threadData);
  Radiator_eqFunction_162(data, threadData);
  Radiator_eqFunction_161(data, threadData);
  Radiator_eqFunction_160(data, threadData);
  Radiator_eqFunction_159(data, threadData);
  Radiator_eqFunction_158(data, threadData);
  TRACE_POP
}

int Radiator_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  Radiator_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}
extern void Radiator_eqFunction_1(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_2(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_3(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_4(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_5(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_6(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_7(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_8(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_9(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_10(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_11(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_12(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_13(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_14(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_15(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_16(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_17(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_18(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_19(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_20(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_21(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_22(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_23(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_24(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_25(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_26(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_27(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_28(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_29(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_30(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_31(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_32(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_33(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_34(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_35(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_36(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_37(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_38(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_325(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_40(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_41(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_42(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_43(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_382(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_388(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_326(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_327(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_328(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_49(DATA *data, threadData_t *threadData);


void Radiator_eqFunction_213(DATA*, threadData_t*);
void Radiator_eqFunction_214(DATA*, threadData_t*);
void Radiator_eqFunction_215(DATA*, threadData_t*);
void Radiator_eqFunction_216(DATA*, threadData_t*);
void Radiator_eqFunction_217(DATA*, threadData_t*);
void Radiator_eqFunction_218(DATA*, threadData_t*);
void Radiator_eqFunction_219(DATA*, threadData_t*);
void Radiator_eqFunction_223(DATA*, threadData_t*);
void Radiator_eqFunction_222(DATA*, threadData_t*);
void Radiator_eqFunction_221(DATA*, threadData_t*);
void Radiator_eqFunction_220(DATA*, threadData_t*);
/*
equation index: 224
indexNonlinear: 1
type: NONLINEAR

vars: {Radiator.QEle_flow_nominal[4], Radiator.TWat_nominal[3], Radiator.TWat_nominal[1], Radiator.UAEle}
eqns: {213, 214, 215, 216, 217, 218, 219, 223, 222, 221, 220}
*/
void Radiator_eqFunction_224(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,224};
  int retValue;
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving nonlinear system 224 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  /* get old value */
  data->simulationInfo->nonlinearSystemData[1].nlsxOld[0] = (0.2) * (data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */);
  data->simulationInfo->nonlinearSystemData[1].nlsxOld[1] = data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ - ((0.6) * (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ - 303.15));
  data->simulationInfo->nonlinearSystemData[1].nlsxOld[2] = data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ - ((0.2) * (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ - 303.15));
  data->simulationInfo->nonlinearSystemData[1].nlsxOld[3] = DIVISION_SIM(data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */,((0.5) * (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */) + 151.575 - ((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (293.15) + (data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */))) * (5.0),"(0.5 * Radiator.T_a_nominal + 151.575 - ((1.0 - Radiator.fraRad) * 293.15 + Radiator.fraRad * Radiator.TRad_nominal)) * 5.0",equationIndexes);
  retValue = solve_nonlinear_system(data, threadData, 1);
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,224};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving non-linear system 224 failed at time=%.15g.\nFor more information please use -lv LOG_NLS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */ = data->simulationInfo->nonlinearSystemData[1].nlsx[0];
  data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ = data->simulationInfo->nonlinearSystemData[1].nlsx[1];
  data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ = data->simulationInfo->nonlinearSystemData[1].nlsx[2];
  data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */ = data->simulationInfo->nonlinearSystemData[1].nlsx[3];
  TRACE_POP
}
extern void Radiator_eqFunction_62(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_63(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_64(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_65(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_66(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_67(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_68(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_69(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_70(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_71(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_72(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_73(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_74(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_372(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_76(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_373(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_389(DATA *data, threadData_t *threadData);


/*
equation index: 242
type: SIMPLE_ASSIGN
Radiator.preRad[5].Q_flow = Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[5]) ^ (Radiator.n - 1.0) * Radiator.dTRad[5]
*/
void Radiator_eqFunction_242(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,242};
  modelica_real tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  tmp0 = fabs(data->simulationInfo->realParameter[32] /* Radiator.dTRad_nominal[5] PARAM */);
  tmp1 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp0 < 0.0 && tmp1 != 0.0)
  {
    tmp3 = modf(tmp1, &tmp4);
    
    if(tmp3 > 0.5)
    {
      tmp3 -= 1.0;
      tmp4 += 1.0;
    }
    else if(tmp3 < -0.5)
    {
      tmp3 += 1.0;
      tmp4 -= 1.0;
    }
    
    if(fabs(tmp3) < 1e-10)
      tmp2 = pow(tmp0, tmp4);
    else
    {
      tmp6 = modf(1.0/tmp1, &tmp5);
      if(tmp6 > 0.5)
      {
        tmp6 -= 1.0;
        tmp5 += 1.0;
      }
      else if(tmp6 < -0.5)
      {
        tmp6 += 1.0;
        tmp5 -= 1.0;
      }
      if(fabs(tmp6) < 1e-10 && ((unsigned long)tmp5 & 1))
      {
        tmp2 = -pow(-tmp0, tmp3)*pow(tmp0, tmp4);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp0, tmp1);
      }
    }
  }
  else
  {
    tmp2 = pow(tmp0, tmp1);
  }
  if(isnan(tmp2) || isinf(tmp2))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp0, tmp1);
  }
  data->localData[0]->realVars[51] /* Radiator.preRad[5].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp2)) * (data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_396(DATA *data, threadData_t *threadData);


/*
equation index: 244
type: SIMPLE_ASSIGN
Radiator.preCon[5].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[5]) ^ (Radiator.n - 1.0) * Radiator.dTCon[5]
*/
void Radiator_eqFunction_244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,244};
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  tmp7 = fabs(data->simulationInfo->realParameter[27] /* Radiator.dTCon_nominal[5] PARAM */);
  tmp8 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp7 < 0.0 && tmp8 != 0.0)
  {
    tmp10 = modf(tmp8, &tmp11);
    
    if(tmp10 > 0.5)
    {
      tmp10 -= 1.0;
      tmp11 += 1.0;
    }
    else if(tmp10 < -0.5)
    {
      tmp10 += 1.0;
      tmp11 -= 1.0;
    }
    
    if(fabs(tmp10) < 1e-10)
      tmp9 = pow(tmp7, tmp11);
    else
    {
      tmp13 = modf(1.0/tmp8, &tmp12);
      if(tmp13 > 0.5)
      {
        tmp13 -= 1.0;
        tmp12 += 1.0;
      }
      else if(tmp13 < -0.5)
      {
        tmp13 += 1.0;
        tmp12 -= 1.0;
      }
      if(fabs(tmp13) < 1e-10 && ((unsigned long)tmp12 & 1))
      {
        tmp9 = -pow(-tmp7, tmp10)*pow(tmp7, tmp11);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp7, tmp8);
      }
    }
  }
  else
  {
    tmp9 = pow(tmp7, tmp8);
  }
  if(isnan(tmp9) || isinf(tmp9))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp7, tmp8);
  }
  data->localData[0]->realVars[46] /* Radiator.preCon[5].Q_flow variable */ = (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp9)) * (data->localData[0]->realVars[34] /* Radiator.dTCon[5] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_394(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_83(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_84(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_85(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_86(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_87(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_88(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_361(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_397(DATA *data, threadData_t *threadData);


/*
equation index: 254
type: SIMPLE_ASSIGN
Radiator.preRad[4].Q_flow = Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[4]) ^ (Radiator.n - 1.0) * Radiator.dTRad[4]
*/
void Radiator_eqFunction_254(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,254};
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  tmp14 = fabs(data->simulationInfo->realParameter[31] /* Radiator.dTRad_nominal[4] PARAM */);
  tmp15 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp14 < 0.0 && tmp15 != 0.0)
  {
    tmp17 = modf(tmp15, &tmp18);
    
    if(tmp17 > 0.5)
    {
      tmp17 -= 1.0;
      tmp18 += 1.0;
    }
    else if(tmp17 < -0.5)
    {
      tmp17 += 1.0;
      tmp18 -= 1.0;
    }
    
    if(fabs(tmp17) < 1e-10)
      tmp16 = pow(tmp14, tmp18);
    else
    {
      tmp20 = modf(1.0/tmp15, &tmp19);
      if(tmp20 > 0.5)
      {
        tmp20 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp20 < -0.5)
      {
        tmp20 += 1.0;
        tmp19 -= 1.0;
      }
      if(fabs(tmp20) < 1e-10 && ((unsigned long)tmp19 & 1))
      {
        tmp16 = -pow(-tmp14, tmp17)*pow(tmp14, tmp18);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp14, tmp15);
      }
    }
  }
  else
  {
    tmp16 = pow(tmp14, tmp15);
  }
  if(isnan(tmp16) || isinf(tmp16))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp14, tmp15);
  }
  data->localData[0]->realVars[50] /* Radiator.preRad[4].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp16)) * (data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_404(DATA *data, threadData_t *threadData);


/*
equation index: 256
type: SIMPLE_ASSIGN
Radiator.preCon[4].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[4]) ^ (Radiator.n - 1.0) * Radiator.dTCon[4]
*/
void Radiator_eqFunction_256(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,256};
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_real tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  tmp21 = fabs(data->simulationInfo->realParameter[26] /* Radiator.dTCon_nominal[4] PARAM */);
  tmp22 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp21 < 0.0 && tmp22 != 0.0)
  {
    tmp24 = modf(tmp22, &tmp25);
    
    if(tmp24 > 0.5)
    {
      tmp24 -= 1.0;
      tmp25 += 1.0;
    }
    else if(tmp24 < -0.5)
    {
      tmp24 += 1.0;
      tmp25 -= 1.0;
    }
    
    if(fabs(tmp24) < 1e-10)
      tmp23 = pow(tmp21, tmp25);
    else
    {
      tmp27 = modf(1.0/tmp22, &tmp26);
      if(tmp27 > 0.5)
      {
        tmp27 -= 1.0;
        tmp26 += 1.0;
      }
      else if(tmp27 < -0.5)
      {
        tmp27 += 1.0;
        tmp26 -= 1.0;
      }
      if(fabs(tmp27) < 1e-10 && ((unsigned long)tmp26 & 1))
      {
        tmp23 = -pow(-tmp21, tmp24)*pow(tmp21, tmp25);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp21, tmp22);
      }
    }
  }
  else
  {
    tmp23 = pow(tmp21, tmp22);
  }
  if(isnan(tmp23) || isinf(tmp23))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp21, tmp22);
  }
  data->localData[0]->realVars[45] /* Radiator.preCon[4].Q_flow variable */ = (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp23)) * (data->localData[0]->realVars[33] /* Radiator.dTCon[4] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_402(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_95(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_380(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_395(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_98(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_99(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_100(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_101(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_102(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_351(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_405(DATA *data, threadData_t *threadData);


/*
equation index: 268
type: SIMPLE_ASSIGN
Radiator.preRad[3].Q_flow = Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[3]) ^ (Radiator.n - 1.0) * Radiator.dTRad[3]
*/
void Radiator_eqFunction_268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,268};
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_real tmp34;
  tmp28 = fabs(data->simulationInfo->realParameter[30] /* Radiator.dTRad_nominal[3] PARAM */);
  tmp29 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp28 < 0.0 && tmp29 != 0.0)
  {
    tmp31 = modf(tmp29, &tmp32);
    
    if(tmp31 > 0.5)
    {
      tmp31 -= 1.0;
      tmp32 += 1.0;
    }
    else if(tmp31 < -0.5)
    {
      tmp31 += 1.0;
      tmp32 -= 1.0;
    }
    
    if(fabs(tmp31) < 1e-10)
      tmp30 = pow(tmp28, tmp32);
    else
    {
      tmp34 = modf(1.0/tmp29, &tmp33);
      if(tmp34 > 0.5)
      {
        tmp34 -= 1.0;
        tmp33 += 1.0;
      }
      else if(tmp34 < -0.5)
      {
        tmp34 += 1.0;
        tmp33 -= 1.0;
      }
      if(fabs(tmp34) < 1e-10 && ((unsigned long)tmp33 & 1))
      {
        tmp30 = -pow(-tmp28, tmp31)*pow(tmp28, tmp32);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp28, tmp29);
      }
    }
  }
  else
  {
    tmp30 = pow(tmp28, tmp29);
  }
  if(isnan(tmp30) || isinf(tmp30))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp28, tmp29);
  }
  data->localData[0]->realVars[49] /* Radiator.preRad[3].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp30)) * (data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_412(DATA *data, threadData_t *threadData);


/*
equation index: 270
type: SIMPLE_ASSIGN
Radiator.preCon[3].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[3]) ^ (Radiator.n - 1.0) * Radiator.dTCon[3]
*/
void Radiator_eqFunction_270(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,270};
  modelica_real tmp35;
  modelica_real tmp36;
  modelica_real tmp37;
  modelica_real tmp38;
  modelica_real tmp39;
  modelica_real tmp40;
  modelica_real tmp41;
  tmp35 = fabs(data->simulationInfo->realParameter[25] /* Radiator.dTCon_nominal[3] PARAM */);
  tmp36 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp35 < 0.0 && tmp36 != 0.0)
  {
    tmp38 = modf(tmp36, &tmp39);
    
    if(tmp38 > 0.5)
    {
      tmp38 -= 1.0;
      tmp39 += 1.0;
    }
    else if(tmp38 < -0.5)
    {
      tmp38 += 1.0;
      tmp39 -= 1.0;
    }
    
    if(fabs(tmp38) < 1e-10)
      tmp37 = pow(tmp35, tmp39);
    else
    {
      tmp41 = modf(1.0/tmp36, &tmp40);
      if(tmp41 > 0.5)
      {
        tmp41 -= 1.0;
        tmp40 += 1.0;
      }
      else if(tmp41 < -0.5)
      {
        tmp41 += 1.0;
        tmp40 -= 1.0;
      }
      if(fabs(tmp41) < 1e-10 && ((unsigned long)tmp40 & 1))
      {
        tmp37 = -pow(-tmp35, tmp38)*pow(tmp35, tmp39);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp35, tmp36);
      }
    }
  }
  else
  {
    tmp37 = pow(tmp35, tmp36);
  }
  if(isnan(tmp37) || isinf(tmp37))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp35, tmp36);
  }
  data->localData[0]->realVars[44] /* Radiator.preCon[3].Q_flow variable */ = (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp37)) * (data->localData[0]->realVars[32] /* Radiator.dTCon[3] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_410(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_109(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_377(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_403(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_112(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_113(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_114(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_115(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_116(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_342(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_413(DATA *data, threadData_t *threadData);


/*
equation index: 282
type: SIMPLE_ASSIGN
Radiator.preRad[2].Q_flow = Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[2]) ^ (Radiator.n - 1.0) * Radiator.dTRad[2]
*/
void Radiator_eqFunction_282(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,282};
  modelica_real tmp42;
  modelica_real tmp43;
  modelica_real tmp44;
  modelica_real tmp45;
  modelica_real tmp46;
  modelica_real tmp47;
  modelica_real tmp48;
  tmp42 = fabs(data->simulationInfo->realParameter[29] /* Radiator.dTRad_nominal[2] PARAM */);
  tmp43 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp42 < 0.0 && tmp43 != 0.0)
  {
    tmp45 = modf(tmp43, &tmp46);
    
    if(tmp45 > 0.5)
    {
      tmp45 -= 1.0;
      tmp46 += 1.0;
    }
    else if(tmp45 < -0.5)
    {
      tmp45 += 1.0;
      tmp46 -= 1.0;
    }
    
    if(fabs(tmp45) < 1e-10)
      tmp44 = pow(tmp42, tmp46);
    else
    {
      tmp48 = modf(1.0/tmp43, &tmp47);
      if(tmp48 > 0.5)
      {
        tmp48 -= 1.0;
        tmp47 += 1.0;
      }
      else if(tmp48 < -0.5)
      {
        tmp48 += 1.0;
        tmp47 -= 1.0;
      }
      if(fabs(tmp48) < 1e-10 && ((unsigned long)tmp47 & 1))
      {
        tmp44 = -pow(-tmp42, tmp45)*pow(tmp42, tmp46);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp42, tmp43);
      }
    }
  }
  else
  {
    tmp44 = pow(tmp42, tmp43);
  }
  if(isnan(tmp44) || isinf(tmp44))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp42, tmp43);
  }
  data->localData[0]->realVars[48] /* Radiator.preRad[2].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp44)) * (data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_420(DATA *data, threadData_t *threadData);


/*
equation index: 284
type: SIMPLE_ASSIGN
Radiator.preCon[2].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[2]) ^ (Radiator.n - 1.0) * Radiator.dTCon[2]
*/
void Radiator_eqFunction_284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,284};
  modelica_real tmp49;
  modelica_real tmp50;
  modelica_real tmp51;
  modelica_real tmp52;
  modelica_real tmp53;
  modelica_real tmp54;
  modelica_real tmp55;
  tmp49 = fabs(data->simulationInfo->realParameter[24] /* Radiator.dTCon_nominal[2] PARAM */);
  tmp50 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp49 < 0.0 && tmp50 != 0.0)
  {
    tmp52 = modf(tmp50, &tmp53);
    
    if(tmp52 > 0.5)
    {
      tmp52 -= 1.0;
      tmp53 += 1.0;
    }
    else if(tmp52 < -0.5)
    {
      tmp52 += 1.0;
      tmp53 -= 1.0;
    }
    
    if(fabs(tmp52) < 1e-10)
      tmp51 = pow(tmp49, tmp53);
    else
    {
      tmp55 = modf(1.0/tmp50, &tmp54);
      if(tmp55 > 0.5)
      {
        tmp55 -= 1.0;
        tmp54 += 1.0;
      }
      else if(tmp55 < -0.5)
      {
        tmp55 += 1.0;
        tmp54 -= 1.0;
      }
      if(fabs(tmp55) < 1e-10 && ((unsigned long)tmp54 & 1))
      {
        tmp51 = -pow(-tmp49, tmp52)*pow(tmp49, tmp53);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp49, tmp50);
      }
    }
  }
  else
  {
    tmp51 = pow(tmp49, tmp50);
  }
  if(isnan(tmp51) || isinf(tmp51))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp49, tmp50);
  }
  data->localData[0]->realVars[43] /* Radiator.preCon[2].Q_flow variable */ = (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp51)) * (data->localData[0]->realVars[31] /* Radiator.dTCon[2] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_418(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_123(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_363(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_411(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_126(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_127(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_128(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_129(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_130(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_336(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_421(DATA *data, threadData_t *threadData);


/*
equation index: 296
type: SIMPLE_ASSIGN
Radiator.preRad[1].Q_flow = Radiator.fraRad * Radiator.UAEle * abs(Radiator.dTRad_nominal[1]) ^ (Radiator.n - 1.0) * Radiator.dTRad[1]
*/
void Radiator_eqFunction_296(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,296};
  modelica_real tmp56;
  modelica_real tmp57;
  modelica_real tmp58;
  modelica_real tmp59;
  modelica_real tmp60;
  modelica_real tmp61;
  modelica_real tmp62;
  tmp56 = fabs(data->simulationInfo->realParameter[28] /* Radiator.dTRad_nominal[1] PARAM */);
  tmp57 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp56 < 0.0 && tmp57 != 0.0)
  {
    tmp59 = modf(tmp57, &tmp60);
    
    if(tmp59 > 0.5)
    {
      tmp59 -= 1.0;
      tmp60 += 1.0;
    }
    else if(tmp59 < -0.5)
    {
      tmp59 += 1.0;
      tmp60 -= 1.0;
    }
    
    if(fabs(tmp59) < 1e-10)
      tmp58 = pow(tmp56, tmp60);
    else
    {
      tmp62 = modf(1.0/tmp57, &tmp61);
      if(tmp62 > 0.5)
      {
        tmp62 -= 1.0;
        tmp61 += 1.0;
      }
      else if(tmp62 < -0.5)
      {
        tmp62 += 1.0;
        tmp61 -= 1.0;
      }
      if(fabs(tmp62) < 1e-10 && ((unsigned long)tmp61 & 1))
      {
        tmp58 = -pow(-tmp56, tmp59)*pow(tmp56, tmp60);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp56, tmp57);
      }
    }
  }
  else
  {
    tmp58 = pow(tmp56, tmp57);
  }
  if(isnan(tmp58) || isinf(tmp58))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp56, tmp57);
  }
  data->localData[0]->realVars[47] /* Radiator.preRad[1].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp58)) * (data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_427(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_434(DATA *data, threadData_t *threadData);


/*
equation index: 299
type: SIMPLE_ASSIGN
Radiator.preCon[1].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * abs(Radiator.dTCon_nominal[1]) ^ (Radiator.n - 1.0) * Radiator.dTCon[1]
*/
void Radiator_eqFunction_299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,299};
  modelica_real tmp63;
  modelica_real tmp64;
  modelica_real tmp65;
  modelica_real tmp66;
  modelica_real tmp67;
  modelica_real tmp68;
  modelica_real tmp69;
  tmp63 = fabs(data->simulationInfo->realParameter[23] /* Radiator.dTCon_nominal[1] PARAM */);
  tmp64 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
  if(tmp63 < 0.0 && tmp64 != 0.0)
  {
    tmp66 = modf(tmp64, &tmp67);
    
    if(tmp66 > 0.5)
    {
      tmp66 -= 1.0;
      tmp67 += 1.0;
    }
    else if(tmp66 < -0.5)
    {
      tmp66 += 1.0;
      tmp67 -= 1.0;
    }
    
    if(fabs(tmp66) < 1e-10)
      tmp65 = pow(tmp63, tmp67);
    else
    {
      tmp69 = modf(1.0/tmp64, &tmp68);
      if(tmp69 > 0.5)
      {
        tmp69 -= 1.0;
        tmp68 += 1.0;
      }
      else if(tmp69 < -0.5)
      {
        tmp69 += 1.0;
        tmp68 -= 1.0;
      }
      if(fabs(tmp69) < 1e-10 && ((unsigned long)tmp68 & 1))
      {
        tmp65 = -pow(-tmp63, tmp66)*pow(tmp63, tmp67);
      }
      else
      {
        throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp63, tmp64);
      }
    }
  }
  else
  {
    tmp65 = pow(tmp63, tmp64);
  }
  if(isnan(tmp65) || isinf(tmp65))
  {
    throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp63, tmp64);
  }
  data->localData[0]->realVars[42] /* Radiator.preCon[1].Q_flow variable */ = (((1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (tmp65)) * (data->localData[0]->realVars[30] /* Radiator.dTCon[1] variable */);
  TRACE_POP
}
extern void Radiator_eqFunction_424(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_428(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_429(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_430(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_431(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_432(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_143(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_353(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_419(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_146(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_385(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_386(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_433(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_150(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_387(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_152(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_153(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_154(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_155(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_156(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_157(DATA *data, threadData_t *threadData);

int Radiator_functionInitialEquations_lambda0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  Radiator_eqFunction_1(data, threadData);

  Radiator_eqFunction_2(data, threadData);

  Radiator_eqFunction_3(data, threadData);

  Radiator_eqFunction_4(data, threadData);

  Radiator_eqFunction_5(data, threadData);

  Radiator_eqFunction_6(data, threadData);

  Radiator_eqFunction_7(data, threadData);

  Radiator_eqFunction_8(data, threadData);

  Radiator_eqFunction_9(data, threadData);

  Radiator_eqFunction_10(data, threadData);

  Radiator_eqFunction_11(data, threadData);

  Radiator_eqFunction_12(data, threadData);

  Radiator_eqFunction_13(data, threadData);

  Radiator_eqFunction_14(data, threadData);

  Radiator_eqFunction_15(data, threadData);

  Radiator_eqFunction_16(data, threadData);

  Radiator_eqFunction_17(data, threadData);

  Radiator_eqFunction_18(data, threadData);

  Radiator_eqFunction_19(data, threadData);

  Radiator_eqFunction_20(data, threadData);

  Radiator_eqFunction_21(data, threadData);

  Radiator_eqFunction_22(data, threadData);

  Radiator_eqFunction_23(data, threadData);

  Radiator_eqFunction_24(data, threadData);

  Radiator_eqFunction_25(data, threadData);

  Radiator_eqFunction_26(data, threadData);

  Radiator_eqFunction_27(data, threadData);

  Radiator_eqFunction_28(data, threadData);

  Radiator_eqFunction_29(data, threadData);

  Radiator_eqFunction_30(data, threadData);

  Radiator_eqFunction_31(data, threadData);

  Radiator_eqFunction_32(data, threadData);

  Radiator_eqFunction_33(data, threadData);

  Radiator_eqFunction_34(data, threadData);

  Radiator_eqFunction_35(data, threadData);

  Radiator_eqFunction_36(data, threadData);

  Radiator_eqFunction_37(data, threadData);

  Radiator_eqFunction_38(data, threadData);

  Radiator_eqFunction_325(data, threadData);

  Radiator_eqFunction_40(data, threadData);

  Radiator_eqFunction_41(data, threadData);

  Radiator_eqFunction_42(data, threadData);

  Radiator_eqFunction_43(data, threadData);

  Radiator_eqFunction_382(data, threadData);

  Radiator_eqFunction_388(data, threadData);

  Radiator_eqFunction_326(data, threadData);

  Radiator_eqFunction_327(data, threadData);

  Radiator_eqFunction_328(data, threadData);

  Radiator_eqFunction_49(data, threadData);

  Radiator_eqFunction_224(data, threadData);

  Radiator_eqFunction_62(data, threadData);

  Radiator_eqFunction_63(data, threadData);

  Radiator_eqFunction_64(data, threadData);

  Radiator_eqFunction_65(data, threadData);

  Radiator_eqFunction_66(data, threadData);

  Radiator_eqFunction_67(data, threadData);

  Radiator_eqFunction_68(data, threadData);

  Radiator_eqFunction_69(data, threadData);

  Radiator_eqFunction_70(data, threadData);

  Radiator_eqFunction_71(data, threadData);

  Radiator_eqFunction_72(data, threadData);

  Radiator_eqFunction_73(data, threadData);

  Radiator_eqFunction_74(data, threadData);

  Radiator_eqFunction_372(data, threadData);

  Radiator_eqFunction_76(data, threadData);

  Radiator_eqFunction_373(data, threadData);

  Radiator_eqFunction_389(data, threadData);

  Radiator_eqFunction_242(data, threadData);

  Radiator_eqFunction_396(data, threadData);

  Radiator_eqFunction_244(data, threadData);

  Radiator_eqFunction_394(data, threadData);

  Radiator_eqFunction_83(data, threadData);

  Radiator_eqFunction_84(data, threadData);

  Radiator_eqFunction_85(data, threadData);

  Radiator_eqFunction_86(data, threadData);

  Radiator_eqFunction_87(data, threadData);

  Radiator_eqFunction_88(data, threadData);

  Radiator_eqFunction_361(data, threadData);

  Radiator_eqFunction_397(data, threadData);

  Radiator_eqFunction_254(data, threadData);

  Radiator_eqFunction_404(data, threadData);

  Radiator_eqFunction_256(data, threadData);

  Radiator_eqFunction_402(data, threadData);

  Radiator_eqFunction_95(data, threadData);

  Radiator_eqFunction_380(data, threadData);

  Radiator_eqFunction_395(data, threadData);

  Radiator_eqFunction_98(data, threadData);

  Radiator_eqFunction_99(data, threadData);

  Radiator_eqFunction_100(data, threadData);

  Radiator_eqFunction_101(data, threadData);

  Radiator_eqFunction_102(data, threadData);

  Radiator_eqFunction_351(data, threadData);

  Radiator_eqFunction_405(data, threadData);

  Radiator_eqFunction_268(data, threadData);

  Radiator_eqFunction_412(data, threadData);

  Radiator_eqFunction_270(data, threadData);

  Radiator_eqFunction_410(data, threadData);

  Radiator_eqFunction_109(data, threadData);

  Radiator_eqFunction_377(data, threadData);

  Radiator_eqFunction_403(data, threadData);

  Radiator_eqFunction_112(data, threadData);

  Radiator_eqFunction_113(data, threadData);

  Radiator_eqFunction_114(data, threadData);

  Radiator_eqFunction_115(data, threadData);

  Radiator_eqFunction_116(data, threadData);

  Radiator_eqFunction_342(data, threadData);

  Radiator_eqFunction_413(data, threadData);

  Radiator_eqFunction_282(data, threadData);

  Radiator_eqFunction_420(data, threadData);

  Radiator_eqFunction_284(data, threadData);

  Radiator_eqFunction_418(data, threadData);

  Radiator_eqFunction_123(data, threadData);

  Radiator_eqFunction_363(data, threadData);

  Radiator_eqFunction_411(data, threadData);

  Radiator_eqFunction_126(data, threadData);

  Radiator_eqFunction_127(data, threadData);

  Radiator_eqFunction_128(data, threadData);

  Radiator_eqFunction_129(data, threadData);

  Radiator_eqFunction_130(data, threadData);

  Radiator_eqFunction_336(data, threadData);

  Radiator_eqFunction_421(data, threadData);

  Radiator_eqFunction_296(data, threadData);

  Radiator_eqFunction_427(data, threadData);

  Radiator_eqFunction_434(data, threadData);

  Radiator_eqFunction_299(data, threadData);

  Radiator_eqFunction_424(data, threadData);

  Radiator_eqFunction_428(data, threadData);

  Radiator_eqFunction_429(data, threadData);

  Radiator_eqFunction_430(data, threadData);

  Radiator_eqFunction_431(data, threadData);

  Radiator_eqFunction_432(data, threadData);

  Radiator_eqFunction_143(data, threadData);

  Radiator_eqFunction_353(data, threadData);

  Radiator_eqFunction_419(data, threadData);

  Radiator_eqFunction_146(data, threadData);

  Radiator_eqFunction_385(data, threadData);

  Radiator_eqFunction_386(data, threadData);

  Radiator_eqFunction_433(data, threadData);

  Radiator_eqFunction_150(data, threadData);

  Radiator_eqFunction_387(data, threadData);

  Radiator_eqFunction_152(data, threadData);

  Radiator_eqFunction_153(data, threadData);

  Radiator_eqFunction_154(data, threadData);

  Radiator_eqFunction_155(data, threadData);

  Radiator_eqFunction_156(data, threadData);

  Radiator_eqFunction_157(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}
int Radiator_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  TRACE_POP
  return 0;
}


#if defined(__cplusplus)
}
#endif

