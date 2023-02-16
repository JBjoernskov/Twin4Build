/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "Radiator_model.h"
#if defined(__cplusplus)
extern "C" {
#endif


/*
equation index: 450
type: SIMPLE_ASSIGN
$START.Radiator.vol[5].dynBal.U = Radiator.vol[5].dynBal.fluidVolume * 995.586 * Radiator.Radiator.vol.dynBal.Medium.specificInternalEnergy(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(300000.0, 293.15, {})) + 20.0 * Radiator.vol[5].dynBal.CSen
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_450(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,450};
  data->modelData->realVarsData[5].attribute /* Radiator.vol[5].dynBal.U STATE(1) */.start = ((data->simulationInfo->realParameter[136] /* Radiator.vol[5].dynBal.fluidVolume PARAM */) * (995.586)) * (omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, 300000.0, 293.15, _OMC_LIT36))) + (20.0) * (data->simulationInfo->realParameter[116] /* Radiator.vol[5].dynBal.CSen PARAM */);
    data->localData[0]->realVars[5] /* Radiator.vol[5].dynBal.U STATE(1) */ = data->modelData->realVarsData[5].attribute /* Radiator.vol[5].dynBal.U STATE(1) */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[5].info /* Radiator.vol[5].dynBal.U */.name, (modelica_real) data->localData[0]->realVars[5] /* Radiator.vol[5].dynBal.U STATE(1) */);
  TRACE_POP
}

/*
equation index: 451
type: SIMPLE_ASSIGN
$START.Radiator.vol[4].dynBal.U = Radiator.vol[4].dynBal.fluidVolume * 995.586 * Radiator.Radiator.vol.dynBal.Medium.specificInternalEnergy(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(300000.0, 293.15, {})) + 20.0 * Radiator.vol[4].dynBal.CSen
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,451};
  data->modelData->realVarsData[4].attribute /* Radiator.vol[4].dynBal.U STATE(1) */.start = ((data->simulationInfo->realParameter[135] /* Radiator.vol[4].dynBal.fluidVolume PARAM */) * (995.586)) * (omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, 300000.0, 293.15, _OMC_LIT36))) + (20.0) * (data->simulationInfo->realParameter[115] /* Radiator.vol[4].dynBal.CSen PARAM */);
    data->localData[0]->realVars[4] /* Radiator.vol[4].dynBal.U STATE(1) */ = data->modelData->realVarsData[4].attribute /* Radiator.vol[4].dynBal.U STATE(1) */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[4].info /* Radiator.vol[4].dynBal.U */.name, (modelica_real) data->localData[0]->realVars[4] /* Radiator.vol[4].dynBal.U STATE(1) */);
  TRACE_POP
}

/*
equation index: 452
type: SIMPLE_ASSIGN
$START.Radiator.vol[3].dynBal.U = Radiator.vol[3].dynBal.fluidVolume * 995.586 * Radiator.Radiator.vol.dynBal.Medium.specificInternalEnergy(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(300000.0, 293.15, {})) + 20.0 * Radiator.vol[3].dynBal.CSen
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,452};
  data->modelData->realVarsData[3].attribute /* Radiator.vol[3].dynBal.U STATE(1) */.start = ((data->simulationInfo->realParameter[134] /* Radiator.vol[3].dynBal.fluidVolume PARAM */) * (995.586)) * (omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, 300000.0, 293.15, _OMC_LIT36))) + (20.0) * (data->simulationInfo->realParameter[114] /* Radiator.vol[3].dynBal.CSen PARAM */);
    data->localData[0]->realVars[3] /* Radiator.vol[3].dynBal.U STATE(1) */ = data->modelData->realVarsData[3].attribute /* Radiator.vol[3].dynBal.U STATE(1) */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[3].info /* Radiator.vol[3].dynBal.U */.name, (modelica_real) data->localData[0]->realVars[3] /* Radiator.vol[3].dynBal.U STATE(1) */);
  TRACE_POP
}

/*
equation index: 453
type: SIMPLE_ASSIGN
$START.Radiator.vol[2].dynBal.U = Radiator.vol[2].dynBal.fluidVolume * 995.586 * Radiator.Radiator.vol.dynBal.Medium.specificInternalEnergy(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(300000.0, 293.15, {})) + 20.0 * Radiator.vol[2].dynBal.CSen
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,453};
  data->modelData->realVarsData[2].attribute /* Radiator.vol[2].dynBal.U STATE(1) */.start = ((data->simulationInfo->realParameter[133] /* Radiator.vol[2].dynBal.fluidVolume PARAM */) * (995.586)) * (omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, 300000.0, 293.15, _OMC_LIT36))) + (20.0) * (data->simulationInfo->realParameter[113] /* Radiator.vol[2].dynBal.CSen PARAM */);
    data->localData[0]->realVars[2] /* Radiator.vol[2].dynBal.U STATE(1) */ = data->modelData->realVarsData[2].attribute /* Radiator.vol[2].dynBal.U STATE(1) */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[2].info /* Radiator.vol[2].dynBal.U */.name, (modelica_real) data->localData[0]->realVars[2] /* Radiator.vol[2].dynBal.U STATE(1) */);
  TRACE_POP
}

/*
equation index: 454
type: SIMPLE_ASSIGN
$START.Radiator.vol[1].dynBal.U = Radiator.vol[1].dynBal.fluidVolume * Radiator.vol[1].dynBal.rho_start * Radiator.Radiator.vol.dynBal.Medium.specificInternalEnergy(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(Radiator.vol[1].dynBal.p_start, Radiator.vol[1].dynBal.T_start, {})) + (Radiator.vol[1].dynBal.T_start - 273.15) * Radiator.vol[1].dynBal.CSen
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_454(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,454};
  data->modelData->realVarsData[1].attribute /* Radiator.vol[1].dynBal.U STATE(1) */.start = ((data->simulationInfo->realParameter[132] /* Radiator.vol[1].dynBal.fluidVolume PARAM */) * (data->simulationInfo->realParameter[177] /* Radiator.vol[1].dynBal.rho_start PARAM */)) * (omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */, data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */, _OMC_LIT36))) + (data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */ - 273.15) * (data->simulationInfo->realParameter[112] /* Radiator.vol[1].dynBal.CSen PARAM */);
    data->localData[0]->realVars[1] /* Radiator.vol[1].dynBal.U STATE(1) */ = data->modelData->realVarsData[1].attribute /* Radiator.vol[1].dynBal.U STATE(1) */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[1].info /* Radiator.vol[1].dynBal.U */.name, (modelica_real) data->localData[0]->realVars[1] /* Radiator.vol[1].dynBal.U STATE(1) */);
  TRACE_POP
}

/*
equation index: 455
type: SIMPLE_ASSIGN
$START.Radiator.vol[1].T = Radiator.vol[1].T_start
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,455};
  data->modelData->realVarsData[55].attribute /* Radiator.vol[1].T variable */.start = data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */;
    data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */ = data->modelData->realVarsData[55].attribute /* Radiator.vol[1].T variable */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[55].info /* Radiator.vol[1].T */.name, (modelica_real) data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */);
  TRACE_POP
}

/*
equation index: 456
type: SIMPLE_ASSIGN
$START.Radiator.port_a.h_outflow = Radiator.vol[1].dynBal.hStart
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_456(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,456};
  data->modelData->realVarsData[41].attribute /* Radiator.port_a.h_outflow variable */.start = data->simulationInfo->realParameter[137] /* Radiator.vol[1].dynBal.hStart PARAM */;
    data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */ = data->modelData->realVarsData[41].attribute /* Radiator.port_a.h_outflow variable */.start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[41].info /* Radiator.port_a.h_outflow */.name, (modelica_real) data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */);
  TRACE_POP
}
OMC_DISABLE_OPT
int Radiator_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  /* min ******************************************************** */
  
  infoStreamPrint(LOG_INIT, 1, "updating min-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* max ******************************************************** */
  
  infoStreamPrint(LOG_INIT, 1, "updating max-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* nominal **************************************************** */
  
  infoStreamPrint(LOG_INIT, 1, "updating nominal-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(LOG_INIT, 1, "updating primary start-values");
  Radiator_eqFunction_450(data, threadData);

  Radiator_eqFunction_451(data, threadData);

  Radiator_eqFunction_452(data, threadData);

  Radiator_eqFunction_453(data, threadData);

  Radiator_eqFunction_454(data, threadData);

  Radiator_eqFunction_455(data, threadData);

  Radiator_eqFunction_456(data, threadData);
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  TRACE_POP
  return 0;
}

void Radiator_updateBoundParameters_0(DATA *data, threadData_t *threadData);

/*
equation index: 457
type: SIMPLE_ASSIGN
Radiator.sta_a.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,457};
  data->simulationInfo->realParameter[85] /* Radiator.sta_a.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 458
type: SIMPLE_ASSIGN
Radiator.res.port_a.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_458(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,458};
  data->simulationInfo->realParameter[81] /* Radiator.res.port_a.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 459
type: SIMPLE_ASSIGN
flow_source.p_in_internal = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,459};
  data->simulationInfo->realParameter[268] /* flow_source.p_in_internal PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 460
type: SIMPLE_ASSIGN
flow_source.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,460};
  data->simulationInfo->realParameter[269] /* flow_source.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 461
type: SIMPLE_ASSIGN
Radiator.port_a.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,461};
  data->simulationInfo->realParameter[43] /* Radiator.port_a.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 462
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.state.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_462(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,462};
  data->simulationInfo->realParameter[156] /* Radiator.vol[5].dynBal.medium.state.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 463
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.state.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,463};
  data->simulationInfo->realParameter[155] /* Radiator.vol[4].dynBal.medium.state.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 464
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.state.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_464(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,464};
  data->simulationInfo->realParameter[154] /* Radiator.vol[3].dynBal.medium.state.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 465
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.state.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,465};
  data->simulationInfo->realParameter[153] /* Radiator.vol[2].dynBal.medium.state.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 466
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.state.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_466(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,466};
  data->simulationInfo->realParameter[152] /* Radiator.vol[1].dynBal.medium.state.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 467
type: SIMPLE_ASSIGN
Radiator.res.port_b.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,467};
  data->simulationInfo->realParameter[82] /* Radiator.res.port_b.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 468
type: SIMPLE_ASSIGN
Radiator.vol[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,468};
  data->simulationInfo->realParameter[207] /* Radiator.vol[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 469
type: SIMPLE_ASSIGN
Radiator.vol[1].ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,469};
  data->simulationInfo->realParameter[217] /* Radiator.vol[1].ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 470
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_470(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,470};
  data->simulationInfo->realParameter[162] /* Radiator.vol[1].dynBal.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 471
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,471};
  data->simulationInfo->realParameter[147] /* Radiator.vol[1].dynBal.medium.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 472
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_472(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,472};
  data->simulationInfo->realParameter[163] /* Radiator.vol[1].dynBal.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 473
type: SIMPLE_ASSIGN
Radiator.vol[1].ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,473};
  data->simulationInfo->realParameter[218] /* Radiator.vol[1].ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 474
type: SIMPLE_ASSIGN
Radiator.vol[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_474(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,474};
  data->simulationInfo->realParameter[208] /* Radiator.vol[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 475
type: SIMPLE_ASSIGN
Radiator.vol[2].ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,475};
  data->simulationInfo->realParameter[219] /* Radiator.vol[2].ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 476
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,476};
  data->simulationInfo->realParameter[164] /* Radiator.vol[2].dynBal.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 477
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,477};
  data->simulationInfo->realParameter[148] /* Radiator.vol[2].dynBal.medium.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 478
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_478(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,478};
  data->simulationInfo->realParameter[165] /* Radiator.vol[2].dynBal.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 479
type: SIMPLE_ASSIGN
Radiator.vol[2].ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,479};
  data->simulationInfo->realParameter[220] /* Radiator.vol[2].ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 480
type: SIMPLE_ASSIGN
Radiator.vol[3].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_480(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,480};
  data->simulationInfo->realParameter[209] /* Radiator.vol[3].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 481
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,481};
  data->simulationInfo->realParameter[221] /* Radiator.vol[3].ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 482
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_482(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,482};
  data->simulationInfo->realParameter[166] /* Radiator.vol[3].dynBal.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 483
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,483};
  data->simulationInfo->realParameter[149] /* Radiator.vol[3].dynBal.medium.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 484
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,484};
  data->simulationInfo->realParameter[167] /* Radiator.vol[3].dynBal.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 485
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,485};
  data->simulationInfo->realParameter[222] /* Radiator.vol[3].ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 486
type: SIMPLE_ASSIGN
Radiator.vol[4].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_486(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,486};
  data->simulationInfo->realParameter[210] /* Radiator.vol[4].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 487
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,487};
  data->simulationInfo->realParameter[223] /* Radiator.vol[4].ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 488
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_488(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,488};
  data->simulationInfo->realParameter[168] /* Radiator.vol[4].dynBal.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 489
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,489};
  data->simulationInfo->realParameter[150] /* Radiator.vol[4].dynBal.medium.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 490
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_490(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,490};
  data->simulationInfo->realParameter[169] /* Radiator.vol[4].dynBal.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 491
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,491};
  data->simulationInfo->realParameter[224] /* Radiator.vol[4].ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 492
type: SIMPLE_ASSIGN
Radiator.vol[5].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,492};
  data->simulationInfo->realParameter[211] /* Radiator.vol[5].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 493
type: SIMPLE_ASSIGN
Radiator.vol[5].ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,493};
  data->simulationInfo->realParameter[225] /* Radiator.vol[5].ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 494
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_494(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,494};
  data->simulationInfo->realParameter[170] /* Radiator.vol[5].dynBal.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 495
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,495};
  data->simulationInfo->realParameter[151] /* Radiator.vol[5].dynBal.medium.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 496
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_496(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,496};
  data->simulationInfo->realParameter[171] /* Radiator.vol[5].dynBal.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 497
type: SIMPLE_ASSIGN
Radiator.vol[5].ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,497};
  data->simulationInfo->realParameter[226] /* Radiator.vol[5].ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 498
type: SIMPLE_ASSIGN
Radiator.sta_b.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_498(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,498};
  data->simulationInfo->realParameter[86] /* Radiator.sta_b.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 499
type: SIMPLE_ASSIGN
Radiator.port_b.p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,499};
  data->simulationInfo->realParameter[44] /* Radiator.port_b.p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 500
type: SIMPLE_ASSIGN
flow_sink.ports[1].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,500};
  data->simulationInfo->realParameter[263] /* flow_sink.ports[1].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 501
type: SIMPLE_ASSIGN
flow_sink.ports[2].p = flow_sink.p
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,501};
  data->simulationInfo->realParameter[264] /* flow_sink.ports[2].p PARAM */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}

/*
equation index: 515
type: SIMPLE_ASSIGN
Radiator.Q_flow_nominal = Q_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,515};
  data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */ = data->simulationInfo->realParameter[0] /* Q_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 516
type: SIMPLE_ASSIGN
Radiator.T_a_nominal = T_a_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,516};
  data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ = data->simulationInfo->realParameter[258] /* T_a_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 518
type: SIMPLE_ASSIGN
Radiator.m_flow_nominal = abs(Radiator.Q_flow_nominal / (Radiator.cp_nominal * (Radiator.T_a_nominal - 303.15)))
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_518(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,518};
  data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */ = fabs(DIVISION_SIM(data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */,(data->simulationInfo->realParameter[22] /* Radiator.cp_nominal PARAM */) * (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */ - 303.15),"Radiator.cp_nominal * (Radiator.T_a_nominal - 303.15)",equationIndexes));
  TRACE_POP
}

/*
equation index: 519
type: SIMPLE_ASSIGN
Radiator.res.m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,519};
  data->simulationInfo->realParameter[77] /* Radiator.res.m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 520
type: SIMPLE_ASSIGN
Radiator.res.m_flow_nominal_pos = abs(Radiator.res.m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_520(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,520};
  data->simulationInfo->realParameter[78] /* Radiator.res.m_flow_nominal_pos PARAM */ = fabs(data->simulationInfo->realParameter[77] /* Radiator.res.m_flow_nominal PARAM */);
  TRACE_POP
}

/*
equation index: 523
type: SIMPLE_ASSIGN
Radiator.res.eta_default = Radiator.Radiator.res.Medium.dynamicViscosity(Radiator.res.sta_default)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,523};
  data->simulationInfo->realParameter[75] /* Radiator.res.eta_default PARAM */ = omc_Radiator_Radiator_res_Medium_dynamicViscosity(threadData, omc_Radiator_Radiator_res_Medium_ThermodynamicState(threadData, data->simulationInfo->realParameter[84] /* Radiator.res.sta_default.p PARAM */, data->simulationInfo->realParameter[83] /* Radiator.res.sta_default.T PARAM */));
  TRACE_POP
}

/*
equation index: 531
type: SIMPLE_ASSIGN
Radiator.res.m_flow_small = 0.0001 * abs(Radiator.res.m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,531};
  data->simulationInfo->realParameter[79] /* Radiator.res.m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[77] /* Radiator.res.m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 585
type: SIMPLE_ASSIGN
Radiator.VWat = 5.8e-06 * abs(Radiator.Q_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,585};
  data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */ = (5.8e-06) * (fabs(data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 586
type: SIMPLE_ASSIGN
Radiator.vol[5].V = 0.2 * Radiator.VWat
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_586(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,586};
  data->simulationInfo->realParameter[106] /* Radiator.vol[5].V PARAM */ = (0.2) * (data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */);
  TRACE_POP
}

/*
equation index: 587
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.fluidVolume = Radiator.vol[5].V
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,587};
  data->simulationInfo->realParameter[136] /* Radiator.vol[5].dynBal.fluidVolume PARAM */ = data->simulationInfo->realParameter[106] /* Radiator.vol[5].V PARAM */;
  TRACE_POP
}

/*
equation index: 588
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.CSen = 2267241.379310345 * Radiator.vol[5].dynBal.fluidVolume
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,588};
  data->simulationInfo->realParameter[116] /* Radiator.vol[5].dynBal.CSen PARAM */ = (2267241.379310345) * (data->simulationInfo->realParameter[136] /* Radiator.vol[5].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 605
type: SIMPLE_ASSIGN
Radiator.vol[5].m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,605};
  data->simulationInfo->realParameter[201] /* Radiator.vol[5].m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 606
type: SIMPLE_ASSIGN
Radiator.vol[5].m_flow_small = 0.0001 * abs(Radiator.vol[5].m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_606(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,606};
  data->simulationInfo->realParameter[206] /* Radiator.vol[5].m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[201] /* Radiator.vol[5].m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 633
type: SIMPLE_ASSIGN
Radiator.vol[4].V = 0.2 * Radiator.VWat
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,633};
  data->simulationInfo->realParameter[105] /* Radiator.vol[4].V PARAM */ = (0.2) * (data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */);
  TRACE_POP
}

/*
equation index: 634
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.fluidVolume = Radiator.vol[4].V
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_634(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,634};
  data->simulationInfo->realParameter[135] /* Radiator.vol[4].dynBal.fluidVolume PARAM */ = data->simulationInfo->realParameter[105] /* Radiator.vol[4].V PARAM */;
  TRACE_POP
}

/*
equation index: 635
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.CSen = 2267241.379310345 * Radiator.vol[4].dynBal.fluidVolume
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,635};
  data->simulationInfo->realParameter[115] /* Radiator.vol[4].dynBal.CSen PARAM */ = (2267241.379310345) * (data->simulationInfo->realParameter[135] /* Radiator.vol[4].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 652
type: SIMPLE_ASSIGN
Radiator.vol[4].m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,652};
  data->simulationInfo->realParameter[200] /* Radiator.vol[4].m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 653
type: SIMPLE_ASSIGN
Radiator.vol[4].m_flow_small = 0.0001 * abs(Radiator.vol[4].m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,653};
  data->simulationInfo->realParameter[205] /* Radiator.vol[4].m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[200] /* Radiator.vol[4].m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 680
type: SIMPLE_ASSIGN
Radiator.vol[3].V = 0.2 * Radiator.VWat
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_680(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,680};
  data->simulationInfo->realParameter[104] /* Radiator.vol[3].V PARAM */ = (0.2) * (data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */);
  TRACE_POP
}

/*
equation index: 681
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.fluidVolume = Radiator.vol[3].V
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,681};
  data->simulationInfo->realParameter[134] /* Radiator.vol[3].dynBal.fluidVolume PARAM */ = data->simulationInfo->realParameter[104] /* Radiator.vol[3].V PARAM */;
  TRACE_POP
}

/*
equation index: 682
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.CSen = 2267241.379310345 * Radiator.vol[3].dynBal.fluidVolume
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_682(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,682};
  data->simulationInfo->realParameter[114] /* Radiator.vol[3].dynBal.CSen PARAM */ = (2267241.379310345) * (data->simulationInfo->realParameter[134] /* Radiator.vol[3].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 699
type: SIMPLE_ASSIGN
Radiator.vol[3].m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,699};
  data->simulationInfo->realParameter[199] /* Radiator.vol[3].m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 700
type: SIMPLE_ASSIGN
Radiator.vol[3].m_flow_small = 0.0001 * abs(Radiator.vol[3].m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,700};
  data->simulationInfo->realParameter[204] /* Radiator.vol[3].m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[199] /* Radiator.vol[3].m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 727
type: SIMPLE_ASSIGN
Radiator.vol[2].V = 0.2 * Radiator.VWat
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,727};
  data->simulationInfo->realParameter[103] /* Radiator.vol[2].V PARAM */ = (0.2) * (data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */);
  TRACE_POP
}

/*
equation index: 728
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.fluidVolume = Radiator.vol[2].V
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_728(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,728};
  data->simulationInfo->realParameter[133] /* Radiator.vol[2].dynBal.fluidVolume PARAM */ = data->simulationInfo->realParameter[103] /* Radiator.vol[2].V PARAM */;
  TRACE_POP
}

/*
equation index: 729
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.CSen = 2267241.379310345 * Radiator.vol[2].dynBal.fluidVolume
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,729};
  data->simulationInfo->realParameter[113] /* Radiator.vol[2].dynBal.CSen PARAM */ = (2267241.379310345) * (data->simulationInfo->realParameter[133] /* Radiator.vol[2].dynBal.fluidVolume PARAM */);
  TRACE_POP
}

/*
equation index: 746
type: SIMPLE_ASSIGN
Radiator.vol[2].m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_746(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,746};
  data->simulationInfo->realParameter[198] /* Radiator.vol[2].m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 747
type: SIMPLE_ASSIGN
Radiator.vol[2].m_flow_small = 0.0001 * abs(Radiator.vol[2].m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,747};
  data->simulationInfo->realParameter[203] /* Radiator.vol[2].m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[198] /* Radiator.vol[2].m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 767
type: SIMPLE_ASSIGN
Radiator.vol[1].p_start = Radiator.p_start
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,767};
  data->simulationInfo->realParameter[212] /* Radiator.vol[1].p_start PARAM */ = data->simulationInfo->realParameter[42] /* Radiator.p_start PARAM */;
  TRACE_POP
}

/*
equation index: 768
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.p_start = Radiator.vol[1].p_start
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_768(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,768};
  data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */ = data->simulationInfo->realParameter[212] /* Radiator.vol[1].p_start PARAM */;
  TRACE_POP
}

/*
equation index: 769
type: SIMPLE_ASSIGN
Radiator.vol[1].T_start = Radiator.T_start
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,769};
  data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */ = data->simulationInfo->realParameter[16] /* Radiator.T_start PARAM */;
  TRACE_POP
}

/*
equation index: 770
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.T_start = Radiator.vol[1].T_start
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_770(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,770};
  data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */ = data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */;
  TRACE_POP
}

/*
equation index: 771
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.hStart = Radiator.Radiator.vol.dynBal.Medium.specificEnthalpy_pTX(Radiator.vol[1].dynBal.p_start, Radiator.vol[1].dynBal.T_start, {1.0})
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,771};
  data->simulationInfo->realParameter[137] /* Radiator.vol[1].dynBal.hStart PARAM */ = omc_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData, data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */, data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */, _OMC_LIT24);
  TRACE_POP
}

/*
equation index: 776
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.rho_start = Radiator.Radiator.vol.dynBal.Medium.density(Radiator.Radiator.vol.dynBal.Medium.setState_pTX(Radiator.vol[1].dynBal.p_start, Radiator.vol[1].dynBal.T_start, {}))
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_776(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,776};
  data->simulationInfo->realParameter[177] /* Radiator.vol[1].dynBal.rho_start PARAM */ = omc_Radiator_Radiator_vol_dynBal_Medium_density(threadData, omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */, data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */, _OMC_LIT36));
  TRACE_POP
}

/*
equation index: 778
type: SIMPLE_ASSIGN
Radiator.mDry = 0.0263 * abs(Radiator.Q_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_778(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,778};
  data->simulationInfo->realParameter[37] /* Radiator.mDry PARAM */ = (0.0263) * (fabs(data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 779
type: SIMPLE_ASSIGN
Radiator.mSenFac = 1.0 + 500.0 * Radiator.mDry / (Radiator.VWat * Radiator.cp_nominal * 995.586)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,779};
  data->simulationInfo->realParameter[38] /* Radiator.mSenFac PARAM */ = 1.0 + DIVISION_SIM((500.0) * (data->simulationInfo->realParameter[37] /* Radiator.mDry PARAM */),((data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */) * (data->simulationInfo->realParameter[22] /* Radiator.cp_nominal PARAM */)) * (995.586),"Radiator.VWat * Radiator.cp_nominal * 995.586",equationIndexes);
  TRACE_POP
}

/*
equation index: 780
type: SIMPLE_ASSIGN
Radiator.vol[1].mSenFac = Radiator.mSenFac
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,780};
  data->simulationInfo->realParameter[192] /* Radiator.vol[1].mSenFac PARAM */ = data->simulationInfo->realParameter[38] /* Radiator.mSenFac PARAM */;
  TRACE_POP
}

/*
equation index: 781
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.mSenFac = Radiator.vol[1].mSenFac
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,781};
  data->simulationInfo->realParameter[142] /* Radiator.vol[1].dynBal.mSenFac PARAM */ = data->simulationInfo->realParameter[192] /* Radiator.vol[1].mSenFac PARAM */;
  TRACE_POP
}

/*
equation index: 782
type: SIMPLE_ASSIGN
Radiator.vol[1].V = 0.2 * Radiator.VWat
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_782(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,782};
  data->simulationInfo->realParameter[102] /* Radiator.vol[1].V PARAM */ = (0.2) * (data->simulationInfo->realParameter[18] /* Radiator.VWat PARAM */);
  TRACE_POP
}

/*
equation index: 783
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.fluidVolume = Radiator.vol[1].V
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,783};
  data->simulationInfo->realParameter[132] /* Radiator.vol[1].dynBal.fluidVolume PARAM */ = data->simulationInfo->realParameter[102] /* Radiator.vol[1].V PARAM */;
  TRACE_POP
}

/*
equation index: 784
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.CSen = 4165531.824 * (-1.0 + Radiator.vol[1].dynBal.mSenFac) * Radiator.vol[1].dynBal.fluidVolume
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_784(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,784};
  data->simulationInfo->realParameter[112] /* Radiator.vol[1].dynBal.CSen PARAM */ = (4165531.824) * ((-1.0 + data->simulationInfo->realParameter[142] /* Radiator.vol[1].dynBal.mSenFac PARAM */) * (data->simulationInfo->realParameter[132] /* Radiator.vol[1].dynBal.fluidVolume PARAM */));
  TRACE_POP
}

/*
equation index: 798
type: SIMPLE_ASSIGN
Radiator.vol[1].m_flow_nominal = Radiator.m_flow_nominal
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_798(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,798};
  data->simulationInfo->realParameter[197] /* Radiator.vol[1].m_flow_nominal PARAM */ = data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */;
  TRACE_POP
}

/*
equation index: 799
type: SIMPLE_ASSIGN
Radiator.vol[1].m_flow_small = 0.0001 * abs(Radiator.vol[1].m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,799};
  data->simulationInfo->realParameter[202] /* Radiator.vol[1].m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[197] /* Radiator.vol[1].m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 824
type: SIMPLE_ASSIGN
Radiator.m_flow_small = 0.0001 * abs(Radiator.m_flow_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_824(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,824};
  data->simulationInfo->realParameter[40] /* Radiator.m_flow_small PARAM */ = (0.0001) * (fabs(data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */));
  TRACE_POP
}

/*
equation index: 834
type: SIMPLE_ASSIGN
m_flow_nominal = 0.0002390057361376673 * Q_flow_nominal / (T_a_nominal - T_b_nominal)
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_834(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,834};
  data->simulationInfo->realParameter[270] /* m_flow_nominal PARAM */ = (0.0002390057361376673) * (DIVISION_SIM(data->simulationInfo->realParameter[0] /* Q_flow_nominal PARAM */,data->simulationInfo->realParameter[258] /* T_a_nominal PARAM */ - data->simulationInfo->realParameter[259] /* T_b_nominal PARAM */,"T_a_nominal - T_b_nominal",equationIndexes));
  TRACE_POP
}
extern void Radiator_eqFunction_157(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_156(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_155(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_154(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_153(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_152(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_38(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_37(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_36(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_35(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_34(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_33(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_32(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_31(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_30(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_29(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_28(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_27(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_26(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_25(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_24(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_23(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_22(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_21(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_20(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_19(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_18(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_17(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_16(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_15(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_14(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_13(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_12(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_11(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_10(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_9(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_8(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_7(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_6(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_5(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_4(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_3(DATA *data, threadData_t *threadData);

extern void Radiator_eqFunction_2(DATA *data, threadData_t *threadData);


/*
equation index: 878
type: ALGORITHM

  assert(flow_sink.p >= 0.0 and flow_sink.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= flow_sink.p <= 100000000.0, has value: " + String(flow_sink.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_878(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,878};
  modelica_boolean tmp0;
  modelica_boolean tmp1;
  static const MMC_DEFSTRINGLIT(tmp2,85,"Variable violating min/max constraint: 0.0 <= flow_sink.p <= 100000000.0, has value: ");
  modelica_string tmp3;
  static int tmp4 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp4)
  {
    tmp0 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    tmp1 = LessEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,100000000.0);
    if(!(tmp0 && tmp1))
    {
      tmp3 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp2),tmp3);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/Boundary_pT.mo",9,3,11,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.p >= 0.0 and flow_sink.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp4 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 879
type: ALGORITHM

  assert(Radiator.sta_a.p >= 0.0 and Radiator.sta_a.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.sta_a.p <= 100000000.0, has value: " + String(Radiator.sta_a.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,879};
  modelica_boolean tmp5;
  modelica_boolean tmp6;
  static const MMC_DEFSTRINGLIT(tmp7,90,"Variable violating min/max constraint: 0.0 <= Radiator.sta_a.p <= 100000000.0, has value: ");
  modelica_string tmp8;
  static int tmp9 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp9)
  {
    tmp5 = GreaterEq(data->simulationInfo->realParameter[85] /* Radiator.sta_a.p PARAM */,0.0);
    tmp6 = LessEq(data->simulationInfo->realParameter[85] /* Radiator.sta_a.p PARAM */,100000000.0);
    if(!(tmp5 && tmp6))
    {
      tmp8 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[85] /* Radiator.sta_a.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp7),tmp8);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.sta_a.p >= 0.0 and Radiator.sta_a.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp9 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 880
type: ALGORITHM

  assert(Radiator.res.port_a.p >= 0.0 and Radiator.res.port_a.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.res.port_a.p <= 100000000.0, has value: " + String(Radiator.res.port_a.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_880(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,880};
  modelica_boolean tmp10;
  modelica_boolean tmp11;
  static const MMC_DEFSTRINGLIT(tmp12,95,"Variable violating min/max constraint: 0.0 <= Radiator.res.port_a.p <= 100000000.0, has value: ");
  modelica_string tmp13;
  static int tmp14 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp14)
  {
    tmp10 = GreaterEq(data->simulationInfo->realParameter[81] /* Radiator.res.port_a.p PARAM */,0.0);
    tmp11 = LessEq(data->simulationInfo->realParameter[81] /* Radiator.res.port_a.p PARAM */,100000000.0);
    if(!(tmp10 && tmp11))
    {
      tmp13 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[81] /* Radiator.res.port_a.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp12),tmp13);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.port_a.p >= 0.0 and Radiator.res.port_a.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp14 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 881
type: ALGORITHM

  assert(flow_source.ports[1].p >= 0.0 and flow_source.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= flow_source.ports[1].p <= 100000000.0, has value: " + String(flow_source.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,881};
  modelica_boolean tmp15;
  modelica_boolean tmp16;
  static const MMC_DEFSTRINGLIT(tmp17,96,"Variable violating min/max constraint: 0.0 <= flow_source.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp18;
  static int tmp19 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp19)
  {
    tmp15 = GreaterEq(data->simulationInfo->realParameter[269] /* flow_source.ports[1].p PARAM */,0.0);
    tmp16 = LessEq(data->simulationInfo->realParameter[269] /* flow_source.ports[1].p PARAM */,100000000.0);
    if(!(tmp15 && tmp16))
    {
      tmp18 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[269] /* flow_source.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp17),tmp18);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_source.ports[1].p >= 0.0 and flow_source.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp19 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 882
type: ALGORITHM

  assert(Radiator.port_a.p >= 0.0 and Radiator.port_a.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.port_a.p <= 100000000.0, has value: " + String(Radiator.port_a.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_882(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,882};
  modelica_boolean tmp20;
  modelica_boolean tmp21;
  static const MMC_DEFSTRINGLIT(tmp22,91,"Variable violating min/max constraint: 0.0 <= Radiator.port_a.p <= 100000000.0, has value: ");
  modelica_string tmp23;
  static int tmp24 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp24)
  {
    tmp20 = GreaterEq(data->simulationInfo->realParameter[43] /* Radiator.port_a.p PARAM */,0.0);
    tmp21 = LessEq(data->simulationInfo->realParameter[43] /* Radiator.port_a.p PARAM */,100000000.0);
    if(!(tmp20 && tmp21))
    {
      tmp23 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[43] /* Radiator.port_a.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp22),tmp23);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.port_a.p >= 0.0 and Radiator.port_a.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp24 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 883
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.medium.state.p >= 0.0 and Radiator.vol[5].dynBal.medium.state.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.medium.state.p <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.medium.state.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,883};
  modelica_boolean tmp25;
  modelica_boolean tmp26;
  static const MMC_DEFSTRINGLIT(tmp27,111,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.medium.state.p <= 100000000.0, has value: ");
  modelica_string tmp28;
  static int tmp29 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp29)
  {
    tmp25 = GreaterEq(data->simulationInfo->realParameter[156] /* Radiator.vol[5].dynBal.medium.state.p PARAM */,0.0);
    tmp26 = LessEq(data->simulationInfo->realParameter[156] /* Radiator.vol[5].dynBal.medium.state.p PARAM */,100000000.0);
    if(!(tmp25 && tmp26))
    {
      tmp28 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[156] /* Radiator.vol[5].dynBal.medium.state.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp27),tmp28);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.medium.state.p >= 0.0 and Radiator.vol[5].dynBal.medium.state.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp29 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 884
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.medium.state.p >= 0.0 and Radiator.vol[4].dynBal.medium.state.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.medium.state.p <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.medium.state.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,884};
  modelica_boolean tmp30;
  modelica_boolean tmp31;
  static const MMC_DEFSTRINGLIT(tmp32,111,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.medium.state.p <= 100000000.0, has value: ");
  modelica_string tmp33;
  static int tmp34 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp34)
  {
    tmp30 = GreaterEq(data->simulationInfo->realParameter[155] /* Radiator.vol[4].dynBal.medium.state.p PARAM */,0.0);
    tmp31 = LessEq(data->simulationInfo->realParameter[155] /* Radiator.vol[4].dynBal.medium.state.p PARAM */,100000000.0);
    if(!(tmp30 && tmp31))
    {
      tmp33 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[155] /* Radiator.vol[4].dynBal.medium.state.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp32),tmp33);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.medium.state.p >= 0.0 and Radiator.vol[4].dynBal.medium.state.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp34 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 885
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.medium.state.p >= 0.0 and Radiator.vol[3].dynBal.medium.state.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.medium.state.p <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.medium.state.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,885};
  modelica_boolean tmp35;
  modelica_boolean tmp36;
  static const MMC_DEFSTRINGLIT(tmp37,111,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.medium.state.p <= 100000000.0, has value: ");
  modelica_string tmp38;
  static int tmp39 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp39)
  {
    tmp35 = GreaterEq(data->simulationInfo->realParameter[154] /* Radiator.vol[3].dynBal.medium.state.p PARAM */,0.0);
    tmp36 = LessEq(data->simulationInfo->realParameter[154] /* Radiator.vol[3].dynBal.medium.state.p PARAM */,100000000.0);
    if(!(tmp35 && tmp36))
    {
      tmp38 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[154] /* Radiator.vol[3].dynBal.medium.state.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp37),tmp38);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.medium.state.p >= 0.0 and Radiator.vol[3].dynBal.medium.state.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp39 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 886
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.medium.state.p >= 0.0 and Radiator.vol[2].dynBal.medium.state.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.medium.state.p <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.medium.state.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_886(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,886};
  modelica_boolean tmp40;
  modelica_boolean tmp41;
  static const MMC_DEFSTRINGLIT(tmp42,111,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.medium.state.p <= 100000000.0, has value: ");
  modelica_string tmp43;
  static int tmp44 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp44)
  {
    tmp40 = GreaterEq(data->simulationInfo->realParameter[153] /* Radiator.vol[2].dynBal.medium.state.p PARAM */,0.0);
    tmp41 = LessEq(data->simulationInfo->realParameter[153] /* Radiator.vol[2].dynBal.medium.state.p PARAM */,100000000.0);
    if(!(tmp40 && tmp41))
    {
      tmp43 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[153] /* Radiator.vol[2].dynBal.medium.state.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp42),tmp43);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.medium.state.p >= 0.0 and Radiator.vol[2].dynBal.medium.state.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp44 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 887
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.medium.state.p >= 0.0 and Radiator.vol[1].dynBal.medium.state.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.medium.state.p <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.medium.state.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,887};
  modelica_boolean tmp45;
  modelica_boolean tmp46;
  static const MMC_DEFSTRINGLIT(tmp47,111,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.medium.state.p <= 100000000.0, has value: ");
  modelica_string tmp48;
  static int tmp49 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp49)
  {
    tmp45 = GreaterEq(data->simulationInfo->realParameter[152] /* Radiator.vol[1].dynBal.medium.state.p PARAM */,0.0);
    tmp46 = LessEq(data->simulationInfo->realParameter[152] /* Radiator.vol[1].dynBal.medium.state.p PARAM */,100000000.0);
    if(!(tmp45 && tmp46))
    {
      tmp48 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[152] /* Radiator.vol[1].dynBal.medium.state.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp47),tmp48);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.medium.state.p >= 0.0 and Radiator.vol[1].dynBal.medium.state.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp49 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 888
type: ALGORITHM

  assert(Radiator.res.port_b.p >= 0.0 and Radiator.res.port_b.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.res.port_b.p <= 100000000.0, has value: " + String(Radiator.res.port_b.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_888(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,888};
  modelica_boolean tmp50;
  modelica_boolean tmp51;
  static const MMC_DEFSTRINGLIT(tmp52,95,"Variable violating min/max constraint: 0.0 <= Radiator.res.port_b.p <= 100000000.0, has value: ");
  modelica_string tmp53;
  static int tmp54 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp54)
  {
    tmp50 = GreaterEq(data->simulationInfo->realParameter[82] /* Radiator.res.port_b.p PARAM */,0.0);
    tmp51 = LessEq(data->simulationInfo->realParameter[82] /* Radiator.res.port_b.p PARAM */,100000000.0);
    if(!(tmp50 && tmp51))
    {
      tmp53 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[82] /* Radiator.res.port_b.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp52),tmp53);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.port_b.p >= 0.0 and Radiator.res.port_b.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp54 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 889
type: ALGORITHM

  assert(Radiator.vol[1].ports[1].p >= 0.0 and Radiator.vol[1].ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[1].ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,889};
  modelica_boolean tmp55;
  modelica_boolean tmp56;
  static const MMC_DEFSTRINGLIT(tmp57,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp58;
  static int tmp59 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp59)
  {
    tmp55 = GreaterEq(data->simulationInfo->realParameter[217] /* Radiator.vol[1].ports[1].p PARAM */,0.0);
    tmp56 = LessEq(data->simulationInfo->realParameter[217] /* Radiator.vol[1].ports[1].p PARAM */,100000000.0);
    if(!(tmp55 && tmp56))
    {
      tmp58 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[217] /* Radiator.vol[1].ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp57),tmp58);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].ports[1].p >= 0.0 and Radiator.vol[1].ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp59 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 890
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.ports[1].p >= 0.0 and Radiator.vol[1].dynBal.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_890(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,890};
  modelica_boolean tmp60;
  modelica_boolean tmp61;
  static const MMC_DEFSTRINGLIT(tmp62,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp63;
  static int tmp64 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp64)
  {
    tmp60 = GreaterEq(data->simulationInfo->realParameter[162] /* Radiator.vol[1].dynBal.ports[1].p PARAM */,0.0);
    tmp61 = LessEq(data->simulationInfo->realParameter[162] /* Radiator.vol[1].dynBal.ports[1].p PARAM */,100000000.0);
    if(!(tmp60 && tmp61))
    {
      tmp63 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[162] /* Radiator.vol[1].dynBal.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp62),tmp63);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.ports[1].p >= 0.0 and Radiator.vol[1].dynBal.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp64 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 891
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.medium.p >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.medium.p, has value: " + String(Radiator.vol[1].dynBal.medium.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,891};
  modelica_boolean tmp65;
  static const MMC_DEFSTRINGLIT(tmp66,86,"Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.medium.p, has value: ");
  modelica_string tmp67;
  static int tmp68 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp68)
  {
    tmp65 = GreaterEq(data->simulationInfo->realParameter[147] /* Radiator.vol[1].dynBal.medium.p PARAM */,0.0);
    if(!tmp65)
    {
      tmp67 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[147] /* Radiator.vol[1].dynBal.medium.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp66),tmp67);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",25,5,25,58,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.medium.p >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp68 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 892
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.ports[2].p >= 0.0 and Radiator.vol[1].dynBal.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,892};
  modelica_boolean tmp69;
  modelica_boolean tmp70;
  static const MMC_DEFSTRINGLIT(tmp71,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp72;
  static int tmp73 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp73)
  {
    tmp69 = GreaterEq(data->simulationInfo->realParameter[163] /* Radiator.vol[1].dynBal.ports[2].p PARAM */,0.0);
    tmp70 = LessEq(data->simulationInfo->realParameter[163] /* Radiator.vol[1].dynBal.ports[2].p PARAM */,100000000.0);
    if(!(tmp69 && tmp70))
    {
      tmp72 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[163] /* Radiator.vol[1].dynBal.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp71),tmp72);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.ports[2].p >= 0.0 and Radiator.vol[1].dynBal.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp73 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 893
type: ALGORITHM

  assert(Radiator.vol[1].ports[2].p >= 0.0 and Radiator.vol[1].ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[1].ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,893};
  modelica_boolean tmp74;
  modelica_boolean tmp75;
  static const MMC_DEFSTRINGLIT(tmp76,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp77;
  static int tmp78 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp78)
  {
    tmp74 = GreaterEq(data->simulationInfo->realParameter[218] /* Radiator.vol[1].ports[2].p PARAM */,0.0);
    tmp75 = LessEq(data->simulationInfo->realParameter[218] /* Radiator.vol[1].ports[2].p PARAM */,100000000.0);
    if(!(tmp74 && tmp75))
    {
      tmp77 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[218] /* Radiator.vol[1].ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp76),tmp77);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].ports[2].p >= 0.0 and Radiator.vol[1].ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp78 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 894
type: ALGORITHM

  assert(Radiator.vol[2].ports[1].p >= 0.0 and Radiator.vol[2].ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[2].ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_894(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,894};
  modelica_boolean tmp79;
  modelica_boolean tmp80;
  static const MMC_DEFSTRINGLIT(tmp81,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp82;
  static int tmp83 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp83)
  {
    tmp79 = GreaterEq(data->simulationInfo->realParameter[219] /* Radiator.vol[2].ports[1].p PARAM */,0.0);
    tmp80 = LessEq(data->simulationInfo->realParameter[219] /* Radiator.vol[2].ports[1].p PARAM */,100000000.0);
    if(!(tmp79 && tmp80))
    {
      tmp82 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[219] /* Radiator.vol[2].ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp81),tmp82);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].ports[1].p >= 0.0 and Radiator.vol[2].ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp83 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 895
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.ports[1].p >= 0.0 and Radiator.vol[2].dynBal.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,895};
  modelica_boolean tmp84;
  modelica_boolean tmp85;
  static const MMC_DEFSTRINGLIT(tmp86,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp87;
  static int tmp88 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp88)
  {
    tmp84 = GreaterEq(data->simulationInfo->realParameter[164] /* Radiator.vol[2].dynBal.ports[1].p PARAM */,0.0);
    tmp85 = LessEq(data->simulationInfo->realParameter[164] /* Radiator.vol[2].dynBal.ports[1].p PARAM */,100000000.0);
    if(!(tmp84 && tmp85))
    {
      tmp87 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[164] /* Radiator.vol[2].dynBal.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp86),tmp87);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.ports[1].p >= 0.0 and Radiator.vol[2].dynBal.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp88 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 896
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.medium.p >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.medium.p, has value: " + String(Radiator.vol[2].dynBal.medium.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_896(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,896};
  modelica_boolean tmp89;
  static const MMC_DEFSTRINGLIT(tmp90,86,"Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.medium.p, has value: ");
  modelica_string tmp91;
  static int tmp92 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp92)
  {
    tmp89 = GreaterEq(data->simulationInfo->realParameter[148] /* Radiator.vol[2].dynBal.medium.p PARAM */,0.0);
    if(!tmp89)
    {
      tmp91 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[148] /* Radiator.vol[2].dynBal.medium.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp90),tmp91);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",25,5,25,58,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.medium.p >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp92 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 897
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.ports[2].p >= 0.0 and Radiator.vol[2].dynBal.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,897};
  modelica_boolean tmp93;
  modelica_boolean tmp94;
  static const MMC_DEFSTRINGLIT(tmp95,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp96;
  static int tmp97 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp97)
  {
    tmp93 = GreaterEq(data->simulationInfo->realParameter[165] /* Radiator.vol[2].dynBal.ports[2].p PARAM */,0.0);
    tmp94 = LessEq(data->simulationInfo->realParameter[165] /* Radiator.vol[2].dynBal.ports[2].p PARAM */,100000000.0);
    if(!(tmp93 && tmp94))
    {
      tmp96 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[165] /* Radiator.vol[2].dynBal.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp95),tmp96);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.ports[2].p >= 0.0 and Radiator.vol[2].dynBal.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp97 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 898
type: ALGORITHM

  assert(Radiator.vol[2].ports[2].p >= 0.0 and Radiator.vol[2].ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[2].ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_898(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,898};
  modelica_boolean tmp98;
  modelica_boolean tmp99;
  static const MMC_DEFSTRINGLIT(tmp100,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp101;
  static int tmp102 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp102)
  {
    tmp98 = GreaterEq(data->simulationInfo->realParameter[220] /* Radiator.vol[2].ports[2].p PARAM */,0.0);
    tmp99 = LessEq(data->simulationInfo->realParameter[220] /* Radiator.vol[2].ports[2].p PARAM */,100000000.0);
    if(!(tmp98 && tmp99))
    {
      tmp101 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[220] /* Radiator.vol[2].ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp100),tmp101);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].ports[2].p >= 0.0 and Radiator.vol[2].ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp102 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 899
type: ALGORITHM

  assert(Radiator.vol[3].ports[1].p >= 0.0 and Radiator.vol[3].ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[3].ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,899};
  modelica_boolean tmp103;
  modelica_boolean tmp104;
  static const MMC_DEFSTRINGLIT(tmp105,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp106;
  static int tmp107 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp107)
  {
    tmp103 = GreaterEq(data->simulationInfo->realParameter[221] /* Radiator.vol[3].ports[1].p PARAM */,0.0);
    tmp104 = LessEq(data->simulationInfo->realParameter[221] /* Radiator.vol[3].ports[1].p PARAM */,100000000.0);
    if(!(tmp103 && tmp104))
    {
      tmp106 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[221] /* Radiator.vol[3].ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp105),tmp106);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].ports[1].p >= 0.0 and Radiator.vol[3].ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp107 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 900
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.ports[1].p >= 0.0 and Radiator.vol[3].dynBal.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,900};
  modelica_boolean tmp108;
  modelica_boolean tmp109;
  static const MMC_DEFSTRINGLIT(tmp110,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp111;
  static int tmp112 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp112)
  {
    tmp108 = GreaterEq(data->simulationInfo->realParameter[166] /* Radiator.vol[3].dynBal.ports[1].p PARAM */,0.0);
    tmp109 = LessEq(data->simulationInfo->realParameter[166] /* Radiator.vol[3].dynBal.ports[1].p PARAM */,100000000.0);
    if(!(tmp108 && tmp109))
    {
      tmp111 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[166] /* Radiator.vol[3].dynBal.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp110),tmp111);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.ports[1].p >= 0.0 and Radiator.vol[3].dynBal.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp112 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 901
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.medium.p >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.medium.p, has value: " + String(Radiator.vol[3].dynBal.medium.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,901};
  modelica_boolean tmp113;
  static const MMC_DEFSTRINGLIT(tmp114,86,"Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.medium.p, has value: ");
  modelica_string tmp115;
  static int tmp116 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp116)
  {
    tmp113 = GreaterEq(data->simulationInfo->realParameter[149] /* Radiator.vol[3].dynBal.medium.p PARAM */,0.0);
    if(!tmp113)
    {
      tmp115 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[149] /* Radiator.vol[3].dynBal.medium.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp114),tmp115);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",25,5,25,58,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.medium.p >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp116 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 902
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.ports[2].p >= 0.0 and Radiator.vol[3].dynBal.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_902(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,902};
  modelica_boolean tmp117;
  modelica_boolean tmp118;
  static const MMC_DEFSTRINGLIT(tmp119,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp120;
  static int tmp121 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp121)
  {
    tmp117 = GreaterEq(data->simulationInfo->realParameter[167] /* Radiator.vol[3].dynBal.ports[2].p PARAM */,0.0);
    tmp118 = LessEq(data->simulationInfo->realParameter[167] /* Radiator.vol[3].dynBal.ports[2].p PARAM */,100000000.0);
    if(!(tmp117 && tmp118))
    {
      tmp120 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[167] /* Radiator.vol[3].dynBal.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp119),tmp120);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.ports[2].p >= 0.0 and Radiator.vol[3].dynBal.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp121 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 903
type: ALGORITHM

  assert(Radiator.vol[3].ports[2].p >= 0.0 and Radiator.vol[3].ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[3].ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,903};
  modelica_boolean tmp122;
  modelica_boolean tmp123;
  static const MMC_DEFSTRINGLIT(tmp124,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp125;
  static int tmp126 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp126)
  {
    tmp122 = GreaterEq(data->simulationInfo->realParameter[222] /* Radiator.vol[3].ports[2].p PARAM */,0.0);
    tmp123 = LessEq(data->simulationInfo->realParameter[222] /* Radiator.vol[3].ports[2].p PARAM */,100000000.0);
    if(!(tmp122 && tmp123))
    {
      tmp125 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[222] /* Radiator.vol[3].ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp124),tmp125);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].ports[2].p >= 0.0 and Radiator.vol[3].ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp126 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 904
type: ALGORITHM

  assert(Radiator.vol[4].ports[1].p >= 0.0 and Radiator.vol[4].ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[4].ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_904(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,904};
  modelica_boolean tmp127;
  modelica_boolean tmp128;
  static const MMC_DEFSTRINGLIT(tmp129,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp130;
  static int tmp131 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp131)
  {
    tmp127 = GreaterEq(data->simulationInfo->realParameter[223] /* Radiator.vol[4].ports[1].p PARAM */,0.0);
    tmp128 = LessEq(data->simulationInfo->realParameter[223] /* Radiator.vol[4].ports[1].p PARAM */,100000000.0);
    if(!(tmp127 && tmp128))
    {
      tmp130 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[223] /* Radiator.vol[4].ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp129),tmp130);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].ports[1].p >= 0.0 and Radiator.vol[4].ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp131 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 905
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.ports[1].p >= 0.0 and Radiator.vol[4].dynBal.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,905};
  modelica_boolean tmp132;
  modelica_boolean tmp133;
  static const MMC_DEFSTRINGLIT(tmp134,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp135;
  static int tmp136 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp136)
  {
    tmp132 = GreaterEq(data->simulationInfo->realParameter[168] /* Radiator.vol[4].dynBal.ports[1].p PARAM */,0.0);
    tmp133 = LessEq(data->simulationInfo->realParameter[168] /* Radiator.vol[4].dynBal.ports[1].p PARAM */,100000000.0);
    if(!(tmp132 && tmp133))
    {
      tmp135 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[168] /* Radiator.vol[4].dynBal.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp134),tmp135);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.ports[1].p >= 0.0 and Radiator.vol[4].dynBal.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp136 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 906
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.medium.p >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.medium.p, has value: " + String(Radiator.vol[4].dynBal.medium.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_906(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,906};
  modelica_boolean tmp137;
  static const MMC_DEFSTRINGLIT(tmp138,86,"Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.medium.p, has value: ");
  modelica_string tmp139;
  static int tmp140 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp140)
  {
    tmp137 = GreaterEq(data->simulationInfo->realParameter[150] /* Radiator.vol[4].dynBal.medium.p PARAM */,0.0);
    if(!tmp137)
    {
      tmp139 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[150] /* Radiator.vol[4].dynBal.medium.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp138),tmp139);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",25,5,25,58,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.medium.p >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp140 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 907
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.ports[2].p >= 0.0 and Radiator.vol[4].dynBal.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,907};
  modelica_boolean tmp141;
  modelica_boolean tmp142;
  static const MMC_DEFSTRINGLIT(tmp143,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp144;
  static int tmp145 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp145)
  {
    tmp141 = GreaterEq(data->simulationInfo->realParameter[169] /* Radiator.vol[4].dynBal.ports[2].p PARAM */,0.0);
    tmp142 = LessEq(data->simulationInfo->realParameter[169] /* Radiator.vol[4].dynBal.ports[2].p PARAM */,100000000.0);
    if(!(tmp141 && tmp142))
    {
      tmp144 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[169] /* Radiator.vol[4].dynBal.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp143),tmp144);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.ports[2].p >= 0.0 and Radiator.vol[4].dynBal.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp145 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 908
type: ALGORITHM

  assert(Radiator.vol[4].ports[2].p >= 0.0 and Radiator.vol[4].ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[4].ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,908};
  modelica_boolean tmp146;
  modelica_boolean tmp147;
  static const MMC_DEFSTRINGLIT(tmp148,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp149;
  static int tmp150 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp150)
  {
    tmp146 = GreaterEq(data->simulationInfo->realParameter[224] /* Radiator.vol[4].ports[2].p PARAM */,0.0);
    tmp147 = LessEq(data->simulationInfo->realParameter[224] /* Radiator.vol[4].ports[2].p PARAM */,100000000.0);
    if(!(tmp146 && tmp147))
    {
      tmp149 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[224] /* Radiator.vol[4].ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp148),tmp149);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].ports[2].p >= 0.0 and Radiator.vol[4].ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp150 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 909
type: ALGORITHM

  assert(Radiator.vol[5].ports[1].p >= 0.0 and Radiator.vol[5].ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[5].ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,909};
  modelica_boolean tmp151;
  modelica_boolean tmp152;
  static const MMC_DEFSTRINGLIT(tmp153,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp154;
  static int tmp155 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp155)
  {
    tmp151 = GreaterEq(data->simulationInfo->realParameter[225] /* Radiator.vol[5].ports[1].p PARAM */,0.0);
    tmp152 = LessEq(data->simulationInfo->realParameter[225] /* Radiator.vol[5].ports[1].p PARAM */,100000000.0);
    if(!(tmp151 && tmp152))
    {
      tmp154 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[225] /* Radiator.vol[5].ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp153),tmp154);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].ports[1].p >= 0.0 and Radiator.vol[5].ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp155 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 910
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.ports[1].p >= 0.0 and Radiator.vol[5].dynBal.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.ports[1].p <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_910(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,910};
  modelica_boolean tmp156;
  modelica_boolean tmp157;
  static const MMC_DEFSTRINGLIT(tmp158,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp159;
  static int tmp160 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp160)
  {
    tmp156 = GreaterEq(data->simulationInfo->realParameter[170] /* Radiator.vol[5].dynBal.ports[1].p PARAM */,0.0);
    tmp157 = LessEq(data->simulationInfo->realParameter[170] /* Radiator.vol[5].dynBal.ports[1].p PARAM */,100000000.0);
    if(!(tmp156 && tmp157))
    {
      tmp159 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[170] /* Radiator.vol[5].dynBal.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp158),tmp159);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.ports[1].p >= 0.0 and Radiator.vol[5].dynBal.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp160 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 911
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.medium.p >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.medium.p, has value: " + String(Radiator.vol[5].dynBal.medium.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,911};
  modelica_boolean tmp161;
  static const MMC_DEFSTRINGLIT(tmp162,86,"Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.medium.p, has value: ");
  modelica_string tmp163;
  static int tmp164 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp164)
  {
    tmp161 = GreaterEq(data->simulationInfo->realParameter[151] /* Radiator.vol[5].dynBal.medium.p PARAM */,0.0);
    if(!tmp161)
    {
      tmp163 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[151] /* Radiator.vol[5].dynBal.medium.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp162),tmp163);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",25,5,25,58,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.medium.p >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp164 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 912
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.ports[2].p >= 0.0 and Radiator.vol[5].dynBal.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_912(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,912};
  modelica_boolean tmp165;
  modelica_boolean tmp166;
  static const MMC_DEFSTRINGLIT(tmp167,107,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp168;
  static int tmp169 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp169)
  {
    tmp165 = GreaterEq(data->simulationInfo->realParameter[171] /* Radiator.vol[5].dynBal.ports[2].p PARAM */,0.0);
    tmp166 = LessEq(data->simulationInfo->realParameter[171] /* Radiator.vol[5].dynBal.ports[2].p PARAM */,100000000.0);
    if(!(tmp165 && tmp166))
    {
      tmp168 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[171] /* Radiator.vol[5].dynBal.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp167),tmp168);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.ports[2].p >= 0.0 and Radiator.vol[5].dynBal.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp169 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 913
type: ALGORITHM

  assert(Radiator.vol[5].ports[2].p >= 0.0 and Radiator.vol[5].ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].ports[2].p <= 100000000.0, has value: " + String(Radiator.vol[5].ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,913};
  modelica_boolean tmp170;
  modelica_boolean tmp171;
  static const MMC_DEFSTRINGLIT(tmp172,100,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp173;
  static int tmp174 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp174)
  {
    tmp170 = GreaterEq(data->simulationInfo->realParameter[226] /* Radiator.vol[5].ports[2].p PARAM */,0.0);
    tmp171 = LessEq(data->simulationInfo->realParameter[226] /* Radiator.vol[5].ports[2].p PARAM */,100000000.0);
    if(!(tmp170 && tmp171))
    {
      tmp173 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[226] /* Radiator.vol[5].ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp172),tmp173);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].ports[2].p >= 0.0 and Radiator.vol[5].ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp174 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 914
type: ALGORITHM

  assert(Radiator.sta_b.p >= 0.0 and Radiator.sta_b.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.sta_b.p <= 100000000.0, has value: " + String(Radiator.sta_b.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_914(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,914};
  modelica_boolean tmp175;
  modelica_boolean tmp176;
  static const MMC_DEFSTRINGLIT(tmp177,90,"Variable violating min/max constraint: 0.0 <= Radiator.sta_b.p <= 100000000.0, has value: ");
  modelica_string tmp178;
  static int tmp179 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp179)
  {
    tmp175 = GreaterEq(data->simulationInfo->realParameter[86] /* Radiator.sta_b.p PARAM */,0.0);
    tmp176 = LessEq(data->simulationInfo->realParameter[86] /* Radiator.sta_b.p PARAM */,100000000.0);
    if(!(tmp175 && tmp176))
    {
      tmp178 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[86] /* Radiator.sta_b.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp177),tmp178);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.sta_b.p >= 0.0 and Radiator.sta_b.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp179 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 915
type: ALGORITHM

  assert(Radiator.port_b.p >= 0.0 and Radiator.port_b.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.port_b.p <= 100000000.0, has value: " + String(Radiator.port_b.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,915};
  modelica_boolean tmp180;
  modelica_boolean tmp181;
  static const MMC_DEFSTRINGLIT(tmp182,91,"Variable violating min/max constraint: 0.0 <= Radiator.port_b.p <= 100000000.0, has value: ");
  modelica_string tmp183;
  static int tmp184 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp184)
  {
    tmp180 = GreaterEq(data->simulationInfo->realParameter[44] /* Radiator.port_b.p PARAM */,0.0);
    tmp181 = LessEq(data->simulationInfo->realParameter[44] /* Radiator.port_b.p PARAM */,100000000.0);
    if(!(tmp180 && tmp181))
    {
      tmp183 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[44] /* Radiator.port_b.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp182),tmp183);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.port_b.p >= 0.0 and Radiator.port_b.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp184 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 916
type: ALGORITHM

  assert(flow_sink.ports[1].p >= 0.0 and flow_sink.ports[1].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= flow_sink.ports[1].p <= 100000000.0, has value: " + String(flow_sink.ports[1].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,916};
  modelica_boolean tmp185;
  modelica_boolean tmp186;
  static const MMC_DEFSTRINGLIT(tmp187,94,"Variable violating min/max constraint: 0.0 <= flow_sink.ports[1].p <= 100000000.0, has value: ");
  modelica_string tmp188;
  static int tmp189 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp189)
  {
    tmp185 = GreaterEq(data->simulationInfo->realParameter[263] /* flow_sink.ports[1].p PARAM */,0.0);
    tmp186 = LessEq(data->simulationInfo->realParameter[263] /* flow_sink.ports[1].p PARAM */,100000000.0);
    if(!(tmp185 && tmp186))
    {
      tmp188 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[263] /* flow_sink.ports[1].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp187),tmp188);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.ports[1].p >= 0.0 and flow_sink.ports[1].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp189 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 917
type: ALGORITHM

  assert(flow_sink.ports[2].p >= 0.0 and flow_sink.ports[2].p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= flow_sink.ports[2].p <= 100000000.0, has value: " + String(flow_sink.ports[2].p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,917};
  modelica_boolean tmp190;
  modelica_boolean tmp191;
  static const MMC_DEFSTRINGLIT(tmp192,94,"Variable violating min/max constraint: 0.0 <= flow_sink.ports[2].p <= 100000000.0, has value: ");
  modelica_string tmp193;
  static int tmp194 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp194)
  {
    tmp190 = GreaterEq(data->simulationInfo->realParameter[264] /* flow_sink.ports[2].p PARAM */,0.0);
    tmp191 = LessEq(data->simulationInfo->realParameter[264] /* flow_sink.ports[2].p PARAM */,100000000.0);
    if(!(tmp190 && tmp191))
    {
      tmp193 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[264] /* flow_sink.ports[2].p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp192),tmp193);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",15,5,15,79,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.ports[2].p >= 0.0 and flow_sink.ports[2].p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp194 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 918
type: ALGORITHM

  assert(flow_sink.T >= 1.0 and flow_sink.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= flow_sink.T <= 10000.0, has value: " + String(flow_sink.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_918(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,918};
  modelica_boolean tmp195;
  modelica_boolean tmp196;
  static const MMC_DEFSTRINGLIT(tmp197,81,"Variable violating min/max constraint: 1.0 <= flow_sink.T <= 10000.0, has value: ");
  modelica_string tmp198;
  static int tmp199 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp199)
  {
    tmp195 = GreaterEq(data->simulationInfo->realParameter[260] /* flow_sink.T PARAM */,1.0);
    tmp196 = LessEq(data->simulationInfo->realParameter[260] /* flow_sink.T PARAM */,10000.0);
    if(!(tmp195 && tmp196))
    {
      tmp198 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[260] /* flow_sink.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp197),tmp198);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/Boundary_pT.mo",16,3,18,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.T >= 1.0 and flow_sink.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp199 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 919
type: ALGORITHM

  assert(flow_sink.X[1] >= 0.0 and flow_sink.X[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= flow_sink.X[1] <= 1.0, has value: " + String(flow_sink.X[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,919};
  modelica_boolean tmp200;
  modelica_boolean tmp201;
  static const MMC_DEFSTRINGLIT(tmp202,80,"Variable violating min/max constraint: 0.0 <= flow_sink.X[1] <= 1.0, has value: ");
  modelica_string tmp203;
  static int tmp204 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp204)
  {
    tmp200 = GreaterEq(data->simulationInfo->realParameter[261] /* flow_sink.X[1] PARAM */,0.0);
    tmp201 = LessEq(data->simulationInfo->realParameter[261] /* flow_sink.X[1] PARAM */,1.0);
    if(!(tmp200 && tmp201))
    {
      tmp203 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[261] /* flow_sink.X[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp202),tmp203);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/BaseClasses/PartialSource_Xi_C.mo",15,3,18,90,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.X[1] >= 0.0 and flow_sink.X[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp204 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 920
type: ALGORITHM

  assert(flow_sink.flowDirection >= Modelica.Fluid.Types.PortFlowDirection.Entering and flow_sink.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, "Variable violating min/max constraint: Modelica.Fluid.Types.PortFlowDirection.Entering <= flow_sink.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, has value: " + String(flow_sink.flowDirection, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_920(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,920};
  modelica_boolean tmp205;
  modelica_boolean tmp206;
  static const MMC_DEFSTRINGLIT(tmp207,182,"Variable violating min/max constraint: Modelica.Fluid.Types.PortFlowDirection.Entering <= flow_sink.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, has value: ");
  modelica_string tmp208;
  static int tmp209 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp209)
  {
    tmp205 = GreaterEq(data->simulationInfo->integerParameter[57] /* flow_sink.flowDirection PARAM */,1);
    tmp206 = LessEq(data->simulationInfo->integerParameter[57] /* flow_sink.flowDirection PARAM */,3);
    if(!(tmp205 && tmp206))
    {
      tmp208 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[57] /* flow_sink.flowDirection PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp207),tmp208);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/BaseClasses/PartialSource.mo",31,3,32,80,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.flowDirection >= Modelica.Fluid.Types.PortFlowDirection.Entering and flow_sink.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp209 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 921
type: ALGORITHM

  assert(Radiator.res.deltaM >= 1e-06, "Variable violating min constraint: 1e-06 <= Radiator.res.deltaM, has value: " + String(Radiator.res.deltaM, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,921};
  modelica_boolean tmp210;
  static const MMC_DEFSTRINGLIT(tmp211,76,"Variable violating min constraint: 1e-06 <= Radiator.res.deltaM, has value: ");
  modelica_string tmp212;
  static int tmp213 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp213)
  {
    tmp210 = GreaterEq(data->simulationInfo->realParameter[72] /* Radiator.res.deltaM PARAM */,1e-06);
    if(!tmp210)
    {
      tmp212 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[72] /* Radiator.res.deltaM PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp211),tmp212);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/FixedResistances/PressureDrop.mo",7,3,11,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.deltaM >= 1e-06", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp213 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 922
type: ALGORITHM

  assert(T_a_nominal >= 0.0, "Variable violating min constraint: 0.0 <= T_a_nominal, has value: " + String(T_a_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_922(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,922};
  modelica_boolean tmp214;
  static const MMC_DEFSTRINGLIT(tmp215,66,"Variable violating min constraint: 0.0 <= T_a_nominal, has value: ");
  modelica_string tmp216;
  static int tmp217 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp217)
  {
    tmp214 = GreaterEq(data->simulationInfo->realParameter[258] /* T_a_nominal PARAM */,0.0);
    if(!tmp214)
    {
      tmp216 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[258] /* T_a_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp215),tmp216);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/Radiator.mo",7,3,8,54,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nT_a_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp217 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 923
type: ALGORITHM

  assert(Radiator.T_a_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.T_a_nominal, has value: " + String(Radiator.T_a_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,923};
  modelica_boolean tmp218;
  static const MMC_DEFSTRINGLIT(tmp219,75,"Variable violating min constraint: 0.0 <= Radiator.T_a_nominal, has value: ");
  modelica_string tmp220;
  static int tmp221 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp221)
  {
    tmp218 = GreaterEq(data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */,0.0);
    if(!tmp218)
    {
      tmp220 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp219),tmp220);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",25,3,27,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.T_a_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp221 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 924
type: ALGORITHM

  assert(Radiator.res.sta_default.p >= 0.0 and Radiator.res.sta_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.res.sta_default.p <= 100000000.0, has value: " + String(Radiator.res.sta_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,924};
  modelica_boolean tmp222;
  modelica_boolean tmp223;
  static const MMC_DEFSTRINGLIT(tmp224,100,"Variable violating min/max constraint: 0.0 <= Radiator.res.sta_default.p <= 100000000.0, has value: ");
  modelica_string tmp225;
  static int tmp226 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp226)
  {
    tmp222 = GreaterEq(data->simulationInfo->realParameter[84] /* Radiator.res.sta_default.p PARAM */,0.0);
    tmp223 = LessEq(data->simulationInfo->realParameter[84] /* Radiator.res.sta_default.p PARAM */,100000000.0);
    if(!(tmp222 && tmp223))
    {
      tmp225 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[84] /* Radiator.res.sta_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp224),tmp225);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.sta_default.p >= 0.0 and Radiator.res.sta_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp226 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 925
type: ALGORITHM

  assert(Radiator.res.sta_default.T >= 1.0 and Radiator.res.sta_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.res.sta_default.T <= 10000.0, has value: " + String(Radiator.res.sta_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,925};
  modelica_boolean tmp227;
  modelica_boolean tmp228;
  static const MMC_DEFSTRINGLIT(tmp229,96,"Variable violating min/max constraint: 1.0 <= Radiator.res.sta_default.T <= 10000.0, has value: ");
  modelica_string tmp230;
  static int tmp231 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp231)
  {
    tmp227 = GreaterEq(data->simulationInfo->realParameter[83] /* Radiator.res.sta_default.T PARAM */,1.0);
    tmp228 = LessEq(data->simulationInfo->realParameter[83] /* Radiator.res.sta_default.T PARAM */,10000.0);
    if(!(tmp227 && tmp228))
    {
      tmp230 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[83] /* Radiator.res.sta_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp229),tmp230);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.sta_default.T >= 1.0 and Radiator.res.sta_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp231 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 926
type: ALGORITHM

  assert(Radiator.res.eta_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.res.eta_default, has value: " + String(Radiator.res.eta_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_926(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,926};
  modelica_boolean tmp232;
  static const MMC_DEFSTRINGLIT(tmp233,79,"Variable violating min constraint: 0.0 <= Radiator.res.eta_default, has value: ");
  modelica_string tmp234;
  static int tmp235 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp235)
  {
    tmp232 = GreaterEq(data->simulationInfo->realParameter[75] /* Radiator.res.eta_default PARAM */,0.0);
    if(!tmp232)
    {
      tmp234 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[75] /* Radiator.res.eta_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp233),tmp234);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/BaseClasses/PartialResistance.mo",33,3,35,77,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.eta_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp235 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 927
type: ALGORITHM

  assert(Radiator.res.m_flow_turbulent >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.res.m_flow_turbulent, has value: " + String(Radiator.res.m_flow_turbulent, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,927};
  modelica_boolean tmp236;
  static const MMC_DEFSTRINGLIT(tmp237,84,"Variable violating min constraint: 0.0 <= Radiator.res.m_flow_turbulent, has value: ");
  modelica_string tmp238;
  static int tmp239 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp239)
  {
    tmp236 = GreaterEq(data->simulationInfo->realParameter[80] /* Radiator.res.m_flow_turbulent PARAM */,0.0);
    if(!tmp236)
    {
      tmp238 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[80] /* Radiator.res.m_flow_turbulent PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp237),tmp238);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/BaseClasses/PartialResistance.mo",27,3,28,53,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.m_flow_turbulent >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp239 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 928
type: ALGORITHM

  assert(Radiator.res.m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.res.m_flow_small, has value: " + String(Radiator.res.m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_928(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,928};
  modelica_boolean tmp240;
  static const MMC_DEFSTRINGLIT(tmp241,80,"Variable violating min constraint: 0.0 <= Radiator.res.m_flow_small, has value: ");
  modelica_string tmp242;
  static int tmp243 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp243)
  {
    tmp240 = GreaterEq(data->simulationInfo->realParameter[79] /* Radiator.res.m_flow_small PARAM */,0.0);
    if(!tmp240)
    {
      tmp242 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[79] /* Radiator.res.m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp241),tmp242);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/PartialTwoPortInterface.mo",10,3,12,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.res.m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp243 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 929
type: ALGORITHM

  assert(Radiator.preSumRad.T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preSumRad.T_ref, has value: " + String(Radiator.preSumRad.T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,929};
  modelica_boolean tmp244;
  static const MMC_DEFSTRINGLIT(tmp245,79,"Variable violating min constraint: 0.0 <= Radiator.preSumRad.T_ref, has value: ");
  modelica_string tmp246;
  static int tmp247 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp247)
  {
    tmp244 = GreaterEq(data->simulationInfo->realParameter[67] /* Radiator.preSumRad.T_ref PARAM */,0.0);
    if(!tmp244)
    {
      tmp246 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[67] /* Radiator.preSumRad.T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp245),tmp246);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preSumRad.T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp247 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 930
type: ALGORITHM

  assert(Radiator.preSumCon.T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preSumCon.T_ref, has value: " + String(Radiator.preSumCon.T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_930(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,930};
  modelica_boolean tmp248;
  static const MMC_DEFSTRINGLIT(tmp249,79,"Variable violating min constraint: 0.0 <= Radiator.preSumCon.T_ref, has value: ");
  modelica_string tmp250;
  static int tmp251 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp251)
  {
    tmp248 = GreaterEq(data->simulationInfo->realParameter[65] /* Radiator.preSumCon.T_ref PARAM */,0.0);
    if(!tmp248)
    {
      tmp250 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[65] /* Radiator.preSumCon.T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp249),tmp250);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preSumCon.T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp251 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 931
type: ALGORITHM

  assert(Radiator.preRad[5].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preRad[5].T_ref, has value: " + String(Radiator.preRad[5].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,931};
  modelica_boolean tmp252;
  static const MMC_DEFSTRINGLIT(tmp253,79,"Variable violating min constraint: 0.0 <= Radiator.preRad[5].T_ref, has value: ");
  modelica_string tmp254;
  static int tmp255 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp255)
  {
    tmp252 = GreaterEq(data->simulationInfo->realParameter[59] /* Radiator.preRad[5].T_ref PARAM */,0.0);
    if(!tmp252)
    {
      tmp254 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[59] /* Radiator.preRad[5].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp253),tmp254);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preRad[5].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp255 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 932
type: ALGORITHM

  assert(Radiator.preRad[4].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preRad[4].T_ref, has value: " + String(Radiator.preRad[4].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,932};
  modelica_boolean tmp256;
  static const MMC_DEFSTRINGLIT(tmp257,79,"Variable violating min constraint: 0.0 <= Radiator.preRad[4].T_ref, has value: ");
  modelica_string tmp258;
  static int tmp259 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp259)
  {
    tmp256 = GreaterEq(data->simulationInfo->realParameter[58] /* Radiator.preRad[4].T_ref PARAM */,0.0);
    if(!tmp256)
    {
      tmp258 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[58] /* Radiator.preRad[4].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp257),tmp258);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preRad[4].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp259 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 933
type: ALGORITHM

  assert(Radiator.preRad[3].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preRad[3].T_ref, has value: " + String(Radiator.preRad[3].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,933};
  modelica_boolean tmp260;
  static const MMC_DEFSTRINGLIT(tmp261,79,"Variable violating min constraint: 0.0 <= Radiator.preRad[3].T_ref, has value: ");
  modelica_string tmp262;
  static int tmp263 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp263)
  {
    tmp260 = GreaterEq(data->simulationInfo->realParameter[57] /* Radiator.preRad[3].T_ref PARAM */,0.0);
    if(!tmp260)
    {
      tmp262 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[57] /* Radiator.preRad[3].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp261),tmp262);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preRad[3].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp263 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 934
type: ALGORITHM

  assert(Radiator.preRad[2].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preRad[2].T_ref, has value: " + String(Radiator.preRad[2].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_934(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,934};
  modelica_boolean tmp264;
  static const MMC_DEFSTRINGLIT(tmp265,79,"Variable violating min constraint: 0.0 <= Radiator.preRad[2].T_ref, has value: ");
  modelica_string tmp266;
  static int tmp267 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp267)
  {
    tmp264 = GreaterEq(data->simulationInfo->realParameter[56] /* Radiator.preRad[2].T_ref PARAM */,0.0);
    if(!tmp264)
    {
      tmp266 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[56] /* Radiator.preRad[2].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp265),tmp266);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preRad[2].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp267 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 935
type: ALGORITHM

  assert(Radiator.preRad[1].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preRad[1].T_ref, has value: " + String(Radiator.preRad[1].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,935};
  modelica_boolean tmp268;
  static const MMC_DEFSTRINGLIT(tmp269,79,"Variable violating min constraint: 0.0 <= Radiator.preRad[1].T_ref, has value: ");
  modelica_string tmp270;
  static int tmp271 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp271)
  {
    tmp268 = GreaterEq(data->simulationInfo->realParameter[55] /* Radiator.preRad[1].T_ref PARAM */,0.0);
    if(!tmp268)
    {
      tmp270 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[55] /* Radiator.preRad[1].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp269),tmp270);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preRad[1].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp271 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 936
type: ALGORITHM

  assert(Radiator.preCon[5].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preCon[5].T_ref, has value: " + String(Radiator.preCon[5].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_936(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,936};
  modelica_boolean tmp272;
  static const MMC_DEFSTRINGLIT(tmp273,79,"Variable violating min constraint: 0.0 <= Radiator.preCon[5].T_ref, has value: ");
  modelica_string tmp274;
  static int tmp275 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp275)
  {
    tmp272 = GreaterEq(data->simulationInfo->realParameter[49] /* Radiator.preCon[5].T_ref PARAM */,0.0);
    if(!tmp272)
    {
      tmp274 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[49] /* Radiator.preCon[5].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp273),tmp274);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preCon[5].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp275 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 937
type: ALGORITHM

  assert(Radiator.preCon[4].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preCon[4].T_ref, has value: " + String(Radiator.preCon[4].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,937};
  modelica_boolean tmp276;
  static const MMC_DEFSTRINGLIT(tmp277,79,"Variable violating min constraint: 0.0 <= Radiator.preCon[4].T_ref, has value: ");
  modelica_string tmp278;
  static int tmp279 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp279)
  {
    tmp276 = GreaterEq(data->simulationInfo->realParameter[48] /* Radiator.preCon[4].T_ref PARAM */,0.0);
    if(!tmp276)
    {
      tmp278 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[48] /* Radiator.preCon[4].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp277),tmp278);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preCon[4].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp279 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 938
type: ALGORITHM

  assert(Radiator.preCon[3].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preCon[3].T_ref, has value: " + String(Radiator.preCon[3].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_938(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,938};
  modelica_boolean tmp280;
  static const MMC_DEFSTRINGLIT(tmp281,79,"Variable violating min constraint: 0.0 <= Radiator.preCon[3].T_ref, has value: ");
  modelica_string tmp282;
  static int tmp283 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp283)
  {
    tmp280 = GreaterEq(data->simulationInfo->realParameter[47] /* Radiator.preCon[3].T_ref PARAM */,0.0);
    if(!tmp280)
    {
      tmp282 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[47] /* Radiator.preCon[3].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp281),tmp282);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preCon[3].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp283 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 939
type: ALGORITHM

  assert(Radiator.preCon[2].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preCon[2].T_ref, has value: " + String(Radiator.preCon[2].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,939};
  modelica_boolean tmp284;
  static const MMC_DEFSTRINGLIT(tmp285,79,"Variable violating min constraint: 0.0 <= Radiator.preCon[2].T_ref, has value: ");
  modelica_string tmp286;
  static int tmp287 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp287)
  {
    tmp284 = GreaterEq(data->simulationInfo->realParameter[46] /* Radiator.preCon[2].T_ref PARAM */,0.0);
    if(!tmp284)
    {
      tmp286 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[46] /* Radiator.preCon[2].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp285),tmp286);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preCon[2].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp287 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 940
type: ALGORITHM

  assert(Radiator.preCon[1].T_ref >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.preCon[1].T_ref, has value: " + String(Radiator.preCon[1].T_ref, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,940};
  modelica_boolean tmp288;
  static const MMC_DEFSTRINGLIT(tmp289,79,"Variable violating min constraint: 0.0 <= Radiator.preCon[1].T_ref, has value: ");
  modelica_string tmp290;
  static int tmp291 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp291)
  {
    tmp288 = GreaterEq(data->simulationInfo->realParameter[45] /* Radiator.preCon[1].T_ref PARAM */,0.0);
    if(!tmp288)
    {
      tmp290 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[45] /* Radiator.preCon[1].T_ref PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp289),tmp290);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Thermal/HeatTransfer/Sources/PrescribedHeatFlow.mo",3,3,4,28,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.preCon[1].T_ref >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp291 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 941
type: ALGORITHM

  assert(Radiator.fraRad >= 0.0 and Radiator.fraRad <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.fraRad <= 1.0, has value: " + String(Radiator.fraRad, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,941};
  modelica_boolean tmp292;
  modelica_boolean tmp293;
  static const MMC_DEFSTRINGLIT(tmp294,81,"Variable violating min/max constraint: 0.0 <= Radiator.fraRad <= 1.0, has value: ");
  modelica_string tmp295;
  static int tmp296 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp296)
  {
    tmp292 = GreaterEq(data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */,0.0);
    tmp293 = LessEq(data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */,1.0);
    if(!(tmp292 && tmp293))
    {
      tmp295 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp294),tmp295);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",19,3,19,78,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.fraRad >= 0.0 and Radiator.fraRad <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp296 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 942
type: ALGORITHM

  assert(Radiator.TRad_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.TRad_nominal, has value: " + String(Radiator.TRad_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_942(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,942};
  modelica_boolean tmp297;
  static const MMC_DEFSTRINGLIT(tmp298,76,"Variable violating min constraint: 0.0 <= Radiator.TRad_nominal, has value: ");
  modelica_string tmp299;
  static int tmp300 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp300)
  {
    tmp297 = GreaterEq(data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,0.0);
    if(!tmp297)
    {
      tmp299 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp298),tmp299);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",34,3,36,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.TRad_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp300 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 943
type: ALGORITHM

  assert(Radiator.vol[5].state_start.T >= 1.0 and Radiator.vol[5].state_start.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].state_start.T <= 10000.0, has value: " + String(Radiator.vol[5].state_start.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,943};
  modelica_boolean tmp301;
  modelica_boolean tmp302;
  static const MMC_DEFSTRINGLIT(tmp303,99,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].state_start.T <= 10000.0, has value: ");
  modelica_string tmp304;
  static int tmp305 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp305)
  {
    tmp301 = GreaterEq(data->simulationInfo->realParameter[251] /* Radiator.vol[5].state_start.T PARAM */,1.0);
    tmp302 = LessEq(data->simulationInfo->realParameter[251] /* Radiator.vol[5].state_start.T PARAM */,10000.0);
    if(!(tmp301 && tmp302))
    {
      tmp304 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[251] /* Radiator.vol[5].state_start.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp303),tmp304);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].state_start.T >= 1.0 and Radiator.vol[5].state_start.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp305 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 944
type: ALGORITHM

  assert(Radiator.vol[5].state_start.p >= 0.0 and Radiator.vol[5].state_start.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].state_start.p <= 100000000.0, has value: " + String(Radiator.vol[5].state_start.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_944(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,944};
  modelica_boolean tmp306;
  modelica_boolean tmp307;
  static const MMC_DEFSTRINGLIT(tmp308,103,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].state_start.p <= 100000000.0, has value: ");
  modelica_string tmp309;
  static int tmp310 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp310)
  {
    tmp306 = GreaterEq(data->simulationInfo->realParameter[256] /* Radiator.vol[5].state_start.p PARAM */,0.0);
    tmp307 = LessEq(data->simulationInfo->realParameter[256] /* Radiator.vol[5].state_start.p PARAM */,100000000.0);
    if(!(tmp306 && tmp307))
    {
      tmp309 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[256] /* Radiator.vol[5].state_start.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp308),tmp309);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].state_start.p >= 0.0 and Radiator.vol[5].state_start.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp310 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 945
type: ALGORITHM

  assert(Radiator.vol[5].rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].rho_default, has value: " + String(Radiator.vol[5].rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,945};
  modelica_boolean tmp311;
  static const MMC_DEFSTRINGLIT(tmp312,82,"Variable violating min constraint: 0.0 <= Radiator.vol[5].rho_default, has value: ");
  modelica_string tmp313;
  static int tmp314 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp314)
  {
    tmp311 = GreaterEq(data->simulationInfo->realParameter[231] /* Radiator.vol[5].rho_default PARAM */,0.0);
    if(!tmp311)
    {
      tmp313 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[231] /* Radiator.vol[5].rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp312),tmp313);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",96,3,97,63,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp314 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 946
type: ALGORITHM

  assert(Radiator.vol[5].state_default.T >= 1.0 and Radiator.vol[5].state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].state_default.T <= 10000.0, has value: " + String(Radiator.vol[5].state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_946(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,946};
  modelica_boolean tmp315;
  modelica_boolean tmp316;
  static const MMC_DEFSTRINGLIT(tmp317,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].state_default.T <= 10000.0, has value: ");
  modelica_string tmp318;
  static int tmp319 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp319)
  {
    tmp315 = GreaterEq(data->simulationInfo->realParameter[241] /* Radiator.vol[5].state_default.T PARAM */,1.0);
    tmp316 = LessEq(data->simulationInfo->realParameter[241] /* Radiator.vol[5].state_default.T PARAM */,10000.0);
    if(!(tmp315 && tmp316))
    {
      tmp318 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[241] /* Radiator.vol[5].state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp317),tmp318);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].state_default.T >= 1.0 and Radiator.vol[5].state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp319 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 947
type: ALGORITHM

  assert(Radiator.vol[5].state_default.p >= 0.0 and Radiator.vol[5].state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].state_default.p <= 100000000.0, has value: " + String(Radiator.vol[5].state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,947};
  modelica_boolean tmp320;
  modelica_boolean tmp321;
  static const MMC_DEFSTRINGLIT(tmp322,105,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].state_default.p <= 100000000.0, has value: ");
  modelica_string tmp323;
  static int tmp324 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp324)
  {
    tmp320 = GreaterEq(data->simulationInfo->realParameter[246] /* Radiator.vol[5].state_default.p PARAM */,0.0);
    tmp321 = LessEq(data->simulationInfo->realParameter[246] /* Radiator.vol[5].state_default.p PARAM */,100000000.0);
    if(!(tmp320 && tmp321))
    {
      tmp323 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[246] /* Radiator.vol[5].state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp322),tmp323);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].state_default.p >= 0.0 and Radiator.vol[5].state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp324 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 948
type: ALGORITHM

  assert(Radiator.vol[5].rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].rho_start, has value: " + String(Radiator.vol[5].rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,948};
  modelica_boolean tmp325;
  static const MMC_DEFSTRINGLIT(tmp326,80,"Variable violating min constraint: 0.0 <= Radiator.vol[5].rho_start, has value: ");
  modelica_string tmp327;
  static int tmp328 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp328)
  {
    tmp325 = GreaterEq(data->simulationInfo->realParameter[236] /* Radiator.vol[5].rho_start PARAM */,0.0);
    if(!tmp325)
    {
      tmp327 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[236] /* Radiator.vol[5].rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp326),tmp327);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",89,3,90,73,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp328 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 949
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.rho_default, has value: " + String(Radiator.vol[5].dynBal.rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,949};
  modelica_boolean tmp329;
  static const MMC_DEFSTRINGLIT(tmp330,89,"Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.rho_default, has value: ");
  modelica_string tmp331;
  static int tmp332 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp332)
  {
    tmp329 = GreaterEq(data->simulationInfo->realParameter[176] /* Radiator.vol[5].dynBal.rho_default PARAM */,0.0);
    if(!tmp329)
    {
      tmp331 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[176] /* Radiator.vol[5].dynBal.rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp330),tmp331);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",145,3,146,59,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp332 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 950
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.state_default.T >= 1.0 and Radiator.vol[5].dynBal.state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.state_default.T <= 10000.0, has value: " + String(Radiator.vol[5].dynBal.state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_950(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,950};
  modelica_boolean tmp333;
  modelica_boolean tmp334;
  static const MMC_DEFSTRINGLIT(tmp335,108,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.state_default.T <= 10000.0, has value: ");
  modelica_string tmp336;
  static int tmp337 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp337)
  {
    tmp333 = GreaterEq(data->simulationInfo->realParameter[186] /* Radiator.vol[5].dynBal.state_default.T PARAM */,1.0);
    tmp334 = LessEq(data->simulationInfo->realParameter[186] /* Radiator.vol[5].dynBal.state_default.T PARAM */,10000.0);
    if(!(tmp333 && tmp334))
    {
      tmp336 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[186] /* Radiator.vol[5].dynBal.state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp335),tmp336);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.state_default.T >= 1.0 and Radiator.vol[5].dynBal.state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp337 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 951
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.state_default.p >= 0.0 and Radiator.vol[5].dynBal.state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.state_default.p <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,951};
  modelica_boolean tmp338;
  modelica_boolean tmp339;
  static const MMC_DEFSTRINGLIT(tmp340,112,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.state_default.p <= 100000000.0, has value: ");
  modelica_string tmp341;
  static int tmp342 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp342)
  {
    tmp338 = GreaterEq(data->simulationInfo->realParameter[191] /* Radiator.vol[5].dynBal.state_default.p PARAM */,0.0);
    tmp339 = LessEq(data->simulationInfo->realParameter[191] /* Radiator.vol[5].dynBal.state_default.p PARAM */,100000000.0);
    if(!(tmp338 && tmp339))
    {
      tmp341 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[191] /* Radiator.vol[5].dynBal.state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp340),tmp341);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.state_default.p >= 0.0 and Radiator.vol[5].dynBal.state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp342 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 952
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.rho_start, has value: " + String(Radiator.vol[5].dynBal.rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_952(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,952};
  modelica_boolean tmp343;
  static const MMC_DEFSTRINGLIT(tmp344,87,"Variable violating min constraint: 0.0 <= Radiator.vol[5].dynBal.rho_start, has value: ");
  modelica_string tmp345;
  static int tmp346 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp346)
  {
    tmp343 = GreaterEq(data->simulationInfo->realParameter[181] /* Radiator.vol[5].dynBal.rho_start PARAM */,0.0);
    if(!tmp343)
    {
      tmp345 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[181] /* Radiator.vol[5].dynBal.rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp344),tmp345);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",131,3,135,70,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp346 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 953
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[5].dynBal.mSenFac, has value: " + String(Radiator.vol[5].dynBal.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,953};
  modelica_boolean tmp347;
  static const MMC_DEFSTRINGLIT(tmp348,85,"Variable violating min constraint: 1.0 <= Radiator.vol[5].dynBal.mSenFac, has value: ");
  modelica_string tmp349;
  static int tmp350 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp350)
  {
    tmp347 = GreaterEq(data->simulationInfo->realParameter[146] /* Radiator.vol[5].dynBal.mSenFac PARAM */,1.0);
    if(!tmp347)
    {
      tmp349 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[146] /* Radiator.vol[5].dynBal.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp348),tmp349);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp350 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 954
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.X_start[1] >= 0.0 and Radiator.vol[5].dynBal.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.X_start[1] <= 1.0, has value: " + String(Radiator.vol[5].dynBal.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_954(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,954};
  modelica_boolean tmp351;
  modelica_boolean tmp352;
  static const MMC_DEFSTRINGLIT(tmp353,99,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.X_start[1] <= 1.0, has value: ");
  modelica_string tmp354;
  static int tmp355 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp355)
  {
    tmp351 = GreaterEq(data->simulationInfo->realParameter[126] /* Radiator.vol[5].dynBal.X_start[1] PARAM */,0.0);
    tmp352 = LessEq(data->simulationInfo->realParameter[126] /* Radiator.vol[5].dynBal.X_start[1] PARAM */,1.0);
    if(!(tmp351 && tmp352))
    {
      tmp354 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[126] /* Radiator.vol[5].dynBal.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp353),tmp354);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.X_start[1] >= 0.0 and Radiator.vol[5].dynBal.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp355 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 955
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.T_start >= 1.0 and Radiator.vol[5].dynBal.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.T_start <= 10000.0, has value: " + String(Radiator.vol[5].dynBal.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,955};
  modelica_boolean tmp356;
  modelica_boolean tmp357;
  static const MMC_DEFSTRINGLIT(tmp358,100,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.T_start <= 10000.0, has value: ");
  modelica_string tmp359;
  static int tmp360 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp360)
  {
    tmp356 = GreaterEq(data->simulationInfo->realParameter[121] /* Radiator.vol[5].dynBal.T_start PARAM */,1.0);
    tmp357 = LessEq(data->simulationInfo->realParameter[121] /* Radiator.vol[5].dynBal.T_start PARAM */,10000.0);
    if(!(tmp356 && tmp357))
    {
      tmp359 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[121] /* Radiator.vol[5].dynBal.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp358),tmp359);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.T_start >= 1.0 and Radiator.vol[5].dynBal.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp360 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 956
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.p_start >= 0.0 and Radiator.vol[5].dynBal.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.p_start <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,956};
  modelica_boolean tmp361;
  modelica_boolean tmp362;
  static const MMC_DEFSTRINGLIT(tmp363,104,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].dynBal.p_start <= 100000000.0, has value: ");
  modelica_string tmp364;
  static int tmp365 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp365)
  {
    tmp361 = GreaterEq(data->simulationInfo->realParameter[161] /* Radiator.vol[5].dynBal.p_start PARAM */,0.0);
    tmp362 = LessEq(data->simulationInfo->realParameter[161] /* Radiator.vol[5].dynBal.p_start PARAM */,100000000.0);
    if(!(tmp361 && tmp362))
    {
      tmp364 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[161] /* Radiator.vol[5].dynBal.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp363),tmp364);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.p_start >= 0.0 and Radiator.vol[5].dynBal.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp365 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 957
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].dynBal.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,957};
  modelica_boolean tmp366;
  modelica_boolean tmp367;
  static const MMC_DEFSTRINGLIT(tmp368,185,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp369;
  static int tmp370 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp370)
  {
    tmp366 = GreaterEq(data->simulationInfo->integerParameter[31] /* Radiator.vol[5].dynBal.traceDynamics PARAM */,1);
    tmp367 = LessEq(data->simulationInfo->integerParameter[31] /* Radiator.vol[5].dynBal.traceDynamics PARAM */,4);
    if(!(tmp366 && tmp367))
    {
      tmp369 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[31] /* Radiator.vol[5].dynBal.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp368),tmp369);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp370 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 958
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].dynBal.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_958(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,958};
  modelica_boolean tmp371;
  modelica_boolean tmp372;
  static const MMC_DEFSTRINGLIT(tmp373,189,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp374;
  static int tmp375 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp375)
  {
    tmp371 = GreaterEq(data->simulationInfo->integerParameter[26] /* Radiator.vol[5].dynBal.substanceDynamics PARAM */,1);
    tmp372 = LessEq(data->simulationInfo->integerParameter[26] /* Radiator.vol[5].dynBal.substanceDynamics PARAM */,4);
    if(!(tmp371 && tmp372))
    {
      tmp374 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[26] /* Radiator.vol[5].dynBal.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp373),tmp374);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp375 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 959
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].dynBal.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,959};
  modelica_boolean tmp376;
  modelica_boolean tmp377;
  static const MMC_DEFSTRINGLIT(tmp378,184,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp379;
  static int tmp380 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp380)
  {
    tmp376 = GreaterEq(data->simulationInfo->integerParameter[16] /* Radiator.vol[5].dynBal.massDynamics PARAM */,1);
    tmp377 = LessEq(data->simulationInfo->integerParameter[16] /* Radiator.vol[5].dynBal.massDynamics PARAM */,4);
    if(!(tmp376 && tmp377))
    {
      tmp379 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[16] /* Radiator.vol[5].dynBal.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp378),tmp379);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp380 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 960
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].dynBal.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_960(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,960};
  modelica_boolean tmp381;
  modelica_boolean tmp382;
  static const MMC_DEFSTRINGLIT(tmp383,186,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp384;
  static int tmp385 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp385)
  {
    tmp381 = GreaterEq(data->simulationInfo->integerParameter[11] /* Radiator.vol[5].dynBal.energyDynamics PARAM */,1);
    tmp382 = LessEq(data->simulationInfo->integerParameter[11] /* Radiator.vol[5].dynBal.energyDynamics PARAM */,4);
    if(!(tmp381 && tmp382))
    {
      tmp384 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[11] /* Radiator.vol[5].dynBal.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp383),tmp384);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp385 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 961
type: ALGORITHM

  assert(Radiator.vol[5].m_flow_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].m_flow_nominal, has value: " + String(Radiator.vol[5].m_flow_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,961};
  modelica_boolean tmp386;
  static const MMC_DEFSTRINGLIT(tmp387,85,"Variable violating min constraint: 0.0 <= Radiator.vol[5].m_flow_nominal, has value: ");
  modelica_string tmp388;
  static int tmp389 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp389)
  {
    tmp386 = GreaterEq(data->simulationInfo->realParameter[201] /* Radiator.vol[5].m_flow_nominal PARAM */,0.0);
    if(!tmp386)
    {
      tmp388 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[201] /* Radiator.vol[5].m_flow_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp387),tmp388);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",20,3,21,76,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].m_flow_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp389 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 962
type: ALGORITHM

  assert(Radiator.vol[5].m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[5].m_flow_small, has value: " + String(Radiator.vol[5].m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_962(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,962};
  modelica_boolean tmp390;
  static const MMC_DEFSTRINGLIT(tmp391,83,"Variable violating min constraint: 0.0 <= Radiator.vol[5].m_flow_small, has value: ");
  modelica_string tmp392;
  static int tmp393 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp393)
  {
    tmp390 = GreaterEq(data->simulationInfo->realParameter[206] /* Radiator.vol[5].m_flow_small PARAM */,0.0);
    if(!tmp390)
    {
      tmp392 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[206] /* Radiator.vol[5].m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp391),tmp392);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",25,3,27,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp393 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 963
type: ALGORITHM

  assert(Radiator.vol[5].mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[5].mSenFac, has value: " + String(Radiator.vol[5].mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,963};
  modelica_boolean tmp394;
  static const MMC_DEFSTRINGLIT(tmp395,78,"Variable violating min constraint: 1.0 <= Radiator.vol[5].mSenFac, has value: ");
  modelica_string tmp396;
  static int tmp397 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp397)
  {
    tmp394 = GreaterEq(data->simulationInfo->realParameter[196] /* Radiator.vol[5].mSenFac PARAM */,1.0);
    if(!tmp394)
    {
      tmp396 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[196] /* Radiator.vol[5].mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp395),tmp396);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp397 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 964
type: ALGORITHM

  assert(Radiator.vol[5].X_start[1] >= 0.0 and Radiator.vol[5].X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].X_start[1] <= 1.0, has value: " + String(Radiator.vol[5].X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,964};
  modelica_boolean tmp398;
  modelica_boolean tmp399;
  static const MMC_DEFSTRINGLIT(tmp400,92,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].X_start[1] <= 1.0, has value: ");
  modelica_string tmp401;
  static int tmp402 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp402)
  {
    tmp398 = GreaterEq(data->simulationInfo->realParameter[111] /* Radiator.vol[5].X_start[1] PARAM */,0.0);
    tmp399 = LessEq(data->simulationInfo->realParameter[111] /* Radiator.vol[5].X_start[1] PARAM */,1.0);
    if(!(tmp398 && tmp399))
    {
      tmp401 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[111] /* Radiator.vol[5].X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp400),tmp401);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].X_start[1] >= 0.0 and Radiator.vol[5].X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp402 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 965
type: ALGORITHM

  assert(Radiator.vol[5].T_start >= 1.0 and Radiator.vol[5].T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].T_start <= 10000.0, has value: " + String(Radiator.vol[5].T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,965};
  modelica_boolean tmp403;
  modelica_boolean tmp404;
  static const MMC_DEFSTRINGLIT(tmp405,93,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].T_start <= 10000.0, has value: ");
  modelica_string tmp406;
  static int tmp407 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp407)
  {
    tmp403 = GreaterEq(data->simulationInfo->realParameter[101] /* Radiator.vol[5].T_start PARAM */,1.0);
    tmp404 = LessEq(data->simulationInfo->realParameter[101] /* Radiator.vol[5].T_start PARAM */,10000.0);
    if(!(tmp403 && tmp404))
    {
      tmp406 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[101] /* Radiator.vol[5].T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp405),tmp406);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].T_start >= 1.0 and Radiator.vol[5].T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp407 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 966
type: ALGORITHM

  assert(Radiator.vol[5].p_start >= 0.0 and Radiator.vol[5].p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[5].p_start <= 100000000.0, has value: " + String(Radiator.vol[5].p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_966(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,966};
  modelica_boolean tmp408;
  modelica_boolean tmp409;
  static const MMC_DEFSTRINGLIT(tmp410,97,"Variable violating min/max constraint: 0.0 <= Radiator.vol[5].p_start <= 100000000.0, has value: ");
  modelica_string tmp411;
  static int tmp412 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp412)
  {
    tmp408 = GreaterEq(data->simulationInfo->realParameter[216] /* Radiator.vol[5].p_start PARAM */,0.0);
    tmp409 = LessEq(data->simulationInfo->realParameter[216] /* Radiator.vol[5].p_start PARAM */,100000000.0);
    if(!(tmp408 && tmp409))
    {
      tmp411 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[216] /* Radiator.vol[5].p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp410),tmp411);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].p_start >= 0.0 and Radiator.vol[5].p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp412 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 967
type: ALGORITHM

  assert(Radiator.vol[5].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,967};
  modelica_boolean tmp413;
  modelica_boolean tmp414;
  static const MMC_DEFSTRINGLIT(tmp415,178,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp416;
  static int tmp417 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp417)
  {
    tmp413 = GreaterEq(data->simulationInfo->integerParameter[56] /* Radiator.vol[5].traceDynamics PARAM */,1);
    tmp414 = LessEq(data->simulationInfo->integerParameter[56] /* Radiator.vol[5].traceDynamics PARAM */,4);
    if(!(tmp413 && tmp414))
    {
      tmp416 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[56] /* Radiator.vol[5].traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp415),tmp416);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp417 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 968
type: ALGORITHM

  assert(Radiator.vol[5].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_968(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,968};
  modelica_boolean tmp418;
  modelica_boolean tmp419;
  static const MMC_DEFSTRINGLIT(tmp420,182,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp421;
  static int tmp422 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp422)
  {
    tmp418 = GreaterEq(data->simulationInfo->integerParameter[51] /* Radiator.vol[5].substanceDynamics PARAM */,1);
    tmp419 = LessEq(data->simulationInfo->integerParameter[51] /* Radiator.vol[5].substanceDynamics PARAM */,4);
    if(!(tmp418 && tmp419))
    {
      tmp421 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[51] /* Radiator.vol[5].substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp420),tmp421);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp422 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 969
type: ALGORITHM

  assert(Radiator.vol[5].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,969};
  modelica_boolean tmp423;
  modelica_boolean tmp424;
  static const MMC_DEFSTRINGLIT(tmp425,177,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp426;
  static int tmp427 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp427)
  {
    tmp423 = GreaterEq(data->simulationInfo->integerParameter[41] /* Radiator.vol[5].massDynamics PARAM */,1);
    tmp424 = LessEq(data->simulationInfo->integerParameter[41] /* Radiator.vol[5].massDynamics PARAM */,4);
    if(!(tmp423 && tmp424))
    {
      tmp426 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[41] /* Radiator.vol[5].massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp425),tmp426);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp427 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 970
type: ALGORITHM

  assert(Radiator.vol[5].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[5].energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_970(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,970};
  modelica_boolean tmp428;
  modelica_boolean tmp429;
  static const MMC_DEFSTRINGLIT(tmp430,179,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[5].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp431;
  static int tmp432 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp432)
  {
    tmp428 = GreaterEq(data->simulationInfo->integerParameter[36] /* Radiator.vol[5].energyDynamics PARAM */,1);
    tmp429 = LessEq(data->simulationInfo->integerParameter[36] /* Radiator.vol[5].energyDynamics PARAM */,4);
    if(!(tmp428 && tmp429))
    {
      tmp431 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[36] /* Radiator.vol[5].energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp430),tmp431);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[5].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp432 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 971
type: ALGORITHM

  assert(Radiator.vol[4].state_start.T >= 1.0 and Radiator.vol[4].state_start.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].state_start.T <= 10000.0, has value: " + String(Radiator.vol[4].state_start.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,971};
  modelica_boolean tmp433;
  modelica_boolean tmp434;
  static const MMC_DEFSTRINGLIT(tmp435,99,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].state_start.T <= 10000.0, has value: ");
  modelica_string tmp436;
  static int tmp437 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp437)
  {
    tmp433 = GreaterEq(data->simulationInfo->realParameter[250] /* Radiator.vol[4].state_start.T PARAM */,1.0);
    tmp434 = LessEq(data->simulationInfo->realParameter[250] /* Radiator.vol[4].state_start.T PARAM */,10000.0);
    if(!(tmp433 && tmp434))
    {
      tmp436 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[250] /* Radiator.vol[4].state_start.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp435),tmp436);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].state_start.T >= 1.0 and Radiator.vol[4].state_start.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp437 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 972
type: ALGORITHM

  assert(Radiator.vol[4].state_start.p >= 0.0 and Radiator.vol[4].state_start.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].state_start.p <= 100000000.0, has value: " + String(Radiator.vol[4].state_start.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,972};
  modelica_boolean tmp438;
  modelica_boolean tmp439;
  static const MMC_DEFSTRINGLIT(tmp440,103,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].state_start.p <= 100000000.0, has value: ");
  modelica_string tmp441;
  static int tmp442 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp442)
  {
    tmp438 = GreaterEq(data->simulationInfo->realParameter[255] /* Radiator.vol[4].state_start.p PARAM */,0.0);
    tmp439 = LessEq(data->simulationInfo->realParameter[255] /* Radiator.vol[4].state_start.p PARAM */,100000000.0);
    if(!(tmp438 && tmp439))
    {
      tmp441 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[255] /* Radiator.vol[4].state_start.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp440),tmp441);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].state_start.p >= 0.0 and Radiator.vol[4].state_start.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp442 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 973
type: ALGORITHM

  assert(Radiator.vol[4].rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].rho_default, has value: " + String(Radiator.vol[4].rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,973};
  modelica_boolean tmp443;
  static const MMC_DEFSTRINGLIT(tmp444,82,"Variable violating min constraint: 0.0 <= Radiator.vol[4].rho_default, has value: ");
  modelica_string tmp445;
  static int tmp446 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp446)
  {
    tmp443 = GreaterEq(data->simulationInfo->realParameter[230] /* Radiator.vol[4].rho_default PARAM */,0.0);
    if(!tmp443)
    {
      tmp445 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[230] /* Radiator.vol[4].rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp444),tmp445);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",96,3,97,63,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp446 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 974
type: ALGORITHM

  assert(Radiator.vol[4].state_default.T >= 1.0 and Radiator.vol[4].state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].state_default.T <= 10000.0, has value: " + String(Radiator.vol[4].state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_974(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,974};
  modelica_boolean tmp447;
  modelica_boolean tmp448;
  static const MMC_DEFSTRINGLIT(tmp449,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].state_default.T <= 10000.0, has value: ");
  modelica_string tmp450;
  static int tmp451 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp451)
  {
    tmp447 = GreaterEq(data->simulationInfo->realParameter[240] /* Radiator.vol[4].state_default.T PARAM */,1.0);
    tmp448 = LessEq(data->simulationInfo->realParameter[240] /* Radiator.vol[4].state_default.T PARAM */,10000.0);
    if(!(tmp447 && tmp448))
    {
      tmp450 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[240] /* Radiator.vol[4].state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp449),tmp450);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].state_default.T >= 1.0 and Radiator.vol[4].state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp451 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 975
type: ALGORITHM

  assert(Radiator.vol[4].state_default.p >= 0.0 and Radiator.vol[4].state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].state_default.p <= 100000000.0, has value: " + String(Radiator.vol[4].state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,975};
  modelica_boolean tmp452;
  modelica_boolean tmp453;
  static const MMC_DEFSTRINGLIT(tmp454,105,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].state_default.p <= 100000000.0, has value: ");
  modelica_string tmp455;
  static int tmp456 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp456)
  {
    tmp452 = GreaterEq(data->simulationInfo->realParameter[245] /* Radiator.vol[4].state_default.p PARAM */,0.0);
    tmp453 = LessEq(data->simulationInfo->realParameter[245] /* Radiator.vol[4].state_default.p PARAM */,100000000.0);
    if(!(tmp452 && tmp453))
    {
      tmp455 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[245] /* Radiator.vol[4].state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp454),tmp455);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].state_default.p >= 0.0 and Radiator.vol[4].state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp456 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 976
type: ALGORITHM

  assert(Radiator.vol[4].rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].rho_start, has value: " + String(Radiator.vol[4].rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_976(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,976};
  modelica_boolean tmp457;
  static const MMC_DEFSTRINGLIT(tmp458,80,"Variable violating min constraint: 0.0 <= Radiator.vol[4].rho_start, has value: ");
  modelica_string tmp459;
  static int tmp460 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp460)
  {
    tmp457 = GreaterEq(data->simulationInfo->realParameter[235] /* Radiator.vol[4].rho_start PARAM */,0.0);
    if(!tmp457)
    {
      tmp459 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[235] /* Radiator.vol[4].rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp458),tmp459);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",89,3,90,73,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp460 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 977
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.rho_default, has value: " + String(Radiator.vol[4].dynBal.rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,977};
  modelica_boolean tmp461;
  static const MMC_DEFSTRINGLIT(tmp462,89,"Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.rho_default, has value: ");
  modelica_string tmp463;
  static int tmp464 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp464)
  {
    tmp461 = GreaterEq(data->simulationInfo->realParameter[175] /* Radiator.vol[4].dynBal.rho_default PARAM */,0.0);
    if(!tmp461)
    {
      tmp463 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[175] /* Radiator.vol[4].dynBal.rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp462),tmp463);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",145,3,146,59,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp464 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 978
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.state_default.T >= 1.0 and Radiator.vol[4].dynBal.state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.state_default.T <= 10000.0, has value: " + String(Radiator.vol[4].dynBal.state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_978(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,978};
  modelica_boolean tmp465;
  modelica_boolean tmp466;
  static const MMC_DEFSTRINGLIT(tmp467,108,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.state_default.T <= 10000.0, has value: ");
  modelica_string tmp468;
  static int tmp469 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp469)
  {
    tmp465 = GreaterEq(data->simulationInfo->realParameter[185] /* Radiator.vol[4].dynBal.state_default.T PARAM */,1.0);
    tmp466 = LessEq(data->simulationInfo->realParameter[185] /* Radiator.vol[4].dynBal.state_default.T PARAM */,10000.0);
    if(!(tmp465 && tmp466))
    {
      tmp468 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[185] /* Radiator.vol[4].dynBal.state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp467),tmp468);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.state_default.T >= 1.0 and Radiator.vol[4].dynBal.state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp469 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 979
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.state_default.p >= 0.0 and Radiator.vol[4].dynBal.state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.state_default.p <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,979};
  modelica_boolean tmp470;
  modelica_boolean tmp471;
  static const MMC_DEFSTRINGLIT(tmp472,112,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.state_default.p <= 100000000.0, has value: ");
  modelica_string tmp473;
  static int tmp474 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp474)
  {
    tmp470 = GreaterEq(data->simulationInfo->realParameter[190] /* Radiator.vol[4].dynBal.state_default.p PARAM */,0.0);
    tmp471 = LessEq(data->simulationInfo->realParameter[190] /* Radiator.vol[4].dynBal.state_default.p PARAM */,100000000.0);
    if(!(tmp470 && tmp471))
    {
      tmp473 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[190] /* Radiator.vol[4].dynBal.state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp472),tmp473);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.state_default.p >= 0.0 and Radiator.vol[4].dynBal.state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp474 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 980
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.rho_start, has value: " + String(Radiator.vol[4].dynBal.rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,980};
  modelica_boolean tmp475;
  static const MMC_DEFSTRINGLIT(tmp476,87,"Variable violating min constraint: 0.0 <= Radiator.vol[4].dynBal.rho_start, has value: ");
  modelica_string tmp477;
  static int tmp478 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp478)
  {
    tmp475 = GreaterEq(data->simulationInfo->realParameter[180] /* Radiator.vol[4].dynBal.rho_start PARAM */,0.0);
    if(!tmp475)
    {
      tmp477 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[180] /* Radiator.vol[4].dynBal.rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp476),tmp477);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",131,3,135,70,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp478 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 981
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[4].dynBal.mSenFac, has value: " + String(Radiator.vol[4].dynBal.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,981};
  modelica_boolean tmp479;
  static const MMC_DEFSTRINGLIT(tmp480,85,"Variable violating min constraint: 1.0 <= Radiator.vol[4].dynBal.mSenFac, has value: ");
  modelica_string tmp481;
  static int tmp482 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp482)
  {
    tmp479 = GreaterEq(data->simulationInfo->realParameter[145] /* Radiator.vol[4].dynBal.mSenFac PARAM */,1.0);
    if(!tmp479)
    {
      tmp481 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[145] /* Radiator.vol[4].dynBal.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp480),tmp481);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp482 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 982
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.X_start[1] >= 0.0 and Radiator.vol[4].dynBal.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.X_start[1] <= 1.0, has value: " + String(Radiator.vol[4].dynBal.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_982(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,982};
  modelica_boolean tmp483;
  modelica_boolean tmp484;
  static const MMC_DEFSTRINGLIT(tmp485,99,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.X_start[1] <= 1.0, has value: ");
  modelica_string tmp486;
  static int tmp487 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp487)
  {
    tmp483 = GreaterEq(data->simulationInfo->realParameter[125] /* Radiator.vol[4].dynBal.X_start[1] PARAM */,0.0);
    tmp484 = LessEq(data->simulationInfo->realParameter[125] /* Radiator.vol[4].dynBal.X_start[1] PARAM */,1.0);
    if(!(tmp483 && tmp484))
    {
      tmp486 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[125] /* Radiator.vol[4].dynBal.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp485),tmp486);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.X_start[1] >= 0.0 and Radiator.vol[4].dynBal.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp487 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 983
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.T_start >= 1.0 and Radiator.vol[4].dynBal.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.T_start <= 10000.0, has value: " + String(Radiator.vol[4].dynBal.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,983};
  modelica_boolean tmp488;
  modelica_boolean tmp489;
  static const MMC_DEFSTRINGLIT(tmp490,100,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.T_start <= 10000.0, has value: ");
  modelica_string tmp491;
  static int tmp492 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp492)
  {
    tmp488 = GreaterEq(data->simulationInfo->realParameter[120] /* Radiator.vol[4].dynBal.T_start PARAM */,1.0);
    tmp489 = LessEq(data->simulationInfo->realParameter[120] /* Radiator.vol[4].dynBal.T_start PARAM */,10000.0);
    if(!(tmp488 && tmp489))
    {
      tmp491 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[120] /* Radiator.vol[4].dynBal.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp490),tmp491);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.T_start >= 1.0 and Radiator.vol[4].dynBal.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp492 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 984
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.p_start >= 0.0 and Radiator.vol[4].dynBal.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.p_start <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_984(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,984};
  modelica_boolean tmp493;
  modelica_boolean tmp494;
  static const MMC_DEFSTRINGLIT(tmp495,104,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].dynBal.p_start <= 100000000.0, has value: ");
  modelica_string tmp496;
  static int tmp497 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp497)
  {
    tmp493 = GreaterEq(data->simulationInfo->realParameter[160] /* Radiator.vol[4].dynBal.p_start PARAM */,0.0);
    tmp494 = LessEq(data->simulationInfo->realParameter[160] /* Radiator.vol[4].dynBal.p_start PARAM */,100000000.0);
    if(!(tmp493 && tmp494))
    {
      tmp496 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[160] /* Radiator.vol[4].dynBal.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp495),tmp496);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.p_start >= 0.0 and Radiator.vol[4].dynBal.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp497 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 985
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].dynBal.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,985};
  modelica_boolean tmp498;
  modelica_boolean tmp499;
  static const MMC_DEFSTRINGLIT(tmp500,185,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp501;
  static int tmp502 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp502)
  {
    tmp498 = GreaterEq(data->simulationInfo->integerParameter[30] /* Radiator.vol[4].dynBal.traceDynamics PARAM */,1);
    tmp499 = LessEq(data->simulationInfo->integerParameter[30] /* Radiator.vol[4].dynBal.traceDynamics PARAM */,4);
    if(!(tmp498 && tmp499))
    {
      tmp501 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[30] /* Radiator.vol[4].dynBal.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp500),tmp501);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp502 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 986
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].dynBal.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_986(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,986};
  modelica_boolean tmp503;
  modelica_boolean tmp504;
  static const MMC_DEFSTRINGLIT(tmp505,189,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp506;
  static int tmp507 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp507)
  {
    tmp503 = GreaterEq(data->simulationInfo->integerParameter[25] /* Radiator.vol[4].dynBal.substanceDynamics PARAM */,1);
    tmp504 = LessEq(data->simulationInfo->integerParameter[25] /* Radiator.vol[4].dynBal.substanceDynamics PARAM */,4);
    if(!(tmp503 && tmp504))
    {
      tmp506 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[25] /* Radiator.vol[4].dynBal.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp505),tmp506);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp507 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 987
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].dynBal.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,987};
  modelica_boolean tmp508;
  modelica_boolean tmp509;
  static const MMC_DEFSTRINGLIT(tmp510,184,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp511;
  static int tmp512 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp512)
  {
    tmp508 = GreaterEq(data->simulationInfo->integerParameter[15] /* Radiator.vol[4].dynBal.massDynamics PARAM */,1);
    tmp509 = LessEq(data->simulationInfo->integerParameter[15] /* Radiator.vol[4].dynBal.massDynamics PARAM */,4);
    if(!(tmp508 && tmp509))
    {
      tmp511 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[15] /* Radiator.vol[4].dynBal.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp510),tmp511);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp512 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 988
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].dynBal.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,988};
  modelica_boolean tmp513;
  modelica_boolean tmp514;
  static const MMC_DEFSTRINGLIT(tmp515,186,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp516;
  static int tmp517 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp517)
  {
    tmp513 = GreaterEq(data->simulationInfo->integerParameter[10] /* Radiator.vol[4].dynBal.energyDynamics PARAM */,1);
    tmp514 = LessEq(data->simulationInfo->integerParameter[10] /* Radiator.vol[4].dynBal.energyDynamics PARAM */,4);
    if(!(tmp513 && tmp514))
    {
      tmp516 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[10] /* Radiator.vol[4].dynBal.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp515),tmp516);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp517 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 989
type: ALGORITHM

  assert(Radiator.vol[4].m_flow_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].m_flow_nominal, has value: " + String(Radiator.vol[4].m_flow_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,989};
  modelica_boolean tmp518;
  static const MMC_DEFSTRINGLIT(tmp519,85,"Variable violating min constraint: 0.0 <= Radiator.vol[4].m_flow_nominal, has value: ");
  modelica_string tmp520;
  static int tmp521 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp521)
  {
    tmp518 = GreaterEq(data->simulationInfo->realParameter[200] /* Radiator.vol[4].m_flow_nominal PARAM */,0.0);
    if(!tmp518)
    {
      tmp520 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[200] /* Radiator.vol[4].m_flow_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp519),tmp520);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",20,3,21,76,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].m_flow_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp521 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 990
type: ALGORITHM

  assert(Radiator.vol[4].m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[4].m_flow_small, has value: " + String(Radiator.vol[4].m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_990(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,990};
  modelica_boolean tmp522;
  static const MMC_DEFSTRINGLIT(tmp523,83,"Variable violating min constraint: 0.0 <= Radiator.vol[4].m_flow_small, has value: ");
  modelica_string tmp524;
  static int tmp525 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp525)
  {
    tmp522 = GreaterEq(data->simulationInfo->realParameter[205] /* Radiator.vol[4].m_flow_small PARAM */,0.0);
    if(!tmp522)
    {
      tmp524 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[205] /* Radiator.vol[4].m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp523),tmp524);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",25,3,27,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp525 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 991
type: ALGORITHM

  assert(Radiator.vol[4].mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[4].mSenFac, has value: " + String(Radiator.vol[4].mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,991};
  modelica_boolean tmp526;
  static const MMC_DEFSTRINGLIT(tmp527,78,"Variable violating min constraint: 1.0 <= Radiator.vol[4].mSenFac, has value: ");
  modelica_string tmp528;
  static int tmp529 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp529)
  {
    tmp526 = GreaterEq(data->simulationInfo->realParameter[195] /* Radiator.vol[4].mSenFac PARAM */,1.0);
    if(!tmp526)
    {
      tmp528 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[195] /* Radiator.vol[4].mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp527),tmp528);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp529 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 992
type: ALGORITHM

  assert(Radiator.vol[4].X_start[1] >= 0.0 and Radiator.vol[4].X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].X_start[1] <= 1.0, has value: " + String(Radiator.vol[4].X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_992(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,992};
  modelica_boolean tmp530;
  modelica_boolean tmp531;
  static const MMC_DEFSTRINGLIT(tmp532,92,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].X_start[1] <= 1.0, has value: ");
  modelica_string tmp533;
  static int tmp534 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp534)
  {
    tmp530 = GreaterEq(data->simulationInfo->realParameter[110] /* Radiator.vol[4].X_start[1] PARAM */,0.0);
    tmp531 = LessEq(data->simulationInfo->realParameter[110] /* Radiator.vol[4].X_start[1] PARAM */,1.0);
    if(!(tmp530 && tmp531))
    {
      tmp533 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[110] /* Radiator.vol[4].X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp532),tmp533);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].X_start[1] >= 0.0 and Radiator.vol[4].X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp534 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 993
type: ALGORITHM

  assert(Radiator.vol[4].T_start >= 1.0 and Radiator.vol[4].T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].T_start <= 10000.0, has value: " + String(Radiator.vol[4].T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,993};
  modelica_boolean tmp535;
  modelica_boolean tmp536;
  static const MMC_DEFSTRINGLIT(tmp537,93,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].T_start <= 10000.0, has value: ");
  modelica_string tmp538;
  static int tmp539 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp539)
  {
    tmp535 = GreaterEq(data->simulationInfo->realParameter[100] /* Radiator.vol[4].T_start PARAM */,1.0);
    tmp536 = LessEq(data->simulationInfo->realParameter[100] /* Radiator.vol[4].T_start PARAM */,10000.0);
    if(!(tmp535 && tmp536))
    {
      tmp538 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[100] /* Radiator.vol[4].T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp537),tmp538);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].T_start >= 1.0 and Radiator.vol[4].T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp539 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 994
type: ALGORITHM

  assert(Radiator.vol[4].p_start >= 0.0 and Radiator.vol[4].p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[4].p_start <= 100000000.0, has value: " + String(Radiator.vol[4].p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_994(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,994};
  modelica_boolean tmp540;
  modelica_boolean tmp541;
  static const MMC_DEFSTRINGLIT(tmp542,97,"Variable violating min/max constraint: 0.0 <= Radiator.vol[4].p_start <= 100000000.0, has value: ");
  modelica_string tmp543;
  static int tmp544 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp544)
  {
    tmp540 = GreaterEq(data->simulationInfo->realParameter[215] /* Radiator.vol[4].p_start PARAM */,0.0);
    tmp541 = LessEq(data->simulationInfo->realParameter[215] /* Radiator.vol[4].p_start PARAM */,100000000.0);
    if(!(tmp540 && tmp541))
    {
      tmp543 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[215] /* Radiator.vol[4].p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp542),tmp543);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].p_start >= 0.0 and Radiator.vol[4].p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp544 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 995
type: ALGORITHM

  assert(Radiator.vol[4].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,995};
  modelica_boolean tmp545;
  modelica_boolean tmp546;
  static const MMC_DEFSTRINGLIT(tmp547,178,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp548;
  static int tmp549 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp549)
  {
    tmp545 = GreaterEq(data->simulationInfo->integerParameter[55] /* Radiator.vol[4].traceDynamics PARAM */,1);
    tmp546 = LessEq(data->simulationInfo->integerParameter[55] /* Radiator.vol[4].traceDynamics PARAM */,4);
    if(!(tmp545 && tmp546))
    {
      tmp548 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[55] /* Radiator.vol[4].traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp547),tmp548);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp549 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 996
type: ALGORITHM

  assert(Radiator.vol[4].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,996};
  modelica_boolean tmp550;
  modelica_boolean tmp551;
  static const MMC_DEFSTRINGLIT(tmp552,182,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp553;
  static int tmp554 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp554)
  {
    tmp550 = GreaterEq(data->simulationInfo->integerParameter[50] /* Radiator.vol[4].substanceDynamics PARAM */,1);
    tmp551 = LessEq(data->simulationInfo->integerParameter[50] /* Radiator.vol[4].substanceDynamics PARAM */,4);
    if(!(tmp550 && tmp551))
    {
      tmp553 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[50] /* Radiator.vol[4].substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp552),tmp553);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp554 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 997
type: ALGORITHM

  assert(Radiator.vol[4].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,997};
  modelica_boolean tmp555;
  modelica_boolean tmp556;
  static const MMC_DEFSTRINGLIT(tmp557,177,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp558;
  static int tmp559 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp559)
  {
    tmp555 = GreaterEq(data->simulationInfo->integerParameter[40] /* Radiator.vol[4].massDynamics PARAM */,1);
    tmp556 = LessEq(data->simulationInfo->integerParameter[40] /* Radiator.vol[4].massDynamics PARAM */,4);
    if(!(tmp555 && tmp556))
    {
      tmp558 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[40] /* Radiator.vol[4].massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp557),tmp558);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp559 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 998
type: ALGORITHM

  assert(Radiator.vol[4].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[4].energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_998(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,998};
  modelica_boolean tmp560;
  modelica_boolean tmp561;
  static const MMC_DEFSTRINGLIT(tmp562,179,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[4].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp563;
  static int tmp564 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp564)
  {
    tmp560 = GreaterEq(data->simulationInfo->integerParameter[35] /* Radiator.vol[4].energyDynamics PARAM */,1);
    tmp561 = LessEq(data->simulationInfo->integerParameter[35] /* Radiator.vol[4].energyDynamics PARAM */,4);
    if(!(tmp560 && tmp561))
    {
      tmp563 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[35] /* Radiator.vol[4].energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp562),tmp563);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[4].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp564 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 999
type: ALGORITHM

  assert(Radiator.vol[3].state_start.T >= 1.0 and Radiator.vol[3].state_start.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].state_start.T <= 10000.0, has value: " + String(Radiator.vol[3].state_start.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,999};
  modelica_boolean tmp565;
  modelica_boolean tmp566;
  static const MMC_DEFSTRINGLIT(tmp567,99,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].state_start.T <= 10000.0, has value: ");
  modelica_string tmp568;
  static int tmp569 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp569)
  {
    tmp565 = GreaterEq(data->simulationInfo->realParameter[249] /* Radiator.vol[3].state_start.T PARAM */,1.0);
    tmp566 = LessEq(data->simulationInfo->realParameter[249] /* Radiator.vol[3].state_start.T PARAM */,10000.0);
    if(!(tmp565 && tmp566))
    {
      tmp568 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[249] /* Radiator.vol[3].state_start.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp567),tmp568);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].state_start.T >= 1.0 and Radiator.vol[3].state_start.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp569 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1000
type: ALGORITHM

  assert(Radiator.vol[3].state_start.p >= 0.0 and Radiator.vol[3].state_start.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].state_start.p <= 100000000.0, has value: " + String(Radiator.vol[3].state_start.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1000(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1000};
  modelica_boolean tmp570;
  modelica_boolean tmp571;
  static const MMC_DEFSTRINGLIT(tmp572,103,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].state_start.p <= 100000000.0, has value: ");
  modelica_string tmp573;
  static int tmp574 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp574)
  {
    tmp570 = GreaterEq(data->simulationInfo->realParameter[254] /* Radiator.vol[3].state_start.p PARAM */,0.0);
    tmp571 = LessEq(data->simulationInfo->realParameter[254] /* Radiator.vol[3].state_start.p PARAM */,100000000.0);
    if(!(tmp570 && tmp571))
    {
      tmp573 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[254] /* Radiator.vol[3].state_start.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp572),tmp573);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].state_start.p >= 0.0 and Radiator.vol[3].state_start.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp574 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1001
type: ALGORITHM

  assert(Radiator.vol[3].rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].rho_default, has value: " + String(Radiator.vol[3].rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1001};
  modelica_boolean tmp575;
  static const MMC_DEFSTRINGLIT(tmp576,82,"Variable violating min constraint: 0.0 <= Radiator.vol[3].rho_default, has value: ");
  modelica_string tmp577;
  static int tmp578 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp578)
  {
    tmp575 = GreaterEq(data->simulationInfo->realParameter[229] /* Radiator.vol[3].rho_default PARAM */,0.0);
    if(!tmp575)
    {
      tmp577 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[229] /* Radiator.vol[3].rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp576),tmp577);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",96,3,97,63,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp578 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1002
type: ALGORITHM

  assert(Radiator.vol[3].state_default.T >= 1.0 and Radiator.vol[3].state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].state_default.T <= 10000.0, has value: " + String(Radiator.vol[3].state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1002(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1002};
  modelica_boolean tmp579;
  modelica_boolean tmp580;
  static const MMC_DEFSTRINGLIT(tmp581,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].state_default.T <= 10000.0, has value: ");
  modelica_string tmp582;
  static int tmp583 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp583)
  {
    tmp579 = GreaterEq(data->simulationInfo->realParameter[239] /* Radiator.vol[3].state_default.T PARAM */,1.0);
    tmp580 = LessEq(data->simulationInfo->realParameter[239] /* Radiator.vol[3].state_default.T PARAM */,10000.0);
    if(!(tmp579 && tmp580))
    {
      tmp582 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[239] /* Radiator.vol[3].state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp581),tmp582);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].state_default.T >= 1.0 and Radiator.vol[3].state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp583 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1003
type: ALGORITHM

  assert(Radiator.vol[3].state_default.p >= 0.0 and Radiator.vol[3].state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].state_default.p <= 100000000.0, has value: " + String(Radiator.vol[3].state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1003};
  modelica_boolean tmp584;
  modelica_boolean tmp585;
  static const MMC_DEFSTRINGLIT(tmp586,105,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].state_default.p <= 100000000.0, has value: ");
  modelica_string tmp587;
  static int tmp588 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp588)
  {
    tmp584 = GreaterEq(data->simulationInfo->realParameter[244] /* Radiator.vol[3].state_default.p PARAM */,0.0);
    tmp585 = LessEq(data->simulationInfo->realParameter[244] /* Radiator.vol[3].state_default.p PARAM */,100000000.0);
    if(!(tmp584 && tmp585))
    {
      tmp587 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[244] /* Radiator.vol[3].state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp586),tmp587);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].state_default.p >= 0.0 and Radiator.vol[3].state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp588 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1004
type: ALGORITHM

  assert(Radiator.vol[3].rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].rho_start, has value: " + String(Radiator.vol[3].rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1004};
  modelica_boolean tmp589;
  static const MMC_DEFSTRINGLIT(tmp590,80,"Variable violating min constraint: 0.0 <= Radiator.vol[3].rho_start, has value: ");
  modelica_string tmp591;
  static int tmp592 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp592)
  {
    tmp589 = GreaterEq(data->simulationInfo->realParameter[234] /* Radiator.vol[3].rho_start PARAM */,0.0);
    if(!tmp589)
    {
      tmp591 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[234] /* Radiator.vol[3].rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp590),tmp591);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",89,3,90,73,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp592 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1005
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.rho_default, has value: " + String(Radiator.vol[3].dynBal.rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1005};
  modelica_boolean tmp593;
  static const MMC_DEFSTRINGLIT(tmp594,89,"Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.rho_default, has value: ");
  modelica_string tmp595;
  static int tmp596 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp596)
  {
    tmp593 = GreaterEq(data->simulationInfo->realParameter[174] /* Radiator.vol[3].dynBal.rho_default PARAM */,0.0);
    if(!tmp593)
    {
      tmp595 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[174] /* Radiator.vol[3].dynBal.rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp594),tmp595);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",145,3,146,59,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp596 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1006
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.state_default.T >= 1.0 and Radiator.vol[3].dynBal.state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.state_default.T <= 10000.0, has value: " + String(Radiator.vol[3].dynBal.state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1006(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1006};
  modelica_boolean tmp597;
  modelica_boolean tmp598;
  static const MMC_DEFSTRINGLIT(tmp599,108,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.state_default.T <= 10000.0, has value: ");
  modelica_string tmp600;
  static int tmp601 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp601)
  {
    tmp597 = GreaterEq(data->simulationInfo->realParameter[184] /* Radiator.vol[3].dynBal.state_default.T PARAM */,1.0);
    tmp598 = LessEq(data->simulationInfo->realParameter[184] /* Radiator.vol[3].dynBal.state_default.T PARAM */,10000.0);
    if(!(tmp597 && tmp598))
    {
      tmp600 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[184] /* Radiator.vol[3].dynBal.state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp599),tmp600);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.state_default.T >= 1.0 and Radiator.vol[3].dynBal.state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp601 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1007
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.state_default.p >= 0.0 and Radiator.vol[3].dynBal.state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.state_default.p <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1007};
  modelica_boolean tmp602;
  modelica_boolean tmp603;
  static const MMC_DEFSTRINGLIT(tmp604,112,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.state_default.p <= 100000000.0, has value: ");
  modelica_string tmp605;
  static int tmp606 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp606)
  {
    tmp602 = GreaterEq(data->simulationInfo->realParameter[189] /* Radiator.vol[3].dynBal.state_default.p PARAM */,0.0);
    tmp603 = LessEq(data->simulationInfo->realParameter[189] /* Radiator.vol[3].dynBal.state_default.p PARAM */,100000000.0);
    if(!(tmp602 && tmp603))
    {
      tmp605 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[189] /* Radiator.vol[3].dynBal.state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp604),tmp605);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.state_default.p >= 0.0 and Radiator.vol[3].dynBal.state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp606 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1008
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.rho_start, has value: " + String(Radiator.vol[3].dynBal.rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1008(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1008};
  modelica_boolean tmp607;
  static const MMC_DEFSTRINGLIT(tmp608,87,"Variable violating min constraint: 0.0 <= Radiator.vol[3].dynBal.rho_start, has value: ");
  modelica_string tmp609;
  static int tmp610 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp610)
  {
    tmp607 = GreaterEq(data->simulationInfo->realParameter[179] /* Radiator.vol[3].dynBal.rho_start PARAM */,0.0);
    if(!tmp607)
    {
      tmp609 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[179] /* Radiator.vol[3].dynBal.rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp608),tmp609);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",131,3,135,70,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp610 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1009
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[3].dynBal.mSenFac, has value: " + String(Radiator.vol[3].dynBal.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1009};
  modelica_boolean tmp611;
  static const MMC_DEFSTRINGLIT(tmp612,85,"Variable violating min constraint: 1.0 <= Radiator.vol[3].dynBal.mSenFac, has value: ");
  modelica_string tmp613;
  static int tmp614 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp614)
  {
    tmp611 = GreaterEq(data->simulationInfo->realParameter[144] /* Radiator.vol[3].dynBal.mSenFac PARAM */,1.0);
    if(!tmp611)
    {
      tmp613 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[144] /* Radiator.vol[3].dynBal.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp612),tmp613);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp614 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1010
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.X_start[1] >= 0.0 and Radiator.vol[3].dynBal.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.X_start[1] <= 1.0, has value: " + String(Radiator.vol[3].dynBal.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1010(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1010};
  modelica_boolean tmp615;
  modelica_boolean tmp616;
  static const MMC_DEFSTRINGLIT(tmp617,99,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.X_start[1] <= 1.0, has value: ");
  modelica_string tmp618;
  static int tmp619 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp619)
  {
    tmp615 = GreaterEq(data->simulationInfo->realParameter[124] /* Radiator.vol[3].dynBal.X_start[1] PARAM */,0.0);
    tmp616 = LessEq(data->simulationInfo->realParameter[124] /* Radiator.vol[3].dynBal.X_start[1] PARAM */,1.0);
    if(!(tmp615 && tmp616))
    {
      tmp618 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[124] /* Radiator.vol[3].dynBal.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp617),tmp618);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.X_start[1] >= 0.0 and Radiator.vol[3].dynBal.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp619 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1011
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.T_start >= 1.0 and Radiator.vol[3].dynBal.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.T_start <= 10000.0, has value: " + String(Radiator.vol[3].dynBal.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1011};
  modelica_boolean tmp620;
  modelica_boolean tmp621;
  static const MMC_DEFSTRINGLIT(tmp622,100,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.T_start <= 10000.0, has value: ");
  modelica_string tmp623;
  static int tmp624 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp624)
  {
    tmp620 = GreaterEq(data->simulationInfo->realParameter[119] /* Radiator.vol[3].dynBal.T_start PARAM */,1.0);
    tmp621 = LessEq(data->simulationInfo->realParameter[119] /* Radiator.vol[3].dynBal.T_start PARAM */,10000.0);
    if(!(tmp620 && tmp621))
    {
      tmp623 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[119] /* Radiator.vol[3].dynBal.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp622),tmp623);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.T_start >= 1.0 and Radiator.vol[3].dynBal.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp624 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1012
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.p_start >= 0.0 and Radiator.vol[3].dynBal.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.p_start <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1012};
  modelica_boolean tmp625;
  modelica_boolean tmp626;
  static const MMC_DEFSTRINGLIT(tmp627,104,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].dynBal.p_start <= 100000000.0, has value: ");
  modelica_string tmp628;
  static int tmp629 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp629)
  {
    tmp625 = GreaterEq(data->simulationInfo->realParameter[159] /* Radiator.vol[3].dynBal.p_start PARAM */,0.0);
    tmp626 = LessEq(data->simulationInfo->realParameter[159] /* Radiator.vol[3].dynBal.p_start PARAM */,100000000.0);
    if(!(tmp625 && tmp626))
    {
      tmp628 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[159] /* Radiator.vol[3].dynBal.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp627),tmp628);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.p_start >= 0.0 and Radiator.vol[3].dynBal.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp629 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1013
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].dynBal.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1013};
  modelica_boolean tmp630;
  modelica_boolean tmp631;
  static const MMC_DEFSTRINGLIT(tmp632,185,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp633;
  static int tmp634 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp634)
  {
    tmp630 = GreaterEq(data->simulationInfo->integerParameter[29] /* Radiator.vol[3].dynBal.traceDynamics PARAM */,1);
    tmp631 = LessEq(data->simulationInfo->integerParameter[29] /* Radiator.vol[3].dynBal.traceDynamics PARAM */,4);
    if(!(tmp630 && tmp631))
    {
      tmp633 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[29] /* Radiator.vol[3].dynBal.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp632),tmp633);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp634 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1014
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].dynBal.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1014(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1014};
  modelica_boolean tmp635;
  modelica_boolean tmp636;
  static const MMC_DEFSTRINGLIT(tmp637,189,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp638;
  static int tmp639 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp639)
  {
    tmp635 = GreaterEq(data->simulationInfo->integerParameter[24] /* Radiator.vol[3].dynBal.substanceDynamics PARAM */,1);
    tmp636 = LessEq(data->simulationInfo->integerParameter[24] /* Radiator.vol[3].dynBal.substanceDynamics PARAM */,4);
    if(!(tmp635 && tmp636))
    {
      tmp638 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[24] /* Radiator.vol[3].dynBal.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp637),tmp638);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp639 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1015
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].dynBal.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1015};
  modelica_boolean tmp640;
  modelica_boolean tmp641;
  static const MMC_DEFSTRINGLIT(tmp642,184,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp643;
  static int tmp644 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp644)
  {
    tmp640 = GreaterEq(data->simulationInfo->integerParameter[14] /* Radiator.vol[3].dynBal.massDynamics PARAM */,1);
    tmp641 = LessEq(data->simulationInfo->integerParameter[14] /* Radiator.vol[3].dynBal.massDynamics PARAM */,4);
    if(!(tmp640 && tmp641))
    {
      tmp643 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[14] /* Radiator.vol[3].dynBal.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp642),tmp643);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp644 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1016
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].dynBal.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1016(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1016};
  modelica_boolean tmp645;
  modelica_boolean tmp646;
  static const MMC_DEFSTRINGLIT(tmp647,186,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp648;
  static int tmp649 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp649)
  {
    tmp645 = GreaterEq(data->simulationInfo->integerParameter[9] /* Radiator.vol[3].dynBal.energyDynamics PARAM */,1);
    tmp646 = LessEq(data->simulationInfo->integerParameter[9] /* Radiator.vol[3].dynBal.energyDynamics PARAM */,4);
    if(!(tmp645 && tmp646))
    {
      tmp648 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[9] /* Radiator.vol[3].dynBal.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp647),tmp648);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp649 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1017
type: ALGORITHM

  assert(Radiator.vol[3].m_flow_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].m_flow_nominal, has value: " + String(Radiator.vol[3].m_flow_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1017};
  modelica_boolean tmp650;
  static const MMC_DEFSTRINGLIT(tmp651,85,"Variable violating min constraint: 0.0 <= Radiator.vol[3].m_flow_nominal, has value: ");
  modelica_string tmp652;
  static int tmp653 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp653)
  {
    tmp650 = GreaterEq(data->simulationInfo->realParameter[199] /* Radiator.vol[3].m_flow_nominal PARAM */,0.0);
    if(!tmp650)
    {
      tmp652 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[199] /* Radiator.vol[3].m_flow_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp651),tmp652);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",20,3,21,76,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].m_flow_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp653 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1018
type: ALGORITHM

  assert(Radiator.vol[3].m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[3].m_flow_small, has value: " + String(Radiator.vol[3].m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1018(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1018};
  modelica_boolean tmp654;
  static const MMC_DEFSTRINGLIT(tmp655,83,"Variable violating min constraint: 0.0 <= Radiator.vol[3].m_flow_small, has value: ");
  modelica_string tmp656;
  static int tmp657 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp657)
  {
    tmp654 = GreaterEq(data->simulationInfo->realParameter[204] /* Radiator.vol[3].m_flow_small PARAM */,0.0);
    if(!tmp654)
    {
      tmp656 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[204] /* Radiator.vol[3].m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp655),tmp656);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",25,3,27,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp657 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1019
type: ALGORITHM

  assert(Radiator.vol[3].mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[3].mSenFac, has value: " + String(Radiator.vol[3].mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1019};
  modelica_boolean tmp658;
  static const MMC_DEFSTRINGLIT(tmp659,78,"Variable violating min constraint: 1.0 <= Radiator.vol[3].mSenFac, has value: ");
  modelica_string tmp660;
  static int tmp661 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp661)
  {
    tmp658 = GreaterEq(data->simulationInfo->realParameter[194] /* Radiator.vol[3].mSenFac PARAM */,1.0);
    if(!tmp658)
    {
      tmp660 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[194] /* Radiator.vol[3].mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp659),tmp660);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp661 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1020
type: ALGORITHM

  assert(Radiator.vol[3].X_start[1] >= 0.0 and Radiator.vol[3].X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].X_start[1] <= 1.0, has value: " + String(Radiator.vol[3].X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1020};
  modelica_boolean tmp662;
  modelica_boolean tmp663;
  static const MMC_DEFSTRINGLIT(tmp664,92,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].X_start[1] <= 1.0, has value: ");
  modelica_string tmp665;
  static int tmp666 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp666)
  {
    tmp662 = GreaterEq(data->simulationInfo->realParameter[109] /* Radiator.vol[3].X_start[1] PARAM */,0.0);
    tmp663 = LessEq(data->simulationInfo->realParameter[109] /* Radiator.vol[3].X_start[1] PARAM */,1.0);
    if(!(tmp662 && tmp663))
    {
      tmp665 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[109] /* Radiator.vol[3].X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp664),tmp665);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].X_start[1] >= 0.0 and Radiator.vol[3].X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp666 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1021
type: ALGORITHM

  assert(Radiator.vol[3].T_start >= 1.0 and Radiator.vol[3].T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].T_start <= 10000.0, has value: " + String(Radiator.vol[3].T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1021};
  modelica_boolean tmp667;
  modelica_boolean tmp668;
  static const MMC_DEFSTRINGLIT(tmp669,93,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].T_start <= 10000.0, has value: ");
  modelica_string tmp670;
  static int tmp671 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp671)
  {
    tmp667 = GreaterEq(data->simulationInfo->realParameter[99] /* Radiator.vol[3].T_start PARAM */,1.0);
    tmp668 = LessEq(data->simulationInfo->realParameter[99] /* Radiator.vol[3].T_start PARAM */,10000.0);
    if(!(tmp667 && tmp668))
    {
      tmp670 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[99] /* Radiator.vol[3].T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp669),tmp670);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].T_start >= 1.0 and Radiator.vol[3].T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp671 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1022
type: ALGORITHM

  assert(Radiator.vol[3].p_start >= 0.0 and Radiator.vol[3].p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[3].p_start <= 100000000.0, has value: " + String(Radiator.vol[3].p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1022(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1022};
  modelica_boolean tmp672;
  modelica_boolean tmp673;
  static const MMC_DEFSTRINGLIT(tmp674,97,"Variable violating min/max constraint: 0.0 <= Radiator.vol[3].p_start <= 100000000.0, has value: ");
  modelica_string tmp675;
  static int tmp676 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp676)
  {
    tmp672 = GreaterEq(data->simulationInfo->realParameter[214] /* Radiator.vol[3].p_start PARAM */,0.0);
    tmp673 = LessEq(data->simulationInfo->realParameter[214] /* Radiator.vol[3].p_start PARAM */,100000000.0);
    if(!(tmp672 && tmp673))
    {
      tmp675 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[214] /* Radiator.vol[3].p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp674),tmp675);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].p_start >= 0.0 and Radiator.vol[3].p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp676 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1023
type: ALGORITHM

  assert(Radiator.vol[3].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1023};
  modelica_boolean tmp677;
  modelica_boolean tmp678;
  static const MMC_DEFSTRINGLIT(tmp679,178,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp680;
  static int tmp681 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp681)
  {
    tmp677 = GreaterEq(data->simulationInfo->integerParameter[54] /* Radiator.vol[3].traceDynamics PARAM */,1);
    tmp678 = LessEq(data->simulationInfo->integerParameter[54] /* Radiator.vol[3].traceDynamics PARAM */,4);
    if(!(tmp677 && tmp678))
    {
      tmp680 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[54] /* Radiator.vol[3].traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp679),tmp680);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp681 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1024
type: ALGORITHM

  assert(Radiator.vol[3].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1024(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1024};
  modelica_boolean tmp682;
  modelica_boolean tmp683;
  static const MMC_DEFSTRINGLIT(tmp684,182,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp685;
  static int tmp686 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp686)
  {
    tmp682 = GreaterEq(data->simulationInfo->integerParameter[49] /* Radiator.vol[3].substanceDynamics PARAM */,1);
    tmp683 = LessEq(data->simulationInfo->integerParameter[49] /* Radiator.vol[3].substanceDynamics PARAM */,4);
    if(!(tmp682 && tmp683))
    {
      tmp685 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[49] /* Radiator.vol[3].substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp684),tmp685);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp686 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1025
type: ALGORITHM

  assert(Radiator.vol[3].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1025};
  modelica_boolean tmp687;
  modelica_boolean tmp688;
  static const MMC_DEFSTRINGLIT(tmp689,177,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp690;
  static int tmp691 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp691)
  {
    tmp687 = GreaterEq(data->simulationInfo->integerParameter[39] /* Radiator.vol[3].massDynamics PARAM */,1);
    tmp688 = LessEq(data->simulationInfo->integerParameter[39] /* Radiator.vol[3].massDynamics PARAM */,4);
    if(!(tmp687 && tmp688))
    {
      tmp690 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[39] /* Radiator.vol[3].massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp689),tmp690);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp691 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1026
type: ALGORITHM

  assert(Radiator.vol[3].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[3].energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1026(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1026};
  modelica_boolean tmp692;
  modelica_boolean tmp693;
  static const MMC_DEFSTRINGLIT(tmp694,179,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[3].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp695;
  static int tmp696 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp696)
  {
    tmp692 = GreaterEq(data->simulationInfo->integerParameter[34] /* Radiator.vol[3].energyDynamics PARAM */,1);
    tmp693 = LessEq(data->simulationInfo->integerParameter[34] /* Radiator.vol[3].energyDynamics PARAM */,4);
    if(!(tmp692 && tmp693))
    {
      tmp695 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[34] /* Radiator.vol[3].energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp694),tmp695);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[3].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp696 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1027
type: ALGORITHM

  assert(Radiator.vol[2].state_start.T >= 1.0 and Radiator.vol[2].state_start.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].state_start.T <= 10000.0, has value: " + String(Radiator.vol[2].state_start.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1027};
  modelica_boolean tmp697;
  modelica_boolean tmp698;
  static const MMC_DEFSTRINGLIT(tmp699,99,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].state_start.T <= 10000.0, has value: ");
  modelica_string tmp700;
  static int tmp701 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp701)
  {
    tmp697 = GreaterEq(data->simulationInfo->realParameter[248] /* Radiator.vol[2].state_start.T PARAM */,1.0);
    tmp698 = LessEq(data->simulationInfo->realParameter[248] /* Radiator.vol[2].state_start.T PARAM */,10000.0);
    if(!(tmp697 && tmp698))
    {
      tmp700 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[248] /* Radiator.vol[2].state_start.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp699),tmp700);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].state_start.T >= 1.0 and Radiator.vol[2].state_start.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp701 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1028
type: ALGORITHM

  assert(Radiator.vol[2].state_start.p >= 0.0 and Radiator.vol[2].state_start.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].state_start.p <= 100000000.0, has value: " + String(Radiator.vol[2].state_start.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1028};
  modelica_boolean tmp702;
  modelica_boolean tmp703;
  static const MMC_DEFSTRINGLIT(tmp704,103,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].state_start.p <= 100000000.0, has value: ");
  modelica_string tmp705;
  static int tmp706 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp706)
  {
    tmp702 = GreaterEq(data->simulationInfo->realParameter[253] /* Radiator.vol[2].state_start.p PARAM */,0.0);
    tmp703 = LessEq(data->simulationInfo->realParameter[253] /* Radiator.vol[2].state_start.p PARAM */,100000000.0);
    if(!(tmp702 && tmp703))
    {
      tmp705 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[253] /* Radiator.vol[2].state_start.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp704),tmp705);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].state_start.p >= 0.0 and Radiator.vol[2].state_start.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp706 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1029
type: ALGORITHM

  assert(Radiator.vol[2].rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].rho_default, has value: " + String(Radiator.vol[2].rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1029};
  modelica_boolean tmp707;
  static const MMC_DEFSTRINGLIT(tmp708,82,"Variable violating min constraint: 0.0 <= Radiator.vol[2].rho_default, has value: ");
  modelica_string tmp709;
  static int tmp710 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp710)
  {
    tmp707 = GreaterEq(data->simulationInfo->realParameter[228] /* Radiator.vol[2].rho_default PARAM */,0.0);
    if(!tmp707)
    {
      tmp709 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[228] /* Radiator.vol[2].rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp708),tmp709);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",96,3,97,63,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp710 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1030
type: ALGORITHM

  assert(Radiator.vol[2].state_default.T >= 1.0 and Radiator.vol[2].state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].state_default.T <= 10000.0, has value: " + String(Radiator.vol[2].state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1030(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1030};
  modelica_boolean tmp711;
  modelica_boolean tmp712;
  static const MMC_DEFSTRINGLIT(tmp713,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].state_default.T <= 10000.0, has value: ");
  modelica_string tmp714;
  static int tmp715 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp715)
  {
    tmp711 = GreaterEq(data->simulationInfo->realParameter[238] /* Radiator.vol[2].state_default.T PARAM */,1.0);
    tmp712 = LessEq(data->simulationInfo->realParameter[238] /* Radiator.vol[2].state_default.T PARAM */,10000.0);
    if(!(tmp711 && tmp712))
    {
      tmp714 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[238] /* Radiator.vol[2].state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp713),tmp714);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].state_default.T >= 1.0 and Radiator.vol[2].state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp715 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1031
type: ALGORITHM

  assert(Radiator.vol[2].state_default.p >= 0.0 and Radiator.vol[2].state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].state_default.p <= 100000000.0, has value: " + String(Radiator.vol[2].state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1031};
  modelica_boolean tmp716;
  modelica_boolean tmp717;
  static const MMC_DEFSTRINGLIT(tmp718,105,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].state_default.p <= 100000000.0, has value: ");
  modelica_string tmp719;
  static int tmp720 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp720)
  {
    tmp716 = GreaterEq(data->simulationInfo->realParameter[243] /* Radiator.vol[2].state_default.p PARAM */,0.0);
    tmp717 = LessEq(data->simulationInfo->realParameter[243] /* Radiator.vol[2].state_default.p PARAM */,100000000.0);
    if(!(tmp716 && tmp717))
    {
      tmp719 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[243] /* Radiator.vol[2].state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp718),tmp719);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].state_default.p >= 0.0 and Radiator.vol[2].state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp720 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1032
type: ALGORITHM

  assert(Radiator.vol[2].rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].rho_start, has value: " + String(Radiator.vol[2].rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1032(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1032};
  modelica_boolean tmp721;
  static const MMC_DEFSTRINGLIT(tmp722,80,"Variable violating min constraint: 0.0 <= Radiator.vol[2].rho_start, has value: ");
  modelica_string tmp723;
  static int tmp724 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp724)
  {
    tmp721 = GreaterEq(data->simulationInfo->realParameter[233] /* Radiator.vol[2].rho_start PARAM */,0.0);
    if(!tmp721)
    {
      tmp723 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[233] /* Radiator.vol[2].rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp722),tmp723);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",89,3,90,73,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp724 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1033
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.rho_default, has value: " + String(Radiator.vol[2].dynBal.rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1033};
  modelica_boolean tmp725;
  static const MMC_DEFSTRINGLIT(tmp726,89,"Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.rho_default, has value: ");
  modelica_string tmp727;
  static int tmp728 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp728)
  {
    tmp725 = GreaterEq(data->simulationInfo->realParameter[173] /* Radiator.vol[2].dynBal.rho_default PARAM */,0.0);
    if(!tmp725)
    {
      tmp727 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[173] /* Radiator.vol[2].dynBal.rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp726),tmp727);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",145,3,146,59,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp728 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1034
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.state_default.T >= 1.0 and Radiator.vol[2].dynBal.state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.state_default.T <= 10000.0, has value: " + String(Radiator.vol[2].dynBal.state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1034(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1034};
  modelica_boolean tmp729;
  modelica_boolean tmp730;
  static const MMC_DEFSTRINGLIT(tmp731,108,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.state_default.T <= 10000.0, has value: ");
  modelica_string tmp732;
  static int tmp733 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp733)
  {
    tmp729 = GreaterEq(data->simulationInfo->realParameter[183] /* Radiator.vol[2].dynBal.state_default.T PARAM */,1.0);
    tmp730 = LessEq(data->simulationInfo->realParameter[183] /* Radiator.vol[2].dynBal.state_default.T PARAM */,10000.0);
    if(!(tmp729 && tmp730))
    {
      tmp732 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[183] /* Radiator.vol[2].dynBal.state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp731),tmp732);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.state_default.T >= 1.0 and Radiator.vol[2].dynBal.state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp733 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1035
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.state_default.p >= 0.0 and Radiator.vol[2].dynBal.state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.state_default.p <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1035};
  modelica_boolean tmp734;
  modelica_boolean tmp735;
  static const MMC_DEFSTRINGLIT(tmp736,112,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.state_default.p <= 100000000.0, has value: ");
  modelica_string tmp737;
  static int tmp738 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp738)
  {
    tmp734 = GreaterEq(data->simulationInfo->realParameter[188] /* Radiator.vol[2].dynBal.state_default.p PARAM */,0.0);
    tmp735 = LessEq(data->simulationInfo->realParameter[188] /* Radiator.vol[2].dynBal.state_default.p PARAM */,100000000.0);
    if(!(tmp734 && tmp735))
    {
      tmp737 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[188] /* Radiator.vol[2].dynBal.state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp736),tmp737);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.state_default.p >= 0.0 and Radiator.vol[2].dynBal.state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp738 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1036
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.rho_start, has value: " + String(Radiator.vol[2].dynBal.rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1036};
  modelica_boolean tmp739;
  static const MMC_DEFSTRINGLIT(tmp740,87,"Variable violating min constraint: 0.0 <= Radiator.vol[2].dynBal.rho_start, has value: ");
  modelica_string tmp741;
  static int tmp742 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp742)
  {
    tmp739 = GreaterEq(data->simulationInfo->realParameter[178] /* Radiator.vol[2].dynBal.rho_start PARAM */,0.0);
    if(!tmp739)
    {
      tmp741 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[178] /* Radiator.vol[2].dynBal.rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp740),tmp741);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",131,3,135,70,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp742 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1037
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[2].dynBal.mSenFac, has value: " + String(Radiator.vol[2].dynBal.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1037};
  modelica_boolean tmp743;
  static const MMC_DEFSTRINGLIT(tmp744,85,"Variable violating min constraint: 1.0 <= Radiator.vol[2].dynBal.mSenFac, has value: ");
  modelica_string tmp745;
  static int tmp746 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp746)
  {
    tmp743 = GreaterEq(data->simulationInfo->realParameter[143] /* Radiator.vol[2].dynBal.mSenFac PARAM */,1.0);
    if(!tmp743)
    {
      tmp745 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[143] /* Radiator.vol[2].dynBal.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp744),tmp745);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp746 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1038
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.X_start[1] >= 0.0 and Radiator.vol[2].dynBal.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.X_start[1] <= 1.0, has value: " + String(Radiator.vol[2].dynBal.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1038(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1038};
  modelica_boolean tmp747;
  modelica_boolean tmp748;
  static const MMC_DEFSTRINGLIT(tmp749,99,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.X_start[1] <= 1.0, has value: ");
  modelica_string tmp750;
  static int tmp751 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp751)
  {
    tmp747 = GreaterEq(data->simulationInfo->realParameter[123] /* Radiator.vol[2].dynBal.X_start[1] PARAM */,0.0);
    tmp748 = LessEq(data->simulationInfo->realParameter[123] /* Radiator.vol[2].dynBal.X_start[1] PARAM */,1.0);
    if(!(tmp747 && tmp748))
    {
      tmp750 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[123] /* Radiator.vol[2].dynBal.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp749),tmp750);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.X_start[1] >= 0.0 and Radiator.vol[2].dynBal.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp751 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1039
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.T_start >= 1.0 and Radiator.vol[2].dynBal.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.T_start <= 10000.0, has value: " + String(Radiator.vol[2].dynBal.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1039};
  modelica_boolean tmp752;
  modelica_boolean tmp753;
  static const MMC_DEFSTRINGLIT(tmp754,100,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.T_start <= 10000.0, has value: ");
  modelica_string tmp755;
  static int tmp756 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp756)
  {
    tmp752 = GreaterEq(data->simulationInfo->realParameter[118] /* Radiator.vol[2].dynBal.T_start PARAM */,1.0);
    tmp753 = LessEq(data->simulationInfo->realParameter[118] /* Radiator.vol[2].dynBal.T_start PARAM */,10000.0);
    if(!(tmp752 && tmp753))
    {
      tmp755 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[118] /* Radiator.vol[2].dynBal.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp754),tmp755);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.T_start >= 1.0 and Radiator.vol[2].dynBal.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp756 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1040
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.p_start >= 0.0 and Radiator.vol[2].dynBal.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.p_start <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1040(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1040};
  modelica_boolean tmp757;
  modelica_boolean tmp758;
  static const MMC_DEFSTRINGLIT(tmp759,104,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].dynBal.p_start <= 100000000.0, has value: ");
  modelica_string tmp760;
  static int tmp761 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp761)
  {
    tmp757 = GreaterEq(data->simulationInfo->realParameter[158] /* Radiator.vol[2].dynBal.p_start PARAM */,0.0);
    tmp758 = LessEq(data->simulationInfo->realParameter[158] /* Radiator.vol[2].dynBal.p_start PARAM */,100000000.0);
    if(!(tmp757 && tmp758))
    {
      tmp760 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[158] /* Radiator.vol[2].dynBal.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp759),tmp760);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.p_start >= 0.0 and Radiator.vol[2].dynBal.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp761 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1041
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].dynBal.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1041};
  modelica_boolean tmp762;
  modelica_boolean tmp763;
  static const MMC_DEFSTRINGLIT(tmp764,185,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp765;
  static int tmp766 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp766)
  {
    tmp762 = GreaterEq(data->simulationInfo->integerParameter[28] /* Radiator.vol[2].dynBal.traceDynamics PARAM */,1);
    tmp763 = LessEq(data->simulationInfo->integerParameter[28] /* Radiator.vol[2].dynBal.traceDynamics PARAM */,4);
    if(!(tmp762 && tmp763))
    {
      tmp765 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[28] /* Radiator.vol[2].dynBal.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp764),tmp765);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp766 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1042
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].dynBal.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1042(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1042};
  modelica_boolean tmp767;
  modelica_boolean tmp768;
  static const MMC_DEFSTRINGLIT(tmp769,189,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp770;
  static int tmp771 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp771)
  {
    tmp767 = GreaterEq(data->simulationInfo->integerParameter[23] /* Radiator.vol[2].dynBal.substanceDynamics PARAM */,1);
    tmp768 = LessEq(data->simulationInfo->integerParameter[23] /* Radiator.vol[2].dynBal.substanceDynamics PARAM */,4);
    if(!(tmp767 && tmp768))
    {
      tmp770 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[23] /* Radiator.vol[2].dynBal.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp769),tmp770);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp771 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1043
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].dynBal.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1043};
  modelica_boolean tmp772;
  modelica_boolean tmp773;
  static const MMC_DEFSTRINGLIT(tmp774,184,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp775;
  static int tmp776 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp776)
  {
    tmp772 = GreaterEq(data->simulationInfo->integerParameter[13] /* Radiator.vol[2].dynBal.massDynamics PARAM */,1);
    tmp773 = LessEq(data->simulationInfo->integerParameter[13] /* Radiator.vol[2].dynBal.massDynamics PARAM */,4);
    if(!(tmp772 && tmp773))
    {
      tmp775 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[13] /* Radiator.vol[2].dynBal.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp774),tmp775);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp776 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1044
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].dynBal.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1044};
  modelica_boolean tmp777;
  modelica_boolean tmp778;
  static const MMC_DEFSTRINGLIT(tmp779,186,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp780;
  static int tmp781 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp781)
  {
    tmp777 = GreaterEq(data->simulationInfo->integerParameter[8] /* Radiator.vol[2].dynBal.energyDynamics PARAM */,1);
    tmp778 = LessEq(data->simulationInfo->integerParameter[8] /* Radiator.vol[2].dynBal.energyDynamics PARAM */,4);
    if(!(tmp777 && tmp778))
    {
      tmp780 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[8] /* Radiator.vol[2].dynBal.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp779),tmp780);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp781 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1045
type: ALGORITHM

  assert(Radiator.vol[2].m_flow_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].m_flow_nominal, has value: " + String(Radiator.vol[2].m_flow_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1045};
  modelica_boolean tmp782;
  static const MMC_DEFSTRINGLIT(tmp783,85,"Variable violating min constraint: 0.0 <= Radiator.vol[2].m_flow_nominal, has value: ");
  modelica_string tmp784;
  static int tmp785 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp785)
  {
    tmp782 = GreaterEq(data->simulationInfo->realParameter[198] /* Radiator.vol[2].m_flow_nominal PARAM */,0.0);
    if(!tmp782)
    {
      tmp784 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[198] /* Radiator.vol[2].m_flow_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp783),tmp784);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",20,3,21,76,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].m_flow_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp785 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1046
type: ALGORITHM

  assert(Radiator.vol[2].m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[2].m_flow_small, has value: " + String(Radiator.vol[2].m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1046(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1046};
  modelica_boolean tmp786;
  static const MMC_DEFSTRINGLIT(tmp787,83,"Variable violating min constraint: 0.0 <= Radiator.vol[2].m_flow_small, has value: ");
  modelica_string tmp788;
  static int tmp789 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp789)
  {
    tmp786 = GreaterEq(data->simulationInfo->realParameter[203] /* Radiator.vol[2].m_flow_small PARAM */,0.0);
    if(!tmp786)
    {
      tmp788 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[203] /* Radiator.vol[2].m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp787),tmp788);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",25,3,27,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp789 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1047
type: ALGORITHM

  assert(Radiator.vol[2].mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[2].mSenFac, has value: " + String(Radiator.vol[2].mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1047};
  modelica_boolean tmp790;
  static const MMC_DEFSTRINGLIT(tmp791,78,"Variable violating min constraint: 1.0 <= Radiator.vol[2].mSenFac, has value: ");
  modelica_string tmp792;
  static int tmp793 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp793)
  {
    tmp790 = GreaterEq(data->simulationInfo->realParameter[193] /* Radiator.vol[2].mSenFac PARAM */,1.0);
    if(!tmp790)
    {
      tmp792 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[193] /* Radiator.vol[2].mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp791),tmp792);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp793 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1048
type: ALGORITHM

  assert(Radiator.vol[2].X_start[1] >= 0.0 and Radiator.vol[2].X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].X_start[1] <= 1.0, has value: " + String(Radiator.vol[2].X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1048(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1048};
  modelica_boolean tmp794;
  modelica_boolean tmp795;
  static const MMC_DEFSTRINGLIT(tmp796,92,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].X_start[1] <= 1.0, has value: ");
  modelica_string tmp797;
  static int tmp798 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp798)
  {
    tmp794 = GreaterEq(data->simulationInfo->realParameter[108] /* Radiator.vol[2].X_start[1] PARAM */,0.0);
    tmp795 = LessEq(data->simulationInfo->realParameter[108] /* Radiator.vol[2].X_start[1] PARAM */,1.0);
    if(!(tmp794 && tmp795))
    {
      tmp797 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[108] /* Radiator.vol[2].X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp796),tmp797);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].X_start[1] >= 0.0 and Radiator.vol[2].X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp798 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1049
type: ALGORITHM

  assert(Radiator.vol[2].T_start >= 1.0 and Radiator.vol[2].T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].T_start <= 10000.0, has value: " + String(Radiator.vol[2].T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1049};
  modelica_boolean tmp799;
  modelica_boolean tmp800;
  static const MMC_DEFSTRINGLIT(tmp801,93,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].T_start <= 10000.0, has value: ");
  modelica_string tmp802;
  static int tmp803 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp803)
  {
    tmp799 = GreaterEq(data->simulationInfo->realParameter[98] /* Radiator.vol[2].T_start PARAM */,1.0);
    tmp800 = LessEq(data->simulationInfo->realParameter[98] /* Radiator.vol[2].T_start PARAM */,10000.0);
    if(!(tmp799 && tmp800))
    {
      tmp802 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[98] /* Radiator.vol[2].T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp801),tmp802);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].T_start >= 1.0 and Radiator.vol[2].T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp803 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1050
type: ALGORITHM

  assert(Radiator.vol[2].p_start >= 0.0 and Radiator.vol[2].p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[2].p_start <= 100000000.0, has value: " + String(Radiator.vol[2].p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1050(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1050};
  modelica_boolean tmp804;
  modelica_boolean tmp805;
  static const MMC_DEFSTRINGLIT(tmp806,97,"Variable violating min/max constraint: 0.0 <= Radiator.vol[2].p_start <= 100000000.0, has value: ");
  modelica_string tmp807;
  static int tmp808 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp808)
  {
    tmp804 = GreaterEq(data->simulationInfo->realParameter[213] /* Radiator.vol[2].p_start PARAM */,0.0);
    tmp805 = LessEq(data->simulationInfo->realParameter[213] /* Radiator.vol[2].p_start PARAM */,100000000.0);
    if(!(tmp804 && tmp805))
    {
      tmp807 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[213] /* Radiator.vol[2].p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp806),tmp807);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].p_start >= 0.0 and Radiator.vol[2].p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp808 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1051
type: ALGORITHM

  assert(Radiator.vol[2].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1051};
  modelica_boolean tmp809;
  modelica_boolean tmp810;
  static const MMC_DEFSTRINGLIT(tmp811,178,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp812;
  static int tmp813 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp813)
  {
    tmp809 = GreaterEq(data->simulationInfo->integerParameter[53] /* Radiator.vol[2].traceDynamics PARAM */,1);
    tmp810 = LessEq(data->simulationInfo->integerParameter[53] /* Radiator.vol[2].traceDynamics PARAM */,4);
    if(!(tmp809 && tmp810))
    {
      tmp812 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[53] /* Radiator.vol[2].traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp811),tmp812);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp813 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1052
type: ALGORITHM

  assert(Radiator.vol[2].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1052};
  modelica_boolean tmp814;
  modelica_boolean tmp815;
  static const MMC_DEFSTRINGLIT(tmp816,182,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp817;
  static int tmp818 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp818)
  {
    tmp814 = GreaterEq(data->simulationInfo->integerParameter[48] /* Radiator.vol[2].substanceDynamics PARAM */,1);
    tmp815 = LessEq(data->simulationInfo->integerParameter[48] /* Radiator.vol[2].substanceDynamics PARAM */,4);
    if(!(tmp814 && tmp815))
    {
      tmp817 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[48] /* Radiator.vol[2].substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp816),tmp817);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp818 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1053
type: ALGORITHM

  assert(Radiator.vol[2].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1053};
  modelica_boolean tmp819;
  modelica_boolean tmp820;
  static const MMC_DEFSTRINGLIT(tmp821,177,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp822;
  static int tmp823 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp823)
  {
    tmp819 = GreaterEq(data->simulationInfo->integerParameter[38] /* Radiator.vol[2].massDynamics PARAM */,1);
    tmp820 = LessEq(data->simulationInfo->integerParameter[38] /* Radiator.vol[2].massDynamics PARAM */,4);
    if(!(tmp819 && tmp820))
    {
      tmp822 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[38] /* Radiator.vol[2].massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp821),tmp822);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp823 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1054
type: ALGORITHM

  assert(Radiator.vol[2].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[2].energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1054(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1054};
  modelica_boolean tmp824;
  modelica_boolean tmp825;
  static const MMC_DEFSTRINGLIT(tmp826,179,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[2].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp827;
  static int tmp828 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp828)
  {
    tmp824 = GreaterEq(data->simulationInfo->integerParameter[33] /* Radiator.vol[2].energyDynamics PARAM */,1);
    tmp825 = LessEq(data->simulationInfo->integerParameter[33] /* Radiator.vol[2].energyDynamics PARAM */,4);
    if(!(tmp824 && tmp825))
    {
      tmp827 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[33] /* Radiator.vol[2].energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp826),tmp827);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[2].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp828 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1055
type: ALGORITHM

  assert(Radiator.vol[1].state_start.T >= 1.0 and Radiator.vol[1].state_start.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].state_start.T <= 10000.0, has value: " + String(Radiator.vol[1].state_start.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1055};
  modelica_boolean tmp829;
  modelica_boolean tmp830;
  static const MMC_DEFSTRINGLIT(tmp831,99,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].state_start.T <= 10000.0, has value: ");
  modelica_string tmp832;
  static int tmp833 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp833)
  {
    tmp829 = GreaterEq(data->simulationInfo->realParameter[247] /* Radiator.vol[1].state_start.T PARAM */,1.0);
    tmp830 = LessEq(data->simulationInfo->realParameter[247] /* Radiator.vol[1].state_start.T PARAM */,10000.0);
    if(!(tmp829 && tmp830))
    {
      tmp832 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[247] /* Radiator.vol[1].state_start.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp831),tmp832);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].state_start.T >= 1.0 and Radiator.vol[1].state_start.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp833 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1056
type: ALGORITHM

  assert(Radiator.vol[1].state_start.p >= 0.0 and Radiator.vol[1].state_start.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].state_start.p <= 100000000.0, has value: " + String(Radiator.vol[1].state_start.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1056(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1056};
  modelica_boolean tmp834;
  modelica_boolean tmp835;
  static const MMC_DEFSTRINGLIT(tmp836,103,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].state_start.p <= 100000000.0, has value: ");
  modelica_string tmp837;
  static int tmp838 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp838)
  {
    tmp834 = GreaterEq(data->simulationInfo->realParameter[252] /* Radiator.vol[1].state_start.p PARAM */,0.0);
    tmp835 = LessEq(data->simulationInfo->realParameter[252] /* Radiator.vol[1].state_start.p PARAM */,100000000.0);
    if(!(tmp834 && tmp835))
    {
      tmp837 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[252] /* Radiator.vol[1].state_start.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp836),tmp837);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].state_start.p >= 0.0 and Radiator.vol[1].state_start.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp838 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1057
type: ALGORITHM

  assert(Radiator.vol[1].rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].rho_default, has value: " + String(Radiator.vol[1].rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1057};
  modelica_boolean tmp839;
  static const MMC_DEFSTRINGLIT(tmp840,82,"Variable violating min constraint: 0.0 <= Radiator.vol[1].rho_default, has value: ");
  modelica_string tmp841;
  static int tmp842 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp842)
  {
    tmp839 = GreaterEq(data->simulationInfo->realParameter[227] /* Radiator.vol[1].rho_default PARAM */,0.0);
    if(!tmp839)
    {
      tmp841 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[227] /* Radiator.vol[1].rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp840),tmp841);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",96,3,97,63,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp842 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1058
type: ALGORITHM

  assert(Radiator.vol[1].state_default.T >= 1.0 and Radiator.vol[1].state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].state_default.T <= 10000.0, has value: " + String(Radiator.vol[1].state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1058(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1058};
  modelica_boolean tmp843;
  modelica_boolean tmp844;
  static const MMC_DEFSTRINGLIT(tmp845,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].state_default.T <= 10000.0, has value: ");
  modelica_string tmp846;
  static int tmp847 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp847)
  {
    tmp843 = GreaterEq(data->simulationInfo->realParameter[237] /* Radiator.vol[1].state_default.T PARAM */,1.0);
    tmp844 = LessEq(data->simulationInfo->realParameter[237] /* Radiator.vol[1].state_default.T PARAM */,10000.0);
    if(!(tmp843 && tmp844))
    {
      tmp846 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[237] /* Radiator.vol[1].state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp845),tmp846);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].state_default.T >= 1.0 and Radiator.vol[1].state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp847 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1059
type: ALGORITHM

  assert(Radiator.vol[1].state_default.p >= 0.0 and Radiator.vol[1].state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].state_default.p <= 100000000.0, has value: " + String(Radiator.vol[1].state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1059};
  modelica_boolean tmp848;
  modelica_boolean tmp849;
  static const MMC_DEFSTRINGLIT(tmp850,105,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].state_default.p <= 100000000.0, has value: ");
  modelica_string tmp851;
  static int tmp852 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp852)
  {
    tmp848 = GreaterEq(data->simulationInfo->realParameter[242] /* Radiator.vol[1].state_default.p PARAM */,0.0);
    tmp849 = LessEq(data->simulationInfo->realParameter[242] /* Radiator.vol[1].state_default.p PARAM */,100000000.0);
    if(!(tmp848 && tmp849))
    {
      tmp851 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[242] /* Radiator.vol[1].state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp850),tmp851);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].state_default.p >= 0.0 and Radiator.vol[1].state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp852 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1060
type: ALGORITHM

  assert(Radiator.vol[1].rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].rho_start, has value: " + String(Radiator.vol[1].rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1060};
  modelica_boolean tmp853;
  static const MMC_DEFSTRINGLIT(tmp854,80,"Variable violating min constraint: 0.0 <= Radiator.vol[1].rho_start, has value: ");
  modelica_string tmp855;
  static int tmp856 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp856)
  {
    tmp853 = GreaterEq(data->simulationInfo->realParameter[232] /* Radiator.vol[1].rho_start PARAM */,0.0);
    if(!tmp853)
    {
      tmp855 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[232] /* Radiator.vol[1].rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp854),tmp855);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",89,3,90,73,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp856 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1061
type: ALGORITHM

  assert(Radiator.p_start >= 0.0 and Radiator.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.p_start <= 100000000.0, has value: " + String(Radiator.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1061};
  modelica_boolean tmp857;
  modelica_boolean tmp858;
  static const MMC_DEFSTRINGLIT(tmp859,90,"Variable violating min/max constraint: 0.0 <= Radiator.p_start <= 100000000.0, has value: ");
  modelica_string tmp860;
  static int tmp861 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp861)
  {
    tmp857 = GreaterEq(data->simulationInfo->realParameter[42] /* Radiator.p_start PARAM */,0.0);
    tmp858 = LessEq(data->simulationInfo->realParameter[42] /* Radiator.p_start PARAM */,100000000.0);
    if(!(tmp857 && tmp858))
    {
      tmp860 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[42] /* Radiator.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp859),tmp860);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.p_start >= 0.0 and Radiator.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp861 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1062
type: ALGORITHM

  assert(Radiator.vol[1].p_start >= 0.0 and Radiator.vol[1].p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].p_start <= 100000000.0, has value: " + String(Radiator.vol[1].p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1062(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1062};
  modelica_boolean tmp862;
  modelica_boolean tmp863;
  static const MMC_DEFSTRINGLIT(tmp864,97,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].p_start <= 100000000.0, has value: ");
  modelica_string tmp865;
  static int tmp866 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp866)
  {
    tmp862 = GreaterEq(data->simulationInfo->realParameter[212] /* Radiator.vol[1].p_start PARAM */,0.0);
    tmp863 = LessEq(data->simulationInfo->realParameter[212] /* Radiator.vol[1].p_start PARAM */,100000000.0);
    if(!(tmp862 && tmp863))
    {
      tmp865 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[212] /* Radiator.vol[1].p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp864),tmp865);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].p_start >= 0.0 and Radiator.vol[1].p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp866 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1063
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.p_start >= 0.0 and Radiator.vol[1].dynBal.p_start <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.p_start <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.p_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1063};
  modelica_boolean tmp867;
  modelica_boolean tmp868;
  static const MMC_DEFSTRINGLIT(tmp869,104,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.p_start <= 100000000.0, has value: ");
  modelica_string tmp870;
  static int tmp871 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp871)
  {
    tmp867 = GreaterEq(data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */,0.0);
    tmp868 = LessEq(data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */,100000000.0);
    if(!(tmp867 && tmp868))
    {
      tmp870 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[157] /* Radiator.vol[1].dynBal.p_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp869),tmp870);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",29,3,31,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.p_start >= 0.0 and Radiator.vol[1].dynBal.p_start <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp871 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1064
type: ALGORITHM

  assert(Radiator.T_start >= 1.0 and Radiator.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.T_start <= 10000.0, has value: " + String(Radiator.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1064(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1064};
  modelica_boolean tmp872;
  modelica_boolean tmp873;
  static const MMC_DEFSTRINGLIT(tmp874,86,"Variable violating min/max constraint: 1.0 <= Radiator.T_start <= 10000.0, has value: ");
  modelica_string tmp875;
  static int tmp876 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp876)
  {
    tmp872 = GreaterEq(data->simulationInfo->realParameter[16] /* Radiator.T_start PARAM */,1.0);
    tmp873 = LessEq(data->simulationInfo->realParameter[16] /* Radiator.T_start PARAM */,10000.0);
    if(!(tmp872 && tmp873))
    {
      tmp875 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[16] /* Radiator.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp874),tmp875);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.T_start >= 1.0 and Radiator.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp876 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1065
type: ALGORITHM

  assert(Radiator.vol[1].T_start >= 1.0 and Radiator.vol[1].T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].T_start <= 10000.0, has value: " + String(Radiator.vol[1].T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1065};
  modelica_boolean tmp877;
  modelica_boolean tmp878;
  static const MMC_DEFSTRINGLIT(tmp879,93,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].T_start <= 10000.0, has value: ");
  modelica_string tmp880;
  static int tmp881 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp881)
  {
    tmp877 = GreaterEq(data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */,1.0);
    tmp878 = LessEq(data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */,10000.0);
    if(!(tmp877 && tmp878))
    {
      tmp880 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[97] /* Radiator.vol[1].T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp879),tmp880);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].T_start >= 1.0 and Radiator.vol[1].T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp881 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1066
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.T_start >= 1.0 and Radiator.vol[1].dynBal.T_start <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.T_start <= 10000.0, has value: " + String(Radiator.vol[1].dynBal.T_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1066(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1066};
  modelica_boolean tmp882;
  modelica_boolean tmp883;
  static const MMC_DEFSTRINGLIT(tmp884,100,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.T_start <= 10000.0, has value: ");
  modelica_string tmp885;
  static int tmp886 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp886)
  {
    tmp882 = GreaterEq(data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */,1.0);
    tmp883 = LessEq(data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */,10000.0);
    if(!(tmp882 && tmp883))
    {
      tmp885 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[117] /* Radiator.vol[1].dynBal.T_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp884),tmp885);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",32,3,34,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.T_start >= 1.0 and Radiator.vol[1].dynBal.T_start <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp886 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1067
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.rho_default >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.rho_default, has value: " + String(Radiator.vol[1].dynBal.rho_default, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1067};
  modelica_boolean tmp887;
  static const MMC_DEFSTRINGLIT(tmp888,89,"Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.rho_default, has value: ");
  modelica_string tmp889;
  static int tmp890 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp890)
  {
    tmp887 = GreaterEq(data->simulationInfo->realParameter[172] /* Radiator.vol[1].dynBal.rho_default PARAM */,0.0);
    if(!tmp887)
    {
      tmp889 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[172] /* Radiator.vol[1].dynBal.rho_default PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp888),tmp889);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",145,3,146,59,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.rho_default >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp890 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1068
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.state_default.T >= 1.0 and Radiator.vol[1].dynBal.state_default.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.state_default.T <= 10000.0, has value: " + String(Radiator.vol[1].dynBal.state_default.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1068};
  modelica_boolean tmp891;
  modelica_boolean tmp892;
  static const MMC_DEFSTRINGLIT(tmp893,108,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.state_default.T <= 10000.0, has value: ");
  modelica_string tmp894;
  static int tmp895 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp895)
  {
    tmp891 = GreaterEq(data->simulationInfo->realParameter[182] /* Radiator.vol[1].dynBal.state_default.T PARAM */,1.0);
    tmp892 = LessEq(data->simulationInfo->realParameter[182] /* Radiator.vol[1].dynBal.state_default.T PARAM */,10000.0);
    if(!(tmp891 && tmp892))
    {
      tmp894 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[182] /* Radiator.vol[1].dynBal.state_default.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp893),tmp894);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.state_default.T >= 1.0 and Radiator.vol[1].dynBal.state_default.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp895 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1069
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.state_default.p >= 0.0 and Radiator.vol[1].dynBal.state_default.p <= 100000000.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.state_default.p <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.state_default.p, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1069};
  modelica_boolean tmp896;
  modelica_boolean tmp897;
  static const MMC_DEFSTRINGLIT(tmp898,112,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.state_default.p <= 100000000.0, has value: ");
  modelica_string tmp899;
  static int tmp900 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp900)
  {
    tmp896 = GreaterEq(data->simulationInfo->realParameter[187] /* Radiator.vol[1].dynBal.state_default.p PARAM */,0.0);
    tmp897 = LessEq(data->simulationInfo->realParameter[187] /* Radiator.vol[1].dynBal.state_default.p PARAM */,100000000.0);
    if(!(tmp896 && tmp897))
    {
      tmp899 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[187] /* Radiator.vol[1].dynBal.state_default.p PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp898),tmp899);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5869,7,5869,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.state_default.p >= 0.0 and Radiator.vol[1].dynBal.state_default.p <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp900 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1070
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.rho_start >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.rho_start, has value: " + String(Radiator.vol[1].dynBal.rho_start, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1070(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1070};
  modelica_boolean tmp901;
  static const MMC_DEFSTRINGLIT(tmp902,87,"Variable violating min constraint: 0.0 <= Radiator.vol[1].dynBal.rho_start, has value: ");
  modelica_string tmp903;
  static int tmp904 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp904)
  {
    tmp901 = GreaterEq(data->simulationInfo->realParameter[177] /* Radiator.vol[1].dynBal.rho_start PARAM */,0.0);
    if(!tmp901)
    {
      tmp903 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[177] /* Radiator.vol[1].dynBal.rho_start PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp902),tmp903);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",131,3,135,70,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.rho_start >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp904 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1071
type: ALGORITHM

  assert(Radiator.mDry >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.mDry, has value: " + String(Radiator.mDry, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1071};
  modelica_boolean tmp905;
  static const MMC_DEFSTRINGLIT(tmp906,68,"Variable violating min constraint: 0.0 <= Radiator.mDry, has value: ");
  modelica_string tmp907;
  static int tmp908 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp908)
  {
    tmp905 = GreaterEq(data->simulationInfo->realParameter[37] /* Radiator.mDry PARAM */,0.0);
    if(!tmp905)
    {
      tmp907 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[37] /* Radiator.mDry PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp906),tmp907);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",42,3,44,114,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.mDry >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp908 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1072
type: ALGORITHM

  assert(Radiator.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.mSenFac, has value: " + String(Radiator.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1072(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1072};
  modelica_boolean tmp909;
  static const MMC_DEFSTRINGLIT(tmp910,71,"Variable violating min constraint: 1.0 <= Radiator.mSenFac, has value: ");
  modelica_string tmp911;
  static int tmp912 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp912)
  {
    tmp909 = GreaterEq(data->simulationInfo->realParameter[38] /* Radiator.mSenFac PARAM */,1.0);
    if(!tmp909)
    {
      tmp911 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[38] /* Radiator.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp910),tmp911);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp912 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1073
type: ALGORITHM

  assert(Radiator.vol[1].mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[1].mSenFac, has value: " + String(Radiator.vol[1].mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1073};
  modelica_boolean tmp913;
  static const MMC_DEFSTRINGLIT(tmp914,78,"Variable violating min constraint: 1.0 <= Radiator.vol[1].mSenFac, has value: ");
  modelica_string tmp915;
  static int tmp916 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp916)
  {
    tmp913 = GreaterEq(data->simulationInfo->realParameter[192] /* Radiator.vol[1].mSenFac PARAM */,1.0);
    if(!tmp913)
    {
      tmp915 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[192] /* Radiator.vol[1].mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp914),tmp915);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp916 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1074
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.mSenFac >= 1.0, "Variable violating min constraint: 1.0 <= Radiator.vol[1].dynBal.mSenFac, has value: " + String(Radiator.vol[1].dynBal.mSenFac, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1074(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1074};
  modelica_boolean tmp917;
  static const MMC_DEFSTRINGLIT(tmp918,85,"Variable violating min constraint: 1.0 <= Radiator.vol[1].dynBal.mSenFac, has value: ");
  modelica_string tmp919;
  static int tmp920 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp920)
  {
    tmp917 = GreaterEq(data->simulationInfo->realParameter[142] /* Radiator.vol[1].dynBal.mSenFac PARAM */,1.0);
    if(!tmp917)
    {
      tmp919 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[142] /* Radiator.vol[1].dynBal.mSenFac PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp918),tmp919);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",47,3,49,39,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.mSenFac >= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp920 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1075
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.X_start[1] >= 0.0 and Radiator.vol[1].dynBal.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.X_start[1] <= 1.0, has value: " + String(Radiator.vol[1].dynBal.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1075};
  modelica_boolean tmp921;
  modelica_boolean tmp922;
  static const MMC_DEFSTRINGLIT(tmp923,99,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].dynBal.X_start[1] <= 1.0, has value: ");
  modelica_string tmp924;
  static int tmp925 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp925)
  {
    tmp921 = GreaterEq(data->simulationInfo->realParameter[122] /* Radiator.vol[1].dynBal.X_start[1] PARAM */,0.0);
    tmp922 = LessEq(data->simulationInfo->realParameter[122] /* Radiator.vol[1].dynBal.X_start[1] PARAM */,1.0);
    if(!(tmp921 && tmp922))
    {
      tmp924 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[122] /* Radiator.vol[1].dynBal.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp923),tmp924);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.X_start[1] >= 0.0 and Radiator.vol[1].dynBal.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp925 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1076
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].dynBal.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1076};
  modelica_boolean tmp926;
  modelica_boolean tmp927;
  static const MMC_DEFSTRINGLIT(tmp928,185,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp929;
  static int tmp930 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp930)
  {
    tmp926 = GreaterEq(data->simulationInfo->integerParameter[27] /* Radiator.vol[1].dynBal.traceDynamics PARAM */,1);
    tmp927 = LessEq(data->simulationInfo->integerParameter[27] /* Radiator.vol[1].dynBal.traceDynamics PARAM */,4);
    if(!(tmp926 && tmp927))
    {
      tmp929 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[27] /* Radiator.vol[1].dynBal.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp928),tmp929);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp930 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1077
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].dynBal.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1077};
  modelica_boolean tmp931;
  modelica_boolean tmp932;
  static const MMC_DEFSTRINGLIT(tmp933,189,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp934;
  static int tmp935 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp935)
  {
    tmp931 = GreaterEq(data->simulationInfo->integerParameter[22] /* Radiator.vol[1].dynBal.substanceDynamics PARAM */,1);
    tmp932 = LessEq(data->simulationInfo->integerParameter[22] /* Radiator.vol[1].dynBal.substanceDynamics PARAM */,4);
    if(!(tmp931 && tmp932))
    {
      tmp934 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[22] /* Radiator.vol[1].dynBal.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp933),tmp934);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp935 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1078
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].dynBal.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1078(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1078};
  modelica_boolean tmp936;
  modelica_boolean tmp937;
  static const MMC_DEFSTRINGLIT(tmp938,184,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp939;
  static int tmp940 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp940)
  {
    tmp936 = GreaterEq(data->simulationInfo->integerParameter[12] /* Radiator.vol[1].dynBal.massDynamics PARAM */,1);
    tmp937 = LessEq(data->simulationInfo->integerParameter[12] /* Radiator.vol[1].dynBal.massDynamics PARAM */,4);
    if(!(tmp936 && tmp937))
    {
      tmp939 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[12] /* Radiator.vol[1].dynBal.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp938),tmp939);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp940 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1079
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].dynBal.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1079};
  modelica_boolean tmp941;
  modelica_boolean tmp942;
  static const MMC_DEFSTRINGLIT(tmp943,186,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp944;
  static int tmp945 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp945)
  {
    tmp941 = GreaterEq(data->simulationInfo->integerParameter[7] /* Radiator.vol[1].dynBal.energyDynamics PARAM */,1);
    tmp942 = LessEq(data->simulationInfo->integerParameter[7] /* Radiator.vol[1].dynBal.energyDynamics PARAM */,4);
    if(!(tmp941 && tmp942))
    {
      tmp944 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[7] /* Radiator.vol[1].dynBal.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp943),tmp944);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].dynBal.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp945 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1080
type: ALGORITHM

  assert(Radiator.vol[1].m_flow_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].m_flow_nominal, has value: " + String(Radiator.vol[1].m_flow_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1080(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1080};
  modelica_boolean tmp946;
  static const MMC_DEFSTRINGLIT(tmp947,85,"Variable violating min constraint: 0.0 <= Radiator.vol[1].m_flow_nominal, has value: ");
  modelica_string tmp948;
  static int tmp949 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp949)
  {
    tmp946 = GreaterEq(data->simulationInfo->realParameter[197] /* Radiator.vol[1].m_flow_nominal PARAM */,0.0);
    if(!tmp946)
    {
      tmp948 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[197] /* Radiator.vol[1].m_flow_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp947),tmp948);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",20,3,21,76,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].m_flow_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp949 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1081
type: ALGORITHM

  assert(Radiator.vol[1].m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.vol[1].m_flow_small, has value: " + String(Radiator.vol[1].m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1081};
  modelica_boolean tmp950;
  static const MMC_DEFSTRINGLIT(tmp951,83,"Variable violating min constraint: 0.0 <= Radiator.vol[1].m_flow_small, has value: ");
  modelica_string tmp952;
  static int tmp953 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp953)
  {
    tmp950 = GreaterEq(data->simulationInfo->realParameter[202] /* Radiator.vol[1].m_flow_small PARAM */,0.0);
    if(!tmp950)
    {
      tmp952 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[202] /* Radiator.vol[1].m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp951),tmp952);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",25,3,27,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp953 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1082
type: ALGORITHM

  assert(Radiator.vol[1].X_start[1] >= 0.0 and Radiator.vol[1].X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.vol[1].X_start[1] <= 1.0, has value: " + String(Radiator.vol[1].X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1082(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1082};
  modelica_boolean tmp954;
  modelica_boolean tmp955;
  static const MMC_DEFSTRINGLIT(tmp956,92,"Variable violating min/max constraint: 0.0 <= Radiator.vol[1].X_start[1] <= 1.0, has value: ");
  modelica_string tmp957;
  static int tmp958 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp958)
  {
    tmp954 = GreaterEq(data->simulationInfo->realParameter[107] /* Radiator.vol[1].X_start[1] PARAM */,0.0);
    tmp955 = LessEq(data->simulationInfo->realParameter[107] /* Radiator.vol[1].X_start[1] PARAM */,1.0);
    if(!(tmp954 && tmp955))
    {
      tmp957 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[107] /* Radiator.vol[1].X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp956),tmp957);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].X_start[1] >= 0.0 and Radiator.vol[1].X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp958 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1083
type: ALGORITHM

  assert(Radiator.vol[1].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1083};
  modelica_boolean tmp959;
  modelica_boolean tmp960;
  static const MMC_DEFSTRINGLIT(tmp961,178,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp962;
  static int tmp963 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp963)
  {
    tmp959 = GreaterEq(data->simulationInfo->integerParameter[52] /* Radiator.vol[1].traceDynamics PARAM */,1);
    tmp960 = LessEq(data->simulationInfo->integerParameter[52] /* Radiator.vol[1].traceDynamics PARAM */,4);
    if(!(tmp959 && tmp960))
    {
      tmp962 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[52] /* Radiator.vol[1].traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp961),tmp962);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp963 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1084
type: ALGORITHM

  assert(Radiator.vol[1].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1084};
  modelica_boolean tmp964;
  modelica_boolean tmp965;
  static const MMC_DEFSTRINGLIT(tmp966,182,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp967;
  static int tmp968 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp968)
  {
    tmp964 = GreaterEq(data->simulationInfo->integerParameter[47] /* Radiator.vol[1].substanceDynamics PARAM */,1);
    tmp965 = LessEq(data->simulationInfo->integerParameter[47] /* Radiator.vol[1].substanceDynamics PARAM */,4);
    if(!(tmp964 && tmp965))
    {
      tmp967 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[47] /* Radiator.vol[1].substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp966),tmp967);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp968 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1085
type: ALGORITHM

  assert(Radiator.vol[1].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1085};
  modelica_boolean tmp969;
  modelica_boolean tmp970;
  static const MMC_DEFSTRINGLIT(tmp971,177,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp972;
  static int tmp973 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp973)
  {
    tmp969 = GreaterEq(data->simulationInfo->integerParameter[37] /* Radiator.vol[1].massDynamics PARAM */,1);
    tmp970 = LessEq(data->simulationInfo->integerParameter[37] /* Radiator.vol[1].massDynamics PARAM */,4);
    if(!(tmp969 && tmp970))
    {
      tmp972 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[37] /* Radiator.vol[1].massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp971),tmp972);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp973 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1086
type: ALGORITHM

  assert(Radiator.vol[1].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.vol[1].energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1086(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1086};
  modelica_boolean tmp974;
  modelica_boolean tmp975;
  static const MMC_DEFSTRINGLIT(tmp976,179,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.vol[1].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp977;
  static int tmp978 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp978)
  {
    tmp974 = GreaterEq(data->simulationInfo->integerParameter[32] /* Radiator.vol[1].energyDynamics PARAM */,1);
    tmp975 = LessEq(data->simulationInfo->integerParameter[32] /* Radiator.vol[1].energyDynamics PARAM */,4);
    if(!(tmp974 && tmp975))
    {
      tmp977 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[32] /* Radiator.vol[1].energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp976),tmp977);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.vol[1].energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp978 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1087
type: ALGORITHM

  assert(Radiator.deltaM >= 0.01, "Variable violating min constraint: 0.01 <= Radiator.deltaM, has value: " + String(Radiator.deltaM, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1087};
  modelica_boolean tmp979;
  static const MMC_DEFSTRINGLIT(tmp980,71,"Variable violating min constraint: 0.01 <= Radiator.deltaM, has value: ");
  modelica_string tmp981;
  static int tmp982 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp982)
  {
    tmp979 = GreaterEq(data->simulationInfo->realParameter[33] /* Radiator.deltaM PARAM */,0.01);
    if(!tmp979)
    {
      tmp981 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[33] /* Radiator.deltaM PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp980),tmp981);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",45,3,49,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.deltaM >= 0.01", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp982 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1088
type: ALGORITHM

  assert(Radiator.TAir_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.TAir_nominal, has value: " + String(Radiator.TAir_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1088(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1088};
  modelica_boolean tmp983;
  static const MMC_DEFSTRINGLIT(tmp984,76,"Variable violating min constraint: 0.0 <= Radiator.TAir_nominal, has value: ");
  modelica_string tmp985;
  static int tmp986 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp986)
  {
    tmp983 = GreaterEq(data->simulationInfo->realParameter[7] /* Radiator.TAir_nominal PARAM */,0.0);
    if(!tmp983)
    {
      tmp985 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[7] /* Radiator.TAir_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp984),tmp985);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",31,3,33,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.TAir_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp986 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1089
type: ALGORITHM

  assert(Radiator.T_b_nominal >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.T_b_nominal, has value: " + String(Radiator.T_b_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1089};
  modelica_boolean tmp987;
  static const MMC_DEFSTRINGLIT(tmp988,75,"Variable violating min constraint: 0.0 <= Radiator.T_b_nominal, has value: ");
  modelica_string tmp989;
  static int tmp990 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp990)
  {
    tmp987 = GreaterEq(data->simulationInfo->realParameter[15] /* Radiator.T_b_nominal PARAM */,0.0);
    if(!tmp987)
    {
      tmp989 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[15] /* Radiator.T_b_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp988),tmp989);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",28,3,30,51,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.T_b_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp990 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1090
type: ALGORITHM

  assert(Radiator.nEle >= 1, "Variable violating min constraint: 1 <= Radiator.nEle, has value: " + String(Radiator.nEle, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1090(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1090};
  modelica_boolean tmp991;
  static const MMC_DEFSTRINGLIT(tmp992,66,"Variable violating min constraint: 1 <= Radiator.nEle, has value: ");
  modelica_string tmp993;
  static int tmp994 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp994)
  {
    tmp991 = GreaterEq(data->simulationInfo->integerParameter[2] /* Radiator.nEle PARAM */,((modelica_integer) 1));
    if(!tmp991)
    {
      tmp993 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[2] /* Radiator.nEle PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp992),tmp993);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/HeatExchangers/Radiators/RadiatorEN442_2.mo",17,3,18,52,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.nEle >= 1", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp994 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1091
type: ALGORITHM

  assert(Radiator.X_start[1] >= 0.0 and Radiator.X_start[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= Radiator.X_start[1] <= 1.0, has value: " + String(Radiator.X_start[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1091};
  modelica_boolean tmp995;
  modelica_boolean tmp996;
  static const MMC_DEFSTRINGLIT(tmp997,85,"Variable violating min/max constraint: 0.0 <= Radiator.X_start[1] <= 1.0, has value: ");
  modelica_string tmp998;
  static int tmp999 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp999)
  {
    tmp995 = GreaterEq(data->simulationInfo->realParameter[19] /* Radiator.X_start[1] PARAM */,0.0);
    tmp996 = LessEq(data->simulationInfo->realParameter[19] /* Radiator.X_start[1] PARAM */,1.0);
    if(!(tmp995 && tmp996))
    {
      tmp998 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[19] /* Radiator.X_start[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp997),tmp998);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",35,3,38,69,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.X_start[1] >= 0.0 and Radiator.X_start[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp999 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1092
type: ALGORITHM

  assert(Radiator.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.traceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1092};
  modelica_boolean tmp1000;
  modelica_boolean tmp1001;
  static const MMC_DEFSTRINGLIT(tmp1002,171,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp1003;
  static int tmp1004 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1004)
  {
    tmp1000 = GreaterEq(data->simulationInfo->integerParameter[6] /* Radiator.traceDynamics PARAM */,1);
    tmp1001 = LessEq(data->simulationInfo->integerParameter[6] /* Radiator.traceDynamics PARAM */,4);
    if(!(tmp1000 && tmp1001))
    {
      tmp1003 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[6] /* Radiator.traceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1002),tmp1003);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",24,3,26,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.traceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.traceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1004 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1093
type: ALGORITHM

  assert(Radiator.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.substanceDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1093};
  modelica_boolean tmp1005;
  modelica_boolean tmp1006;
  static const MMC_DEFSTRINGLIT(tmp1007,175,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp1008;
  static int tmp1009 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1009)
  {
    tmp1005 = GreaterEq(data->simulationInfo->integerParameter[3] /* Radiator.substanceDynamics PARAM */,1);
    tmp1006 = LessEq(data->simulationInfo->integerParameter[3] /* Radiator.substanceDynamics PARAM */,4);
    if(!(tmp1005 && tmp1006))
    {
      tmp1008 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[3] /* Radiator.substanceDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1007),tmp1008);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",21,3,23,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.substanceDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.substanceDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1009 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1094
type: ALGORITHM

  assert(Radiator.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.massDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1094(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1094};
  modelica_boolean tmp1010;
  modelica_boolean tmp1011;
  static const MMC_DEFSTRINGLIT(tmp1012,170,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp1013;
  static int tmp1014 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1014)
  {
    tmp1010 = GreaterEq(data->simulationInfo->integerParameter[1] /* Radiator.massDynamics PARAM */,1);
    tmp1011 = LessEq(data->simulationInfo->integerParameter[1] /* Radiator.massDynamics PARAM */,4);
    if(!(tmp1010 && tmp1011))
    {
      tmp1013 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[1] /* Radiator.massDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1012),tmp1013);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",18,3,20,74,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.massDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.massDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1014 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1095
type: ALGORITHM

  assert(Radiator.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, "Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: " + String(Radiator.energyDynamics, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1095};
  modelica_boolean tmp1015;
  modelica_boolean tmp1016;
  static const MMC_DEFSTRINGLIT(tmp1017,172,"Variable violating min/max constraint: Modelica.Fluid.Types.Dynamics.DynamicFreeInitial <= Radiator.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState, has value: ");
  modelica_string tmp1018;
  static int tmp1019 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1019)
  {
    tmp1015 = GreaterEq(data->simulationInfo->integerParameter[0] /* Radiator.energyDynamics PARAM */,1);
    tmp1016 = LessEq(data->simulationInfo->integerParameter[0] /* Radiator.energyDynamics PARAM */,4);
    if(!(tmp1015 && tmp1016))
    {
      tmp1018 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[0] /* Radiator.energyDynamics PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1017),tmp1018);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/LumpedVolumeDeclarations.mo",15,3,17,88,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.energyDynamics >= Modelica.Fluid.Types.Dynamics.DynamicFreeInitial and Radiator.energyDynamics <= Modelica.Fluid.Types.Dynamics.SteadyState", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1019 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1096
type: ALGORITHM

  assert(Radiator.m_flow_small >= 0.0, "Variable violating min constraint: 0.0 <= Radiator.m_flow_small, has value: " + String(Radiator.m_flow_small, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1096(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1096};
  modelica_boolean tmp1020;
  static const MMC_DEFSTRINGLIT(tmp1021,76,"Variable violating min constraint: 0.0 <= Radiator.m_flow_small, has value: ");
  modelica_string tmp1022;
  static int tmp1023 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1023)
  {
    tmp1020 = GreaterEq(data->simulationInfo->realParameter[40] /* Radiator.m_flow_small PARAM */,0.0);
    if(!tmp1020)
    {
      tmp1022 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[40] /* Radiator.m_flow_small PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1021),tmp1022);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/PartialTwoPortInterface.mo",10,3,12,40,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.m_flow_small >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1023 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1097
type: ALGORITHM

  assert(flow_source.T >= 1.0 and flow_source.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= flow_source.T <= 10000.0, has value: " + String(flow_source.T, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1097};
  modelica_boolean tmp1024;
  modelica_boolean tmp1025;
  static const MMC_DEFSTRINGLIT(tmp1026,83,"Variable violating min/max constraint: 1.0 <= flow_source.T <= 10000.0, has value: ");
  modelica_string tmp1027;
  static int tmp1028 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1028)
  {
    tmp1024 = GreaterEq(data->simulationInfo->realParameter[265] /* flow_source.T PARAM */,1.0);
    tmp1025 = LessEq(data->simulationInfo->realParameter[265] /* flow_source.T PARAM */,10000.0);
    if(!(tmp1024 && tmp1025))
    {
      tmp1027 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[265] /* flow_source.T PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1026),tmp1027);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/MassFlowSource_T.mo",15,3,17,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_source.T >= 1.0 and flow_source.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1028 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1098
type: ALGORITHM

  assert(flow_source.X[1] >= 0.0 and flow_source.X[1] <= 1.0, "Variable violating min/max constraint: 0.0 <= flow_source.X[1] <= 1.0, has value: " + String(flow_source.X[1], "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1098(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1098};
  modelica_boolean tmp1029;
  modelica_boolean tmp1030;
  static const MMC_DEFSTRINGLIT(tmp1031,82,"Variable violating min/max constraint: 0.0 <= flow_source.X[1] <= 1.0, has value: ");
  modelica_string tmp1032;
  static int tmp1033 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1033)
  {
    tmp1029 = GreaterEq(data->simulationInfo->realParameter[266] /* flow_source.X[1] PARAM */,0.0);
    tmp1030 = LessEq(data->simulationInfo->realParameter[266] /* flow_source.X[1] PARAM */,1.0);
    if(!(tmp1029 && tmp1030))
    {
      tmp1032 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[266] /* flow_source.X[1] PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1031),tmp1032);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/BaseClasses/PartialSource_Xi_C.mo",15,3,18,90,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_source.X[1] >= 0.0 and flow_source.X[1] <= 1.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1033 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1099
type: ALGORITHM

  assert(flow_source.flowDirection >= Modelica.Fluid.Types.PortFlowDirection.Entering and flow_source.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, "Variable violating min/max constraint: Modelica.Fluid.Types.PortFlowDirection.Entering <= flow_source.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, has value: " + String(flow_source.flowDirection, "d"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1099};
  modelica_boolean tmp1034;
  modelica_boolean tmp1035;
  static const MMC_DEFSTRINGLIT(tmp1036,184,"Variable violating min/max constraint: Modelica.Fluid.Types.PortFlowDirection.Entering <= flow_source.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional, has value: ");
  modelica_string tmp1037;
  static int tmp1038 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1038)
  {
    tmp1034 = GreaterEq(data->simulationInfo->integerParameter[59] /* flow_source.flowDirection PARAM */,1);
    tmp1035 = LessEq(data->simulationInfo->integerParameter[59] /* flow_source.flowDirection PARAM */,3);
    if(!(tmp1034 && tmp1035))
    {
      tmp1037 = modelica_integer_to_modelica_string_format(data->simulationInfo->integerParameter[59] /* flow_source.flowDirection PARAM */, (modelica_string) mmc_strings_len1[100]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1036),tmp1037);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Sources/BaseClasses/PartialSource.mo",31,3,32,80,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_source.flowDirection >= Modelica.Fluid.Types.PortFlowDirection.Entering and flow_source.flowDirection <= Modelica.Fluid.Types.PortFlowDirection.Bidirectional", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1038 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1100
type: ALGORITHM

  assert(T_b_nominal >= 0.0, "Variable violating min constraint: 0.0 <= T_b_nominal, has value: " + String(T_b_nominal, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1100};
  modelica_boolean tmp1039;
  static const MMC_DEFSTRINGLIT(tmp1040,66,"Variable violating min constraint: 0.0 <= T_b_nominal, has value: ");
  modelica_string tmp1041;
  static int tmp1042 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1042)
  {
    tmp1039 = GreaterEq(data->simulationInfo->realParameter[259] /* T_b_nominal PARAM */,0.0);
    if(!tmp1039)
    {
      tmp1041 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[259] /* T_b_nominal PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1040),tmp1041);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/Radiator.mo",9,3,10,55,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nT_b_nominal >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1042 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1101
type: ALGORITHM

  assert(TRoo >= 0.0, "Variable violating min constraint: 0.0 <= TRoo, has value: " + String(TRoo, "g"));
*/
OMC_DISABLE_OPT
static void Radiator_eqFunction_1101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1101};
  modelica_boolean tmp1043;
  static const MMC_DEFSTRINGLIT(tmp1044,59,"Variable violating min constraint: 0.0 <= TRoo, has value: ");
  modelica_string tmp1045;
  static int tmp1046 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp1046)
  {
    tmp1043 = GreaterEq(data->simulationInfo->realParameter[257] /* TRoo PARAM */,0.0);
    if(!tmp1043)
    {
      tmp1045 = modelica_real_to_modelica_string_format(data->simulationInfo->realParameter[257] /* TRoo PARAM */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp1044),tmp1045);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/Radiator.mo",4,3,5,32,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nTRoo >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp1046 = 1;
    }
  }
  TRACE_POP
}
OMC_DISABLE_OPT
void Radiator_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  Radiator_eqFunction_457(data, threadData);
  Radiator_eqFunction_458(data, threadData);
  Radiator_eqFunction_459(data, threadData);
  Radiator_eqFunction_460(data, threadData);
  Radiator_eqFunction_461(data, threadData);
  Radiator_eqFunction_462(data, threadData);
  Radiator_eqFunction_463(data, threadData);
  Radiator_eqFunction_464(data, threadData);
  Radiator_eqFunction_465(data, threadData);
  Radiator_eqFunction_466(data, threadData);
  Radiator_eqFunction_467(data, threadData);
  Radiator_eqFunction_468(data, threadData);
  Radiator_eqFunction_469(data, threadData);
  Radiator_eqFunction_470(data, threadData);
  Radiator_eqFunction_471(data, threadData);
  Radiator_eqFunction_472(data, threadData);
  Radiator_eqFunction_473(data, threadData);
  Radiator_eqFunction_474(data, threadData);
  Radiator_eqFunction_475(data, threadData);
  Radiator_eqFunction_476(data, threadData);
  Radiator_eqFunction_477(data, threadData);
  Radiator_eqFunction_478(data, threadData);
  Radiator_eqFunction_479(data, threadData);
  Radiator_eqFunction_480(data, threadData);
  Radiator_eqFunction_481(data, threadData);
  Radiator_eqFunction_482(data, threadData);
  Radiator_eqFunction_483(data, threadData);
  Radiator_eqFunction_484(data, threadData);
  Radiator_eqFunction_485(data, threadData);
  Radiator_eqFunction_486(data, threadData);
  Radiator_eqFunction_487(data, threadData);
  Radiator_eqFunction_488(data, threadData);
  Radiator_eqFunction_489(data, threadData);
  Radiator_eqFunction_490(data, threadData);
  Radiator_eqFunction_491(data, threadData);
  Radiator_eqFunction_492(data, threadData);
  Radiator_eqFunction_493(data, threadData);
  Radiator_eqFunction_494(data, threadData);
  Radiator_eqFunction_495(data, threadData);
  Radiator_eqFunction_496(data, threadData);
  Radiator_eqFunction_497(data, threadData);
  Radiator_eqFunction_498(data, threadData);
  Radiator_eqFunction_499(data, threadData);
  Radiator_eqFunction_500(data, threadData);
  Radiator_eqFunction_501(data, threadData);
  Radiator_eqFunction_515(data, threadData);
  Radiator_eqFunction_516(data, threadData);
  Radiator_eqFunction_518(data, threadData);
  Radiator_eqFunction_519(data, threadData);
  Radiator_eqFunction_520(data, threadData);
  Radiator_eqFunction_523(data, threadData);
  Radiator_eqFunction_531(data, threadData);
  Radiator_eqFunction_585(data, threadData);
  Radiator_eqFunction_586(data, threadData);
  Radiator_eqFunction_587(data, threadData);
  Radiator_eqFunction_588(data, threadData);
  Radiator_eqFunction_605(data, threadData);
  Radiator_eqFunction_606(data, threadData);
  Radiator_eqFunction_633(data, threadData);
  Radiator_eqFunction_634(data, threadData);
  Radiator_eqFunction_635(data, threadData);
  Radiator_eqFunction_652(data, threadData);
  Radiator_eqFunction_653(data, threadData);
  Radiator_eqFunction_680(data, threadData);
  Radiator_eqFunction_681(data, threadData);
  Radiator_eqFunction_682(data, threadData);
  Radiator_eqFunction_699(data, threadData);
  Radiator_eqFunction_700(data, threadData);
  Radiator_eqFunction_727(data, threadData);
  Radiator_eqFunction_728(data, threadData);
  Radiator_eqFunction_729(data, threadData);
  Radiator_eqFunction_746(data, threadData);
  Radiator_eqFunction_747(data, threadData);
  Radiator_eqFunction_767(data, threadData);
  Radiator_eqFunction_768(data, threadData);
  Radiator_eqFunction_769(data, threadData);
  Radiator_eqFunction_770(data, threadData);
  Radiator_eqFunction_771(data, threadData);
  Radiator_eqFunction_776(data, threadData);
  Radiator_eqFunction_778(data, threadData);
  Radiator_eqFunction_779(data, threadData);
  Radiator_eqFunction_780(data, threadData);
  Radiator_eqFunction_781(data, threadData);
  Radiator_eqFunction_782(data, threadData);
  Radiator_eqFunction_783(data, threadData);
  Radiator_eqFunction_784(data, threadData);
  Radiator_eqFunction_798(data, threadData);
  Radiator_eqFunction_799(data, threadData);
  Radiator_eqFunction_824(data, threadData);
  Radiator_eqFunction_834(data, threadData);
  Radiator_eqFunction_157(data, threadData);
  Radiator_eqFunction_156(data, threadData);
  Radiator_eqFunction_155(data, threadData);
  Radiator_eqFunction_154(data, threadData);
  Radiator_eqFunction_153(data, threadData);
  Radiator_eqFunction_152(data, threadData);
  Radiator_eqFunction_38(data, threadData);
  Radiator_eqFunction_37(data, threadData);
  Radiator_eqFunction_36(data, threadData);
  Radiator_eqFunction_35(data, threadData);
  Radiator_eqFunction_34(data, threadData);
  Radiator_eqFunction_33(data, threadData);
  Radiator_eqFunction_32(data, threadData);
  Radiator_eqFunction_31(data, threadData);
  Radiator_eqFunction_30(data, threadData);
  Radiator_eqFunction_29(data, threadData);
  Radiator_eqFunction_28(data, threadData);
  Radiator_eqFunction_27(data, threadData);
  Radiator_eqFunction_26(data, threadData);
  Radiator_eqFunction_25(data, threadData);
  Radiator_eqFunction_24(data, threadData);
  Radiator_eqFunction_23(data, threadData);
  Radiator_eqFunction_22(data, threadData);
  Radiator_eqFunction_21(data, threadData);
  Radiator_eqFunction_20(data, threadData);
  Radiator_eqFunction_19(data, threadData);
  Radiator_eqFunction_18(data, threadData);
  Radiator_eqFunction_17(data, threadData);
  Radiator_eqFunction_16(data, threadData);
  Radiator_eqFunction_15(data, threadData);
  Radiator_eqFunction_14(data, threadData);
  Radiator_eqFunction_13(data, threadData);
  Radiator_eqFunction_12(data, threadData);
  Radiator_eqFunction_11(data, threadData);
  Radiator_eqFunction_10(data, threadData);
  Radiator_eqFunction_9(data, threadData);
  Radiator_eqFunction_8(data, threadData);
  Radiator_eqFunction_7(data, threadData);
  Radiator_eqFunction_6(data, threadData);
  Radiator_eqFunction_5(data, threadData);
  Radiator_eqFunction_4(data, threadData);
  Radiator_eqFunction_3(data, threadData);
  Radiator_eqFunction_2(data, threadData);
  Radiator_eqFunction_878(data, threadData);
  Radiator_eqFunction_879(data, threadData);
  Radiator_eqFunction_880(data, threadData);
  Radiator_eqFunction_881(data, threadData);
  Radiator_eqFunction_882(data, threadData);
  Radiator_eqFunction_883(data, threadData);
  Radiator_eqFunction_884(data, threadData);
  Radiator_eqFunction_885(data, threadData);
  Radiator_eqFunction_886(data, threadData);
  Radiator_eqFunction_887(data, threadData);
  Radiator_eqFunction_888(data, threadData);
  Radiator_eqFunction_889(data, threadData);
  Radiator_eqFunction_890(data, threadData);
  Radiator_eqFunction_891(data, threadData);
  Radiator_eqFunction_892(data, threadData);
  Radiator_eqFunction_893(data, threadData);
  Radiator_eqFunction_894(data, threadData);
  Radiator_eqFunction_895(data, threadData);
  Radiator_eqFunction_896(data, threadData);
  Radiator_eqFunction_897(data, threadData);
  Radiator_eqFunction_898(data, threadData);
  Radiator_eqFunction_899(data, threadData);
  Radiator_eqFunction_900(data, threadData);
  Radiator_eqFunction_901(data, threadData);
  Radiator_eqFunction_902(data, threadData);
  Radiator_eqFunction_903(data, threadData);
  Radiator_eqFunction_904(data, threadData);
  Radiator_eqFunction_905(data, threadData);
  Radiator_eqFunction_906(data, threadData);
  Radiator_eqFunction_907(data, threadData);
  Radiator_eqFunction_908(data, threadData);
  Radiator_eqFunction_909(data, threadData);
  Radiator_eqFunction_910(data, threadData);
  Radiator_eqFunction_911(data, threadData);
  Radiator_eqFunction_912(data, threadData);
  Radiator_eqFunction_913(data, threadData);
  Radiator_eqFunction_914(data, threadData);
  Radiator_eqFunction_915(data, threadData);
  Radiator_eqFunction_916(data, threadData);
  Radiator_eqFunction_917(data, threadData);
  Radiator_eqFunction_918(data, threadData);
  Radiator_eqFunction_919(data, threadData);
  Radiator_eqFunction_920(data, threadData);
  Radiator_eqFunction_921(data, threadData);
  Radiator_eqFunction_922(data, threadData);
  Radiator_eqFunction_923(data, threadData);
  Radiator_eqFunction_924(data, threadData);
  Radiator_eqFunction_925(data, threadData);
  Radiator_eqFunction_926(data, threadData);
  Radiator_eqFunction_927(data, threadData);
  Radiator_eqFunction_928(data, threadData);
  Radiator_eqFunction_929(data, threadData);
  Radiator_eqFunction_930(data, threadData);
  Radiator_eqFunction_931(data, threadData);
  Radiator_eqFunction_932(data, threadData);
  Radiator_eqFunction_933(data, threadData);
  Radiator_eqFunction_934(data, threadData);
  Radiator_eqFunction_935(data, threadData);
  Radiator_eqFunction_936(data, threadData);
  Radiator_eqFunction_937(data, threadData);
  Radiator_eqFunction_938(data, threadData);
  Radiator_eqFunction_939(data, threadData);
  Radiator_eqFunction_940(data, threadData);
  Radiator_eqFunction_941(data, threadData);
  Radiator_eqFunction_942(data, threadData);
  Radiator_eqFunction_943(data, threadData);
  Radiator_eqFunction_944(data, threadData);
  Radiator_eqFunction_945(data, threadData);
  Radiator_eqFunction_946(data, threadData);
  Radiator_eqFunction_947(data, threadData);
  Radiator_eqFunction_948(data, threadData);
  Radiator_eqFunction_949(data, threadData);
  Radiator_eqFunction_950(data, threadData);
  Radiator_eqFunction_951(data, threadData);
  Radiator_eqFunction_952(data, threadData);
  Radiator_eqFunction_953(data, threadData);
  Radiator_eqFunction_954(data, threadData);
  Radiator_eqFunction_955(data, threadData);
  Radiator_eqFunction_956(data, threadData);
  Radiator_eqFunction_957(data, threadData);
  Radiator_eqFunction_958(data, threadData);
  Radiator_eqFunction_959(data, threadData);
  Radiator_eqFunction_960(data, threadData);
  Radiator_eqFunction_961(data, threadData);
  Radiator_eqFunction_962(data, threadData);
  Radiator_eqFunction_963(data, threadData);
  Radiator_eqFunction_964(data, threadData);
  Radiator_eqFunction_965(data, threadData);
  Radiator_eqFunction_966(data, threadData);
  Radiator_eqFunction_967(data, threadData);
  Radiator_eqFunction_968(data, threadData);
  Radiator_eqFunction_969(data, threadData);
  Radiator_eqFunction_970(data, threadData);
  Radiator_eqFunction_971(data, threadData);
  Radiator_eqFunction_972(data, threadData);
  Radiator_eqFunction_973(data, threadData);
  Radiator_eqFunction_974(data, threadData);
  Radiator_eqFunction_975(data, threadData);
  Radiator_eqFunction_976(data, threadData);
  Radiator_eqFunction_977(data, threadData);
  Radiator_eqFunction_978(data, threadData);
  Radiator_eqFunction_979(data, threadData);
  Radiator_eqFunction_980(data, threadData);
  Radiator_eqFunction_981(data, threadData);
  Radiator_eqFunction_982(data, threadData);
  Radiator_eqFunction_983(data, threadData);
  Radiator_eqFunction_984(data, threadData);
  Radiator_eqFunction_985(data, threadData);
  Radiator_eqFunction_986(data, threadData);
  Radiator_eqFunction_987(data, threadData);
  Radiator_eqFunction_988(data, threadData);
  Radiator_eqFunction_989(data, threadData);
  Radiator_eqFunction_990(data, threadData);
  Radiator_eqFunction_991(data, threadData);
  Radiator_eqFunction_992(data, threadData);
  Radiator_eqFunction_993(data, threadData);
  Radiator_eqFunction_994(data, threadData);
  Radiator_eqFunction_995(data, threadData);
  Radiator_eqFunction_996(data, threadData);
  Radiator_eqFunction_997(data, threadData);
  Radiator_eqFunction_998(data, threadData);
  Radiator_eqFunction_999(data, threadData);
  Radiator_eqFunction_1000(data, threadData);
  Radiator_eqFunction_1001(data, threadData);
  Radiator_eqFunction_1002(data, threadData);
  Radiator_eqFunction_1003(data, threadData);
  Radiator_eqFunction_1004(data, threadData);
  Radiator_eqFunction_1005(data, threadData);
  Radiator_eqFunction_1006(data, threadData);
  Radiator_eqFunction_1007(data, threadData);
  Radiator_eqFunction_1008(data, threadData);
  Radiator_eqFunction_1009(data, threadData);
  Radiator_eqFunction_1010(data, threadData);
  Radiator_eqFunction_1011(data, threadData);
  Radiator_eqFunction_1012(data, threadData);
  Radiator_eqFunction_1013(data, threadData);
  Radiator_eqFunction_1014(data, threadData);
  Radiator_eqFunction_1015(data, threadData);
  Radiator_eqFunction_1016(data, threadData);
  Radiator_eqFunction_1017(data, threadData);
  Radiator_eqFunction_1018(data, threadData);
  Radiator_eqFunction_1019(data, threadData);
  Radiator_eqFunction_1020(data, threadData);
  Radiator_eqFunction_1021(data, threadData);
  Radiator_eqFunction_1022(data, threadData);
  Radiator_eqFunction_1023(data, threadData);
  Radiator_eqFunction_1024(data, threadData);
  Radiator_eqFunction_1025(data, threadData);
  Radiator_eqFunction_1026(data, threadData);
  Radiator_eqFunction_1027(data, threadData);
  Radiator_eqFunction_1028(data, threadData);
  Radiator_eqFunction_1029(data, threadData);
  Radiator_eqFunction_1030(data, threadData);
  Radiator_eqFunction_1031(data, threadData);
  Radiator_eqFunction_1032(data, threadData);
  Radiator_eqFunction_1033(data, threadData);
  Radiator_eqFunction_1034(data, threadData);
  Radiator_eqFunction_1035(data, threadData);
  Radiator_eqFunction_1036(data, threadData);
  Radiator_eqFunction_1037(data, threadData);
  Radiator_eqFunction_1038(data, threadData);
  Radiator_eqFunction_1039(data, threadData);
  Radiator_eqFunction_1040(data, threadData);
  Radiator_eqFunction_1041(data, threadData);
  Radiator_eqFunction_1042(data, threadData);
  Radiator_eqFunction_1043(data, threadData);
  Radiator_eqFunction_1044(data, threadData);
  Radiator_eqFunction_1045(data, threadData);
  Radiator_eqFunction_1046(data, threadData);
  Radiator_eqFunction_1047(data, threadData);
  Radiator_eqFunction_1048(data, threadData);
  Radiator_eqFunction_1049(data, threadData);
  Radiator_eqFunction_1050(data, threadData);
  Radiator_eqFunction_1051(data, threadData);
  Radiator_eqFunction_1052(data, threadData);
  Radiator_eqFunction_1053(data, threadData);
  Radiator_eqFunction_1054(data, threadData);
  Radiator_eqFunction_1055(data, threadData);
  Radiator_eqFunction_1056(data, threadData);
  Radiator_eqFunction_1057(data, threadData);
  Radiator_eqFunction_1058(data, threadData);
  Radiator_eqFunction_1059(data, threadData);
  Radiator_eqFunction_1060(data, threadData);
  Radiator_eqFunction_1061(data, threadData);
  Radiator_eqFunction_1062(data, threadData);
  Radiator_eqFunction_1063(data, threadData);
  Radiator_eqFunction_1064(data, threadData);
  Radiator_eqFunction_1065(data, threadData);
  Radiator_eqFunction_1066(data, threadData);
  Radiator_eqFunction_1067(data, threadData);
  Radiator_eqFunction_1068(data, threadData);
  Radiator_eqFunction_1069(data, threadData);
  Radiator_eqFunction_1070(data, threadData);
  Radiator_eqFunction_1071(data, threadData);
  Radiator_eqFunction_1072(data, threadData);
  Radiator_eqFunction_1073(data, threadData);
  Radiator_eqFunction_1074(data, threadData);
  Radiator_eqFunction_1075(data, threadData);
  Radiator_eqFunction_1076(data, threadData);
  Radiator_eqFunction_1077(data, threadData);
  Radiator_eqFunction_1078(data, threadData);
  Radiator_eqFunction_1079(data, threadData);
  Radiator_eqFunction_1080(data, threadData);
  Radiator_eqFunction_1081(data, threadData);
  Radiator_eqFunction_1082(data, threadData);
  Radiator_eqFunction_1083(data, threadData);
  Radiator_eqFunction_1084(data, threadData);
  Radiator_eqFunction_1085(data, threadData);
  Radiator_eqFunction_1086(data, threadData);
  Radiator_eqFunction_1087(data, threadData);
  Radiator_eqFunction_1088(data, threadData);
  Radiator_eqFunction_1089(data, threadData);
  Radiator_eqFunction_1090(data, threadData);
  Radiator_eqFunction_1091(data, threadData);
  Radiator_eqFunction_1092(data, threadData);
  Radiator_eqFunction_1093(data, threadData);
  Radiator_eqFunction_1094(data, threadData);
  Radiator_eqFunction_1095(data, threadData);
  Radiator_eqFunction_1096(data, threadData);
  Radiator_eqFunction_1097(data, threadData);
  Radiator_eqFunction_1098(data, threadData);
  Radiator_eqFunction_1099(data, threadData);
  Radiator_eqFunction_1100(data, threadData);
  Radiator_eqFunction_1101(data, threadData);
  TRACE_POP
}
OMC_DISABLE_OPT
int Radiator_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  data->simulationInfo->integerParameter[2] /* Radiator.nEle PARAM */ = ((modelica_integer) 5);
  data->modelData->integerParameterData[2].time_unvarying = 1;
  data->simulationInfo->integerParameter[4] /* Radiator.sumCon.nin PARAM */ = ((modelica_integer) 5);
  data->modelData->integerParameterData[4].time_unvarying = 1;
  data->simulationInfo->integerParameter[5] /* Radiator.sumRad.nin PARAM */ = ((modelica_integer) 5);
  data->modelData->integerParameterData[5].time_unvarying = 1;
  data->simulationInfo->integerParameter[17] /* Radiator.vol[1].dynBal.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[17].time_unvarying = 1;
  data->simulationInfo->integerParameter[18] /* Radiator.vol[2].dynBal.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[18].time_unvarying = 1;
  data->simulationInfo->integerParameter[19] /* Radiator.vol[3].dynBal.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[19].time_unvarying = 1;
  data->simulationInfo->integerParameter[20] /* Radiator.vol[4].dynBal.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[20].time_unvarying = 1;
  data->simulationInfo->integerParameter[21] /* Radiator.vol[5].dynBal.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[21].time_unvarying = 1;
  data->simulationInfo->integerParameter[42] /* Radiator.vol[1].nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[42].time_unvarying = 1;
  data->simulationInfo->integerParameter[43] /* Radiator.vol[2].nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[43].time_unvarying = 1;
  data->simulationInfo->integerParameter[44] /* Radiator.vol[3].nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[44].time_unvarying = 1;
  data->simulationInfo->integerParameter[45] /* Radiator.vol[4].nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[45].time_unvarying = 1;
  data->simulationInfo->integerParameter[46] /* Radiator.vol[5].nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[46].time_unvarying = 1;
  data->simulationInfo->integerParameter[58] /* flow_sink.nPorts PARAM */ = ((modelica_integer) 2);
  data->modelData->integerParameterData[58].time_unvarying = 1;
  data->simulationInfo->integerParameter[60] /* flow_source.nPorts PARAM */ = ((modelica_integer) 1);
  data->modelData->integerParameterData[60].time_unvarying = 1;
  data->simulationInfo->realParameter[7] /* Radiator.TAir_nominal PARAM */ = 293.15;
  data->modelData->realParameterData[7].time_unvarying = 1;
  data->simulationInfo->realParameter[15] /* Radiator.T_b_nominal PARAM */ = 303.15;
  data->modelData->realParameterData[15].time_unvarying = 1;
  data->simulationInfo->realParameter[19] /* Radiator.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[19].time_unvarying = 1;
  data->simulationInfo->realParameter[20] /* Radiator._dp_start PARAM */ = 0.0;
  data->modelData->realParameterData[20].time_unvarying = 1;
  data->simulationInfo->realParameter[21] /* Radiator._m_flow_start PARAM */ = 0.0;
  data->modelData->realParameterData[21].time_unvarying = 1;
  data->simulationInfo->realParameter[22] /* Radiator.cp_nominal PARAM */ = 4184.0;
  data->modelData->realParameterData[22].time_unvarying = 1;
  data->simulationInfo->realParameter[33] /* Radiator.deltaM PARAM */ = 0.3;
  data->modelData->realParameterData[33].time_unvarying = 1;
  data->simulationInfo->realParameter[34] /* Radiator.dp_nominal PARAM */ = 0.0;
  data->modelData->realParameterData[34].time_unvarying = 1;
  data->simulationInfo->realParameter[36] /* Radiator.k PARAM */ = 1.0;
  data->modelData->realParameterData[36].time_unvarying = 1;
  data->simulationInfo->realParameter[45] /* Radiator.preCon[1].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[45].time_unvarying = 1;
  data->simulationInfo->realParameter[46] /* Radiator.preCon[2].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[46].time_unvarying = 1;
  data->simulationInfo->realParameter[47] /* Radiator.preCon[3].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[47].time_unvarying = 1;
  data->simulationInfo->realParameter[48] /* Radiator.preCon[4].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[48].time_unvarying = 1;
  data->simulationInfo->realParameter[49] /* Radiator.preCon[5].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[49].time_unvarying = 1;
  data->simulationInfo->realParameter[50] /* Radiator.preCon[1].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[50].time_unvarying = 1;
  data->simulationInfo->realParameter[51] /* Radiator.preCon[2].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[51].time_unvarying = 1;
  data->simulationInfo->realParameter[52] /* Radiator.preCon[3].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[52].time_unvarying = 1;
  data->simulationInfo->realParameter[53] /* Radiator.preCon[4].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[53].time_unvarying = 1;
  data->simulationInfo->realParameter[54] /* Radiator.preCon[5].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[54].time_unvarying = 1;
  data->simulationInfo->realParameter[55] /* Radiator.preRad[1].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[55].time_unvarying = 1;
  data->simulationInfo->realParameter[56] /* Radiator.preRad[2].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[56].time_unvarying = 1;
  data->simulationInfo->realParameter[57] /* Radiator.preRad[3].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[57].time_unvarying = 1;
  data->simulationInfo->realParameter[58] /* Radiator.preRad[4].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[58].time_unvarying = 1;
  data->simulationInfo->realParameter[59] /* Radiator.preRad[5].T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[59].time_unvarying = 1;
  data->simulationInfo->realParameter[60] /* Radiator.preRad[1].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[60].time_unvarying = 1;
  data->simulationInfo->realParameter[61] /* Radiator.preRad[2].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[61].time_unvarying = 1;
  data->simulationInfo->realParameter[62] /* Radiator.preRad[3].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[62].time_unvarying = 1;
  data->simulationInfo->realParameter[63] /* Radiator.preRad[4].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[63].time_unvarying = 1;
  data->simulationInfo->realParameter[64] /* Radiator.preRad[5].alpha PARAM */ = 0.0;
  data->modelData->realParameterData[64].time_unvarying = 1;
  data->simulationInfo->realParameter[65] /* Radiator.preSumCon.T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[65].time_unvarying = 1;
  data->simulationInfo->realParameter[66] /* Radiator.preSumCon.alpha PARAM */ = 0.0;
  data->modelData->realParameterData[66].time_unvarying = 1;
  data->simulationInfo->realParameter[67] /* Radiator.preSumRad.T_ref PARAM */ = 293.15;
  data->modelData->realParameterData[67].time_unvarying = 1;
  data->simulationInfo->realParameter[68] /* Radiator.preSumRad.alpha PARAM */ = 0.0;
  data->modelData->realParameterData[68].time_unvarying = 1;
  data->simulationInfo->realParameter[69] /* Radiator.res._dp_start PARAM */ = 0.0;
  data->modelData->realParameterData[69].time_unvarying = 1;
  data->simulationInfo->realParameter[70] /* Radiator.res._m_flow_start PARAM */ = 0.0;
  data->modelData->realParameterData[70].time_unvarying = 1;
  data->simulationInfo->realParameter[71] /* Radiator.res.coeff PARAM */ = 0.0;
  data->modelData->realParameterData[71].time_unvarying = 1;
  data->simulationInfo->realParameter[72] /* Radiator.res.deltaM PARAM */ = 0.3;
  data->modelData->realParameterData[72].time_unvarying = 1;
  data->simulationInfo->realParameter[73] /* Radiator.res.dp_nominal PARAM */ = 0.0;
  data->modelData->realParameterData[73].time_unvarying = 1;
  data->simulationInfo->realParameter[74] /* Radiator.res.dp_nominal_pos PARAM */ = 0.0;
  data->modelData->realParameterData[74].time_unvarying = 1;
  data->simulationInfo->realParameter[76] /* Radiator.res.k PARAM */ = 0.0;
  data->modelData->realParameterData[76].time_unvarying = 1;
  data->simulationInfo->realParameter[80] /* Radiator.res.m_flow_turbulent PARAM */ = 0.0;
  data->modelData->realParameterData[80].time_unvarying = 1;
  data->simulationInfo->realParameter[83] /* Radiator.res.sta_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[83].time_unvarying = 1;
  data->simulationInfo->realParameter[84] /* Radiator.res.sta_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[84].time_unvarying = 1;
  data->simulationInfo->realParameter[87] /* Radiator.sumCon.k[1] PARAM */ = -1.0;
  data->modelData->realParameterData[87].time_unvarying = 1;
  data->simulationInfo->realParameter[88] /* Radiator.sumCon.k[2] PARAM */ = -1.0;
  data->modelData->realParameterData[88].time_unvarying = 1;
  data->simulationInfo->realParameter[89] /* Radiator.sumCon.k[3] PARAM */ = -1.0;
  data->modelData->realParameterData[89].time_unvarying = 1;
  data->simulationInfo->realParameter[90] /* Radiator.sumCon.k[4] PARAM */ = -1.0;
  data->modelData->realParameterData[90].time_unvarying = 1;
  data->simulationInfo->realParameter[91] /* Radiator.sumCon.k[5] PARAM */ = -1.0;
  data->modelData->realParameterData[91].time_unvarying = 1;
  data->simulationInfo->realParameter[92] /* Radiator.sumRad.k[1] PARAM */ = -1.0;
  data->modelData->realParameterData[92].time_unvarying = 1;
  data->simulationInfo->realParameter[93] /* Radiator.sumRad.k[2] PARAM */ = -1.0;
  data->modelData->realParameterData[93].time_unvarying = 1;
  data->simulationInfo->realParameter[94] /* Radiator.sumRad.k[3] PARAM */ = -1.0;
  data->modelData->realParameterData[94].time_unvarying = 1;
  data->simulationInfo->realParameter[95] /* Radiator.sumRad.k[4] PARAM */ = -1.0;
  data->modelData->realParameterData[95].time_unvarying = 1;
  data->simulationInfo->realParameter[96] /* Radiator.sumRad.k[5] PARAM */ = -1.0;
  data->modelData->realParameterData[96].time_unvarying = 1;
  data->simulationInfo->realParameter[98] /* Radiator.vol[2].T_start PARAM */ = 293.15;
  data->modelData->realParameterData[98].time_unvarying = 1;
  data->simulationInfo->realParameter[99] /* Radiator.vol[3].T_start PARAM */ = 293.15;
  data->modelData->realParameterData[99].time_unvarying = 1;
  data->simulationInfo->realParameter[100] /* Radiator.vol[4].T_start PARAM */ = 293.15;
  data->modelData->realParameterData[100].time_unvarying = 1;
  data->simulationInfo->realParameter[101] /* Radiator.vol[5].T_start PARAM */ = 293.15;
  data->modelData->realParameterData[101].time_unvarying = 1;
  data->simulationInfo->realParameter[107] /* Radiator.vol[1].X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[107].time_unvarying = 1;
  data->simulationInfo->realParameter[108] /* Radiator.vol[2].X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[108].time_unvarying = 1;
  data->simulationInfo->realParameter[109] /* Radiator.vol[3].X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[109].time_unvarying = 1;
  data->simulationInfo->realParameter[110] /* Radiator.vol[4].X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[110].time_unvarying = 1;
  data->simulationInfo->realParameter[111] /* Radiator.vol[5].X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[111].time_unvarying = 1;
  data->simulationInfo->realParameter[118] /* Radiator.vol[2].dynBal.T_start PARAM */ = 293.15;
  data->modelData->realParameterData[118].time_unvarying = 1;
  data->simulationInfo->realParameter[119] /* Radiator.vol[3].dynBal.T_start PARAM */ = 293.15;
  data->modelData->realParameterData[119].time_unvarying = 1;
  data->simulationInfo->realParameter[120] /* Radiator.vol[4].dynBal.T_start PARAM */ = 293.15;
  data->modelData->realParameterData[120].time_unvarying = 1;
  data->simulationInfo->realParameter[121] /* Radiator.vol[5].dynBal.T_start PARAM */ = 293.15;
  data->modelData->realParameterData[121].time_unvarying = 1;
  data->simulationInfo->realParameter[122] /* Radiator.vol[1].dynBal.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[122].time_unvarying = 1;
  data->simulationInfo->realParameter[123] /* Radiator.vol[2].dynBal.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[123].time_unvarying = 1;
  data->simulationInfo->realParameter[124] /* Radiator.vol[3].dynBal.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[124].time_unvarying = 1;
  data->simulationInfo->realParameter[125] /* Radiator.vol[4].dynBal.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[125].time_unvarying = 1;
  data->simulationInfo->realParameter[126] /* Radiator.vol[5].dynBal.X_start[1] PARAM */ = 1.0;
  data->modelData->realParameterData[126].time_unvarying = 1;
  data->simulationInfo->realParameter[127] /* Radiator.vol[1].dynBal.cp_default PARAM */ = 4184.0;
  data->modelData->realParameterData[127].time_unvarying = 1;
  data->simulationInfo->realParameter[128] /* Radiator.vol[2].dynBal.cp_default PARAM */ = 4184.0;
  data->modelData->realParameterData[128].time_unvarying = 1;
  data->simulationInfo->realParameter[129] /* Radiator.vol[3].dynBal.cp_default PARAM */ = 4184.0;
  data->modelData->realParameterData[129].time_unvarying = 1;
  data->simulationInfo->realParameter[130] /* Radiator.vol[4].dynBal.cp_default PARAM */ = 4184.0;
  data->modelData->realParameterData[130].time_unvarying = 1;
  data->simulationInfo->realParameter[131] /* Radiator.vol[5].dynBal.cp_default PARAM */ = 4184.0;
  data->modelData->realParameterData[131].time_unvarying = 1;
  data->simulationInfo->realParameter[138] /* Radiator.vol[2].dynBal.hStart PARAM */ = 83680.0;
  data->modelData->realParameterData[138].time_unvarying = 1;
  data->simulationInfo->realParameter[139] /* Radiator.vol[3].dynBal.hStart PARAM */ = 83680.0;
  data->modelData->realParameterData[139].time_unvarying = 1;
  data->simulationInfo->realParameter[140] /* Radiator.vol[4].dynBal.hStart PARAM */ = 83680.0;
  data->modelData->realParameterData[140].time_unvarying = 1;
  data->simulationInfo->realParameter[141] /* Radiator.vol[5].dynBal.hStart PARAM */ = 83680.0;
  data->modelData->realParameterData[141].time_unvarying = 1;
  data->simulationInfo->realParameter[143] /* Radiator.vol[2].dynBal.mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[143].time_unvarying = 1;
  data->simulationInfo->realParameter[144] /* Radiator.vol[3].dynBal.mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[144].time_unvarying = 1;
  data->simulationInfo->realParameter[145] /* Radiator.vol[4].dynBal.mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[145].time_unvarying = 1;
  data->simulationInfo->realParameter[146] /* Radiator.vol[5].dynBal.mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[146].time_unvarying = 1;
  data->simulationInfo->realParameter[158] /* Radiator.vol[2].dynBal.p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[158].time_unvarying = 1;
  data->simulationInfo->realParameter[159] /* Radiator.vol[3].dynBal.p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[159].time_unvarying = 1;
  data->simulationInfo->realParameter[160] /* Radiator.vol[4].dynBal.p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[160].time_unvarying = 1;
  data->simulationInfo->realParameter[161] /* Radiator.vol[5].dynBal.p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[161].time_unvarying = 1;
  data->simulationInfo->realParameter[172] /* Radiator.vol[1].dynBal.rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[172].time_unvarying = 1;
  data->simulationInfo->realParameter[173] /* Radiator.vol[2].dynBal.rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[173].time_unvarying = 1;
  data->simulationInfo->realParameter[174] /* Radiator.vol[3].dynBal.rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[174].time_unvarying = 1;
  data->simulationInfo->realParameter[175] /* Radiator.vol[4].dynBal.rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[175].time_unvarying = 1;
  data->simulationInfo->realParameter[176] /* Radiator.vol[5].dynBal.rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[176].time_unvarying = 1;
  data->simulationInfo->realParameter[178] /* Radiator.vol[2].dynBal.rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[178].time_unvarying = 1;
  data->simulationInfo->realParameter[179] /* Radiator.vol[3].dynBal.rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[179].time_unvarying = 1;
  data->simulationInfo->realParameter[180] /* Radiator.vol[4].dynBal.rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[180].time_unvarying = 1;
  data->simulationInfo->realParameter[181] /* Radiator.vol[5].dynBal.rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[181].time_unvarying = 1;
  data->simulationInfo->realParameter[182] /* Radiator.vol[1].dynBal.state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[182].time_unvarying = 1;
  data->simulationInfo->realParameter[183] /* Radiator.vol[2].dynBal.state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[183].time_unvarying = 1;
  data->simulationInfo->realParameter[184] /* Radiator.vol[3].dynBal.state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[184].time_unvarying = 1;
  data->simulationInfo->realParameter[185] /* Radiator.vol[4].dynBal.state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[185].time_unvarying = 1;
  data->simulationInfo->realParameter[186] /* Radiator.vol[5].dynBal.state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[186].time_unvarying = 1;
  data->simulationInfo->realParameter[187] /* Radiator.vol[1].dynBal.state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[187].time_unvarying = 1;
  data->simulationInfo->realParameter[188] /* Radiator.vol[2].dynBal.state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[188].time_unvarying = 1;
  data->simulationInfo->realParameter[189] /* Radiator.vol[3].dynBal.state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[189].time_unvarying = 1;
  data->simulationInfo->realParameter[190] /* Radiator.vol[4].dynBal.state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[190].time_unvarying = 1;
  data->simulationInfo->realParameter[191] /* Radiator.vol[5].dynBal.state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[191].time_unvarying = 1;
  data->simulationInfo->realParameter[193] /* Radiator.vol[2].mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[193].time_unvarying = 1;
  data->simulationInfo->realParameter[194] /* Radiator.vol[3].mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[194].time_unvarying = 1;
  data->simulationInfo->realParameter[195] /* Radiator.vol[4].mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[195].time_unvarying = 1;
  data->simulationInfo->realParameter[196] /* Radiator.vol[5].mSenFac PARAM */ = 1.544286174036044;
  data->modelData->realParameterData[196].time_unvarying = 1;
  data->simulationInfo->realParameter[213] /* Radiator.vol[2].p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[213].time_unvarying = 1;
  data->simulationInfo->realParameter[214] /* Radiator.vol[3].p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[214].time_unvarying = 1;
  data->simulationInfo->realParameter[215] /* Radiator.vol[4].p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[215].time_unvarying = 1;
  data->simulationInfo->realParameter[216] /* Radiator.vol[5].p_start PARAM */ = 300000.0;
  data->modelData->realParameterData[216].time_unvarying = 1;
  data->simulationInfo->realParameter[227] /* Radiator.vol[1].rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[227].time_unvarying = 1;
  data->simulationInfo->realParameter[228] /* Radiator.vol[2].rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[228].time_unvarying = 1;
  data->simulationInfo->realParameter[229] /* Radiator.vol[3].rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[229].time_unvarying = 1;
  data->simulationInfo->realParameter[230] /* Radiator.vol[4].rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[230].time_unvarying = 1;
  data->simulationInfo->realParameter[231] /* Radiator.vol[5].rho_default PARAM */ = 995.586;
  data->modelData->realParameterData[231].time_unvarying = 1;
  data->simulationInfo->realParameter[232] /* Radiator.vol[1].rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[232].time_unvarying = 1;
  data->simulationInfo->realParameter[233] /* Radiator.vol[2].rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[233].time_unvarying = 1;
  data->simulationInfo->realParameter[234] /* Radiator.vol[3].rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[234].time_unvarying = 1;
  data->simulationInfo->realParameter[235] /* Radiator.vol[4].rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[235].time_unvarying = 1;
  data->simulationInfo->realParameter[236] /* Radiator.vol[5].rho_start PARAM */ = 995.586;
  data->modelData->realParameterData[236].time_unvarying = 1;
  data->simulationInfo->realParameter[237] /* Radiator.vol[1].state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[237].time_unvarying = 1;
  data->simulationInfo->realParameter[238] /* Radiator.vol[2].state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[238].time_unvarying = 1;
  data->simulationInfo->realParameter[239] /* Radiator.vol[3].state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[239].time_unvarying = 1;
  data->simulationInfo->realParameter[240] /* Radiator.vol[4].state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[240].time_unvarying = 1;
  data->simulationInfo->realParameter[241] /* Radiator.vol[5].state_default.T PARAM */ = 293.15;
  data->modelData->realParameterData[241].time_unvarying = 1;
  data->simulationInfo->realParameter[242] /* Radiator.vol[1].state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[242].time_unvarying = 1;
  data->simulationInfo->realParameter[243] /* Radiator.vol[2].state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[243].time_unvarying = 1;
  data->simulationInfo->realParameter[244] /* Radiator.vol[3].state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[244].time_unvarying = 1;
  data->simulationInfo->realParameter[245] /* Radiator.vol[4].state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[245].time_unvarying = 1;
  data->simulationInfo->realParameter[246] /* Radiator.vol[5].state_default.p PARAM */ = 300000.0;
  data->modelData->realParameterData[246].time_unvarying = 1;
  data->simulationInfo->realParameter[247] /* Radiator.vol[1].state_start.T PARAM */ = 293.15;
  data->modelData->realParameterData[247].time_unvarying = 1;
  data->simulationInfo->realParameter[248] /* Radiator.vol[2].state_start.T PARAM */ = 293.15;
  data->modelData->realParameterData[248].time_unvarying = 1;
  data->simulationInfo->realParameter[249] /* Radiator.vol[3].state_start.T PARAM */ = 293.15;
  data->modelData->realParameterData[249].time_unvarying = 1;
  data->simulationInfo->realParameter[250] /* Radiator.vol[4].state_start.T PARAM */ = 293.15;
  data->modelData->realParameterData[250].time_unvarying = 1;
  data->simulationInfo->realParameter[251] /* Radiator.vol[5].state_start.T PARAM */ = 293.15;
  data->modelData->realParameterData[251].time_unvarying = 1;
  data->simulationInfo->realParameter[252] /* Radiator.vol[1].state_start.p PARAM */ = 300000.0;
  data->modelData->realParameterData[252].time_unvarying = 1;
  data->simulationInfo->realParameter[253] /* Radiator.vol[2].state_start.p PARAM */ = 300000.0;
  data->modelData->realParameterData[253].time_unvarying = 1;
  data->simulationInfo->realParameter[254] /* Radiator.vol[3].state_start.p PARAM */ = 300000.0;
  data->modelData->realParameterData[254].time_unvarying = 1;
  data->simulationInfo->realParameter[255] /* Radiator.vol[4].state_start.p PARAM */ = 300000.0;
  data->modelData->realParameterData[255].time_unvarying = 1;
  data->simulationInfo->realParameter[256] /* Radiator.vol[5].state_start.p PARAM */ = 300000.0;
  data->modelData->realParameterData[256].time_unvarying = 1;
  data->simulationInfo->booleanParameter[0] /* Radiator.allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[0].time_unvarying = 1;
  data->simulationInfo->booleanParameter[1] /* Radiator.from_dp PARAM */ = 0;
  data->modelData->booleanParameterData[1].time_unvarying = 1;
  data->simulationInfo->booleanParameter[2] /* Radiator.linearized PARAM */ = 0;
  data->modelData->booleanParameterData[2].time_unvarying = 1;
  data->simulationInfo->booleanParameter[3] /* Radiator.res.allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[3].time_unvarying = 1;
  data->simulationInfo->booleanParameter[4] /* Radiator.res.computeFlowResistance PARAM */ = 0;
  data->modelData->booleanParameterData[4].time_unvarying = 1;
  data->simulationInfo->booleanParameter[5] /* Radiator.res.from_dp PARAM */ = 0;
  data->modelData->booleanParameterData[5].time_unvarying = 1;
  data->simulationInfo->booleanParameter[6] /* Radiator.res.linearized PARAM */ = 0;
  data->modelData->booleanParameterData[6].time_unvarying = 1;
  data->simulationInfo->booleanParameter[7] /* Radiator.res.show_T PARAM */ = 0;
  data->modelData->booleanParameterData[7].time_unvarying = 1;
  data->simulationInfo->booleanParameter[8] /* Radiator.show_T PARAM */ = 1;
  data->modelData->booleanParameterData[8].time_unvarying = 1;
  data->simulationInfo->booleanParameter[9] /* Radiator.vol[1].allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[9].time_unvarying = 1;
  data->simulationInfo->booleanParameter[10] /* Radiator.vol[2].allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[10].time_unvarying = 1;
  data->simulationInfo->booleanParameter[11] /* Radiator.vol[3].allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[11].time_unvarying = 1;
  data->simulationInfo->booleanParameter[12] /* Radiator.vol[4].allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[12].time_unvarying = 1;
  data->simulationInfo->booleanParameter[13] /* Radiator.vol[5].allowFlowReversal PARAM */ = 1;
  data->modelData->booleanParameterData[13].time_unvarying = 1;
  data->simulationInfo->booleanParameter[14] /* Radiator.vol[1].dynBal.computeCSen PARAM */ = 1;
  data->modelData->booleanParameterData[14].time_unvarying = 1;
  data->simulationInfo->booleanParameter[15] /* Radiator.vol[2].dynBal.computeCSen PARAM */ = 1;
  data->modelData->booleanParameterData[15].time_unvarying = 1;
  data->simulationInfo->booleanParameter[16] /* Radiator.vol[3].dynBal.computeCSen PARAM */ = 1;
  data->modelData->booleanParameterData[16].time_unvarying = 1;
  data->simulationInfo->booleanParameter[17] /* Radiator.vol[4].dynBal.computeCSen PARAM */ = 1;
  data->modelData->booleanParameterData[17].time_unvarying = 1;
  data->simulationInfo->booleanParameter[18] /* Radiator.vol[5].dynBal.computeCSen PARAM */ = 1;
  data->modelData->booleanParameterData[18].time_unvarying = 1;
  data->simulationInfo->booleanParameter[19] /* Radiator.vol[1].dynBal.initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[19].time_unvarying = 1;
  data->simulationInfo->booleanParameter[20] /* Radiator.vol[2].dynBal.initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[20].time_unvarying = 1;
  data->simulationInfo->booleanParameter[21] /* Radiator.vol[3].dynBal.initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[21].time_unvarying = 1;
  data->simulationInfo->booleanParameter[22] /* Radiator.vol[4].dynBal.initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[22].time_unvarying = 1;
  data->simulationInfo->booleanParameter[23] /* Radiator.vol[5].dynBal.initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[23].time_unvarying = 1;
  data->simulationInfo->booleanParameter[24] /* Radiator.vol[1].dynBal.medium.preferredMediumStates PARAM */ = 0;
  data->modelData->booleanParameterData[24].time_unvarying = 1;
  data->simulationInfo->booleanParameter[25] /* Radiator.vol[2].dynBal.medium.preferredMediumStates PARAM */ = 0;
  data->modelData->booleanParameterData[25].time_unvarying = 1;
  data->simulationInfo->booleanParameter[26] /* Radiator.vol[3].dynBal.medium.preferredMediumStates PARAM */ = 0;
  data->modelData->booleanParameterData[26].time_unvarying = 1;
  data->simulationInfo->booleanParameter[27] /* Radiator.vol[4].dynBal.medium.preferredMediumStates PARAM */ = 0;
  data->modelData->booleanParameterData[27].time_unvarying = 1;
  data->simulationInfo->booleanParameter[28] /* Radiator.vol[5].dynBal.medium.preferredMediumStates PARAM */ = 0;
  data->modelData->booleanParameterData[28].time_unvarying = 1;
  data->simulationInfo->booleanParameter[29] /* Radiator.vol[1].dynBal.medium.standardOrderComponents PARAM */ = 1;
  data->modelData->booleanParameterData[29].time_unvarying = 1;
  data->simulationInfo->booleanParameter[30] /* Radiator.vol[2].dynBal.medium.standardOrderComponents PARAM */ = 1;
  data->modelData->booleanParameterData[30].time_unvarying = 1;
  data->simulationInfo->booleanParameter[31] /* Radiator.vol[3].dynBal.medium.standardOrderComponents PARAM */ = 1;
  data->modelData->booleanParameterData[31].time_unvarying = 1;
  data->simulationInfo->booleanParameter[32] /* Radiator.vol[4].dynBal.medium.standardOrderComponents PARAM */ = 1;
  data->modelData->booleanParameterData[32].time_unvarying = 1;
  data->simulationInfo->booleanParameter[33] /* Radiator.vol[5].dynBal.medium.standardOrderComponents PARAM */ = 1;
  data->modelData->booleanParameterData[33].time_unvarying = 1;
  data->simulationInfo->booleanParameter[34] /* Radiator.vol[1].dynBal.use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[34].time_unvarying = 1;
  data->simulationInfo->booleanParameter[35] /* Radiator.vol[2].dynBal.use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[35].time_unvarying = 1;
  data->simulationInfo->booleanParameter[36] /* Radiator.vol[3].dynBal.use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[36].time_unvarying = 1;
  data->simulationInfo->booleanParameter[37] /* Radiator.vol[4].dynBal.use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[37].time_unvarying = 1;
  data->simulationInfo->booleanParameter[38] /* Radiator.vol[5].dynBal.use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[38].time_unvarying = 1;
  data->simulationInfo->booleanParameter[39] /* Radiator.vol[1].dynBal.use_mWat_flow PARAM */ = 0;
  data->modelData->booleanParameterData[39].time_unvarying = 1;
  data->simulationInfo->booleanParameter[40] /* Radiator.vol[2].dynBal.use_mWat_flow PARAM */ = 0;
  data->modelData->booleanParameterData[40].time_unvarying = 1;
  data->simulationInfo->booleanParameter[41] /* Radiator.vol[3].dynBal.use_mWat_flow PARAM */ = 0;
  data->modelData->booleanParameterData[41].time_unvarying = 1;
  data->simulationInfo->booleanParameter[42] /* Radiator.vol[4].dynBal.use_mWat_flow PARAM */ = 0;
  data->modelData->booleanParameterData[42].time_unvarying = 1;
  data->simulationInfo->booleanParameter[43] /* Radiator.vol[5].dynBal.use_mWat_flow PARAM */ = 0;
  data->modelData->booleanParameterData[43].time_unvarying = 1;
  data->simulationInfo->booleanParameter[44] /* Radiator.vol[1].dynBal.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[44].time_unvarying = 1;
  data->simulationInfo->booleanParameter[45] /* Radiator.vol[2].dynBal.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[45].time_unvarying = 1;
  data->simulationInfo->booleanParameter[46] /* Radiator.vol[3].dynBal.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[46].time_unvarying = 1;
  data->simulationInfo->booleanParameter[47] /* Radiator.vol[4].dynBal.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[47].time_unvarying = 1;
  data->simulationInfo->booleanParameter[48] /* Radiator.vol[5].dynBal.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[48].time_unvarying = 1;
  data->simulationInfo->booleanParameter[49] /* Radiator.vol[1].initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[49].time_unvarying = 1;
  data->simulationInfo->booleanParameter[50] /* Radiator.vol[2].initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[50].time_unvarying = 1;
  data->simulationInfo->booleanParameter[51] /* Radiator.vol[3].initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[51].time_unvarying = 1;
  data->simulationInfo->booleanParameter[52] /* Radiator.vol[4].initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[52].time_unvarying = 1;
  data->simulationInfo->booleanParameter[53] /* Radiator.vol[5].initialize_p PARAM */ = 0;
  data->modelData->booleanParameterData[53].time_unvarying = 1;
  data->simulationInfo->booleanParameter[54] /* Radiator.vol[1].useSteadyStateTwoPort PARAM */ = 0;
  data->modelData->booleanParameterData[54].time_unvarying = 1;
  data->simulationInfo->booleanParameter[55] /* Radiator.vol[2].useSteadyStateTwoPort PARAM */ = 0;
  data->modelData->booleanParameterData[55].time_unvarying = 1;
  data->simulationInfo->booleanParameter[56] /* Radiator.vol[3].useSteadyStateTwoPort PARAM */ = 0;
  data->modelData->booleanParameterData[56].time_unvarying = 1;
  data->simulationInfo->booleanParameter[57] /* Radiator.vol[4].useSteadyStateTwoPort PARAM */ = 0;
  data->modelData->booleanParameterData[57].time_unvarying = 1;
  data->simulationInfo->booleanParameter[58] /* Radiator.vol[5].useSteadyStateTwoPort PARAM */ = 0;
  data->modelData->booleanParameterData[58].time_unvarying = 1;
  data->simulationInfo->booleanParameter[59] /* Radiator.vol[1].use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[59].time_unvarying = 1;
  data->simulationInfo->booleanParameter[60] /* Radiator.vol[2].use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[60].time_unvarying = 1;
  data->simulationInfo->booleanParameter[61] /* Radiator.vol[3].use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[61].time_unvarying = 1;
  data->simulationInfo->booleanParameter[62] /* Radiator.vol[4].use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[62].time_unvarying = 1;
  data->simulationInfo->booleanParameter[63] /* Radiator.vol[5].use_C_flow PARAM */ = 0;
  data->modelData->booleanParameterData[63].time_unvarying = 1;
  data->simulationInfo->booleanParameter[64] /* Radiator.vol[1].wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[64].time_unvarying = 1;
  data->simulationInfo->booleanParameter[65] /* Radiator.vol[2].wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[65].time_unvarying = 1;
  data->simulationInfo->booleanParameter[66] /* Radiator.vol[3].wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[66].time_unvarying = 1;
  data->simulationInfo->booleanParameter[67] /* Radiator.vol[4].wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[67].time_unvarying = 1;
  data->simulationInfo->booleanParameter[68] /* Radiator.vol[5].wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[68].time_unvarying = 1;
  data->simulationInfo->booleanParameter[69] /* Radiator.wrongEnergyMassBalanceConfiguration PARAM */ = 0;
  data->modelData->booleanParameterData[69].time_unvarying = 1;
  data->simulationInfo->booleanParameter[70] /* flow_sink.use_C_in PARAM */ = 0;
  data->modelData->booleanParameterData[70].time_unvarying = 1;
  data->simulationInfo->booleanParameter[71] /* flow_sink.use_T_in PARAM */ = 1;
  data->modelData->booleanParameterData[71].time_unvarying = 1;
  data->simulationInfo->booleanParameter[72] /* flow_sink.use_X_in PARAM */ = 0;
  data->modelData->booleanParameterData[72].time_unvarying = 1;
  data->simulationInfo->booleanParameter[73] /* flow_sink.use_Xi_in PARAM */ = 0;
  data->modelData->booleanParameterData[73].time_unvarying = 1;
  data->simulationInfo->booleanParameter[74] /* flow_sink.use_p_in PARAM */ = 0;
  data->modelData->booleanParameterData[74].time_unvarying = 1;
  data->simulationInfo->booleanParameter[75] /* flow_sink.verifyInputs PARAM */ = 0;
  data->modelData->booleanParameterData[75].time_unvarying = 1;
  data->simulationInfo->booleanParameter[76] /* flow_source.use_C_in PARAM */ = 0;
  data->modelData->booleanParameterData[76].time_unvarying = 1;
  data->simulationInfo->booleanParameter[77] /* flow_source.use_T_in PARAM */ = 1;
  data->modelData->booleanParameterData[77].time_unvarying = 1;
  data->simulationInfo->booleanParameter[78] /* flow_source.use_X_in PARAM */ = 0;
  data->modelData->booleanParameterData[78].time_unvarying = 1;
  data->simulationInfo->booleanParameter[79] /* flow_source.use_Xi_in PARAM */ = 0;
  data->modelData->booleanParameterData[79].time_unvarying = 1;
  data->simulationInfo->booleanParameter[80] /* flow_source.use_m_flow_in PARAM */ = 1;
  data->modelData->booleanParameterData[80].time_unvarying = 1;
  data->simulationInfo->booleanParameter[81] /* flow_source.verifyInputs PARAM */ = 0;
  data->modelData->booleanParameterData[81].time_unvarying = 1;
  data->simulationInfo->integerParameter[0] /* Radiator.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[0].time_unvarying = 1;
  data->simulationInfo->integerParameter[1] /* Radiator.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[1].time_unvarying = 1;
  data->simulationInfo->integerParameter[3] /* Radiator.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[3].time_unvarying = 1;
  data->simulationInfo->integerParameter[6] /* Radiator.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[6].time_unvarying = 1;
  data->simulationInfo->integerParameter[7] /* Radiator.vol[1].dynBal.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[7].time_unvarying = 1;
  data->simulationInfo->integerParameter[8] /* Radiator.vol[2].dynBal.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[8].time_unvarying = 1;
  data->simulationInfo->integerParameter[9] /* Radiator.vol[3].dynBal.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[9].time_unvarying = 1;
  data->simulationInfo->integerParameter[10] /* Radiator.vol[4].dynBal.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[10].time_unvarying = 1;
  data->simulationInfo->integerParameter[11] /* Radiator.vol[5].dynBal.energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[11].time_unvarying = 1;
  data->simulationInfo->integerParameter[12] /* Radiator.vol[1].dynBal.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[12].time_unvarying = 1;
  data->simulationInfo->integerParameter[13] /* Radiator.vol[2].dynBal.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[13].time_unvarying = 1;
  data->simulationInfo->integerParameter[14] /* Radiator.vol[3].dynBal.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[14].time_unvarying = 1;
  data->simulationInfo->integerParameter[15] /* Radiator.vol[4].dynBal.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[15].time_unvarying = 1;
  data->simulationInfo->integerParameter[16] /* Radiator.vol[5].dynBal.massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[16].time_unvarying = 1;
  data->simulationInfo->integerParameter[22] /* Radiator.vol[1].dynBal.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[22].time_unvarying = 1;
  data->simulationInfo->integerParameter[23] /* Radiator.vol[2].dynBal.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[23].time_unvarying = 1;
  data->simulationInfo->integerParameter[24] /* Radiator.vol[3].dynBal.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[24].time_unvarying = 1;
  data->simulationInfo->integerParameter[25] /* Radiator.vol[4].dynBal.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[25].time_unvarying = 1;
  data->simulationInfo->integerParameter[26] /* Radiator.vol[5].dynBal.substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[26].time_unvarying = 1;
  data->simulationInfo->integerParameter[27] /* Radiator.vol[1].dynBal.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[27].time_unvarying = 1;
  data->simulationInfo->integerParameter[28] /* Radiator.vol[2].dynBal.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[28].time_unvarying = 1;
  data->simulationInfo->integerParameter[29] /* Radiator.vol[3].dynBal.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[29].time_unvarying = 1;
  data->simulationInfo->integerParameter[30] /* Radiator.vol[4].dynBal.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[30].time_unvarying = 1;
  data->simulationInfo->integerParameter[31] /* Radiator.vol[5].dynBal.traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[31].time_unvarying = 1;
  data->simulationInfo->integerParameter[32] /* Radiator.vol[1].energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[32].time_unvarying = 1;
  data->simulationInfo->integerParameter[33] /* Radiator.vol[2].energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[33].time_unvarying = 1;
  data->simulationInfo->integerParameter[34] /* Radiator.vol[3].energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[34].time_unvarying = 1;
  data->simulationInfo->integerParameter[35] /* Radiator.vol[4].energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[35].time_unvarying = 1;
  data->simulationInfo->integerParameter[36] /* Radiator.vol[5].energyDynamics PARAM */ = 2;
  data->modelData->integerParameterData[36].time_unvarying = 1;
  data->simulationInfo->integerParameter[37] /* Radiator.vol[1].massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[37].time_unvarying = 1;
  data->simulationInfo->integerParameter[38] /* Radiator.vol[2].massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[38].time_unvarying = 1;
  data->simulationInfo->integerParameter[39] /* Radiator.vol[3].massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[39].time_unvarying = 1;
  data->simulationInfo->integerParameter[40] /* Radiator.vol[4].massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[40].time_unvarying = 1;
  data->simulationInfo->integerParameter[41] /* Radiator.vol[5].massDynamics PARAM */ = 2;
  data->modelData->integerParameterData[41].time_unvarying = 1;
  data->simulationInfo->integerParameter[47] /* Radiator.vol[1].substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[47].time_unvarying = 1;
  data->simulationInfo->integerParameter[48] /* Radiator.vol[2].substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[48].time_unvarying = 1;
  data->simulationInfo->integerParameter[49] /* Radiator.vol[3].substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[49].time_unvarying = 1;
  data->simulationInfo->integerParameter[50] /* Radiator.vol[4].substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[50].time_unvarying = 1;
  data->simulationInfo->integerParameter[51] /* Radiator.vol[5].substanceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[51].time_unvarying = 1;
  data->simulationInfo->integerParameter[52] /* Radiator.vol[1].traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[52].time_unvarying = 1;
  data->simulationInfo->integerParameter[53] /* Radiator.vol[2].traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[53].time_unvarying = 1;
  data->simulationInfo->integerParameter[54] /* Radiator.vol[3].traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[54].time_unvarying = 1;
  data->simulationInfo->integerParameter[55] /* Radiator.vol[4].traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[55].time_unvarying = 1;
  data->simulationInfo->integerParameter[56] /* Radiator.vol[5].traceDynamics PARAM */ = 2;
  data->modelData->integerParameterData[56].time_unvarying = 1;
  data->simulationInfo->integerParameter[57] /* flow_sink.flowDirection PARAM */ = 3;
  data->modelData->integerParameterData[57].time_unvarying = 1;
  data->simulationInfo->integerParameter[59] /* flow_source.flowDirection PARAM */ = 3;
  data->modelData->integerParameterData[59].time_unvarying = 1;
  Radiator_updateBoundParameters_0(data, threadData);
  TRACE_POP
  return 0;
}

#if defined(__cplusplus)
}
#endif

