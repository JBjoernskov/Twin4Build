/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "Radiator_model.h"
#include "simulation/solver/events.h"

#define prefixedName_performSimulation Radiator_performSimulation
#define prefixedName_updateContinuousSystem Radiator_updateContinuousSystem
#include <simulation/solver/perform_simulation.c.inc>

#define prefixedName_performQSSSimulation Radiator_performQSSSimulation
#include <simulation/solver/perform_qss_simulation.c.inc>


/* dummy VARINFO and FILEINFO */
const FILE_INFO dummyFILE_INFO = omc_dummyFileInfo;
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int Radiator_input_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->localData[0]->realVars[148] /* indoorTemperature variable */ = data->simulationInfo->inputVars[0];
  data->localData[0]->realVars[150] /* supplyWaterTemperature variable */ = data->simulationInfo->inputVars[1];
  data->localData[0]->realVars[151] /* waterFlowRate variable */ = data->simulationInfo->inputVars[2];
  
  TRACE_POP
  return 0;
}

int Radiator_input_function_init(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->inputVars[0] = data->modelData->realVarsData[148].attribute.start;
  data->simulationInfo->inputVars[1] = data->modelData->realVarsData[150].attribute.start;
  data->simulationInfo->inputVars[2] = data->modelData->realVarsData[151].attribute.start;
  
  TRACE_POP
  return 0;
}

int Radiator_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->modelData->realVarsData[148].attribute.start = data->simulationInfo->inputVars[0];
  data->modelData->realVarsData[150].attribute.start = data->simulationInfo->inputVars[1];
  data->modelData->realVarsData[151].attribute.start = data->simulationInfo->inputVars[2];
  
  TRACE_POP
  return 0;
}

int Radiator_inputNames(DATA *data, char ** names){
  TRACE_PUSH

  names[0] = (char *) data->modelData->realVarsData[148].info.name;
  names[1] = (char *) data->modelData->realVarsData[150].info.name;
  names[2] = (char *) data->modelData->realVarsData[151].info.name;
  
  TRACE_POP
  return 0;
}

int Radiator_data_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  TRACE_POP
  return 0;
}

int Radiator_dataReconciliationInputNames(DATA *data, char ** names){
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int Radiator_output_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->outputVars[0] = data->localData[0]->realVars[0] /* Energy STATE(1,Power) */;
  data->simulationInfo->outputVars[1] = data->localData[0]->realVars[26] /* Power variable */;
  data->simulationInfo->outputVars[2] = data->localData[0]->realVars[27] /* Q_con variable */;
  data->simulationInfo->outputVars[3] = data->localData[0]->realVars[28] /* Q_rad variable */;
  data->simulationInfo->outputVars[4] = data->localData[0]->realVars[149] /* outletWaterTemperature variable */;
  
  TRACE_POP
  return 0;
}

int Radiator_setc_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/*
equation index: 321
type: SIMPLE_ASSIGN
flow_sink.ports[1].m_flow = waterFlowRate
*/
void Radiator_eqFunction_321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,321};
  data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */ = data->localData[0]->realVars[151] /* waterFlowRate variable */;
  TRACE_POP
}
/*
equation index: 322
type: SIMPLE_ASSIGN
Radiator.vol[5].ports[1].m_flow = waterFlowRate
*/
void Radiator_eqFunction_322(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,322};
  data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */ = data->localData[0]->realVars[151] /* waterFlowRate variable */;
  TRACE_POP
}
/*
equation index: 323
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[1].m_flow = waterFlowRate
*/
void Radiator_eqFunction_323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,323};
  data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */ = data->localData[0]->realVars[151] /* waterFlowRate variable */;
  TRACE_POP
}
/*
equation index: 324
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[1].m_flow = waterFlowRate
*/
void Radiator_eqFunction_324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,324};
  data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */ = data->localData[0]->realVars[151] /* waterFlowRate variable */;
  TRACE_POP
}
/*
equation index: 325
type: SIMPLE_ASSIGN
Radiator.vol[2].ports[1].m_flow = waterFlowRate
*/
void Radiator_eqFunction_325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,325};
  data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */ = data->localData[0]->realVars[151] /* waterFlowRate variable */;
  TRACE_POP
}
/*
equation index: 326
type: SIMPLE_ASSIGN
flow_sink.p_in_internal = flow_sink.p
*/
void Radiator_eqFunction_326(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,326};
  data->localData[0]->realVars[141] /* flow_sink.p_in_internal variable */ = data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */;
  TRACE_POP
}
/*
equation index: 327
type: SIMPLE_ASSIGN
flow_sink.X_in_internal[1] = flow_sink.X[1]
*/
void Radiator_eqFunction_327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,327};
  data->localData[0]->realVars[140] /* flow_sink.X_in_internal[1] variable */ = data->simulationInfo->realParameter[261] /* flow_sink.X[1] PARAM */;
  TRACE_POP
}
/*
equation index: 328
type: SIMPLE_ASSIGN
flow_source.X_in_internal[1] = flow_source.X[1]
*/
void Radiator_eqFunction_328(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,328};
  data->localData[0]->realVars[146] /* flow_source.X_in_internal[1] variable */ = data->simulationInfo->realParameter[266] /* flow_source.X[1] PARAM */;
  TRACE_POP
}
/*
equation index: 335
type: LINEAR

<var>Radiator.vol[1].dynBal.medium.T</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Radiator_eqFunction_335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,335};
  /* Linear equation system */
  int retValue;
  double aux_x[1] = { data->localData[1]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */ };
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving linear system 335 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  
  retValue = solve_linear_system(data, threadData, 0, &aux_x[0]);
  
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,335};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving linear system 335 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */ = aux_x[0];

  TRACE_POP
}
/*
equation index: 336
type: SIMPLE_ASSIGN
Radiator.vol[1].T = Radiator.Radiator.vol.Medium.temperature_phX(flow_sink.p, Radiator.port_a.h_outflow, {1.0})
*/
void Radiator_eqFunction_336(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,336};
  data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */ = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */, _OMC_LIT24);
  TRACE_POP
}
/*
equation index: 341
type: LINEAR

<var>Radiator.vol[2].dynBal.medium.T_degC</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Radiator_eqFunction_341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,341};
  /* Linear equation system */
  int retValue;
  double aux_x[1] = { data->localData[1]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */ };
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving linear system 341 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  
  retValue = solve_linear_system(data, threadData, 1, &aux_x[0]);
  
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,341};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving linear system 341 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */ = aux_x[0];

  TRACE_POP
}
/*
equation index: 342
type: SIMPLE_ASSIGN
Radiator.vol[2].T = Radiator.Radiator.vol.Medium.temperature_phX(flow_sink.p, Radiator.vol[2].ports[2].h_outflow, {1.0})
*/
void Radiator_eqFunction_342(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,342};
  data->localData[0]->realVars[56] /* Radiator.vol[2].T variable */ = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, _OMC_LIT24);
  TRACE_POP
}
/*
equation index: 343
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.ports_H_flow[2] = -semiLinear(waterFlowRate, Radiator.port_a.h_outflow, Radiator.vol[2].ports[2].h_outflow)
*/
void Radiator_eqFunction_343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,343};
  data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */ = (-semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 344
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports_H_flow[1] = semiLinear(waterFlowRate, Radiator.port_a.h_outflow, Radiator.vol[2].ports[2].h_outflow)
*/
void Radiator_eqFunction_344(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,344};
  data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */);
  TRACE_POP
}
/*
equation index: 345
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.medium.T = Radiator.vol[2].dynBal.medium.T_degC - -273.15
*/
void Radiator_eqFunction_345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,345};
  data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */ = data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */ - (-273.15);
  TRACE_POP
}
/*
equation index: 350
type: LINEAR

<var>Radiator.vol[3].dynBal.medium.T_degC</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Radiator_eqFunction_350(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,350};
  /* Linear equation system */
  int retValue;
  double aux_x[1] = { data->localData[1]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */ };
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving linear system 350 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  
  retValue = solve_linear_system(data, threadData, 2, &aux_x[0]);
  
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,350};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving linear system 350 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */ = aux_x[0];

  TRACE_POP
}
/*
equation index: 351
type: SIMPLE_ASSIGN
Radiator.vol[3].T = Radiator.Radiator.vol.Medium.temperature_phX(flow_sink.p, Radiator.vol[3].ports[2].h_outflow, {1.0})
*/
void Radiator_eqFunction_351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,351};
  data->localData[0]->realVars[57] /* Radiator.vol[3].T variable */ = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, _OMC_LIT24);
  TRACE_POP
}
/*
equation index: 352
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.ports_H_flow[2] = -semiLinear(waterFlowRate, Radiator.vol[2].ports[2].h_outflow, Radiator.vol[3].ports[2].h_outflow)
*/
void Radiator_eqFunction_352(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,352};
  data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */ = (-semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 353
type: SIMPLE_ASSIGN
Radiator.vol[2].dynBal.Hb_flow = Radiator.vol[2].dynBal.ports_H_flow[1] + Radiator.vol[2].dynBal.ports_H_flow[2]
*/
void Radiator_eqFunction_353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,353};
  data->localData[0]->realVars[61] /* Radiator.vol[2].dynBal.Hb_flow variable */ = data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */ + data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */;
  TRACE_POP
}
/*
equation index: 354
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports_H_flow[1] = semiLinear(waterFlowRate, Radiator.vol[2].ports[2].h_outflow, Radiator.vol[3].ports[2].h_outflow)
*/
void Radiator_eqFunction_354(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,354};
  data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */);
  TRACE_POP
}
/*
equation index: 355
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.medium.T = Radiator.vol[3].dynBal.medium.T_degC - -273.15
*/
void Radiator_eqFunction_355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,355};
  data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */ = data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */ - (-273.15);
  TRACE_POP
}
/*
equation index: 360
type: LINEAR

<var>Radiator.vol[4].dynBal.medium.T_degC</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Radiator_eqFunction_360(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,360};
  /* Linear equation system */
  int retValue;
  double aux_x[1] = { data->localData[1]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */ };
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving linear system 360 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  
  retValue = solve_linear_system(data, threadData, 3, &aux_x[0]);
  
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,360};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving linear system 360 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */ = aux_x[0];

  TRACE_POP
}
/*
equation index: 361
type: SIMPLE_ASSIGN
Radiator.vol[4].T = Radiator.Radiator.vol.Medium.temperature_phX(flow_sink.p, Radiator.vol[4].ports[2].h_outflow, {1.0})
*/
void Radiator_eqFunction_361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,361};
  data->localData[0]->realVars[58] /* Radiator.vol[4].T variable */ = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, _OMC_LIT24);
  TRACE_POP
}
/*
equation index: 362
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.ports_H_flow[2] = -semiLinear(waterFlowRate, Radiator.vol[3].ports[2].h_outflow, Radiator.vol[4].ports[2].h_outflow)
*/
void Radiator_eqFunction_362(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,362};
  data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */ = (-semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 363
type: SIMPLE_ASSIGN
Radiator.vol[3].dynBal.Hb_flow = Radiator.vol[3].dynBal.ports_H_flow[1] + Radiator.vol[3].dynBal.ports_H_flow[2]
*/
void Radiator_eqFunction_363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,363};
  data->localData[0]->realVars[62] /* Radiator.vol[3].dynBal.Hb_flow variable */ = data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */ + data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */;
  TRACE_POP
}
/*
equation index: 364
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports_H_flow[1] = semiLinear(waterFlowRate, Radiator.vol[3].ports[2].h_outflow, Radiator.vol[4].ports[2].h_outflow)
*/
void Radiator_eqFunction_364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,364};
  data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */);
  TRACE_POP
}
/*
equation index: 365
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.medium.T = Radiator.vol[4].dynBal.medium.T_degC - -273.15
*/
void Radiator_eqFunction_365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,365};
  data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */ = data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */ - (-273.15);
  TRACE_POP
}
/*
equation index: 370
type: LINEAR

<var>Radiator.vol[5].ports[2].h_outflow</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Radiator_eqFunction_370(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,370};
  /* Linear equation system */
  int retValue;
  double aux_x[1] = { data->localData[1]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */ };
  if(ACTIVE_STREAM(LOG_DT))
  {
    infoStreamPrint(LOG_DT, 1, "Solving linear system 370 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);
    messageClose(LOG_DT);
  }
  
  retValue = solve_linear_system(data, threadData, 4, &aux_x[0]);
  
  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,370};
    throwStreamPrintWithEquationIndexes(threadData, indexes, "Solving linear system 370 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */ = aux_x[0];

  TRACE_POP
}
/*
equation index: 371
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.T = Radiator.vol[5].dynBal.medium.T_degC - -273.15
*/
void Radiator_eqFunction_371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,371};
  data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */ = data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */ - (-273.15);
  TRACE_POP
}
/*
equation index: 372
type: SIMPLE_ASSIGN
Radiator.vol[5].T = Radiator.Radiator.vol.Medium.temperature_phX(flow_sink.p, Radiator.vol[5].ports[2].h_outflow, {1.0})
*/
void Radiator_eqFunction_372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,372};
  data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */ = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */, _OMC_LIT24);
  TRACE_POP
}
/*
equation index: 373
type: SIMPLE_ASSIGN
outletWaterTemperature = -273.15 + Radiator.vol[5].T
*/
void Radiator_eqFunction_373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,373};
  data->localData[0]->realVars[149] /* outletWaterTemperature variable */ = -273.15 + data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */;
  TRACE_POP
}
/*
equation index: 374
type: ALGORITHM

  $cse12 := Radiator.flow_sink.Medium.setState_pTX(flow_sink.p, Radiator.vol[5].T, {flow_sink.X[1]});
*/
void Radiator_eqFunction_374(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,374};
  real_array tmp0;
  Radiator_flow__sink_Medium_ThermodynamicState tmp1;
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)data->simulationInfo->realParameter[261] /* flow_sink.X[1] PARAM */);
  tmp1 = omc_Radiator_flow__sink_Medium_setState__pTX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */, tmp0);
  data->localData[0]->realVars[17] /* $cse12.p variable */ = tmp1._p;
  data->localData[0]->realVars[16] /* $cse12.T variable */ = tmp1._T;
  ;
  TRACE_POP
}
/*
equation index: 375
type: SIMPLE_ASSIGN
flow_sink.ports[2].h_outflow = Radiator.flow_sink.Medium.specificEnthalpy($cse12)
*/
void Radiator_eqFunction_375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,375};
  data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */ = omc_Radiator_flow__sink_Medium_specificEnthalpy(threadData, omc_Radiator_flow__sink_Medium_ThermodynamicState(threadData, data->localData[0]->realVars[17] /* $cse12.p variable */, data->localData[0]->realVars[16] /* $cse12.T variable */));
  TRACE_POP
}
/*
equation index: 376
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.ports_H_flow[2] = -semiLinear(waterFlowRate, Radiator.vol[4].ports[2].h_outflow, Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_376(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,376};
  data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */ = (-semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 377
type: SIMPLE_ASSIGN
Radiator.vol[4].dynBal.Hb_flow = Radiator.vol[4].dynBal.ports_H_flow[1] + Radiator.vol[4].dynBal.ports_H_flow[2]
*/
void Radiator_eqFunction_377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,377};
  data->localData[0]->realVars[63] /* Radiator.vol[4].dynBal.Hb_flow variable */ = data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */ + data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */;
  TRACE_POP
}
/*
equation index: 378
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports_H_flow[1] = semiLinear(waterFlowRate, Radiator.vol[4].ports[2].h_outflow, Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_378(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,378};
  data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */);
  TRACE_POP
}
/*
equation index: 379
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.ports_H_flow[2] = -semiLinear(waterFlowRate, Radiator.vol[5].ports[2].h_outflow, flow_sink.ports[2].h_outflow)
*/
void Radiator_eqFunction_379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,379};
  data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */ = (-semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */, data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 380
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.Hb_flow = Radiator.vol[5].dynBal.ports_H_flow[1] + Radiator.vol[5].dynBal.ports_H_flow[2]
*/
void Radiator_eqFunction_380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,380};
  data->localData[0]->realVars[64] /* Radiator.vol[5].dynBal.Hb_flow variable */ = data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */ + data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */;
  TRACE_POP
}
/*
equation index: 381
type: SIMPLE_ASSIGN
Radiator.sta_b.T = 273.15 + 0.0002390057361376673 * (if noEvent((-waterFlowRate) > 0.0) then flow_sink.ports[2].h_outflow else Radiator.vol[5].ports[2].h_outflow)
*/
void Radiator_eqFunction_381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,381};
  modelica_boolean tmp2;
  tmp2 = Greater((-data->localData[0]->realVars[151] /* waterFlowRate variable */),0.0);
  data->localData[0]->realVars[54] /* Radiator.sta_b.T variable */ = 273.15 + (0.0002390057361376673) * ((tmp2?data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */:data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */));
  TRACE_POP
}
/*
equation index: 382
type: SIMPLE_ASSIGN
flow_source.T_in = 273.15 + supplyWaterTemperature
*/
void Radiator_eqFunction_382(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,382};
  data->localData[0]->realVars[145] /* flow_source.T_in variable */ = 273.15 + data->localData[0]->realVars[150] /* supplyWaterTemperature variable */;
  TRACE_POP
}
/*
equation index: 383
type: ALGORITHM

  $cse1 := Radiator.flow_source.Medium.setState_pTX(flow_sink.p, flow_source.T_in, {flow_source.X[1]});
*/
void Radiator_eqFunction_383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,383};
  real_array tmp3;
  Radiator_flow__source_Medium_ThermodynamicState tmp4;
  array_alloc_scalar_real_array(&tmp3, 1, (modelica_real)data->simulationInfo->realParameter[266] /* flow_source.X[1] PARAM */);
  tmp4 = omc_Radiator_flow__source_Medium_setState__pTX(threadData, data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, data->localData[0]->realVars[145] /* flow_source.T_in variable */, tmp3);
  data->localData[0]->realVars[13] /* $cse1.p variable */ = tmp4._p;
  data->localData[0]->realVars[12] /* $cse1.T variable */ = tmp4._T;
  ;
  TRACE_POP
}
/*
equation index: 384
type: SIMPLE_ASSIGN
flow_source.ports[1].h_outflow = Radiator.flow_source.Medium.specificEnthalpy($cse1)
*/
void Radiator_eqFunction_384(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,384};
  data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */ = omc_Radiator_flow__source_Medium_specificEnthalpy(threadData, omc_Radiator_flow__source_Medium_ThermodynamicState(threadData, data->localData[0]->realVars[13] /* $cse1.p variable */, data->localData[0]->realVars[12] /* $cse1.T variable */));
  TRACE_POP
}
/*
equation index: 385
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.ports_H_flow[1] = semiLinear(waterFlowRate, flow_source.ports[1].h_outflow, Radiator.port_a.h_outflow)
*/
void Radiator_eqFunction_385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,385};
  data->localData[0]->realVars[115] /* Radiator.vol[1].dynBal.ports_H_flow[1] variable */ = semiLinear(data->localData[0]->realVars[151] /* waterFlowRate variable */, data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */, data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */);
  TRACE_POP
}
/*
equation index: 386
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.Hb_flow = Radiator.vol[1].dynBal.ports_H_flow[1] + Radiator.vol[1].dynBal.ports_H_flow[2]
*/
void Radiator_eqFunction_386(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,386};
  data->localData[0]->realVars[60] /* Radiator.vol[1].dynBal.Hb_flow variable */ = data->localData[0]->realVars[115] /* Radiator.vol[1].dynBal.ports_H_flow[1] variable */ + data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */;
  TRACE_POP
}
/*
equation index: 387
type: SIMPLE_ASSIGN
Radiator.sta_a.T = 273.15 + 0.0002390057361376673 * (if noEvent(waterFlowRate > 0.0) then flow_source.ports[1].h_outflow else Radiator.port_a.h_outflow)
*/
void Radiator_eqFunction_387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,387};
  modelica_boolean tmp5;
  tmp5 = Greater(data->localData[0]->realVars[151] /* waterFlowRate variable */,0.0);
  data->localData[0]->realVars[53] /* Radiator.sta_a.T variable */ = 273.15 + (0.0002390057361376673) * ((tmp5?data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */:data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */));
  TRACE_POP
}
/*
equation index: 388
type: SIMPLE_ASSIGN
T_z_source.T = 273.15 + indoorTemperature
*/
void Radiator_eqFunction_388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,388};
  data->localData[0]->realVars[138] /* T_z_source.T variable */ = 273.15 + data->localData[0]->realVars[148] /* indoorTemperature variable */;
  TRACE_POP
}
/*
equation index: 389
type: SIMPLE_ASSIGN
Radiator.dTRad[5] = T_z_source.T - Radiator.vol[5].T
*/
void Radiator_eqFunction_389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,389};
  data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */ = data->localData[0]->realVars[138] /* T_z_source.T variable */ - data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */;
  TRACE_POP
}
/*
equation index: 390
type: SIMPLE_ASSIGN
$cse6 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[5], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_390(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,390};
  data->localData[0]->realVars[22] /* $cse6 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 391
type: SIMPLE_ASSIGN
Radiator.preCon[5].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTRad[5] * $cse6
*/
void Radiator_eqFunction_391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,391};
  data->localData[0]->realVars[46] /* Radiator.preCon[5].Q_flow variable */ = (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * ((data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */) * (data->localData[0]->realVars[22] /* $cse6 variable */)));
  TRACE_POP
}
/*
equation index: 392
type: SIMPLE_ASSIGN
$cse11 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[5], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_392(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,392};
  data->localData[0]->realVars[15] /* $cse11 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 393
type: SIMPLE_ASSIGN
Radiator.preRad[5].Q_flow = Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[5] * $cse11
*/
void Radiator_eqFunction_393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,393};
  data->localData[0]->realVars[51] /* Radiator.preRad[5].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */)) * (data->localData[0]->realVars[15] /* $cse11 variable */);
  TRACE_POP
}
/*
equation index: 394
type: SIMPLE_ASSIGN
Radiator.vol[5].heatPort.Q_flow = Radiator.preRad[5].Q_flow + Radiator.preCon[5].Q_flow
*/
void Radiator_eqFunction_394(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,394};
  data->localData[0]->realVars[129] /* Radiator.vol[5].heatPort.Q_flow variable */ = data->localData[0]->realVars[51] /* Radiator.preRad[5].Q_flow variable */ + data->localData[0]->realVars[46] /* Radiator.preCon[5].Q_flow variable */;
  TRACE_POP
}
/*
equation index: 395
type: SIMPLE_ASSIGN
$DER.Radiator.vol[5].dynBal.U = Radiator.vol[5].dynBal.Hb_flow + Radiator.vol[5].heatPort.Q_flow
*/
void Radiator_eqFunction_395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,395};
  data->localData[0]->realVars[11] /* der(Radiator.vol[5].dynBal.U) STATE_DER */ = data->localData[0]->realVars[64] /* Radiator.vol[5].dynBal.Hb_flow variable */ + data->localData[0]->realVars[129] /* Radiator.vol[5].heatPort.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 396
type: SIMPLE_ASSIGN
Radiator.dTCon[5] = Radiator.dTRad[5]
*/
void Radiator_eqFunction_396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,396};
  data->localData[0]->realVars[34] /* Radiator.dTCon[5] variable */ = data->localData[0]->realVars[39] /* Radiator.dTRad[5] variable */;
  TRACE_POP
}
/*
equation index: 397
type: SIMPLE_ASSIGN
Radiator.dTRad[4] = T_z_source.T - Radiator.vol[4].T
*/
void Radiator_eqFunction_397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,397};
  data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */ = data->localData[0]->realVars[138] /* T_z_source.T variable */ - data->localData[0]->realVars[58] /* Radiator.vol[4].T variable */;
  TRACE_POP
}
/*
equation index: 398
type: SIMPLE_ASSIGN
$cse5 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[4], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_398(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,398};
  data->localData[0]->realVars[21] /* $cse5 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 399
type: SIMPLE_ASSIGN
Radiator.preCon[4].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTRad[4] * $cse5
*/
void Radiator_eqFunction_399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,399};
  data->localData[0]->realVars[45] /* Radiator.preCon[4].Q_flow variable */ = (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * ((data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */) * (data->localData[0]->realVars[21] /* $cse5 variable */)));
  TRACE_POP
}
/*
equation index: 400
type: SIMPLE_ASSIGN
$cse10 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[4], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_400(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,400};
  data->localData[0]->realVars[14] /* $cse10 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 401
type: SIMPLE_ASSIGN
Radiator.preRad[4].Q_flow = Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[4] * $cse10
*/
void Radiator_eqFunction_401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,401};
  data->localData[0]->realVars[50] /* Radiator.preRad[4].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */)) * (data->localData[0]->realVars[14] /* $cse10 variable */);
  TRACE_POP
}
/*
equation index: 402
type: SIMPLE_ASSIGN
Radiator.vol[4].heatPort.Q_flow = Radiator.preRad[4].Q_flow + Radiator.preCon[4].Q_flow
*/
void Radiator_eqFunction_402(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,402};
  data->localData[0]->realVars[128] /* Radiator.vol[4].heatPort.Q_flow variable */ = data->localData[0]->realVars[50] /* Radiator.preRad[4].Q_flow variable */ + data->localData[0]->realVars[45] /* Radiator.preCon[4].Q_flow variable */;
  TRACE_POP
}
/*
equation index: 403
type: SIMPLE_ASSIGN
$DER.Radiator.vol[4].dynBal.U = Radiator.vol[4].dynBal.Hb_flow + Radiator.vol[4].heatPort.Q_flow
*/
void Radiator_eqFunction_403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,403};
  data->localData[0]->realVars[10] /* der(Radiator.vol[4].dynBal.U) STATE_DER */ = data->localData[0]->realVars[63] /* Radiator.vol[4].dynBal.Hb_flow variable */ + data->localData[0]->realVars[128] /* Radiator.vol[4].heatPort.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 404
type: SIMPLE_ASSIGN
Radiator.dTCon[4] = Radiator.dTRad[4]
*/
void Radiator_eqFunction_404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,404};
  data->localData[0]->realVars[33] /* Radiator.dTCon[4] variable */ = data->localData[0]->realVars[38] /* Radiator.dTRad[4] variable */;
  TRACE_POP
}
/*
equation index: 405
type: SIMPLE_ASSIGN
Radiator.dTRad[3] = T_z_source.T - Radiator.vol[3].T
*/
void Radiator_eqFunction_405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,405};
  data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */ = data->localData[0]->realVars[138] /* T_z_source.T variable */ - data->localData[0]->realVars[57] /* Radiator.vol[3].T variable */;
  TRACE_POP
}
/*
equation index: 406
type: SIMPLE_ASSIGN
$cse4 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[3], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_406(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,406};
  data->localData[0]->realVars[20] /* $cse4 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 407
type: SIMPLE_ASSIGN
Radiator.preCon[3].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTRad[3] * $cse4
*/
void Radiator_eqFunction_407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,407};
  data->localData[0]->realVars[44] /* Radiator.preCon[3].Q_flow variable */ = (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * ((data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */) * (data->localData[0]->realVars[20] /* $cse4 variable */)));
  TRACE_POP
}
/*
equation index: 408
type: SIMPLE_ASSIGN
$cse9 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[3], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_408(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,408};
  data->localData[0]->realVars[25] /* $cse9 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 409
type: SIMPLE_ASSIGN
Radiator.preRad[3].Q_flow = Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[3] * $cse9
*/
void Radiator_eqFunction_409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,409};
  data->localData[0]->realVars[49] /* Radiator.preRad[3].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */)) * (data->localData[0]->realVars[25] /* $cse9 variable */);
  TRACE_POP
}
/*
equation index: 410
type: SIMPLE_ASSIGN
Radiator.vol[3].heatPort.Q_flow = Radiator.preRad[3].Q_flow + Radiator.preCon[3].Q_flow
*/
void Radiator_eqFunction_410(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,410};
  data->localData[0]->realVars[127] /* Radiator.vol[3].heatPort.Q_flow variable */ = data->localData[0]->realVars[49] /* Radiator.preRad[3].Q_flow variable */ + data->localData[0]->realVars[44] /* Radiator.preCon[3].Q_flow variable */;
  TRACE_POP
}
/*
equation index: 411
type: SIMPLE_ASSIGN
$DER.Radiator.vol[3].dynBal.U = Radiator.vol[3].dynBal.Hb_flow + Radiator.vol[3].heatPort.Q_flow
*/
void Radiator_eqFunction_411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,411};
  data->localData[0]->realVars[9] /* der(Radiator.vol[3].dynBal.U) STATE_DER */ = data->localData[0]->realVars[62] /* Radiator.vol[3].dynBal.Hb_flow variable */ + data->localData[0]->realVars[127] /* Radiator.vol[3].heatPort.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 412
type: SIMPLE_ASSIGN
Radiator.dTCon[3] = Radiator.dTRad[3]
*/
void Radiator_eqFunction_412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,412};
  data->localData[0]->realVars[32] /* Radiator.dTCon[3] variable */ = data->localData[0]->realVars[37] /* Radiator.dTRad[3] variable */;
  TRACE_POP
}
/*
equation index: 413
type: SIMPLE_ASSIGN
Radiator.dTRad[2] = T_z_source.T - Radiator.vol[2].T
*/
void Radiator_eqFunction_413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,413};
  data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */ = data->localData[0]->realVars[138] /* T_z_source.T variable */ - data->localData[0]->realVars[56] /* Radiator.vol[2].T variable */;
  TRACE_POP
}
/*
equation index: 414
type: SIMPLE_ASSIGN
$cse3 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[2], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_414(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,414};
  data->localData[0]->realVars[19] /* $cse3 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 415
type: SIMPLE_ASSIGN
Radiator.preCon[2].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTRad[2] * $cse3
*/
void Radiator_eqFunction_415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,415};
  data->localData[0]->realVars[43] /* Radiator.preCon[2].Q_flow variable */ = (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * ((data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */) * (data->localData[0]->realVars[19] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 416
type: SIMPLE_ASSIGN
$cse8 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[2], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_416(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,416};
  data->localData[0]->realVars[24] /* $cse8 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 417
type: SIMPLE_ASSIGN
Radiator.preRad[2].Q_flow = Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[2] * $cse8
*/
void Radiator_eqFunction_417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,417};
  data->localData[0]->realVars[48] /* Radiator.preRad[2].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */)) * (data->localData[0]->realVars[24] /* $cse8 variable */);
  TRACE_POP
}
/*
equation index: 418
type: SIMPLE_ASSIGN
Radiator.vol[2].heatPort.Q_flow = Radiator.preRad[2].Q_flow + Radiator.preCon[2].Q_flow
*/
void Radiator_eqFunction_418(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,418};
  data->localData[0]->realVars[126] /* Radiator.vol[2].heatPort.Q_flow variable */ = data->localData[0]->realVars[48] /* Radiator.preRad[2].Q_flow variable */ + data->localData[0]->realVars[43] /* Radiator.preCon[2].Q_flow variable */;
  TRACE_POP
}
/*
equation index: 419
type: SIMPLE_ASSIGN
$DER.Radiator.vol[2].dynBal.U = Radiator.vol[2].dynBal.Hb_flow + Radiator.vol[2].heatPort.Q_flow
*/
void Radiator_eqFunction_419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,419};
  data->localData[0]->realVars[8] /* der(Radiator.vol[2].dynBal.U) STATE_DER */ = data->localData[0]->realVars[61] /* Radiator.vol[2].dynBal.Hb_flow variable */ + data->localData[0]->realVars[126] /* Radiator.vol[2].heatPort.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 420
type: SIMPLE_ASSIGN
Radiator.dTCon[2] = Radiator.dTRad[2]
*/
void Radiator_eqFunction_420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,420};
  data->localData[0]->realVars[31] /* Radiator.dTCon[2] variable */ = data->localData[0]->realVars[36] /* Radiator.dTRad[2] variable */;
  TRACE_POP
}
/*
equation index: 421
type: SIMPLE_ASSIGN
Radiator.dTRad[1] = T_z_source.T - Radiator.vol[1].T
*/
void Radiator_eqFunction_421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,421};
  data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */ = data->localData[0]->realVars[138] /* T_z_source.T variable */ - data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */;
  TRACE_POP
}
/*
equation index: 422
type: SIMPLE_ASSIGN
$cse2 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[1], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_422(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,422};
  data->localData[0]->realVars[18] /* $cse2 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 423
type: SIMPLE_ASSIGN
Radiator.preCon[1].Q_flow = (1.0 - Radiator.fraRad) * Radiator.UAEle * Radiator.dTRad[1] * $cse2
*/
void Radiator_eqFunction_423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,423};
  data->localData[0]->realVars[42] /* Radiator.preCon[1].Q_flow variable */ = (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * ((data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */) * (data->localData[0]->realVars[18] /* $cse2 variable */)));
  TRACE_POP
}
/*
equation index: 424
type: SIMPLE_ASSIGN
Q_con = Radiator.sumCon.k[1] * Radiator.preCon[1].Q_flow + Radiator.sumCon.k[2] * Radiator.preCon[2].Q_flow + Radiator.sumCon.k[3] * Radiator.preCon[3].Q_flow + Radiator.sumCon.k[4] * Radiator.preCon[4].Q_flow + Radiator.sumCon.k[5] * Radiator.preCon[5].Q_flow
*/
void Radiator_eqFunction_424(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,424};
  data->localData[0]->realVars[27] /* Q_con variable */ = (data->simulationInfo->realParameter[87] /* Radiator.sumCon.k[1] PARAM */) * (data->localData[0]->realVars[42] /* Radiator.preCon[1].Q_flow variable */) + (data->simulationInfo->realParameter[88] /* Radiator.sumCon.k[2] PARAM */) * (data->localData[0]->realVars[43] /* Radiator.preCon[2].Q_flow variable */) + (data->simulationInfo->realParameter[89] /* Radiator.sumCon.k[3] PARAM */) * (data->localData[0]->realVars[44] /* Radiator.preCon[3].Q_flow variable */) + (data->simulationInfo->realParameter[90] /* Radiator.sumCon.k[4] PARAM */) * (data->localData[0]->realVars[45] /* Radiator.preCon[4].Q_flow variable */) + (data->simulationInfo->realParameter[91] /* Radiator.sumCon.k[5] PARAM */) * (data->localData[0]->realVars[46] /* Radiator.preCon[5].Q_flow variable */);
  TRACE_POP
}
/*
equation index: 425
type: SIMPLE_ASSIGN
$cse7 = Buildings.Utilities.Math.Functions.regNonZeroPower(Radiator.dTRad[1], Radiator.n - 1.0, 0.05)
*/
void Radiator_eqFunction_425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,425};
  data->localData[0]->realVars[23] /* $cse7 variable */ = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */, data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0, 0.05);
  TRACE_POP
}
/*
equation index: 426
type: SIMPLE_ASSIGN
Radiator.preRad[1].Q_flow = Radiator.fraRad * Radiator.UAEle * Radiator.dTRad[1] * $cse7
*/
void Radiator_eqFunction_426(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,426};
  data->localData[0]->realVars[47] /* Radiator.preRad[1].Q_flow variable */ = (((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */)) * (data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */)) * (data->localData[0]->realVars[23] /* $cse7 variable */);
  TRACE_POP
}
/*
equation index: 427
type: SIMPLE_ASSIGN
Q_rad = Radiator.sumRad.k[1] * Radiator.preRad[1].Q_flow + Radiator.sumRad.k[2] * Radiator.preRad[2].Q_flow + Radiator.sumRad.k[3] * Radiator.preRad[3].Q_flow + Radiator.sumRad.k[4] * Radiator.preRad[4].Q_flow + Radiator.sumRad.k[5] * Radiator.preRad[5].Q_flow
*/
void Radiator_eqFunction_427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,427};
  data->localData[0]->realVars[28] /* Q_rad variable */ = (data->simulationInfo->realParameter[92] /* Radiator.sumRad.k[1] PARAM */) * (data->localData[0]->realVars[47] /* Radiator.preRad[1].Q_flow variable */) + (data->simulationInfo->realParameter[93] /* Radiator.sumRad.k[2] PARAM */) * (data->localData[0]->realVars[48] /* Radiator.preRad[2].Q_flow variable */) + (data->simulationInfo->realParameter[94] /* Radiator.sumRad.k[3] PARAM */) * (data->localData[0]->realVars[49] /* Radiator.preRad[3].Q_flow variable */) + (data->simulationInfo->realParameter[95] /* Radiator.sumRad.k[4] PARAM */) * (data->localData[0]->realVars[50] /* Radiator.preRad[4].Q_flow variable */) + (data->simulationInfo->realParameter[96] /* Radiator.sumRad.k[5] PARAM */) * (data->localData[0]->realVars[51] /* Radiator.preRad[5].Q_flow variable */);
  TRACE_POP
}
/*
equation index: 428
type: SIMPLE_ASSIGN
Power = Q_con + Q_rad
*/
void Radiator_eqFunction_428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,428};
  data->localData[0]->realVars[26] /* Power variable */ = data->localData[0]->realVars[27] /* Q_con variable */ + data->localData[0]->realVars[28] /* Q_rad variable */;
  TRACE_POP
}
/*
equation index: 429
type: SIMPLE_ASSIGN
$DER.Energy = Power
*/
void Radiator_eqFunction_429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,429};
  data->localData[0]->realVars[6] /* der(Energy) STATE_DER */ = data->localData[0]->realVars[26] /* Power variable */;
  TRACE_POP
}
/*
equation index: 430
type: SIMPLE_ASSIGN
T_z_source.port.Q_flow = Q_rad + Q_con
*/
void Radiator_eqFunction_430(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,430};
  data->localData[0]->realVars[139] /* T_z_source.port.Q_flow variable */ = data->localData[0]->realVars[28] /* Q_rad variable */ + data->localData[0]->realVars[27] /* Q_con variable */;
  TRACE_POP
}
/*
equation index: 431
type: SIMPLE_ASSIGN
Radiator.Q_flow = Q_con + Q_rad - Power - T_z_source.port.Q_flow
*/
void Radiator_eqFunction_431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,431};
  data->localData[0]->realVars[29] /* Radiator.Q_flow variable */ = data->localData[0]->realVars[27] /* Q_con variable */ + data->localData[0]->realVars[28] /* Q_rad variable */ - data->localData[0]->realVars[26] /* Power variable */ - data->localData[0]->realVars[139] /* T_z_source.port.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 432
type: SIMPLE_ASSIGN
Radiator.vol[1].heatPort.Q_flow = Radiator.preRad[1].Q_flow + Radiator.preCon[1].Q_flow
*/
void Radiator_eqFunction_432(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,432};
  data->localData[0]->realVars[125] /* Radiator.vol[1].heatPort.Q_flow variable */ = data->localData[0]->realVars[47] /* Radiator.preRad[1].Q_flow variable */ + data->localData[0]->realVars[42] /* Radiator.preCon[1].Q_flow variable */;
  TRACE_POP
}
/*
equation index: 433
type: SIMPLE_ASSIGN
$DER.Radiator.vol[1].dynBal.U = Radiator.vol[1].dynBal.Hb_flow + Radiator.vol[1].heatPort.Q_flow
*/
void Radiator_eqFunction_433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,433};
  data->localData[0]->realVars[7] /* der(Radiator.vol[1].dynBal.U) STATE_DER */ = data->localData[0]->realVars[60] /* Radiator.vol[1].dynBal.Hb_flow variable */ + data->localData[0]->realVars[125] /* Radiator.vol[1].heatPort.Q_flow variable */;
  TRACE_POP
}
/*
equation index: 434
type: SIMPLE_ASSIGN
Radiator.dTCon[1] = Radiator.dTRad[1]
*/
void Radiator_eqFunction_434(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,434};
  data->localData[0]->realVars[30] /* Radiator.dTCon[1] variable */ = data->localData[0]->realVars[35] /* Radiator.dTRad[1] variable */;
  TRACE_POP
}
/*
equation index: 449
type: ALGORITHM

  assert(noEvent(flow_sink.p >= 0.0), "Pressure (= " + String(flow_sink.p, 6, 0, true) + " Pa) of medium \"Buildings.Media.Water\" is negative
(Temperature = " + String(Radiator.vol[5].dynBal.medium.T, 6, 0, true) + " K)");
*/
void Radiator_eqFunction_449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,449};
  modelica_boolean tmp6;
  static const MMC_DEFSTRINGLIT(tmp7,12,"Pressure (= ");
  modelica_string tmp8;
  static const MMC_DEFSTRINGLIT(tmp9,66," Pa) of medium \"Buildings.Media.Water\" is negative\n(Temperature = ");
  modelica_string tmp10;
  static const MMC_DEFSTRINGLIT(tmp11,3," K)");
  static int tmp12 = 0;
  modelica_metatype tmpMeta[4] __attribute__((unused)) = {0};
  {
    tmp6 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    if(!tmp6)
    {
      tmp8 = modelica_real_to_modelica_string(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp7),tmp8);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp9));
      tmp10 = modelica_real_to_modelica_string(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp10);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp11));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",68,5,68,152,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(flow_sink.p >= 0.0)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 448
type: ALGORITHM

  assert(noEvent(Radiator.vol[5].dynBal.medium.T <= 403.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[5].dynBal.medium.T, 6, 0, true) + " K exceeded its maximum allowed value of " + String(130.0, 6, 0, true) + " degC (" + String(403.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_448(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,448};
  modelica_boolean tmp13;
  static const MMC_DEFSTRINGLIT(tmp14,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp15;
  static const MMC_DEFSTRINGLIT(tmp16,41," K exceeded its maximum allowed value of ");
  modelica_string tmp17;
  static const MMC_DEFSTRINGLIT(tmp18,7," degC (");
  modelica_string tmp19;
  static const MMC_DEFSTRINGLIT(tmp20,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp21 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp13 = LessEq(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */,403.15);
    if(!tmp13)
    {
      tmp15 = modelica_real_to_modelica_string(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp14),tmp15);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp16));
      tmp17 = modelica_real_to_modelica_string(130.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp17);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp18));
      tmp19 = modelica_real_to_modelica_string(403.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp19);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp20));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",65,5,66,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[5].dynBal.medium.T <= 403.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 447
type: ALGORITHM

  assert(noEvent(Radiator.vol[5].dynBal.medium.T >= 272.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[5].dynBal.medium.T, 6, 0, true) + " K exceeded its minimum allowed value of " + String(-1.0, 6, 0, true) + " degC (" + String(272.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,447};
  modelica_boolean tmp22;
  static const MMC_DEFSTRINGLIT(tmp23,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp24;
  static const MMC_DEFSTRINGLIT(tmp25,41," K exceeded its minimum allowed value of ");
  modelica_string tmp26;
  static const MMC_DEFSTRINGLIT(tmp27,7," degC (");
  modelica_string tmp28;
  static const MMC_DEFSTRINGLIT(tmp29,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp30 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp22 = GreaterEq(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */,272.15);
    if(!tmp22)
    {
      tmp24 = modelica_real_to_modelica_string(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp23),tmp24);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp25));
      tmp26 = modelica_real_to_modelica_string(-1.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp26);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp27));
      tmp28 = modelica_real_to_modelica_string(272.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp28);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp29));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",62,5,63,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[5].dynBal.medium.T >= 272.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 446
type: ALGORITHM

  assert(noEvent(flow_sink.p >= 0.0), "Pressure (= " + String(flow_sink.p, 6, 0, true) + " Pa) of medium \"Buildings.Media.Water\" is negative
(Temperature = " + String(Radiator.vol[4].dynBal.medium.T, 6, 0, true) + " K)");
*/
void Radiator_eqFunction_446(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,446};
  modelica_boolean tmp31;
  static const MMC_DEFSTRINGLIT(tmp32,12,"Pressure (= ");
  modelica_string tmp33;
  static const MMC_DEFSTRINGLIT(tmp34,66," Pa) of medium \"Buildings.Media.Water\" is negative\n(Temperature = ");
  modelica_string tmp35;
  static const MMC_DEFSTRINGLIT(tmp36,3," K)");
  static int tmp37 = 0;
  modelica_metatype tmpMeta[4] __attribute__((unused)) = {0};
  {
    tmp31 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    if(!tmp31)
    {
      tmp33 = modelica_real_to_modelica_string(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp32),tmp33);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp34));
      tmp35 = modelica_real_to_modelica_string(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp35);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp36));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",68,5,68,152,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(flow_sink.p >= 0.0)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 445
type: ALGORITHM

  assert(noEvent(Radiator.vol[4].dynBal.medium.T <= 403.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[4].dynBal.medium.T, 6, 0, true) + " K exceeded its maximum allowed value of " + String(130.0, 6, 0, true) + " degC (" + String(403.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,445};
  modelica_boolean tmp38;
  static const MMC_DEFSTRINGLIT(tmp39,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp40;
  static const MMC_DEFSTRINGLIT(tmp41,41," K exceeded its maximum allowed value of ");
  modelica_string tmp42;
  static const MMC_DEFSTRINGLIT(tmp43,7," degC (");
  modelica_string tmp44;
  static const MMC_DEFSTRINGLIT(tmp45,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp46 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp38 = LessEq(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */,403.15);
    if(!tmp38)
    {
      tmp40 = modelica_real_to_modelica_string(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp39),tmp40);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp41));
      tmp42 = modelica_real_to_modelica_string(130.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp42);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp43));
      tmp44 = modelica_real_to_modelica_string(403.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp44);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp45));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",65,5,66,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[4].dynBal.medium.T <= 403.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 444
type: ALGORITHM

  assert(noEvent(Radiator.vol[4].dynBal.medium.T >= 272.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[4].dynBal.medium.T, 6, 0, true) + " K exceeded its minimum allowed value of " + String(-1.0, 6, 0, true) + " degC (" + String(272.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,444};
  modelica_boolean tmp47;
  static const MMC_DEFSTRINGLIT(tmp48,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp49;
  static const MMC_DEFSTRINGLIT(tmp50,41," K exceeded its minimum allowed value of ");
  modelica_string tmp51;
  static const MMC_DEFSTRINGLIT(tmp52,7," degC (");
  modelica_string tmp53;
  static const MMC_DEFSTRINGLIT(tmp54,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp55 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp47 = GreaterEq(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */,272.15);
    if(!tmp47)
    {
      tmp49 = modelica_real_to_modelica_string(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp48),tmp49);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp50));
      tmp51 = modelica_real_to_modelica_string(-1.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp51);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp52));
      tmp53 = modelica_real_to_modelica_string(272.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp53);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp54));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",62,5,63,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[4].dynBal.medium.T >= 272.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 443
type: ALGORITHM

  assert(noEvent(flow_sink.p >= 0.0), "Pressure (= " + String(flow_sink.p, 6, 0, true) + " Pa) of medium \"Buildings.Media.Water\" is negative
(Temperature = " + String(Radiator.vol[3].dynBal.medium.T, 6, 0, true) + " K)");
*/
void Radiator_eqFunction_443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,443};
  modelica_boolean tmp56;
  static const MMC_DEFSTRINGLIT(tmp57,12,"Pressure (= ");
  modelica_string tmp58;
  static const MMC_DEFSTRINGLIT(tmp59,66," Pa) of medium \"Buildings.Media.Water\" is negative\n(Temperature = ");
  modelica_string tmp60;
  static const MMC_DEFSTRINGLIT(tmp61,3," K)");
  static int tmp62 = 0;
  modelica_metatype tmpMeta[4] __attribute__((unused)) = {0};
  {
    tmp56 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    if(!tmp56)
    {
      tmp58 = modelica_real_to_modelica_string(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp57),tmp58);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp59));
      tmp60 = modelica_real_to_modelica_string(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp60);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp61));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",68,5,68,152,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(flow_sink.p >= 0.0)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 442
type: ALGORITHM

  assert(noEvent(Radiator.vol[3].dynBal.medium.T <= 403.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[3].dynBal.medium.T, 6, 0, true) + " K exceeded its maximum allowed value of " + String(130.0, 6, 0, true) + " degC (" + String(403.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_442(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,442};
  modelica_boolean tmp63;
  static const MMC_DEFSTRINGLIT(tmp64,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp65;
  static const MMC_DEFSTRINGLIT(tmp66,41," K exceeded its maximum allowed value of ");
  modelica_string tmp67;
  static const MMC_DEFSTRINGLIT(tmp68,7," degC (");
  modelica_string tmp69;
  static const MMC_DEFSTRINGLIT(tmp70,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp71 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp63 = LessEq(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */,403.15);
    if(!tmp63)
    {
      tmp65 = modelica_real_to_modelica_string(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp64),tmp65);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp66));
      tmp67 = modelica_real_to_modelica_string(130.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp67);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp68));
      tmp69 = modelica_real_to_modelica_string(403.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp69);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp70));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",65,5,66,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[3].dynBal.medium.T <= 403.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 441
type: ALGORITHM

  assert(noEvent(Radiator.vol[3].dynBal.medium.T >= 272.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[3].dynBal.medium.T, 6, 0, true) + " K exceeded its minimum allowed value of " + String(-1.0, 6, 0, true) + " degC (" + String(272.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,441};
  modelica_boolean tmp72;
  static const MMC_DEFSTRINGLIT(tmp73,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp74;
  static const MMC_DEFSTRINGLIT(tmp75,41," K exceeded its minimum allowed value of ");
  modelica_string tmp76;
  static const MMC_DEFSTRINGLIT(tmp77,7," degC (");
  modelica_string tmp78;
  static const MMC_DEFSTRINGLIT(tmp79,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp80 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp72 = GreaterEq(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */,272.15);
    if(!tmp72)
    {
      tmp74 = modelica_real_to_modelica_string(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp73),tmp74);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp75));
      tmp76 = modelica_real_to_modelica_string(-1.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp76);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp77));
      tmp78 = modelica_real_to_modelica_string(272.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp78);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp79));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",62,5,63,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[3].dynBal.medium.T >= 272.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 440
type: ALGORITHM

  assert(noEvent(flow_sink.p >= 0.0), "Pressure (= " + String(flow_sink.p, 6, 0, true) + " Pa) of medium \"Buildings.Media.Water\" is negative
(Temperature = " + String(Radiator.vol[2].dynBal.medium.T, 6, 0, true) + " K)");
*/
void Radiator_eqFunction_440(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,440};
  modelica_boolean tmp81;
  static const MMC_DEFSTRINGLIT(tmp82,12,"Pressure (= ");
  modelica_string tmp83;
  static const MMC_DEFSTRINGLIT(tmp84,66," Pa) of medium \"Buildings.Media.Water\" is negative\n(Temperature = ");
  modelica_string tmp85;
  static const MMC_DEFSTRINGLIT(tmp86,3," K)");
  static int tmp87 = 0;
  modelica_metatype tmpMeta[4] __attribute__((unused)) = {0};
  {
    tmp81 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    if(!tmp81)
    {
      tmp83 = modelica_real_to_modelica_string(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp82),tmp83);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp84));
      tmp85 = modelica_real_to_modelica_string(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp85);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp86));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",68,5,68,152,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(flow_sink.p >= 0.0)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 439
type: ALGORITHM

  assert(noEvent(Radiator.vol[2].dynBal.medium.T <= 403.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[2].dynBal.medium.T, 6, 0, true) + " K exceeded its maximum allowed value of " + String(130.0, 6, 0, true) + " degC (" + String(403.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,439};
  modelica_boolean tmp88;
  static const MMC_DEFSTRINGLIT(tmp89,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp90;
  static const MMC_DEFSTRINGLIT(tmp91,41," K exceeded its maximum allowed value of ");
  modelica_string tmp92;
  static const MMC_DEFSTRINGLIT(tmp93,7," degC (");
  modelica_string tmp94;
  static const MMC_DEFSTRINGLIT(tmp95,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp96 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp88 = LessEq(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */,403.15);
    if(!tmp88)
    {
      tmp90 = modelica_real_to_modelica_string(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp89),tmp90);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp91));
      tmp92 = modelica_real_to_modelica_string(130.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp92);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp93));
      tmp94 = modelica_real_to_modelica_string(403.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp94);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp95));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",65,5,66,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[2].dynBal.medium.T <= 403.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 438
type: ALGORITHM

  assert(noEvent(Radiator.vol[2].dynBal.medium.T >= 272.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[2].dynBal.medium.T, 6, 0, true) + " K exceeded its minimum allowed value of " + String(-1.0, 6, 0, true) + " degC (" + String(272.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_438(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,438};
  modelica_boolean tmp97;
  static const MMC_DEFSTRINGLIT(tmp98,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp99;
  static const MMC_DEFSTRINGLIT(tmp100,41," K exceeded its minimum allowed value of ");
  modelica_string tmp101;
  static const MMC_DEFSTRINGLIT(tmp102,7," degC (");
  modelica_string tmp103;
  static const MMC_DEFSTRINGLIT(tmp104,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp105 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp97 = GreaterEq(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */,272.15);
    if(!tmp97)
    {
      tmp99 = modelica_real_to_modelica_string(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp98),tmp99);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp100));
      tmp101 = modelica_real_to_modelica_string(-1.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp101);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp102));
      tmp103 = modelica_real_to_modelica_string(272.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp103);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp104));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",62,5,63,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[2].dynBal.medium.T >= 272.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 437
type: ALGORITHM

  assert(noEvent(flow_sink.p >= 0.0), "Pressure (= " + String(flow_sink.p, 6, 0, true) + " Pa) of medium \"Buildings.Media.Water\" is negative
(Temperature = " + String(Radiator.vol[1].dynBal.medium.T, 6, 0, true) + " K)");
*/
void Radiator_eqFunction_437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,437};
  modelica_boolean tmp106;
  static const MMC_DEFSTRINGLIT(tmp107,12,"Pressure (= ");
  modelica_string tmp108;
  static const MMC_DEFSTRINGLIT(tmp109,66," Pa) of medium \"Buildings.Media.Water\" is negative\n(Temperature = ");
  modelica_string tmp110;
  static const MMC_DEFSTRINGLIT(tmp111,3," K)");
  static int tmp112 = 0;
  modelica_metatype tmpMeta[4] __attribute__((unused)) = {0};
  {
    tmp106 = GreaterEq(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */,0.0);
    if(!tmp106)
    {
      tmp108 = modelica_real_to_modelica_string(data->simulationInfo->realParameter[262] /* flow_sink.p PARAM */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp107),tmp108);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp109));
      tmp110 = modelica_real_to_modelica_string(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp110);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp111));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",68,5,68,152,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(flow_sink.p >= 0.0)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 436
type: ALGORITHM

  assert(noEvent(Radiator.vol[1].dynBal.medium.T <= 403.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[1].dynBal.medium.T, 6, 0, true) + " K exceeded its maximum allowed value of " + String(130.0, 6, 0, true) + " degC (" + String(403.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,436};
  modelica_boolean tmp113;
  static const MMC_DEFSTRINGLIT(tmp114,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp115;
  static const MMC_DEFSTRINGLIT(tmp116,41," K exceeded its maximum allowed value of ");
  modelica_string tmp117;
  static const MMC_DEFSTRINGLIT(tmp118,7," degC (");
  modelica_string tmp119;
  static const MMC_DEFSTRINGLIT(tmp120,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp121 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp113 = LessEq(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */,403.15);
    if(!tmp113)
    {
      tmp115 = modelica_real_to_modelica_string(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp114),tmp115);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp116));
      tmp117 = modelica_real_to_modelica_string(130.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp117);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp118));
      tmp119 = modelica_real_to_modelica_string(403.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp119);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp120));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",65,5,66,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[1].dynBal.medium.T <= 403.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}
/*
equation index: 435
type: ALGORITHM

  assert(noEvent(Radiator.vol[1].dynBal.medium.T >= 272.15), "In Radiator.Radiator.vol.dynBal.medium: Temperature T = " + String(Radiator.vol[1].dynBal.medium.T, 6, 0, true) + " K exceeded its minimum allowed value of " + String(-1.0, 6, 0, true) + " degC (" + String(272.15, 6, 0, true) + " Kelvin) as required from medium model \"Buildings.Media.Water\".");
*/
void Radiator_eqFunction_435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,435};
  modelica_boolean tmp122;
  static const MMC_DEFSTRINGLIT(tmp123,56,"In Radiator.Radiator.vol.dynBal.medium: Temperature T = ");
  modelica_string tmp124;
  static const MMC_DEFSTRINGLIT(tmp125,41," K exceeded its minimum allowed value of ");
  modelica_string tmp126;
  static const MMC_DEFSTRINGLIT(tmp127,7," degC (");
  modelica_string tmp128;
  static const MMC_DEFSTRINGLIT(tmp129,63," Kelvin) as required from medium model \"Buildings.Media.Water\".");
  static int tmp130 = 0;
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
  {
    tmp122 = GreaterEq(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */,272.15);
    if(!tmp122)
    {
      tmp124 = modelica_real_to_modelica_string(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp123),tmp124);
      tmpMeta[1] = stringAppend(tmpMeta[0],MMC_REFSTRINGLIT(tmp125));
      tmp126 = modelica_real_to_modelica_string(-1.0, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[2] = stringAppend(tmpMeta[1],tmp126);
      tmpMeta[3] = stringAppend(tmpMeta[2],MMC_REFSTRINGLIT(tmp127));
      tmp128 = modelica_real_to_modelica_string(272.15, ((modelica_integer) 6), ((modelica_integer) 0), 1);
      tmpMeta[4] = stringAppend(tmpMeta[3],tmp128);
      tmpMeta[5] = stringAppend(tmpMeta[4],MMC_REFSTRINGLIT(tmp129));
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",62,5,63,122,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nnoEvent(Radiator.vol[1].dynBal.medium.T >= 272.15)", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_withEquationIndexes(threadData, info, equationIndexes, MMC_STRINGDATA(tmpMeta[5]));
      }
    }
  }
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionDAE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  int equationIndexes[1] = {0};
  modelica_metatype tmpMeta[6] __attribute__((unused)) = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  Radiator_functionLocalKnownVars(data, threadData);
  Radiator_eqFunction_321(data, threadData);

  Radiator_eqFunction_322(data, threadData);

  Radiator_eqFunction_323(data, threadData);

  Radiator_eqFunction_324(data, threadData);

  Radiator_eqFunction_325(data, threadData);

  Radiator_eqFunction_326(data, threadData);

  Radiator_eqFunction_327(data, threadData);

  Radiator_eqFunction_328(data, threadData);

  Radiator_eqFunction_335(data, threadData);

  Radiator_eqFunction_336(data, threadData);

  Radiator_eqFunction_341(data, threadData);

  Radiator_eqFunction_342(data, threadData);

  Radiator_eqFunction_343(data, threadData);

  Radiator_eqFunction_344(data, threadData);

  Radiator_eqFunction_345(data, threadData);

  Radiator_eqFunction_350(data, threadData);

  Radiator_eqFunction_351(data, threadData);

  Radiator_eqFunction_352(data, threadData);

  Radiator_eqFunction_353(data, threadData);

  Radiator_eqFunction_354(data, threadData);

  Radiator_eqFunction_355(data, threadData);

  Radiator_eqFunction_360(data, threadData);

  Radiator_eqFunction_361(data, threadData);

  Radiator_eqFunction_362(data, threadData);

  Radiator_eqFunction_363(data, threadData);

  Radiator_eqFunction_364(data, threadData);

  Radiator_eqFunction_365(data, threadData);

  Radiator_eqFunction_370(data, threadData);

  Radiator_eqFunction_371(data, threadData);

  Radiator_eqFunction_372(data, threadData);

  Radiator_eqFunction_373(data, threadData);

  Radiator_eqFunction_374(data, threadData);

  Radiator_eqFunction_375(data, threadData);

  Radiator_eqFunction_376(data, threadData);

  Radiator_eqFunction_377(data, threadData);

  Radiator_eqFunction_378(data, threadData);

  Radiator_eqFunction_379(data, threadData);

  Radiator_eqFunction_380(data, threadData);

  Radiator_eqFunction_381(data, threadData);

  Radiator_eqFunction_382(data, threadData);

  Radiator_eqFunction_383(data, threadData);

  Radiator_eqFunction_384(data, threadData);

  Radiator_eqFunction_385(data, threadData);

  Radiator_eqFunction_386(data, threadData);

  Radiator_eqFunction_387(data, threadData);

  Radiator_eqFunction_388(data, threadData);

  Radiator_eqFunction_389(data, threadData);

  Radiator_eqFunction_390(data, threadData);

  Radiator_eqFunction_391(data, threadData);

  Radiator_eqFunction_392(data, threadData);

  Radiator_eqFunction_393(data, threadData);

  Radiator_eqFunction_394(data, threadData);

  Radiator_eqFunction_395(data, threadData);

  Radiator_eqFunction_396(data, threadData);

  Radiator_eqFunction_397(data, threadData);

  Radiator_eqFunction_398(data, threadData);

  Radiator_eqFunction_399(data, threadData);

  Radiator_eqFunction_400(data, threadData);

  Radiator_eqFunction_401(data, threadData);

  Radiator_eqFunction_402(data, threadData);

  Radiator_eqFunction_403(data, threadData);

  Radiator_eqFunction_404(data, threadData);

  Radiator_eqFunction_405(data, threadData);

  Radiator_eqFunction_406(data, threadData);

  Radiator_eqFunction_407(data, threadData);

  Radiator_eqFunction_408(data, threadData);

  Radiator_eqFunction_409(data, threadData);

  Radiator_eqFunction_410(data, threadData);

  Radiator_eqFunction_411(data, threadData);

  Radiator_eqFunction_412(data, threadData);

  Radiator_eqFunction_413(data, threadData);

  Radiator_eqFunction_414(data, threadData);

  Radiator_eqFunction_415(data, threadData);

  Radiator_eqFunction_416(data, threadData);

  Radiator_eqFunction_417(data, threadData);

  Radiator_eqFunction_418(data, threadData);

  Radiator_eqFunction_419(data, threadData);

  Radiator_eqFunction_420(data, threadData);

  Radiator_eqFunction_421(data, threadData);

  Radiator_eqFunction_422(data, threadData);

  Radiator_eqFunction_423(data, threadData);

  Radiator_eqFunction_424(data, threadData);

  Radiator_eqFunction_425(data, threadData);

  Radiator_eqFunction_426(data, threadData);

  Radiator_eqFunction_427(data, threadData);

  Radiator_eqFunction_428(data, threadData);

  Radiator_eqFunction_429(data, threadData);

  Radiator_eqFunction_430(data, threadData);

  Radiator_eqFunction_431(data, threadData);

  Radiator_eqFunction_432(data, threadData);

  Radiator_eqFunction_433(data, threadData);

  Radiator_eqFunction_434(data, threadData);

  Radiator_eqFunction_449(data, threadData);

  Radiator_eqFunction_448(data, threadData);

  Radiator_eqFunction_447(data, threadData);

  Radiator_eqFunction_446(data, threadData);

  Radiator_eqFunction_445(data, threadData);

  Radiator_eqFunction_444(data, threadData);

  Radiator_eqFunction_443(data, threadData);

  Radiator_eqFunction_442(data, threadData);

  Radiator_eqFunction_441(data, threadData);

  Radiator_eqFunction_440(data, threadData);

  Radiator_eqFunction_439(data, threadData);

  Radiator_eqFunction_438(data, threadData);

  Radiator_eqFunction_437(data, threadData);

  Radiator_eqFunction_436(data, threadData);

  Radiator_eqFunction_435(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  TRACE_POP
  return 0;
}


int Radiator_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/* forwarded equations */
extern void Radiator_eqFunction_335(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_336(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_341(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_342(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_343(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_344(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_350(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_351(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_352(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_353(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_354(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_360(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_361(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_362(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_363(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_364(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_370(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_372(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_374(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_375(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_376(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_377(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_378(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_379(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_380(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_382(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_383(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_384(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_385(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_386(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_388(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_389(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_390(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_391(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_392(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_393(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_394(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_395(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_397(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_398(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_399(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_400(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_401(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_402(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_403(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_405(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_406(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_407(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_408(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_409(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_410(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_411(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_413(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_414(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_415(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_416(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_417(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_418(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_419(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_421(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_422(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_423(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_424(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_425(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_426(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_427(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_428(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_429(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_432(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_433(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  Radiator_eqFunction_335(data, threadData);
  threadData->lastEquationSolved = 335;
  Radiator_eqFunction_336(data, threadData);
  threadData->lastEquationSolved = 336;
  Radiator_eqFunction_341(data, threadData);
  threadData->lastEquationSolved = 341;
  Radiator_eqFunction_342(data, threadData);
  threadData->lastEquationSolved = 342;
  Radiator_eqFunction_343(data, threadData);
  threadData->lastEquationSolved = 343;
  Radiator_eqFunction_344(data, threadData);
  threadData->lastEquationSolved = 344;
  Radiator_eqFunction_350(data, threadData);
  threadData->lastEquationSolved = 350;
  Radiator_eqFunction_351(data, threadData);
  threadData->lastEquationSolved = 351;
  Radiator_eqFunction_352(data, threadData);
  threadData->lastEquationSolved = 352;
  Radiator_eqFunction_353(data, threadData);
  threadData->lastEquationSolved = 353;
  Radiator_eqFunction_354(data, threadData);
  threadData->lastEquationSolved = 354;
  Radiator_eqFunction_360(data, threadData);
  threadData->lastEquationSolved = 360;
  Radiator_eqFunction_361(data, threadData);
  threadData->lastEquationSolved = 361;
  Radiator_eqFunction_362(data, threadData);
  threadData->lastEquationSolved = 362;
  Radiator_eqFunction_363(data, threadData);
  threadData->lastEquationSolved = 363;
  Radiator_eqFunction_364(data, threadData);
  threadData->lastEquationSolved = 364;
  Radiator_eqFunction_370(data, threadData);
  threadData->lastEquationSolved = 370;
  Radiator_eqFunction_372(data, threadData);
  threadData->lastEquationSolved = 372;
  Radiator_eqFunction_374(data, threadData);
  threadData->lastEquationSolved = 374;
  Radiator_eqFunction_375(data, threadData);
  threadData->lastEquationSolved = 375;
  Radiator_eqFunction_376(data, threadData);
  threadData->lastEquationSolved = 376;
  Radiator_eqFunction_377(data, threadData);
  threadData->lastEquationSolved = 377;
  Radiator_eqFunction_378(data, threadData);
  threadData->lastEquationSolved = 378;
  Radiator_eqFunction_379(data, threadData);
  threadData->lastEquationSolved = 379;
  Radiator_eqFunction_380(data, threadData);
  threadData->lastEquationSolved = 380;
  Radiator_eqFunction_382(data, threadData);
  threadData->lastEquationSolved = 382;
  Radiator_eqFunction_383(data, threadData);
  threadData->lastEquationSolved = 383;
  Radiator_eqFunction_384(data, threadData);
  threadData->lastEquationSolved = 384;
  Radiator_eqFunction_385(data, threadData);
  threadData->lastEquationSolved = 385;
  Radiator_eqFunction_386(data, threadData);
  threadData->lastEquationSolved = 386;
  Radiator_eqFunction_388(data, threadData);
  threadData->lastEquationSolved = 388;
  Radiator_eqFunction_389(data, threadData);
  threadData->lastEquationSolved = 389;
  Radiator_eqFunction_390(data, threadData);
  threadData->lastEquationSolved = 390;
  Radiator_eqFunction_391(data, threadData);
  threadData->lastEquationSolved = 391;
  Radiator_eqFunction_392(data, threadData);
  threadData->lastEquationSolved = 392;
  Radiator_eqFunction_393(data, threadData);
  threadData->lastEquationSolved = 393;
  Radiator_eqFunction_394(data, threadData);
  threadData->lastEquationSolved = 394;
  Radiator_eqFunction_395(data, threadData);
  threadData->lastEquationSolved = 395;
  Radiator_eqFunction_397(data, threadData);
  threadData->lastEquationSolved = 397;
  Radiator_eqFunction_398(data, threadData);
  threadData->lastEquationSolved = 398;
  Radiator_eqFunction_399(data, threadData);
  threadData->lastEquationSolved = 399;
  Radiator_eqFunction_400(data, threadData);
  threadData->lastEquationSolved = 400;
  Radiator_eqFunction_401(data, threadData);
  threadData->lastEquationSolved = 401;
  Radiator_eqFunction_402(data, threadData);
  threadData->lastEquationSolved = 402;
  Radiator_eqFunction_403(data, threadData);
  threadData->lastEquationSolved = 403;
  Radiator_eqFunction_405(data, threadData);
  threadData->lastEquationSolved = 405;
  Radiator_eqFunction_406(data, threadData);
  threadData->lastEquationSolved = 406;
  Radiator_eqFunction_407(data, threadData);
  threadData->lastEquationSolved = 407;
  Radiator_eqFunction_408(data, threadData);
  threadData->lastEquationSolved = 408;
  Radiator_eqFunction_409(data, threadData);
  threadData->lastEquationSolved = 409;
  Radiator_eqFunction_410(data, threadData);
  threadData->lastEquationSolved = 410;
  Radiator_eqFunction_411(data, threadData);
  threadData->lastEquationSolved = 411;
  Radiator_eqFunction_413(data, threadData);
  threadData->lastEquationSolved = 413;
  Radiator_eqFunction_414(data, threadData);
  threadData->lastEquationSolved = 414;
  Radiator_eqFunction_415(data, threadData);
  threadData->lastEquationSolved = 415;
  Radiator_eqFunction_416(data, threadData);
  threadData->lastEquationSolved = 416;
  Radiator_eqFunction_417(data, threadData);
  threadData->lastEquationSolved = 417;
  Radiator_eqFunction_418(data, threadData);
  threadData->lastEquationSolved = 418;
  Radiator_eqFunction_419(data, threadData);
  threadData->lastEquationSolved = 419;
  Radiator_eqFunction_421(data, threadData);
  threadData->lastEquationSolved = 421;
  Radiator_eqFunction_422(data, threadData);
  threadData->lastEquationSolved = 422;
  Radiator_eqFunction_423(data, threadData);
  threadData->lastEquationSolved = 423;
  Radiator_eqFunction_424(data, threadData);
  threadData->lastEquationSolved = 424;
  Radiator_eqFunction_425(data, threadData);
  threadData->lastEquationSolved = 425;
  Radiator_eqFunction_426(data, threadData);
  threadData->lastEquationSolved = 426;
  Radiator_eqFunction_427(data, threadData);
  threadData->lastEquationSolved = 427;
  Radiator_eqFunction_428(data, threadData);
  threadData->lastEquationSolved = 428;
  Radiator_eqFunction_429(data, threadData);
  threadData->lastEquationSolved = 429;
  Radiator_eqFunction_432(data, threadData);
  threadData->lastEquationSolved = 432;
  Radiator_eqFunction_433(data, threadData);
  threadData->lastEquationSolved = 433;
}

int Radiator_functionODE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  Radiator_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  TRACE_POP
  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char**argv, DATA *data, threadData_t *threadData);

#include "Radiator_12jac.h"
#include "Radiator_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks Radiator_callback = {
   (int (*)(DATA *, threadData_t *, void *)) Radiator_performSimulation,    /* performSimulation */
   (int (*)(DATA *, threadData_t *, void *)) Radiator_performQSSSimulation,    /* performQSSSimulation */
   Radiator_updateContinuousSystem,    /* updateContinuousSystem */
   Radiator_callExternalObjectDestructors,    /* callExternalObjectDestructors */
   Radiator_initialNonLinearSystem,    /* initialNonLinearSystem */
   Radiator_initialLinearSystem,    /* initialLinearSystem */
   NULL,    /* initialMixedSystem */
   #if !defined(OMC_NO_STATESELECTION)
   Radiator_initializeStateSets,
   #else
   NULL,
   #endif    /* initializeStateSets */
   Radiator_initializeDAEmodeData,
   Radiator_functionODE,
   Radiator_functionAlgebraics,
   Radiator_functionDAE,
   Radiator_functionLocalKnownVars,
   Radiator_input_function,
   Radiator_input_function_init,
   Radiator_input_function_updateStartValues,
   Radiator_data_function,
   Radiator_output_function,
   Radiator_setc_function,
   Radiator_function_storeDelayed,
   Radiator_function_storeSpatialDistribution,
   Radiator_function_initSpatialDistribution,
   Radiator_updateBoundVariableAttributes,
   Radiator_functionInitialEquations,
   1, /* useHomotopy - 0: local homotopy (equidistant lambda), 1: global homotopy (equidistant lambda), 2: new global homotopy approach (adaptive lambda), 3: new local homotopy approach (adaptive lambda)*/
   Radiator_functionInitialEquations_lambda0,
   Radiator_functionRemovedInitialEquations,
   Radiator_updateBoundParameters,
   Radiator_checkForAsserts,
   Radiator_function_ZeroCrossingsEquations,
   Radiator_function_ZeroCrossings,
   Radiator_function_updateRelations,
   Radiator_zeroCrossingDescription,
   Radiator_relationDescription,
   Radiator_function_initSample,
   Radiator_INDEX_JAC_A,
   Radiator_INDEX_JAC_B,
   Radiator_INDEX_JAC_C,
   Radiator_INDEX_JAC_D,
   Radiator_INDEX_JAC_F,
   Radiator_initialAnalyticJacobianA,
   Radiator_initialAnalyticJacobianB,
   Radiator_initialAnalyticJacobianC,
   Radiator_initialAnalyticJacobianD,
   Radiator_initialAnalyticJacobianF,
   Radiator_functionJacA_column,
   Radiator_functionJacB_column,
   Radiator_functionJacC_column,
   Radiator_functionJacD_column,
   Radiator_functionJacF_column,
   Radiator_linear_model_frame,
   Radiator_linear_model_datarecovery_frame,
   Radiator_mayer,
   Radiator_lagrange,
   Radiator_pickUpBoundsForInputsInOptimization,
   Radiator_setInputData,
   Radiator_getTimeGrid,
   Radiator_symbolicInlineSystem,
   Radiator_function_initSynchronous,
   Radiator_function_updateSynchronous,
   Radiator_function_equationsSynchronous,
   Radiator_inputNames,
   Radiator_dataReconciliationInputNames,
   NULL,
   NULL,
   NULL,
   -1

};

#define _OMC_LIT_RESOURCE_0_name_data "Buildings"
#define _OMC_LIT_RESOURCE_0_dir_data "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,9,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,116,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Complex"
#define _OMC_LIT_RESOURCE_1_dir_data "C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,7,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,55,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "Modelica"
#define _OMC_LIT_RESOURCE_2_dir_data "C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,8,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,70,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/ModelicaServices 4.0.0"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,16,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,78,_OMC_LIT_RESOURCE_3_dir_data);

#define _OMC_LIT_RESOURCE_4_name_data "Radiator"
#define _OMC_LIT_RESOURCE_4_dir_data "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_4_name,8,_OMC_LIT_RESOURCE_4_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_4_dir,101,_OMC_LIT_RESOURCE_4_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,10,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_4_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_4_dir)}};
void Radiator_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &Radiator_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "Radiator";
  data->modelData->modelFilePrefix = "Radiator";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels";
  data->modelData->modelGUID = "{6c5c0562-fb73-4eb7-b8a9-4ed4c0f705ad}";
  #if defined(OPENMODELICA_XML_FROM_FILE_AT_RUNTIME)
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  #else
  #if defined(_MSC_VER) /* handle joke compilers */
  {
  /* for MSVC we encode a string like char x[] = {'a', 'b', 'c', '\0'} */
  /* because the string constant limit is 65535 bytes */
  static const char contents_init[] =
    #include "Radiator_init.c"
    ;
  static const char contents_info[] =
    #include "Radiator_info.c"
    ;
    data->modelData->initXMLData = contents_init;
    data->modelData->modelDataXml.infoXMLData = contents_info;
  }
  #else /* handle real compilers */
  data->modelData->initXMLData =
  #include "Radiator_init.c"
    ;
  data->modelData->modelDataXml.infoXMLData =
  #include "Radiator_info.c"
    ;
  #endif /* defined(_MSC_VER) */
  #endif /* defined(OPENMODELICA_XML_FROM_FILE_AT_RUNTIME) */
  data->modelData->runTestsuite = 0;
  
  data->modelData->nStates = 6;
  data->modelData->nVariablesReal = 152;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesInteger = 0;
  data->modelData->nVariablesBoolean = 0;
  data->modelData->nVariablesString = 0;
  data->modelData->nParametersReal = 271;
  data->modelData->nParametersInteger = 61;
  data->modelData->nParametersBoolean = 82;
  data->modelData->nParametersString = 0;
  data->modelData->nInputVars = 3;
  data->modelData->nOutputVars = 5;
  
  data->modelData->nAliasReal = 205;
  data->modelData->nAliasInteger = 0;
  data->modelData->nAliasBoolean = 0;
  data->modelData->nAliasString = 0;
  
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  
  data->modelData->modelDataXml.fileName = "Radiator_info.json";
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 21;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 1137;
  data->modelData->nMixedSystems = 0;
  data->modelData->nLinearSystems = 5;
  data->modelData->nNonLinearSystems = 2;
  data->modelData->nStateSets = 0;
  data->modelData->nJacobians = 10;
  data->modelData->nOptimizeConstraints = 0;
  data->modelData->nOptimizeFinalConstraints = 0;
  
  data->modelData->nDelayExpressions = 0;
  
  data->modelData->nClocks = 0;
  data->modelData->nSubClocks = 0;
  
  data->modelData->nSpatialDistributions = 0;
  
  data->modelData->nSensitivityVars = 0;
  data->modelData->nSensitivityParamVars = 0;
  data->modelData->nSetcVars = 0;
  data->modelData->ndataReconVars = 0;
  data->modelData->linearizationDumpLanguage =
  OMC_LINEARIZE_DUMP_LANGUAGE_MODELICA;
}

static int rml_execution_failed()
{
  fflush(NULL);
  fprintf(stderr, "Execution failed!\n");
  fflush(NULL);
  return 1;
}

#if defined(threadData)
#undef threadData
#endif
/* call the simulation runtime main from our main! */
int main(int argc, char**argv)
{
  int res;
  DATA data;
  MODEL_DATA modelData;
  SIMULATION_INFO simInfo;
  data.modelData = &modelData;
  data.simulationInfo = &simInfo;
  measure_time_flag = 0;
  compiledInDAEMode = 0;
  compiledWithSymSolver = 0;
  MMC_INIT(0);
  omc_alloc_interface.init();
  {
    MMC_TRY_TOP()
  
    MMC_TRY_STACK()
  
    Radiator_setupDataStruc(&data, threadData);
    res = _main_SimulationRuntime(argc, argv, &data, threadData);
    
    MMC_ELSE()
    rml_execution_failed();
    fprintf(stderr, "Stack overflow detected and was not caught.\nSend us a bug report at https://trac.openmodelica.org/OpenModelica/newticket\n    Include the following trace:\n");
    printStacktraceMessages();
    fflush(NULL);
    return 1;
    MMC_CATCH_STACK()
    
    MMC_CATCH_TOP(return rml_execution_failed());
  }

  fflush(NULL);
  EXIT(res);
  return res;
}

#ifdef __cplusplus
}
#endif


