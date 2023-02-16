/* Linear Systems */
#include "Radiator_model.h"
#include "Radiator_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

/* linear systems */

/*
equation index: 366
type: SIMPLE_ASSIGN
Radiator.vol[5].dynBal.medium.T_degC = (Radiator.vol[5].dynBal.U - Radiator.vol[5].dynBal.m * Radiator.vol[5].ports[2].h_outflow) / Radiator.vol[5].dynBal.CSen
*/
void Radiator_eqFunction_366(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,366};
  data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */ = DIVISION_SIM(data->localData[0]->realVars[5] /* Radiator.vol[5].dynBal.U STATE(1) */ - ((data->localData[0]->realVars[69] /* Radiator.vol[5].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */)),data->simulationInfo->realParameter[116] /* Radiator.vol[5].dynBal.CSen PARAM */,"Radiator.vol[5].dynBal.CSen",equationIndexes);
  TRACE_POP
}

void residualFunc370(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,370};
  ANALYTIC_JACOBIAN* jacobian = NULL;
  data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */ = xloc[0];
  /* local constraints */
  Radiator_eqFunction_366(data, threadData);
  res[0] = (4184.0) * (data->localData[0]->realVars[99] /* Radiator.vol[5].dynBal.medium.T_degC variable */) - data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */;
  TRACE_POP
}
OMC_DISABLE_OPT
void initializeStaticLSData370(void *inData, threadData_t *threadData, void *systemData)
{
  DATA* data = (DATA*) inData;
  LINEAR_SYSTEM_DATA* linearSystemData = (LINEAR_SYSTEM_DATA*) systemData;
  int i=0;
  /* static ls data for Radiator.vol[5].ports[2].h_outflow */
  linearSystemData->nominal[i] = data->modelData->realVarsData[133].attribute /* Radiator.vol[5].ports[2].h_outflow */.nominal;
  linearSystemData->min[i]     = data->modelData->realVarsData[133].attribute /* Radiator.vol[5].ports[2].h_outflow */.min;
  linearSystemData->max[i++]   = data->modelData->realVarsData[133].attribute /* Radiator.vol[5].ports[2].h_outflow */.max;
}


/*
equation index: 356
type: SIMPLE_ASSIGN
Radiator.vol[4].ports[2].h_outflow = (Radiator.vol[4].dynBal.U - Radiator.vol[4].dynBal.CSen * Radiator.vol[4].dynBal.medium.T_degC) / Radiator.vol[4].dynBal.m
*/
void Radiator_eqFunction_356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,356};
  data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */ = DIVISION_SIM(data->localData[0]->realVars[4] /* Radiator.vol[4].dynBal.U STATE(1) */ - ((data->simulationInfo->realParameter[115] /* Radiator.vol[4].dynBal.CSen PARAM */) * (data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */)),data->localData[0]->realVars[68] /* Radiator.vol[4].dynBal.m DUMMY_STATE */,"Radiator.vol[4].dynBal.m",equationIndexes);
  TRACE_POP
}

void residualFunc360(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,360};
  ANALYTIC_JACOBIAN* jacobian = NULL;
  data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */ = xloc[0];
  /* local constraints */
  Radiator_eqFunction_356(data, threadData);
  res[0] = (4184.0) * (data->localData[0]->realVars[98] /* Radiator.vol[4].dynBal.medium.T_degC variable */) - data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */;
  TRACE_POP
}
OMC_DISABLE_OPT
void initializeStaticLSData360(void *inData, threadData_t *threadData, void *systemData)
{
  DATA* data = (DATA*) inData;
  LINEAR_SYSTEM_DATA* linearSystemData = (LINEAR_SYSTEM_DATA*) systemData;
  int i=0;
  /* static ls data for Radiator.vol[4].dynBal.medium.T_degC */
  linearSystemData->nominal[i] = data->modelData->realVarsData[98].attribute /* Radiator.vol[4].dynBal.medium.T_degC */.nominal;
  linearSystemData->min[i]     = data->modelData->realVarsData[98].attribute /* Radiator.vol[4].dynBal.medium.T_degC */.min;
  linearSystemData->max[i++]   = data->modelData->realVarsData[98].attribute /* Radiator.vol[4].dynBal.medium.T_degC */.max;
}


/*
equation index: 346
type: SIMPLE_ASSIGN
Radiator.vol[3].ports[2].h_outflow = (Radiator.vol[3].dynBal.U - Radiator.vol[3].dynBal.CSen * Radiator.vol[3].dynBal.medium.T_degC) / Radiator.vol[3].dynBal.m
*/
void Radiator_eqFunction_346(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,346};
  data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */ = DIVISION_SIM(data->localData[0]->realVars[3] /* Radiator.vol[3].dynBal.U STATE(1) */ - ((data->simulationInfo->realParameter[114] /* Radiator.vol[3].dynBal.CSen PARAM */) * (data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */)),data->localData[0]->realVars[67] /* Radiator.vol[3].dynBal.m DUMMY_STATE */,"Radiator.vol[3].dynBal.m",equationIndexes);
  TRACE_POP
}

void residualFunc350(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,350};
  ANALYTIC_JACOBIAN* jacobian = NULL;
  data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */ = xloc[0];
  /* local constraints */
  Radiator_eqFunction_346(data, threadData);
  res[0] = (4184.0) * (data->localData[0]->realVars[97] /* Radiator.vol[3].dynBal.medium.T_degC variable */) - data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */;
  TRACE_POP
}
OMC_DISABLE_OPT
void initializeStaticLSData350(void *inData, threadData_t *threadData, void *systemData)
{
  DATA* data = (DATA*) inData;
  LINEAR_SYSTEM_DATA* linearSystemData = (LINEAR_SYSTEM_DATA*) systemData;
  int i=0;
  /* static ls data for Radiator.vol[3].dynBal.medium.T_degC */
  linearSystemData->nominal[i] = data->modelData->realVarsData[97].attribute /* Radiator.vol[3].dynBal.medium.T_degC */.nominal;
  linearSystemData->min[i]     = data->modelData->realVarsData[97].attribute /* Radiator.vol[3].dynBal.medium.T_degC */.min;
  linearSystemData->max[i++]   = data->modelData->realVarsData[97].attribute /* Radiator.vol[3].dynBal.medium.T_degC */.max;
}


/*
equation index: 337
type: SIMPLE_ASSIGN
Radiator.vol[2].ports[2].h_outflow = (Radiator.vol[2].dynBal.U - Radiator.vol[2].dynBal.CSen * Radiator.vol[2].dynBal.medium.T_degC) / Radiator.vol[2].dynBal.m
*/
void Radiator_eqFunction_337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,337};
  data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */ = DIVISION_SIM(data->localData[0]->realVars[2] /* Radiator.vol[2].dynBal.U STATE(1) */ - ((data->simulationInfo->realParameter[113] /* Radiator.vol[2].dynBal.CSen PARAM */) * (data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */)),data->localData[0]->realVars[66] /* Radiator.vol[2].dynBal.m DUMMY_STATE */,"Radiator.vol[2].dynBal.m",equationIndexes);
  TRACE_POP
}

void residualFunc341(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,341};
  ANALYTIC_JACOBIAN* jacobian = NULL;
  data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */ = xloc[0];
  /* local constraints */
  Radiator_eqFunction_337(data, threadData);
  res[0] = (4184.0) * (data->localData[0]->realVars[96] /* Radiator.vol[2].dynBal.medium.T_degC variable */) - data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */;
  TRACE_POP
}
OMC_DISABLE_OPT
void initializeStaticLSData341(void *inData, threadData_t *threadData, void *systemData)
{
  DATA* data = (DATA*) inData;
  LINEAR_SYSTEM_DATA* linearSystemData = (LINEAR_SYSTEM_DATA*) systemData;
  int i=0;
  /* static ls data for Radiator.vol[2].dynBal.medium.T_degC */
  linearSystemData->nominal[i] = data->modelData->realVarsData[96].attribute /* Radiator.vol[2].dynBal.medium.T_degC */.nominal;
  linearSystemData->min[i]     = data->modelData->realVarsData[96].attribute /* Radiator.vol[2].dynBal.medium.T_degC */.min;
  linearSystemData->max[i++]   = data->modelData->realVarsData[96].attribute /* Radiator.vol[2].dynBal.medium.T_degC */.max;
}


/*
equation index: 329
type: SIMPLE_ASSIGN
Radiator.port_a.h_outflow = -1142859.6 - (-4184.0) * Radiator.vol[1].dynBal.medium.T
*/
void Radiator_eqFunction_329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,329};
  data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */ = -1142859.6 - ((-4184.0) * (data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */));
  TRACE_POP
}
/*
equation index: 330
type: SIMPLE_ASSIGN
Radiator.vol[1].dynBal.medium.T_degC = (Radiator.vol[1].dynBal.U - Radiator.vol[1].dynBal.m * Radiator.port_a.h_outflow) / Radiator.vol[1].dynBal.CSen
*/
void Radiator_eqFunction_330(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,330};
  data->localData[0]->realVars[95] /* Radiator.vol[1].dynBal.medium.T_degC variable */ = DIVISION_SIM(data->localData[0]->realVars[1] /* Radiator.vol[1].dynBal.U STATE(1) */ - ((data->localData[0]->realVars[65] /* Radiator.vol[1].dynBal.m DUMMY_STATE */) * (data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */)),data->simulationInfo->realParameter[112] /* Radiator.vol[1].dynBal.CSen PARAM */,"Radiator.vol[1].dynBal.CSen",equationIndexes);
  TRACE_POP
}

void residualFunc335(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,335};
  ANALYTIC_JACOBIAN* jacobian = NULL;
  data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */ = xloc[0];
  /* local constraints */
  Radiator_eqFunction_329(data, threadData);

  /* local constraints */
  Radiator_eqFunction_330(data, threadData);
  res[0] = -273.15 + data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */ - data->localData[0]->realVars[95] /* Radiator.vol[1].dynBal.medium.T_degC variable */;
  TRACE_POP
}
OMC_DISABLE_OPT
void initializeStaticLSData335(void *inData, threadData_t *threadData, void *systemData)
{
  DATA* data = (DATA*) inData;
  LINEAR_SYSTEM_DATA* linearSystemData = (LINEAR_SYSTEM_DATA*) systemData;
  int i=0;
  /* static ls data for Radiator.vol[1].dynBal.medium.T */
  linearSystemData->nominal[i] = data->modelData->realVarsData[90].attribute /* Radiator.vol[1].dynBal.medium.T */.nominal;
  linearSystemData->min[i]     = data->modelData->realVarsData[90].attribute /* Radiator.vol[1].dynBal.medium.T */.min;
  linearSystemData->max[i++]   = data->modelData->realVarsData[90].attribute /* Radiator.vol[1].dynBal.medium.T */.max;
}

/* Prototypes for the strict sets (Dynamic Tearing) */

/* Global constraints for the casual sets */
/* function initialize linear systems */
void Radiator_initialLinearSystem(int nLinearSystems, LINEAR_SYSTEM_DATA* linearSystemData)
{
  /* linear systems */
  assertStreamPrint(NULL, nLinearSystems > 4, "Internal Error: indexlinearSystem mismatch!");
  linearSystemData[4].equationIndex = 370;
  linearSystemData[4].size = 1;
  linearSystemData[4].nnz = 0;
  linearSystemData[4].method = 1;   /* Symbolic Jacobian available */
  linearSystemData[4].residualFunc = residualFunc370;
  linearSystemData[4].strictTearingFunctionCall = NULL;
  linearSystemData[4].analyticalJacobianColumn = Radiator_functionJacLSJac6_column;
  linearSystemData[4].initialAnalyticalJacobian = Radiator_initialAnalyticJacobianLSJac6;
  linearSystemData[4].jacobianIndex = 4 /*jacInx*/;
  linearSystemData[4].setA = NULL;  //setLinearMatrixA370;
  linearSystemData[4].setb = NULL;  //setLinearVectorb370;
  linearSystemData[4].initializeStaticLSData = initializeStaticLSData370;
  
  assertStreamPrint(NULL, nLinearSystems > 3, "Internal Error: indexlinearSystem mismatch!");
  linearSystemData[3].equationIndex = 360;
  linearSystemData[3].size = 1;
  linearSystemData[3].nnz = 0;
  linearSystemData[3].method = 1;   /* Symbolic Jacobian available */
  linearSystemData[3].residualFunc = residualFunc360;
  linearSystemData[3].strictTearingFunctionCall = NULL;
  linearSystemData[3].analyticalJacobianColumn = Radiator_functionJacLSJac5_column;
  linearSystemData[3].initialAnalyticalJacobian = Radiator_initialAnalyticJacobianLSJac5;
  linearSystemData[3].jacobianIndex = 3 /*jacInx*/;
  linearSystemData[3].setA = NULL;  //setLinearMatrixA360;
  linearSystemData[3].setb = NULL;  //setLinearVectorb360;
  linearSystemData[3].initializeStaticLSData = initializeStaticLSData360;
  
  assertStreamPrint(NULL, nLinearSystems > 2, "Internal Error: indexlinearSystem mismatch!");
  linearSystemData[2].equationIndex = 350;
  linearSystemData[2].size = 1;
  linearSystemData[2].nnz = 0;
  linearSystemData[2].method = 1;   /* Symbolic Jacobian available */
  linearSystemData[2].residualFunc = residualFunc350;
  linearSystemData[2].strictTearingFunctionCall = NULL;
  linearSystemData[2].analyticalJacobianColumn = Radiator_functionJacLSJac4_column;
  linearSystemData[2].initialAnalyticalJacobian = Radiator_initialAnalyticJacobianLSJac4;
  linearSystemData[2].jacobianIndex = 2 /*jacInx*/;
  linearSystemData[2].setA = NULL;  //setLinearMatrixA350;
  linearSystemData[2].setb = NULL;  //setLinearVectorb350;
  linearSystemData[2].initializeStaticLSData = initializeStaticLSData350;
  
  assertStreamPrint(NULL, nLinearSystems > 1, "Internal Error: indexlinearSystem mismatch!");
  linearSystemData[1].equationIndex = 341;
  linearSystemData[1].size = 1;
  linearSystemData[1].nnz = 0;
  linearSystemData[1].method = 1;   /* Symbolic Jacobian available */
  linearSystemData[1].residualFunc = residualFunc341;
  linearSystemData[1].strictTearingFunctionCall = NULL;
  linearSystemData[1].analyticalJacobianColumn = Radiator_functionJacLSJac3_column;
  linearSystemData[1].initialAnalyticalJacobian = Radiator_initialAnalyticJacobianLSJac3;
  linearSystemData[1].jacobianIndex = 1 /*jacInx*/;
  linearSystemData[1].setA = NULL;  //setLinearMatrixA341;
  linearSystemData[1].setb = NULL;  //setLinearVectorb341;
  linearSystemData[1].initializeStaticLSData = initializeStaticLSData341;
  
  assertStreamPrint(NULL, nLinearSystems > 0, "Internal Error: indexlinearSystem mismatch!");
  linearSystemData[0].equationIndex = 335;
  linearSystemData[0].size = 1;
  linearSystemData[0].nnz = 0;
  linearSystemData[0].method = 1;   /* Symbolic Jacobian available */
  linearSystemData[0].residualFunc = residualFunc335;
  linearSystemData[0].strictTearingFunctionCall = NULL;
  linearSystemData[0].analyticalJacobianColumn = Radiator_functionJacLSJac2_column;
  linearSystemData[0].initialAnalyticalJacobian = Radiator_initialAnalyticJacobianLSJac2;
  linearSystemData[0].jacobianIndex = 0 /*jacInx*/;
  linearSystemData[0].setA = NULL;  //setLinearMatrixA335;
  linearSystemData[0].setb = NULL;  //setLinearVectorb335;
  linearSystemData[0].initializeStaticLSData = initializeStaticLSData335;
}

#if defined(__cplusplus)
}
#endif

