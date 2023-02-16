/* Jacobians 10 */
#include "Radiator_model.h"
#include "Radiator_12jac.h"
/* constant equations */
/* dynamic equations */

/*
equation index: 332
type: SIMPLE_ASSIGN
Radiator.port_a.h_outflow.$pDERLSJac2.dummyVarLSJac2 = 4184.0 * Radiator.vol[1].dynBal.medium.T.SeedLSJac2
*/
void Radiator_eqFunction_332(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,332};
  jacobian->tmpVars[0] /* Radiator.port_a.h_outflow.$pDERLSJac2.dummyVarLSJac2 JACOBIAN_DIFF_VAR */ = (4184.0) * (jacobian->seedVars[0] /* Radiator.vol[1].dynBal.medium.T.SeedLSJac2 SEED_VAR */);
  TRACE_POP
}

/*
equation index: 333
type: SIMPLE_ASSIGN
Radiator.vol.1.dynBal.medium.T_degC.$pDERLSJac2.dummyVarLSJac2 = (-Radiator.vol[1].dynBal.m) * Radiator.port_a.h_outflow.$pDERLSJac2.dummyVarLSJac2 / Radiator.vol[1].dynBal.CSen
*/
void Radiator_eqFunction_333(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,333};
  jacobian->tmpVars[1] /* Radiator.vol.1.dynBal.medium.T_degC.$pDERLSJac2.dummyVarLSJac2 JACOBIAN_DIFF_VAR */ = DIVISION(((-data->localData[0]->realVars[65] /* Radiator.vol[1].dynBal.m DUMMY_STATE */)) * (jacobian->tmpVars[0] /* Radiator.port_a.h_outflow.$pDERLSJac2.dummyVarLSJac2 JACOBIAN_DIFF_VAR */),data->simulationInfo->realParameter[112] /* Radiator.vol[1].dynBal.CSen PARAM */,"Radiator.vol[1].dynBal.CSen");
  TRACE_POP
}

/*
equation index: 334
type: SIMPLE_ASSIGN
$res_LSJac2_1.$pDERLSJac2.dummyVarLSJac2 = Radiator.vol[1].dynBal.medium.T.SeedLSJac2 - Radiator.vol.1.dynBal.medium.T_degC.$pDERLSJac2.dummyVarLSJac2
*/
void Radiator_eqFunction_334(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,334};
  jacobian->resultVars[0] /* $res_LSJac2_1.$pDERLSJac2.dummyVarLSJac2 JACOBIAN_VAR */ = jacobian->seedVars[0] /* Radiator.vol[1].dynBal.medium.T.SeedLSJac2 SEED_VAR */ - jacobian->tmpVars[1] /* Radiator.vol.1.dynBal.medium.T_degC.$pDERLSJac2.dummyVarLSJac2 JACOBIAN_DIFF_VAR */;
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionJacLSJac2_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac2;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacLSJac2_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac2;
  Radiator_eqFunction_332(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_333(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_334(data, threadData, jacobian, parentJacobian);
  TRACE_POP
  return 0;
}
/* constant equations */
/* dynamic equations */

/*
equation index: 339
type: SIMPLE_ASSIGN
Radiator.vol.2.ports.2.h_outflow.$pDERLSJac3.dummyVarLSJac3 = (-Radiator.vol[2].dynBal.CSen) * Radiator.vol[2].dynBal.medium.T_degC.SeedLSJac3 / Radiator.vol[2].dynBal.m
*/
void Radiator_eqFunction_339(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,339};
  jacobian->tmpVars[0] /* Radiator.vol.2.ports.2.h_outflow.$pDERLSJac3.dummyVarLSJac3 JACOBIAN_DIFF_VAR */ = DIVISION(((-data->simulationInfo->realParameter[113] /* Radiator.vol[2].dynBal.CSen PARAM */)) * (jacobian->seedVars[0] /* Radiator.vol[2].dynBal.medium.T_degC.SeedLSJac3 SEED_VAR */),data->localData[0]->realVars[66] /* Radiator.vol[2].dynBal.m DUMMY_STATE */,"Radiator.vol[2].dynBal.m");
  TRACE_POP
}

/*
equation index: 340
type: SIMPLE_ASSIGN
$res_LSJac3_1.$pDERLSJac3.dummyVarLSJac3 = 4184.0 * Radiator.vol[2].dynBal.medium.T_degC.SeedLSJac3 - Radiator.vol.2.ports.2.h_outflow.$pDERLSJac3.dummyVarLSJac3
*/
void Radiator_eqFunction_340(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,340};
  jacobian->resultVars[0] /* $res_LSJac3_1.$pDERLSJac3.dummyVarLSJac3 JACOBIAN_VAR */ = (4184.0) * (jacobian->seedVars[0] /* Radiator.vol[2].dynBal.medium.T_degC.SeedLSJac3 SEED_VAR */) - jacobian->tmpVars[0] /* Radiator.vol.2.ports.2.h_outflow.$pDERLSJac3.dummyVarLSJac3 JACOBIAN_DIFF_VAR */;
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionJacLSJac3_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac3;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacLSJac3_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac3;
  Radiator_eqFunction_339(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_340(data, threadData, jacobian, parentJacobian);
  TRACE_POP
  return 0;
}
/* constant equations */
/* dynamic equations */

/*
equation index: 348
type: SIMPLE_ASSIGN
Radiator.vol.3.ports.2.h_outflow.$pDERLSJac4.dummyVarLSJac4 = (-Radiator.vol[3].dynBal.CSen) * Radiator.vol[3].dynBal.medium.T_degC.SeedLSJac4 / Radiator.vol[3].dynBal.m
*/
void Radiator_eqFunction_348(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,348};
  jacobian->tmpVars[0] /* Radiator.vol.3.ports.2.h_outflow.$pDERLSJac4.dummyVarLSJac4 JACOBIAN_DIFF_VAR */ = DIVISION(((-data->simulationInfo->realParameter[114] /* Radiator.vol[3].dynBal.CSen PARAM */)) * (jacobian->seedVars[0] /* Radiator.vol[3].dynBal.medium.T_degC.SeedLSJac4 SEED_VAR */),data->localData[0]->realVars[67] /* Radiator.vol[3].dynBal.m DUMMY_STATE */,"Radiator.vol[3].dynBal.m");
  TRACE_POP
}

/*
equation index: 349
type: SIMPLE_ASSIGN
$res_LSJac4_1.$pDERLSJac4.dummyVarLSJac4 = 4184.0 * Radiator.vol[3].dynBal.medium.T_degC.SeedLSJac4 - Radiator.vol.3.ports.2.h_outflow.$pDERLSJac4.dummyVarLSJac4
*/
void Radiator_eqFunction_349(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,349};
  jacobian->resultVars[0] /* $res_LSJac4_1.$pDERLSJac4.dummyVarLSJac4 JACOBIAN_VAR */ = (4184.0) * (jacobian->seedVars[0] /* Radiator.vol[3].dynBal.medium.T_degC.SeedLSJac4 SEED_VAR */) - jacobian->tmpVars[0] /* Radiator.vol.3.ports.2.h_outflow.$pDERLSJac4.dummyVarLSJac4 JACOBIAN_DIFF_VAR */;
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionJacLSJac4_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac4;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacLSJac4_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac4;
  Radiator_eqFunction_348(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_349(data, threadData, jacobian, parentJacobian);
  TRACE_POP
  return 0;
}
/* constant equations */
/* dynamic equations */

/*
equation index: 358
type: SIMPLE_ASSIGN
Radiator.vol.4.ports.2.h_outflow.$pDERLSJac5.dummyVarLSJac5 = (-Radiator.vol[4].dynBal.CSen) * Radiator.vol[4].dynBal.medium.T_degC.SeedLSJac5 / Radiator.vol[4].dynBal.m
*/
void Radiator_eqFunction_358(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,358};
  jacobian->tmpVars[0] /* Radiator.vol.4.ports.2.h_outflow.$pDERLSJac5.dummyVarLSJac5 JACOBIAN_DIFF_VAR */ = DIVISION(((-data->simulationInfo->realParameter[115] /* Radiator.vol[4].dynBal.CSen PARAM */)) * (jacobian->seedVars[0] /* Radiator.vol[4].dynBal.medium.T_degC.SeedLSJac5 SEED_VAR */),data->localData[0]->realVars[68] /* Radiator.vol[4].dynBal.m DUMMY_STATE */,"Radiator.vol[4].dynBal.m");
  TRACE_POP
}

/*
equation index: 359
type: SIMPLE_ASSIGN
$res_LSJac5_1.$pDERLSJac5.dummyVarLSJac5 = 4184.0 * Radiator.vol[4].dynBal.medium.T_degC.SeedLSJac5 - Radiator.vol.4.ports.2.h_outflow.$pDERLSJac5.dummyVarLSJac5
*/
void Radiator_eqFunction_359(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,359};
  jacobian->resultVars[0] /* $res_LSJac5_1.$pDERLSJac5.dummyVarLSJac5 JACOBIAN_VAR */ = (4184.0) * (jacobian->seedVars[0] /* Radiator.vol[4].dynBal.medium.T_degC.SeedLSJac5 SEED_VAR */) - jacobian->tmpVars[0] /* Radiator.vol.4.ports.2.h_outflow.$pDERLSJac5.dummyVarLSJac5 JACOBIAN_DIFF_VAR */;
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionJacLSJac5_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac5;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacLSJac5_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac5;
  Radiator_eqFunction_358(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_359(data, threadData, jacobian, parentJacobian);
  TRACE_POP
  return 0;
}
/* constant equations */
/* dynamic equations */

/*
equation index: 368
type: SIMPLE_ASSIGN
Radiator.vol.5.dynBal.medium.T_degC.$pDERLSJac6.dummyVarLSJac6 = (-Radiator.vol[5].dynBal.m) * Radiator.vol[5].ports[2].h_outflow.SeedLSJac6 / Radiator.vol[5].dynBal.CSen
*/
void Radiator_eqFunction_368(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,368};
  jacobian->tmpVars[0] /* Radiator.vol.5.dynBal.medium.T_degC.$pDERLSJac6.dummyVarLSJac6 JACOBIAN_DIFF_VAR */ = DIVISION(((-data->localData[0]->realVars[69] /* Radiator.vol[5].dynBal.m DUMMY_STATE */)) * (jacobian->seedVars[0] /* Radiator.vol[5].ports[2].h_outflow.SeedLSJac6 SEED_VAR */),data->simulationInfo->realParameter[116] /* Radiator.vol[5].dynBal.CSen PARAM */,"Radiator.vol[5].dynBal.CSen");
  TRACE_POP
}

/*
equation index: 369
type: SIMPLE_ASSIGN
$res_LSJac6_1.$pDERLSJac6.dummyVarLSJac6 = 4184.0 * Radiator.vol.5.dynBal.medium.T_degC.$pDERLSJac6.dummyVarLSJac6 - Radiator.vol[5].ports[2].h_outflow.SeedLSJac6
*/
void Radiator_eqFunction_369(DATA *data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  const int clockIndex = 0;
  const int equationIndexes[2] = {1,369};
  jacobian->resultVars[0] /* $res_LSJac6_1.$pDERLSJac6.dummyVarLSJac6 JACOBIAN_VAR */ = (4184.0) * (jacobian->tmpVars[0] /* Radiator.vol.5.dynBal.medium.T_degC.$pDERLSJac6.dummyVarLSJac6 JACOBIAN_DIFF_VAR */) - jacobian->seedVars[0] /* Radiator.vol[5].ports[2].h_outflow.SeedLSJac6 SEED_VAR */;
  TRACE_POP
}

OMC_DISABLE_OPT
int Radiator_functionJacLSJac6_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac6;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacLSJac6_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_LSJac6;
  Radiator_eqFunction_368(data, threadData, jacobian, parentJacobian);
  Radiator_eqFunction_369(data, threadData, jacobian, parentJacobian);
  TRACE_POP
  return 0;
}
int Radiator_functionJacF_column(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 0;
}
int Radiator_functionJacD_column(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 0;
}
int Radiator_functionJacC_column(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 0;
}
int Radiator_functionJacB_column(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 0;
}
/* constant equations */
/* dynamic equations */

OMC_DISABLE_OPT
int Radiator_functionJacA_constantEqns(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_A;
  
  
  TRACE_POP
  return 0;
}

int Radiator_functionJacA_column(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian, ANALYTIC_JACOBIAN *parentJacobian)
{
  TRACE_PUSH

  DATA* data = ((DATA*)inData);
  int index = Radiator_INDEX_JAC_A;
  TRACE_POP
  return 0;
}

OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianLSJac2(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+1] = {0,1};
  const int rowIndex[1] = {0};
  int i = 0;
  
  jacobian->sizeCols = 1;
  jacobian->sizeRows = 1;
  jacobian->sizeTmpVars = 3;
  jacobian->seedVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(3,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((1+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 1;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 1;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (1+1)*sizeof(unsigned int));
  
  for(i=2;i<1+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 1*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[0] = 1;
  TRACE_POP
  return 0;
}
OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianLSJac3(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+1] = {0,1};
  const int rowIndex[1] = {0};
  int i = 0;
  
  jacobian->sizeCols = 1;
  jacobian->sizeRows = 1;
  jacobian->sizeTmpVars = 2;
  jacobian->seedVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(2,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((1+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 1;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 1;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (1+1)*sizeof(unsigned int));
  
  for(i=2;i<1+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 1*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[0] = 1;
  TRACE_POP
  return 0;
}
OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianLSJac4(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+1] = {0,1};
  const int rowIndex[1] = {0};
  int i = 0;
  
  jacobian->sizeCols = 1;
  jacobian->sizeRows = 1;
  jacobian->sizeTmpVars = 2;
  jacobian->seedVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(2,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((1+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 1;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 1;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (1+1)*sizeof(unsigned int));
  
  for(i=2;i<1+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 1*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[0] = 1;
  TRACE_POP
  return 0;
}
OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianLSJac5(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+1] = {0,1};
  const int rowIndex[1] = {0};
  int i = 0;
  
  jacobian->sizeCols = 1;
  jacobian->sizeRows = 1;
  jacobian->sizeTmpVars = 2;
  jacobian->seedVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(2,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((1+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 1;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 1;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (1+1)*sizeof(unsigned int));
  
  for(i=2;i<1+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 1*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[0] = 1;
  TRACE_POP
  return 0;
}
OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianLSJac6(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+1] = {0,1};
  const int rowIndex[1] = {0};
  int i = 0;
  
  jacobian->sizeCols = 1;
  jacobian->sizeRows = 1;
  jacobian->sizeTmpVars = 2;
  jacobian->seedVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(1,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(2,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((1+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 1;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(1*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 1;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (1+1)*sizeof(unsigned int));
  
  for(i=2;i<1+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 1*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[0] = 1;
  TRACE_POP
  return 0;
}
int Radiator_initialAnalyticJacobianF(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 1;
}
int Radiator_initialAnalyticJacobianD(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 1;
}
int Radiator_initialAnalyticJacobianC(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 1;
}
int Radiator_initialAnalyticJacobianB(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  TRACE_POP
  return 1;
}
OMC_DISABLE_OPT
int Radiator_initialAnalyticJacobianA(void* inData, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian)
{
  TRACE_PUSH
  DATA* data = ((DATA*)inData);
  const int colPtrIndex[1+6] = {0,0,3,4,4,4,3};
  const int rowIndex[18] = {0,1,2,0,1,2,3,0,2,3,4,0,3,4,5,0,4,5};
  int i = 0;
  
  jacobian->sizeCols = 6;
  jacobian->sizeRows = 6;
  jacobian->sizeTmpVars = 0;
  jacobian->seedVars = (modelica_real*) calloc(6,sizeof(modelica_real));
  jacobian->resultVars = (modelica_real*) calloc(6,sizeof(modelica_real));
  jacobian->tmpVars = (modelica_real*) calloc(0,sizeof(modelica_real));
  jacobian->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  jacobian->sparsePattern->leadindex = (unsigned int*) malloc((6+1)*sizeof(unsigned int));
  jacobian->sparsePattern->index = (unsigned int*) malloc(18*sizeof(unsigned int));
  jacobian->sparsePattern->numberOfNoneZeros = 18;
  jacobian->sparsePattern->colorCols = (unsigned int*) malloc(6*sizeof(unsigned int));
  jacobian->sparsePattern->maxColors = 5;
  jacobian->constantEqns = NULL;
  
  /* write lead index of compressed sparse column */
  memcpy(jacobian->sparsePattern->leadindex, colPtrIndex, (6+1)*sizeof(unsigned int));
  
  for(i=2;i<6+1;++i)
    jacobian->sparsePattern->leadindex[i] += jacobian->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(jacobian->sparsePattern->index, rowIndex, 18*sizeof(unsigned int));
  
  /* write color array */
  jacobian->sparsePattern->colorCols[5] = 1;
  jacobian->sparsePattern->colorCols[4] = 2;
  jacobian->sparsePattern->colorCols[3] = 3;
  jacobian->sparsePattern->colorCols[2] = 4;
  jacobian->sparsePattern->colorCols[1] = 5;
  jacobian->sparsePattern->colorCols[0] = 5;
  TRACE_POP
  return 0;
}


