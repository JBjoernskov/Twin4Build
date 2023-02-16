/* Algebraic */
#include "Radiator_model.h"

#ifdef __cplusplus
extern "C" {
#endif


/* forwarded equations */
extern void Radiator_eqFunction_321(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_322(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_323(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_324(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_325(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_326(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_327(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_328(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_345(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_355(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_365(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_371(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_373(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_381(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_387(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_396(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_404(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_412(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_420(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_430(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_431(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_434(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_435(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_436(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_437(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_438(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_439(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_440(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_441(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_442(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_443(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_444(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_445(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_446(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_447(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_448(DATA* data, threadData_t *threadData);
extern void Radiator_eqFunction_449(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  Radiator_eqFunction_321(data, threadData);
  threadData->lastEquationSolved = 321;
  Radiator_eqFunction_322(data, threadData);
  threadData->lastEquationSolved = 322;
  Radiator_eqFunction_323(data, threadData);
  threadData->lastEquationSolved = 323;
  Radiator_eqFunction_324(data, threadData);
  threadData->lastEquationSolved = 324;
  Radiator_eqFunction_325(data, threadData);
  threadData->lastEquationSolved = 325;
  Radiator_eqFunction_326(data, threadData);
  threadData->lastEquationSolved = 326;
  Radiator_eqFunction_327(data, threadData);
  threadData->lastEquationSolved = 327;
  Radiator_eqFunction_328(data, threadData);
  threadData->lastEquationSolved = 328;
  Radiator_eqFunction_345(data, threadData);
  threadData->lastEquationSolved = 345;
  Radiator_eqFunction_355(data, threadData);
  threadData->lastEquationSolved = 355;
  Radiator_eqFunction_365(data, threadData);
  threadData->lastEquationSolved = 365;
  Radiator_eqFunction_371(data, threadData);
  threadData->lastEquationSolved = 371;
  Radiator_eqFunction_373(data, threadData);
  threadData->lastEquationSolved = 373;
  Radiator_eqFunction_381(data, threadData);
  threadData->lastEquationSolved = 381;
  Radiator_eqFunction_387(data, threadData);
  threadData->lastEquationSolved = 387;
  Radiator_eqFunction_396(data, threadData);
  threadData->lastEquationSolved = 396;
  Radiator_eqFunction_404(data, threadData);
  threadData->lastEquationSolved = 404;
  Radiator_eqFunction_412(data, threadData);
  threadData->lastEquationSolved = 412;
  Radiator_eqFunction_420(data, threadData);
  threadData->lastEquationSolved = 420;
  Radiator_eqFunction_430(data, threadData);
  threadData->lastEquationSolved = 430;
  Radiator_eqFunction_431(data, threadData);
  threadData->lastEquationSolved = 431;
  Radiator_eqFunction_434(data, threadData);
  threadData->lastEquationSolved = 434;
  Radiator_eqFunction_435(data, threadData);
  threadData->lastEquationSolved = 435;
  Radiator_eqFunction_436(data, threadData);
  threadData->lastEquationSolved = 436;
  Radiator_eqFunction_437(data, threadData);
  threadData->lastEquationSolved = 437;
  Radiator_eqFunction_438(data, threadData);
  threadData->lastEquationSolved = 438;
  Radiator_eqFunction_439(data, threadData);
  threadData->lastEquationSolved = 439;
  Radiator_eqFunction_440(data, threadData);
  threadData->lastEquationSolved = 440;
  Radiator_eqFunction_441(data, threadData);
  threadData->lastEquationSolved = 441;
  Radiator_eqFunction_442(data, threadData);
  threadData->lastEquationSolved = 442;
  Radiator_eqFunction_443(data, threadData);
  threadData->lastEquationSolved = 443;
  Radiator_eqFunction_444(data, threadData);
  threadData->lastEquationSolved = 444;
  Radiator_eqFunction_445(data, threadData);
  threadData->lastEquationSolved = 445;
  Radiator_eqFunction_446(data, threadData);
  threadData->lastEquationSolved = 446;
  Radiator_eqFunction_447(data, threadData);
  threadData->lastEquationSolved = 447;
  Radiator_eqFunction_448(data, threadData);
  threadData->lastEquationSolved = 448;
  Radiator_eqFunction_449(data, threadData);
  threadData->lastEquationSolved = 449;
}
/* for continuous time variables */
int Radiator_functionAlgebraics(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  functionAlg_system0(data, threadData);

  Radiator_function_savePreSynchronous(data, threadData);
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ALGEBRAICS);
#endif

  TRACE_POP
  return 0;
}

#ifdef __cplusplus
}
#endif
