#ifndef Radiator__H
#define Radiator__H
#include "meta/meta_modelica.h"
#include "util/modelica.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "simulation/simulation_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  modelica_real _p;
  modelica_real _T;
} Radiator_Radiator_Medium_ThermodynamicState;
extern struct record_description Radiator_Radiator_Medium_ThermodynamicState__desc;

void Radiator_Radiator_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_Radiator_Medium_ThermodynamicState_construct(td, ths ) Radiator_Radiator_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_Radiator_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_Radiator_Medium_ThermodynamicState_copy(src,dst) Radiator_Radiator_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_Radiator_Medium_ThermodynamicState omc_Radiator_Radiator_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_Radiator_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_Radiator_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_Radiator_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_Radiator_Medium_ThermodynamicState_array;
#define alloc_Radiator_Radiator_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_Radiator_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_Radiator_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_Radiator_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_Radiator_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_Medium_ThermodynamicState))
#define Radiator_Radiator_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_Radiator_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_Medium_ThermodynamicState))
#define Radiator_Radiator_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_Radiator_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_Radiator_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_Radiator_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_Radiator_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_Medium_ThermodynamicState), __VA_ARGS__)

typedef Radiator_Radiator_Medium_ThermodynamicState Radiator_Radiator_res_Medium_ThermodynamicState;
extern struct record_description Radiator_Radiator_res_Medium_ThermodynamicState__desc;

void Radiator_Radiator_res_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_Radiator_res_Medium_ThermodynamicState_construct(td, ths ) Radiator_Radiator_res_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_Radiator_res_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_Radiator_res_Medium_ThermodynamicState_copy(src,dst) Radiator_Radiator_res_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_Radiator_res_Medium_ThermodynamicState omc_Radiator_Radiator_res_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_Radiator_res_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_Radiator_res_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_Radiator_res_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_Radiator_res_Medium_ThermodynamicState_array;
#define alloc_Radiator_Radiator_res_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_Radiator_res_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_Radiator_res_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_Radiator_res_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_Radiator_res_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_res_Medium_ThermodynamicState))
#define Radiator_Radiator_res_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_Radiator_res_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_res_Medium_ThermodynamicState))
#define Radiator_Radiator_res_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_Radiator_res_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_Radiator_res_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_Radiator_res_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_Radiator_res_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_res_Medium_ThermodynamicState), __VA_ARGS__)

typedef Radiator_Radiator_Medium_ThermodynamicState Radiator_Radiator_vol_Medium_ThermodynamicState;
extern struct record_description Radiator_Radiator_vol_Medium_ThermodynamicState__desc;

void Radiator_Radiator_vol_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_Radiator_vol_Medium_ThermodynamicState_construct(td, ths ) Radiator_Radiator_vol_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_Radiator_vol_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_Radiator_vol_Medium_ThermodynamicState_copy(src,dst) Radiator_Radiator_vol_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_Radiator_vol_Medium_ThermodynamicState omc_Radiator_Radiator_vol_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_Radiator_vol_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_Radiator_vol_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_Radiator_vol_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_Radiator_vol_Medium_ThermodynamicState_array;
#define alloc_Radiator_Radiator_vol_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_Radiator_vol_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_Radiator_vol_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_Radiator_vol_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_Radiator_vol_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_Medium_ThermodynamicState))
#define Radiator_Radiator_vol_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_Radiator_vol_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_Medium_ThermodynamicState))
#define Radiator_Radiator_vol_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_Radiator_vol_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_Radiator_vol_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_Radiator_vol_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_Radiator_vol_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_Medium_ThermodynamicState), __VA_ARGS__)

typedef Radiator_Radiator_Medium_ThermodynamicState Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState;
extern struct record_description Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState__desc;

void Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_construct(td, ths ) Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy(src,dst) Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState omc_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_array;
#define alloc_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState))
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState))
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy_p, sizeof(Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState), __VA_ARGS__)

typedef Radiator_Radiator_Medium_ThermodynamicState Radiator_flow__sink_Medium_ThermodynamicState;
extern struct record_description Radiator_flow__sink_Medium_ThermodynamicState__desc;

void Radiator_flow__sink_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_flow__sink_Medium_ThermodynamicState_construct(td, ths ) Radiator_flow__sink_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_flow__sink_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_flow__sink_Medium_ThermodynamicState_copy(src,dst) Radiator_flow__sink_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_flow__sink_Medium_ThermodynamicState omc_Radiator_flow__sink_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_flow__sink_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_flow__sink_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_flow__sink_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_flow__sink_Medium_ThermodynamicState_array;
#define alloc_Radiator_flow__sink_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_flow__sink_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_flow__sink_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_flow__sink_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_flow__sink_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__sink_Medium_ThermodynamicState))
#define Radiator_flow__sink_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_flow__sink_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__sink_Medium_ThermodynamicState))
#define Radiator_flow__sink_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_flow__sink_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_flow__sink_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_flow__sink_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_flow__sink_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__sink_Medium_ThermodynamicState), __VA_ARGS__)

typedef Radiator_Radiator_Medium_ThermodynamicState Radiator_flow__source_Medium_ThermodynamicState;
extern struct record_description Radiator_flow__source_Medium_ThermodynamicState__desc;

void Radiator_flow__source_Medium_ThermodynamicState_construct_p(threadData_t *threadData, void* v_ths );
#define Radiator_flow__source_Medium_ThermodynamicState_construct(td, ths ) Radiator_flow__source_Medium_ThermodynamicState_construct_p(td, &ths )
void Radiator_flow__source_Medium_ThermodynamicState_copy_p(void* v_src, void* v_dst);
#define Radiator_flow__source_Medium_ThermodynamicState_copy(src,dst) Radiator_flow__source_Medium_ThermodynamicState_copy_p(&src, &dst)

// This function should eventualy replace the default 'modelica' record constructor funcition
// that omc used to generate, i.e., replace functionBodyRecordConstructor template.
// Radiator_flow__source_Medium_ThermodynamicState omc_Radiator_flow__source_Medium_ThermodynamicState(threadData_t *threadData , modelica_real in_p, modelica_real in_T);

// This function is not needed anymore. If you want to know how a record
// is 'assigned to' in simulation context see assignRhsExpToRecordCrefSimContext and
// splitRecordAssignmentToMemberAssignments (simCode). Basically the record is
// split up assignments generated for each memeber individualy.
// void Radiator_flow__source_Medium_ThermodynamicState_copy_to_vars_p(void* v_src , modelica_real* in_p, modelica_real* in_T);
// #define Radiator_flow__source_Medium_ThermodynamicState_copy_to_vars(src,...) Radiator_flow__source_Medium_ThermodynamicState_copy_to_vars_p(&src, __VA_ARGS__)

typedef base_array_t Radiator_flow__source_Medium_ThermodynamicState_array;
#define alloc_Radiator_flow__source_Medium_ThermodynamicState_array(dst,ndims,...) generic_array_create(NULL, dst, Radiator_flow__source_Medium_ThermodynamicState_construct_p, ndims, sizeof(Radiator_flow__source_Medium_ThermodynamicState), __VA_ARGS__)
#define Radiator_flow__source_Medium_ThermodynamicState_array_copy_data(src,dst)   generic_array_copy_data(src, &dst, Radiator_flow__source_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__source_Medium_ThermodynamicState))
#define Radiator_flow__source_Medium_ThermodynamicState_array_alloc_copy(src,dst)  generic_array_alloc_copy(src, &dst, Radiator_flow__source_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__source_Medium_ThermodynamicState))
#define Radiator_flow__source_Medium_ThermodynamicState_array_get(src,ndims,...)   (*(Radiator_flow__source_Medium_ThermodynamicState*)(generic_array_get(&src, sizeof(Radiator_flow__source_Medium_ThermodynamicState), __VA_ARGS__)))
#define Radiator_flow__source_Medium_ThermodynamicState_set(dst,val,...)           generic_array_set(&dst, &val, Radiator_flow__source_Medium_ThermodynamicState_copy_p, sizeof(Radiator_flow__source_Medium_ThermodynamicState), __VA_ARGS__)

DLLExport
modelica_real omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData_t *threadData, modelica_real _x, modelica_real _n, modelica_real _delta);
DLLExport
modelica_metatype boxptr_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData_t *threadData, modelica_metatype _x, modelica_metatype _n, modelica_metatype _delta);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Buildings_Utilities_Math_Functions_regNonZeroPower,2,0) {(void*) boxptr_Buildings_Utilities_Math_Functions_regNonZeroPower,0}};
#define boxvar_Buildings_Utilities_Math_Functions_regNonZeroPower MMC_REFSTRUCTLIT(boxvar_lit_Buildings_Utilities_Math_Functions_regNonZeroPower)


DLLExport
void omc_Modelica_Fluid_Utilities_checkBoundary(threadData_t *threadData, modelica_string _mediumName, string_array _substanceNames, modelica_boolean _singleState, modelica_boolean _define_p, real_array _X_boundary, modelica_string _modelName);
DLLExport
void boxptr_Modelica_Fluid_Utilities_checkBoundary(threadData_t *threadData, modelica_metatype _mediumName, modelica_metatype _substanceNames, modelica_metatype _singleState, modelica_metatype _define_p, modelica_metatype _X_boundary, modelica_metatype _modelName);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Modelica_Fluid_Utilities_checkBoundary,2,0) {(void*) boxptr_Modelica_Fluid_Utilities_checkBoundary,0}};
#define boxvar_Modelica_Fluid_Utilities_checkBoundary MMC_REFSTRUCTLIT(boxvar_lit_Modelica_Fluid_Utilities_checkBoundary)


DLLExport
void omc_Modelica_Utilities_Streams_error(threadData_t *threadData, modelica_string _string);
#define boxptr_Modelica_Utilities_Streams_error omc_Modelica_Utilities_Streams_error
static const MMC_DEFSTRUCTLIT(boxvar_lit_Modelica_Utilities_Streams_error,2,0) {(void*) boxptr_Modelica_Utilities_Streams_error,0}};
#define boxvar_Modelica_Utilities_Streams_error MMC_REFSTRUCTLIT(boxvar_lit_Modelica_Utilities_Streams_error)

/*
 * The function has annotation(Include=...>) or is builtin
 * the external function definition should be present
 * in one of these files and have this prototype:
 * extern void ModelicaError(const char* (*_string*));
 */

DLLExport
Radiator_Radiator_Medium_ThermodynamicState omc_Radiator_Radiator_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_Radiator_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_Radiator_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_Radiator_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_ThermodynamicState)


DLLExport
Radiator_Radiator_Medium_ThermodynamicState omc_Radiator_Radiator_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_setState__pTX,2,0) {(void*) boxptr_Radiator_Radiator_Medium_setState__pTX,0}};
#define boxvar_Radiator_Radiator_Medium_setState__pTX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_setState__pTX)


DLLExport
modelica_real omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData_t *threadData, Radiator_Radiator_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_specificHeatCapacityCp,2,0) {(void*) boxptr_Radiator_Radiator_Medium_specificHeatCapacityCp,0}};
#define boxvar_Radiator_Radiator_Medium_specificHeatCapacityCp MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_Medium_specificHeatCapacityCp)


DLLExport
Radiator_Radiator_res_Medium_ThermodynamicState omc_Radiator_Radiator_res_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_Radiator_res_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_res_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_Radiator_res_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_Radiator_res_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_res_Medium_ThermodynamicState)


DLLExport
modelica_real omc_Radiator_Radiator_res_Medium_dynamicViscosity(threadData_t *threadData, Radiator_Radiator_res_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_res_Medium_dynamicViscosity(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_res_Medium_dynamicViscosity,2,0) {(void*) boxptr_Radiator_Radiator_res_Medium_dynamicViscosity,0}};
#define boxvar_Radiator_Radiator_res_Medium_dynamicViscosity MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_res_Medium_dynamicViscosity)


DLLExport
Radiator_Radiator_vol_Medium_ThermodynamicState omc_Radiator_Radiator_vol_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_Radiator_vol_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_Radiator_vol_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_Medium_ThermodynamicState)


DLLExport
modelica_real omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData_t *threadData, modelica_real _p, modelica_real _h, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_Medium_temperature__phX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _h, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_Medium_temperature__phX,2,0) {(void*) boxptr_Radiator_Radiator_vol_Medium_temperature__phX,0}};
#define boxvar_Radiator_Radiator_vol_Medium_temperature__phX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_Medium_temperature__phX)


DLLExport
Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState omc_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState)


DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_density(threadData_t *threadData, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_density(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_density,2,0) {(void*) boxptr_Radiator_Radiator_vol_dynBal_Medium_density,0}};
#define boxvar_Radiator_Radiator_vol_dynBal_Medium_density MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_density)


DLLExport
Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_setState__pTX,2,0) {(void*) boxptr_Radiator_Radiator_vol_dynBal_Medium_setState__pTX,0}};
#define boxvar_Radiator_Radiator_vol_dynBal_Medium_setState__pTX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_setState__pTX)


DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX,2,0) {(void*) boxptr_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX,0}};
#define boxvar_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX)


DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData_t *threadData, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy,2,0) {(void*) boxptr_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy,0}};
#define boxvar_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy MMC_REFSTRUCTLIT(boxvar_lit_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy)


DLLExport
Radiator_flow__sink_Medium_ThermodynamicState omc_Radiator_flow__sink_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_flow__sink_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_flow__sink_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_flow__sink_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_ThermodynamicState)


DLLExport
Radiator_flow__sink_Medium_ThermodynamicState omc_Radiator_flow__sink_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_flow__sink_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_setState__pTX,2,0) {(void*) boxptr_Radiator_flow__sink_Medium_setState__pTX,0}};
#define boxvar_Radiator_flow__sink_Medium_setState__pTX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_setState__pTX)


DLLExport
modelica_real omc_Radiator_flow__sink_Medium_specificEnthalpy(threadData_t *threadData, Radiator_flow__sink_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_flow__sink_Medium_specificEnthalpy(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_specificEnthalpy,2,0) {(void*) boxptr_Radiator_flow__sink_Medium_specificEnthalpy,0}};
#define boxvar_Radiator_flow__sink_Medium_specificEnthalpy MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__sink_Medium_specificEnthalpy)


DLLExport
Radiator_flow__source_Medium_ThermodynamicState omc_Radiator_flow__source_Medium_ThermodynamicState (threadData_t *threadData, modelica_real omc_p, modelica_real omc_T);

DLLExport
modelica_metatype boxptr_Radiator_flow__source_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_ThermodynamicState,2,0) {(void*) boxptr_Radiator_flow__source_Medium_ThermodynamicState,0}};
#define boxvar_Radiator_flow__source_Medium_ThermodynamicState MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_ThermodynamicState)


DLLExport
Radiator_flow__source_Medium_ThermodynamicState omc_Radiator_flow__source_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X);
DLLExport
modelica_metatype boxptr_Radiator_flow__source_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_setState__pTX,2,0) {(void*) boxptr_Radiator_flow__source_Medium_setState__pTX,0}};
#define boxvar_Radiator_flow__source_Medium_setState__pTX MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_setState__pTX)


DLLExport
modelica_real omc_Radiator_flow__source_Medium_specificEnthalpy(threadData_t *threadData, Radiator_flow__source_Medium_ThermodynamicState _state);
DLLExport
modelica_metatype boxptr_Radiator_flow__source_Medium_specificEnthalpy(threadData_t *threadData, modelica_metatype _state);
static const MMC_DEFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_specificEnthalpy,2,0) {(void*) boxptr_Radiator_flow__source_Medium_specificEnthalpy,0}};
#define boxvar_Radiator_flow__source_Medium_specificEnthalpy MMC_REFSTRUCTLIT(boxvar_lit_Radiator_flow__source_Medium_specificEnthalpy)
#include "Radiator_model.h"


#ifdef __cplusplus
}
#endif
#endif

