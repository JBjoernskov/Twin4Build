/* Non Linear Systems */
#include "Radiator_model.h"
#include "Radiator_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

/* inner equations */

/*
equation index: 50
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[1] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[1] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[1] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[1] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[1] > 1.0) then (-293.15 + Radiator.TWat_nominal[1]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[1])))
*/
void Radiator_eqFunction_50(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,50};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */);
  }
  data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 51
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[3] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[3] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[3] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[3] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[3] > 1.0) then (-293.15 + Radiator.TWat_nominal[3]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[3])))
*/
void Radiator_eqFunction_51(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,51};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */);
  }
  data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 52
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[2] = Radiator.TWat_nominal[3] + Radiator.QEle_flow_nominal[3] / (Radiator.m_flow_nominal * 4184.0)
*/
void Radiator_eqFunction_52(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,52};
  data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ + DIVISION_SIM(data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */,(data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */) * (4184.0),"Radiator.m_flow_nominal * 4184.0",equationIndexes);
  TRACE_POP
}
/*
equation index: 53
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[2] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[2] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[2] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[2] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[2] > 1.0) then (-293.15 + Radiator.TWat_nominal[2]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[2])))
*/
void Radiator_eqFunction_53(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,53};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */);
  }
  data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 54
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[4] = Radiator.TWat_nominal[3] - Radiator.QEle_flow_nominal[4] / (Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[3], {1.0})) * Radiator.m_flow_nominal)
*/
void Radiator_eqFunction_54(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,54};
  real_array tmp0;
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - (DIVISION_SIM(data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */,(omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */),"Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[3], {1.0})) * Radiator.m_flow_nominal",equationIndexes));
  TRACE_POP
}
/*
equation index: 55
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[5] = Radiator.Q_flow_nominal - (Radiator.QEle_flow_nominal[1] + Radiator.QEle_flow_nominal[2] + Radiator.QEle_flow_nominal[3] + Radiator.QEle_flow_nominal[4])
*/
void Radiator_eqFunction_55(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,55};
  data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */ = data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */ - (data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */ + data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */ + data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */ + data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */);
  TRACE_POP
}
/*
equation index: 56
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[5] = Radiator.TWat_nominal[4] - Radiator.QEle_flow_nominal[5] / (Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[4], {1.0})) * Radiator.m_flow_nominal)
*/
void Radiator_eqFunction_56(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,56};
  real_array tmp0;
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ = data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - (DIVISION_SIM(data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */,(omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */),"Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[4], {1.0})) * Radiator.m_flow_nominal",equationIndexes));
  TRACE_POP
}

void residualFunc61(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,61};
  int i;
  real_array tmp0;
  real_array tmp1;
  modelica_boolean tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_boolean tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_boolean tmp34;
  modelica_real tmp35;
  modelica_boolean tmp36;
  modelica_real tmp37;
  modelica_real tmp38;
  modelica_real tmp39;
  modelica_real tmp40;
  modelica_real tmp41;
  modelica_real tmp42;
  modelica_real tmp43;
  modelica_real tmp44;
  modelica_real tmp45;
  modelica_real tmp46;
  modelica_real tmp47;
  modelica_real tmp48;
  modelica_real tmp49;
  modelica_real tmp50;
  modelica_real tmp51;
  modelica_real tmp52;
  modelica_real tmp53;
  modelica_real tmp54;
  modelica_real tmp55;
  modelica_real tmp56;
  modelica_real tmp57;
  modelica_boolean tmp58;
  modelica_real tmp59;
  modelica_boolean tmp60;
  modelica_real tmp61;
  modelica_real tmp62;
  modelica_real tmp63;
  modelica_real tmp64;
  modelica_real tmp65;
  modelica_real tmp66;
  modelica_real tmp67;
  modelica_boolean tmp68;
  modelica_real tmp69;
  real_array tmp70;
  real_array tmp71;
  /* iteration variables */
  for (i=0; i<4; i++) {
    if (isinf(xloc[i]) || isnan(xloc[i])) {
      for (i=0; i<4; i++) {
        res[i] = NAN;
      }
      return;
    }
  }
  data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */ = xloc[0];
  data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ = xloc[1];
  data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ = xloc[2];
  data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */ = xloc[3];
  /* backup outputs */
  /* pre body */
  /* local constraints */
  Radiator_eqFunction_50(data, threadData);

  /* local constraints */
  Radiator_eqFunction_51(data, threadData);

  /* local constraints */
  Radiator_eqFunction_52(data, threadData);

  /* local constraints */
  Radiator_eqFunction_53(data, threadData);

  /* local constraints */
  Radiator_eqFunction_54(data, threadData);

  /* local constraints */
  Radiator_eqFunction_55(data, threadData);

  /* local constraints */
  Radiator_eqFunction_56(data, threadData);
  /* body */
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  array_alloc_scalar_real_array(&tmp1, 1, (modelica_real)1.0);
  res[0] = (data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)) + (-data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */) - ((data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */, tmp1))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)));

  tmp2 = Greater(data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp24 = (modelica_boolean)tmp2;
  if(tmp24)
  {
    tmp3 = data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp4 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp3 < 0.0 && tmp4 != 0.0)
    {
      tmp6 = modf(tmp4, &tmp7);
      
      if(tmp6 > 0.5)
      {
        tmp6 -= 1.0;
        tmp7 += 1.0;
      }
      else if(tmp6 < -0.5)
      {
        tmp6 += 1.0;
        tmp7 -= 1.0;
      }
      
      if(fabs(tmp6) < 1e-10)
        tmp5 = pow(tmp3, tmp7);
      else
      {
        tmp9 = modf(1.0/tmp4, &tmp8);
        if(tmp9 > 0.5)
        {
          tmp9 -= 1.0;
          tmp8 += 1.0;
        }
        else if(tmp9 < -0.5)
        {
          tmp9 += 1.0;
          tmp8 -= 1.0;
        }
        if(fabs(tmp9) < 1e-10 && ((unsigned long)tmp8 & 1))
        {
          tmp5 = -pow(-tmp3, tmp6)*pow(tmp3, tmp7);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3, tmp4);
        }
      }
    }
    else
    {
      tmp5 = pow(tmp3, tmp4);
    }
    if(isnan(tmp5) || isinf(tmp5))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3, tmp4);
    }
    tmp25 = tmp5;
  }
  else
  {
    tmp10 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp11 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp10 < 0.0 && tmp11 != 0.0)
    {
      tmp13 = modf(tmp11, &tmp14);
      
      if(tmp13 > 0.5)
      {
        tmp13 -= 1.0;
        tmp14 += 1.0;
      }
      else if(tmp13 < -0.5)
      {
        tmp13 += 1.0;
        tmp14 -= 1.0;
      }
      
      if(fabs(tmp13) < 1e-10)
        tmp12 = pow(tmp10, tmp14);
      else
      {
        tmp16 = modf(1.0/tmp11, &tmp15);
        if(tmp16 > 0.5)
        {
          tmp16 -= 1.0;
          tmp15 += 1.0;
        }
        else if(tmp16 < -0.5)
        {
          tmp16 += 1.0;
          tmp15 -= 1.0;
        }
        if(fabs(tmp16) < 1e-10 && ((unsigned long)tmp15 & 1))
        {
          tmp12 = -pow(-tmp10, tmp13)*pow(tmp10, tmp14);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp10, tmp11);
        }
      }
    }
    else
    {
      tmp12 = pow(tmp10, tmp11);
    }
    if(isnan(tmp12) || isinf(tmp12))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp10, tmp11);
    }tmp17 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp18 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp17 < 0.0 && tmp18 != 0.0)
    {
      tmp20 = modf(tmp18, &tmp21);
      
      if(tmp20 > 0.5)
      {
        tmp20 -= 1.0;
        tmp21 += 1.0;
      }
      else if(tmp20 < -0.5)
      {
        tmp20 += 1.0;
        tmp21 -= 1.0;
      }
      
      if(fabs(tmp20) < 1e-10)
        tmp19 = pow(tmp17, tmp21);
      else
      {
        tmp23 = modf(1.0/tmp18, &tmp22);
        if(tmp23 > 0.5)
        {
          tmp23 -= 1.0;
          tmp22 += 1.0;
        }
        else if(tmp23 < -0.5)
        {
          tmp23 += 1.0;
          tmp22 -= 1.0;
        }
        if(fabs(tmp23) < 1e-10 && ((unsigned long)tmp22 & 1))
        {
          tmp19 = -pow(-tmp17, tmp20)*pow(tmp17, tmp21);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp17, tmp18);
        }
      }
    }
    else
    {
      tmp19 = pow(tmp17, tmp18);
    }
    if(isnan(tmp19) || isinf(tmp19))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp17, tmp18);
    }
    tmp25 = (tmp12) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp19)) * (data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp26 = Greater(-293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */,1.0);
  tmp34 = (modelica_boolean)tmp26;
  if(tmp34)
  {
    tmp27 = -293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */;
    tmp28 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp27 < 0.0 && tmp28 != 0.0)
    {
      tmp30 = modf(tmp28, &tmp31);
      
      if(tmp30 > 0.5)
      {
        tmp30 -= 1.0;
        tmp31 += 1.0;
      }
      else if(tmp30 < -0.5)
      {
        tmp30 += 1.0;
        tmp31 -= 1.0;
      }
      
      if(fabs(tmp30) < 1e-10)
        tmp29 = pow(tmp27, tmp31);
      else
      {
        tmp33 = modf(1.0/tmp28, &tmp32);
        if(tmp33 > 0.5)
        {
          tmp33 -= 1.0;
          tmp32 += 1.0;
        }
        else if(tmp33 < -0.5)
        {
          tmp33 += 1.0;
          tmp32 -= 1.0;
        }
        if(fabs(tmp33) < 1e-10 && ((unsigned long)tmp32 & 1))
        {
          tmp29 = -pow(-tmp27, tmp30)*pow(tmp27, tmp31);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp27, tmp28);
        }
      }
    }
    else
    {
      tmp29 = pow(tmp27, tmp28);
    }
    if(isnan(tmp29) || isinf(tmp29))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp27, tmp28);
    }
    tmp35 = tmp29;
  }
  else
  {
    tmp35 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */);
  }
  res[1] = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp25) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp35)) - data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */;

  tmp36 = Greater(data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp58 = (modelica_boolean)tmp36;
  if(tmp58)
  {
    tmp37 = data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp38 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp59 = tmp39;
  }
  else
  {
    tmp44 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp45 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp51 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
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
    tmp59 = (tmp46) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp53)) * (data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp60 = Greater(-293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */,1.0);
  tmp68 = (modelica_boolean)tmp60;
  if(tmp68)
  {
    tmp61 = -293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */;
    tmp62 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp61 < 0.0 && tmp62 != 0.0)
    {
      tmp64 = modf(tmp62, &tmp65);
      
      if(tmp64 > 0.5)
      {
        tmp64 -= 1.0;
        tmp65 += 1.0;
      }
      else if(tmp64 < -0.5)
      {
        tmp64 += 1.0;
        tmp65 -= 1.0;
      }
      
      if(fabs(tmp64) < 1e-10)
        tmp63 = pow(tmp61, tmp65);
      else
      {
        tmp67 = modf(1.0/tmp62, &tmp66);
        if(tmp67 > 0.5)
        {
          tmp67 -= 1.0;
          tmp66 += 1.0;
        }
        else if(tmp67 < -0.5)
        {
          tmp67 += 1.0;
          tmp66 -= 1.0;
        }
        if(fabs(tmp67) < 1e-10 && ((unsigned long)tmp66 & 1))
        {
          tmp63 = -pow(-tmp61, tmp64)*pow(tmp61, tmp65);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp61, tmp62);
        }
      }
    }
    else
    {
      tmp63 = pow(tmp61, tmp62);
    }
    if(isnan(tmp63) || isinf(tmp63))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp61, tmp62);
    }
    tmp69 = tmp63;
  }
  else
  {
    tmp69 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */);
  }
  res[2] = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp59) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp69)) - data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */;

  array_alloc_scalar_real_array(&tmp70, 1, (modelica_real)1.0);
  array_alloc_scalar_real_array(&tmp71, 1, (modelica_real)1.0);
  res[3] = (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */, tmp70))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)) + (-data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */) - ((data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */, tmp71))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)));
  /* restore known outputs */
  TRACE_POP
}

OMC_DISABLE_OPT
void initializeSparsePatternNLS61(NONLINEAR_SYSTEM_DATA* inSysData)
{
  int i=0;
  const int colPtrIndex[1+4] = {0,2,3,3,4};
  const int rowIndex[12] = {1,2,0,1,2,0,1,3,0,1,2,3};
  /* sparsity pattern available */
  inSysData->isPatternAvailable = 'T';
  inSysData->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  inSysData->sparsePattern->leadindex = (unsigned int*) malloc((4+1)*sizeof(unsigned int));
  inSysData->sparsePattern->index = (unsigned int*) malloc(12*sizeof(unsigned int));
  inSysData->sparsePattern->numberOfNoneZeros = 12;
  inSysData->sparsePattern->colorCols = (unsigned int*) malloc(4*sizeof(unsigned int));
  inSysData->sparsePattern->maxColors = 4;
  
  /* write lead index of compressed sparse column */
  memcpy(inSysData->sparsePattern->leadindex, colPtrIndex, (4+1)*sizeof(unsigned int));
  
  for(i=2;i<4+1;++i)
    inSysData->sparsePattern->leadindex[i] += inSysData->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(inSysData->sparsePattern->index, rowIndex, 12*sizeof(unsigned int));
  
  /* write color array */
  inSysData->sparsePattern->colorCols[3] = 1;
  inSysData->sparsePattern->colorCols[2] = 2;
  inSysData->sparsePattern->colorCols[1] = 3;
  inSysData->sparsePattern->colorCols[0] = 4;
}

OMC_DISABLE_OPT
void initializeStaticDataNLS61(void *inData, threadData_t *threadData, void *inSystemData)
{
  DATA* data = (DATA*) inData;
  NONLINEAR_SYSTEM_DATA* sysData = (NONLINEAR_SYSTEM_DATA*) inSystemData;
  int i=0;
  /* static nls data for Radiator.QEle_flow_nominal[4] */
  sysData->nominal[i] = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.max;
  /* static nls data for Radiator.TWat_nominal[3] */
  sysData->nominal[i] = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.max;
  /* static nls data for Radiator.TWat_nominal[1] */
  sysData->nominal[i] = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.max;
  /* static nls data for Radiator.UAEle */
  sysData->nominal[i] = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.min;
  sysData->max[i++]   = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.max;
  /* initial sparse pattern */
  initializeSparsePatternNLS61(sysData);
}

OMC_DISABLE_OPT
void getIterationVarsNLS61(struct DATA *inData, double *array)
{
  DATA* data = (DATA*) inData;
  array[0] = data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */;
  array[1] = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
  array[2] = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
  array[3] = data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */;
}


/* inner equations */

/*
equation index: 213
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[1] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[1] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[1] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[1] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[1] > 1.0) then (-293.15 + Radiator.TWat_nominal[1]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[1])))
*/
void Radiator_eqFunction_213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,213};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */);
  }
  data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 214
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[3] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[3] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[3] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[3] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[3] > 1.0) then (-293.15 + Radiator.TWat_nominal[3]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[3])))
*/
void Radiator_eqFunction_214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,214};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */);
  }
  data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 215
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[2] = Radiator.TWat_nominal[3] + Radiator.QEle_flow_nominal[3] / (Radiator.m_flow_nominal * 4184.0)
*/
void Radiator_eqFunction_215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,215};
  data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ + DIVISION_SIM(data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */,(data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */) * (4184.0),"Radiator.m_flow_nominal * 4184.0",equationIndexes);
  TRACE_POP
}
/*
equation index: 216
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[2] = Radiator.UAEle * (Radiator.fraRad * (if noEvent(Radiator.TWat_nominal[2] - Radiator.TRad_nominal > 0.1 * (303.15 - Radiator.TRad_nominal)) then (Radiator.TWat_nominal[2] - Radiator.TRad_nominal) ^ Radiator.n else (0.1 * (303.15 - Radiator.TRad_nominal)) ^ Radiator.n * (1.0 - Radiator.n) + Radiator.n * (0.1 * (303.15 - Radiator.TRad_nominal)) ^ (Radiator.n - 1.0) * (Radiator.TWat_nominal[2] - Radiator.TRad_nominal)) + (1.0 - Radiator.fraRad) * (if noEvent(-293.15 + Radiator.TWat_nominal[2] > 1.0) then (-293.15 + Radiator.TWat_nominal[2]) ^ Radiator.n else 1.0 - Radiator.n + Radiator.n * (-293.15 + Radiator.TWat_nominal[2])))
*/
void Radiator_eqFunction_216(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,216};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_boolean tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  tmp0 = Greater(data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp22 = (modelica_boolean)tmp0;
  if(tmp22)
  {
    tmp1 = data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp2 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp23 = tmp3;
  }
  else
  {
    tmp8 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp9 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp15 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp16 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp15 < 0.0 && tmp16 != 0.0)
    {
      tmp18 = modf(tmp16, &tmp19);
      
      if(tmp18 > 0.5)
      {
        tmp18 -= 1.0;
        tmp19 += 1.0;
      }
      else if(tmp18 < -0.5)
      {
        tmp18 += 1.0;
        tmp19 -= 1.0;
      }
      
      if(fabs(tmp18) < 1e-10)
        tmp17 = pow(tmp15, tmp19);
      else
      {
        tmp21 = modf(1.0/tmp16, &tmp20);
        if(tmp21 > 0.5)
        {
          tmp21 -= 1.0;
          tmp20 += 1.0;
        }
        else if(tmp21 < -0.5)
        {
          tmp21 += 1.0;
          tmp20 -= 1.0;
        }
        if(fabs(tmp21) < 1e-10 && ((unsigned long)tmp20 & 1))
        {
          tmp17 = -pow(-tmp15, tmp18)*pow(tmp15, tmp19);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
        }
      }
    }
    else
    {
      tmp17 = pow(tmp15, tmp16);
    }
    if(isnan(tmp17) || isinf(tmp17))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp15, tmp16);
    }
    tmp23 = (tmp10) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp17)) * (data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp24 = Greater(-293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */,1.0);
  tmp32 = (modelica_boolean)tmp24;
  if(tmp32)
  {
    tmp25 = -293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */;
    tmp26 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp25 < 0.0 && tmp26 != 0.0)
    {
      tmp28 = modf(tmp26, &tmp29);
      
      if(tmp28 > 0.5)
      {
        tmp28 -= 1.0;
        tmp29 += 1.0;
      }
      else if(tmp28 < -0.5)
      {
        tmp28 += 1.0;
        tmp29 -= 1.0;
      }
      
      if(fabs(tmp28) < 1e-10)
        tmp27 = pow(tmp25, tmp29);
      else
      {
        tmp31 = modf(1.0/tmp26, &tmp30);
        if(tmp31 > 0.5)
        {
          tmp31 -= 1.0;
          tmp30 += 1.0;
        }
        else if(tmp31 < -0.5)
        {
          tmp31 += 1.0;
          tmp30 -= 1.0;
        }
        if(fabs(tmp31) < 1e-10 && ((unsigned long)tmp30 & 1))
        {
          tmp27 = -pow(-tmp25, tmp28)*pow(tmp25, tmp29);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
        }
      }
    }
    else
    {
      tmp27 = pow(tmp25, tmp26);
    }
    if(isnan(tmp27) || isinf(tmp27))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp25, tmp26);
    }
    tmp33 = tmp27;
  }
  else
  {
    tmp33 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */);
  }
  data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */ = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp23) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp33));
  TRACE_POP
}
/*
equation index: 217
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[4] = Radiator.TWat_nominal[3] - Radiator.QEle_flow_nominal[4] / (Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[3], {1.0})) * Radiator.m_flow_nominal)
*/
void Radiator_eqFunction_217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,217};
  real_array tmp0;
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ - (DIVISION_SIM(data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */,(omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */),"Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[3], {1.0})) * Radiator.m_flow_nominal",equationIndexes));
  TRACE_POP
}
/*
equation index: 218
type: SIMPLE_ASSIGN
Radiator.QEle_flow_nominal[5] = Radiator.Q_flow_nominal - (Radiator.QEle_flow_nominal[1] + Radiator.QEle_flow_nominal[2] + Radiator.QEle_flow_nominal[3] + Radiator.QEle_flow_nominal[4])
*/
void Radiator_eqFunction_218(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,218};
  data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */ = data->simulationInfo->realParameter[6] /* Radiator.Q_flow_nominal PARAM */ - (data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */ + data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */ + data->simulationInfo->realParameter[3] /* Radiator.QEle_flow_nominal[3] PARAM */ + data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */);
  TRACE_POP
}
/*
equation index: 219
type: SIMPLE_ASSIGN
Radiator.TWat_nominal[5] = Radiator.TWat_nominal[4] - Radiator.QEle_flow_nominal[5] / (Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[4], {1.0})) * Radiator.m_flow_nominal)
*/
void Radiator_eqFunction_219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,219};
  real_array tmp0;
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ = data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - (DIVISION_SIM(data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */,(omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */),"Radiator.Radiator.Medium.specificHeatCapacityCp(Radiator.Radiator.Medium.setState_pTX(300000.0, Radiator.TWat_nominal[4], {1.0})) * Radiator.m_flow_nominal",equationIndexes));
  TRACE_POP
}

void residualFunc224(void** dataIn, const double* xloc, double* res, const int* iflag)
{
  TRACE_PUSH
  DATA *data = (DATA*) ((void**)dataIn[0]);
  threadData_t *threadData = (threadData_t*) ((void**)dataIn[1]);
  const int equationIndexes[2] = {1,224};
  int i;
  real_array tmp0;
  real_array tmp1;
  modelica_boolean tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_real tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_real tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_boolean tmp24;
  modelica_real tmp25;
  modelica_boolean tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_boolean tmp34;
  modelica_real tmp35;
  modelica_boolean tmp36;
  modelica_real tmp37;
  modelica_real tmp38;
  modelica_real tmp39;
  modelica_real tmp40;
  modelica_real tmp41;
  modelica_real tmp42;
  modelica_real tmp43;
  modelica_real tmp44;
  modelica_real tmp45;
  modelica_real tmp46;
  modelica_real tmp47;
  modelica_real tmp48;
  modelica_real tmp49;
  modelica_real tmp50;
  modelica_real tmp51;
  modelica_real tmp52;
  modelica_real tmp53;
  modelica_real tmp54;
  modelica_real tmp55;
  modelica_real tmp56;
  modelica_real tmp57;
  modelica_boolean tmp58;
  modelica_real tmp59;
  modelica_boolean tmp60;
  modelica_real tmp61;
  modelica_real tmp62;
  modelica_real tmp63;
  modelica_real tmp64;
  modelica_real tmp65;
  modelica_real tmp66;
  modelica_real tmp67;
  modelica_boolean tmp68;
  modelica_real tmp69;
  real_array tmp70;
  real_array tmp71;
  /* iteration variables */
  for (i=0; i<4; i++) {
    if (isinf(xloc[i]) || isnan(xloc[i])) {
      for (i=0; i<4; i++) {
        res[i] = NAN;
      }
      return;
    }
  }
  data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */ = xloc[0];
  data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */ = xloc[1];
  data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */ = xloc[2];
  data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */ = xloc[3];
  /* backup outputs */
  /* pre body */
  /* local constraints */
  Radiator_eqFunction_213(data, threadData);

  /* local constraints */
  Radiator_eqFunction_214(data, threadData);

  /* local constraints */
  Radiator_eqFunction_215(data, threadData);

  /* local constraints */
  Radiator_eqFunction_216(data, threadData);

  /* local constraints */
  Radiator_eqFunction_217(data, threadData);

  /* local constraints */
  Radiator_eqFunction_218(data, threadData);

  /* local constraints */
  Radiator_eqFunction_219(data, threadData);
  /* body */
  array_alloc_scalar_real_array(&tmp0, 1, (modelica_real)1.0);
  array_alloc_scalar_real_array(&tmp1, 1, (modelica_real)1.0);
  res[0] = (data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */, tmp0))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)) + (-data->simulationInfo->realParameter[2] /* Radiator.QEle_flow_nominal[2] PARAM */) - ((data->simulationInfo->realParameter[10] /* Radiator.TWat_nominal[2] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */, tmp1))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)));

  tmp2 = Greater(data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp24 = (modelica_boolean)tmp2;
  if(tmp24)
  {
    tmp3 = data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp4 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp3 < 0.0 && tmp4 != 0.0)
    {
      tmp6 = modf(tmp4, &tmp7);
      
      if(tmp6 > 0.5)
      {
        tmp6 -= 1.0;
        tmp7 += 1.0;
      }
      else if(tmp6 < -0.5)
      {
        tmp6 += 1.0;
        tmp7 -= 1.0;
      }
      
      if(fabs(tmp6) < 1e-10)
        tmp5 = pow(tmp3, tmp7);
      else
      {
        tmp9 = modf(1.0/tmp4, &tmp8);
        if(tmp9 > 0.5)
        {
          tmp9 -= 1.0;
          tmp8 += 1.0;
        }
        else if(tmp9 < -0.5)
        {
          tmp9 += 1.0;
          tmp8 -= 1.0;
        }
        if(fabs(tmp9) < 1e-10 && ((unsigned long)tmp8 & 1))
        {
          tmp5 = -pow(-tmp3, tmp6)*pow(tmp3, tmp7);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3, tmp4);
        }
      }
    }
    else
    {
      tmp5 = pow(tmp3, tmp4);
    }
    if(isnan(tmp5) || isinf(tmp5))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp3, tmp4);
    }
    tmp25 = tmp5;
  }
  else
  {
    tmp10 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp11 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp10 < 0.0 && tmp11 != 0.0)
    {
      tmp13 = modf(tmp11, &tmp14);
      
      if(tmp13 > 0.5)
      {
        tmp13 -= 1.0;
        tmp14 += 1.0;
      }
      else if(tmp13 < -0.5)
      {
        tmp13 += 1.0;
        tmp14 -= 1.0;
      }
      
      if(fabs(tmp13) < 1e-10)
        tmp12 = pow(tmp10, tmp14);
      else
      {
        tmp16 = modf(1.0/tmp11, &tmp15);
        if(tmp16 > 0.5)
        {
          tmp16 -= 1.0;
          tmp15 += 1.0;
        }
        else if(tmp16 < -0.5)
        {
          tmp16 += 1.0;
          tmp15 -= 1.0;
        }
        if(fabs(tmp16) < 1e-10 && ((unsigned long)tmp15 & 1))
        {
          tmp12 = -pow(-tmp10, tmp13)*pow(tmp10, tmp14);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp10, tmp11);
        }
      }
    }
    else
    {
      tmp12 = pow(tmp10, tmp11);
    }
    if(isnan(tmp12) || isinf(tmp12))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp10, tmp11);
    }tmp17 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp18 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ - 1.0;
    if(tmp17 < 0.0 && tmp18 != 0.0)
    {
      tmp20 = modf(tmp18, &tmp21);
      
      if(tmp20 > 0.5)
      {
        tmp20 -= 1.0;
        tmp21 += 1.0;
      }
      else if(tmp20 < -0.5)
      {
        tmp20 += 1.0;
        tmp21 -= 1.0;
      }
      
      if(fabs(tmp20) < 1e-10)
        tmp19 = pow(tmp17, tmp21);
      else
      {
        tmp23 = modf(1.0/tmp18, &tmp22);
        if(tmp23 > 0.5)
        {
          tmp23 -= 1.0;
          tmp22 += 1.0;
        }
        else if(tmp23 < -0.5)
        {
          tmp23 += 1.0;
          tmp22 -= 1.0;
        }
        if(fabs(tmp23) < 1e-10 && ((unsigned long)tmp22 & 1))
        {
          tmp19 = -pow(-tmp17, tmp20)*pow(tmp17, tmp21);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp17, tmp18);
        }
      }
    }
    else
    {
      tmp19 = pow(tmp17, tmp18);
    }
    if(isnan(tmp19) || isinf(tmp19))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp17, tmp18);
    }
    tmp25 = (tmp12) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp19)) * (data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp26 = Greater(-293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */,1.0);
  tmp34 = (modelica_boolean)tmp26;
  if(tmp34)
  {
    tmp27 = -293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */;
    tmp28 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp27 < 0.0 && tmp28 != 0.0)
    {
      tmp30 = modf(tmp28, &tmp31);
      
      if(tmp30 > 0.5)
      {
        tmp30 -= 1.0;
        tmp31 += 1.0;
      }
      else if(tmp30 < -0.5)
      {
        tmp30 += 1.0;
        tmp31 -= 1.0;
      }
      
      if(fabs(tmp30) < 1e-10)
        tmp29 = pow(tmp27, tmp31);
      else
      {
        tmp33 = modf(1.0/tmp28, &tmp32);
        if(tmp33 > 0.5)
        {
          tmp33 -= 1.0;
          tmp32 += 1.0;
        }
        else if(tmp33 < -0.5)
        {
          tmp33 += 1.0;
          tmp32 -= 1.0;
        }
        if(fabs(tmp33) < 1e-10 && ((unsigned long)tmp32 & 1))
        {
          tmp29 = -pow(-tmp27, tmp30)*pow(tmp27, tmp31);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp27, tmp28);
        }
      }
    }
    else
    {
      tmp29 = pow(tmp27, tmp28);
    }
    if(isnan(tmp29) || isinf(tmp29))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp27, tmp28);
    }
    tmp35 = tmp29;
  }
  else
  {
    tmp35 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[13] /* Radiator.TWat_nominal[5] PARAM */);
  }
  res[1] = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp25) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp35)) - data->simulationInfo->realParameter[5] /* Radiator.QEle_flow_nominal[5] PARAM */;

  tmp36 = Greater(data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */,(0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */));
  tmp58 = (modelica_boolean)tmp36;
  if(tmp58)
  {
    tmp37 = data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */;
    tmp38 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    tmp59 = tmp39;
  }
  else
  {
    tmp44 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
    tmp45 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
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
    }tmp51 = (0.1) * (303.15 - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
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
    tmp59 = (tmp46) * (1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) + ((data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (tmp53)) * (data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */ - data->simulationInfo->realParameter[8] /* Radiator.TRad_nominal PARAM */);
  }
  tmp60 = Greater(-293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */,1.0);
  tmp68 = (modelica_boolean)tmp60;
  if(tmp68)
  {
    tmp61 = -293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */;
    tmp62 = data->simulationInfo->realParameter[41] /* Radiator.n PARAM */;
    if(tmp61 < 0.0 && tmp62 != 0.0)
    {
      tmp64 = modf(tmp62, &tmp65);
      
      if(tmp64 > 0.5)
      {
        tmp64 -= 1.0;
        tmp65 += 1.0;
      }
      else if(tmp64 < -0.5)
      {
        tmp64 += 1.0;
        tmp65 -= 1.0;
      }
      
      if(fabs(tmp64) < 1e-10)
        tmp63 = pow(tmp61, tmp65);
      else
      {
        tmp67 = modf(1.0/tmp62, &tmp66);
        if(tmp67 > 0.5)
        {
          tmp67 -= 1.0;
          tmp66 += 1.0;
        }
        else if(tmp67 < -0.5)
        {
          tmp67 += 1.0;
          tmp66 -= 1.0;
        }
        if(fabs(tmp67) < 1e-10 && ((unsigned long)tmp66 & 1))
        {
          tmp63 = -pow(-tmp61, tmp64)*pow(tmp61, tmp65);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp61, tmp62);
        }
      }
    }
    else
    {
      tmp63 = pow(tmp61, tmp62);
    }
    if(isnan(tmp63) || isinf(tmp63))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp61, tmp62);
    }
    tmp69 = tmp63;
  }
  else
  {
    tmp69 = 1.0 - data->simulationInfo->realParameter[41] /* Radiator.n PARAM */ + (data->simulationInfo->realParameter[41] /* Radiator.n PARAM */) * (-293.15 + data->simulationInfo->realParameter[12] /* Radiator.TWat_nominal[4] PARAM */);
  }
  res[2] = (data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */) * ((data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp59) + (1.0 - data->simulationInfo->realParameter[35] /* Radiator.fraRad PARAM */) * (tmp69)) - data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */;

  array_alloc_scalar_real_array(&tmp70, 1, (modelica_real)1.0);
  array_alloc_scalar_real_array(&tmp71, 1, (modelica_real)1.0);
  res[3] = (data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */, tmp70))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)) + (-data->simulationInfo->realParameter[1] /* Radiator.QEle_flow_nominal[1] PARAM */) - ((data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */) * ((omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, omc_Radiator_Radiator_Medium_setState__pTX(threadData, 300000.0, data->simulationInfo->realParameter[14] /* Radiator.T_a_nominal PARAM */, tmp71))) * (data->simulationInfo->realParameter[39] /* Radiator.m_flow_nominal PARAM */)));
  /* restore known outputs */
  TRACE_POP
}

OMC_DISABLE_OPT
void initializeSparsePatternNLS224(NONLINEAR_SYSTEM_DATA* inSysData)
{
  int i=0;
  const int colPtrIndex[1+4] = {0,2,3,3,4};
  const int rowIndex[12] = {1,2,0,1,2,0,1,3,0,1,2,3};
  /* sparsity pattern available */
  inSysData->isPatternAvailable = 'T';
  inSysData->sparsePattern = (SPARSE_PATTERN*) malloc(sizeof(SPARSE_PATTERN));
  inSysData->sparsePattern->leadindex = (unsigned int*) malloc((4+1)*sizeof(unsigned int));
  inSysData->sparsePattern->index = (unsigned int*) malloc(12*sizeof(unsigned int));
  inSysData->sparsePattern->numberOfNoneZeros = 12;
  inSysData->sparsePattern->colorCols = (unsigned int*) malloc(4*sizeof(unsigned int));
  inSysData->sparsePattern->maxColors = 4;
  
  /* write lead index of compressed sparse column */
  memcpy(inSysData->sparsePattern->leadindex, colPtrIndex, (4+1)*sizeof(unsigned int));
  
  for(i=2;i<4+1;++i)
    inSysData->sparsePattern->leadindex[i] += inSysData->sparsePattern->leadindex[i-1];
  
  /* call sparse index */
  memcpy(inSysData->sparsePattern->index, rowIndex, 12*sizeof(unsigned int));
  
  /* write color array */
  inSysData->sparsePattern->colorCols[3] = 1;
  inSysData->sparsePattern->colorCols[2] = 2;
  inSysData->sparsePattern->colorCols[1] = 3;
  inSysData->sparsePattern->colorCols[0] = 4;
}

OMC_DISABLE_OPT
void initializeStaticDataNLS224(void *inData, threadData_t *threadData, void *inSystemData)
{
  DATA* data = (DATA*) inData;
  NONLINEAR_SYSTEM_DATA* sysData = (NONLINEAR_SYSTEM_DATA*) inSystemData;
  int i=0;
  /* static nls data for Radiator.QEle_flow_nominal[4] */
  sysData->nominal[i] = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[4].attribute /* Radiator.QEle_flow_nominal[4] */.max;
  /* static nls data for Radiator.TWat_nominal[3] */
  sysData->nominal[i] = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[11].attribute /* Radiator.TWat_nominal[3] */.max;
  /* static nls data for Radiator.TWat_nominal[1] */
  sysData->nominal[i] = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.min;
  sysData->max[i++]   = data->modelData->realParameterData[9].attribute /* Radiator.TWat_nominal[1] */.max;
  /* static nls data for Radiator.UAEle */
  sysData->nominal[i] = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.nominal;
  sysData->min[i]     = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.min;
  sysData->max[i++]   = data->modelData->realParameterData[17].attribute /* Radiator.UAEle */.max;
  /* initial sparse pattern */
  initializeSparsePatternNLS224(sysData);
}

OMC_DISABLE_OPT
void getIterationVarsNLS224(struct DATA *inData, double *array)
{
  DATA* data = (DATA*) inData;
  array[0] = data->simulationInfo->realParameter[4] /* Radiator.QEle_flow_nominal[4] PARAM */;
  array[1] = data->simulationInfo->realParameter[11] /* Radiator.TWat_nominal[3] PARAM */;
  array[2] = data->simulationInfo->realParameter[9] /* Radiator.TWat_nominal[1] PARAM */;
  array[3] = data->simulationInfo->realParameter[17] /* Radiator.UAEle PARAM */;
}

/* Prototypes for the strict sets (Dynamic Tearing) */

/* Global constraints for the casual sets */
/* function initialize non-linear systems */
void Radiator_initialNonLinearSystem(int nNonLinearSystems, NONLINEAR_SYSTEM_DATA* nonLinearSystemData)
{
  
  nonLinearSystemData[1].equationIndex = 224;
  nonLinearSystemData[1].size = 4;
  nonLinearSystemData[1].homotopySupport = 0;
  nonLinearSystemData[1].mixedSystem = 1;
  nonLinearSystemData[1].residualFunc = residualFunc224;
  nonLinearSystemData[1].strictTearingFunctionCall = NULL;
  nonLinearSystemData[1].analyticalJacobianColumn = NULL;
  nonLinearSystemData[1].initialAnalyticalJacobian = NULL;
  nonLinearSystemData[1].jacobianIndex = -1;
  nonLinearSystemData[1].initializeStaticNLSData = initializeStaticDataNLS224;
  nonLinearSystemData[1].getIterationVars = getIterationVarsNLS224;
  nonLinearSystemData[1].checkConstraints = NULL;
  
  
  nonLinearSystemData[0].equationIndex = 61;
  nonLinearSystemData[0].size = 4;
  nonLinearSystemData[0].homotopySupport = 0;
  nonLinearSystemData[0].mixedSystem = 1;
  nonLinearSystemData[0].residualFunc = residualFunc61;
  nonLinearSystemData[0].strictTearingFunctionCall = NULL;
  nonLinearSystemData[0].analyticalJacobianColumn = NULL;
  nonLinearSystemData[0].initialAnalyticalJacobian = NULL;
  nonLinearSystemData[0].jacobianIndex = -1;
  nonLinearSystemData[0].initializeStaticNLSData = initializeStaticDataNLS61;
  nonLinearSystemData[0].getIterationVars = getIterationVarsNLS61;
  nonLinearSystemData[0].checkConstraints = NULL;
}

#if defined(__cplusplus)
}
#endif

