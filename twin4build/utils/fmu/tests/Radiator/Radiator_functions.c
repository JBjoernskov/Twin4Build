#include "omc_simulation_settings.h"
#include "Radiator_functions.h"
#ifdef __cplusplus
extern "C" {
#endif

#include "Radiator_includes.h"


DLLExport
modelica_real omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData_t *threadData, modelica_real _x, modelica_real _n, modelica_real _delta)
{
  modelica_real _y;
  modelica_real _a1;
  modelica_real _a3;
  modelica_real _a5;
  modelica_real _delta2;
  modelica_real _x2;
  modelica_real _y_d;
  modelica_real _yP_d;
  modelica_real _yPP_d;
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
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_real tmp24;
  modelica_real tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  static int tmp33 = 0;
  _tailrecursive: OMC_LABEL_UNUSED
  // _y has no default value.
  // _a1 has no default value.
  // _a3 has no default value.
  // _a5 has no default value.
  // _delta2 has no default value.
  // _x2 has no default value.
  // _y_d has no default value.
  // _yP_d has no default value.
  // _yPP_d has no default value.
  if((fabs(_x) > _delta))
  {
    tmp1 = fabs(_x);
    tmp2 = _n;
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
    _y = tmp3;
  }
  else
  {
    _delta2 = (_delta) * (_delta);

    _x2 = (_x) * (_x);

    tmp8 = _delta;
    tmp9 = _n;
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
    _y_d = tmp10;

    tmp15 = _delta;
    tmp16 = _n - 1.0;
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
    _yP_d = (_n) * (tmp17);

    tmp22 = _delta;
    tmp23 = _n - 2.0;
    if(tmp22 < 0.0 && tmp23 != 0.0)
    {
      tmp25 = modf(tmp23, &tmp26);
      
      if(tmp25 > 0.5)
      {
        tmp25 -= 1.0;
        tmp26 += 1.0;
      }
      else if(tmp25 < -0.5)
      {
        tmp25 += 1.0;
        tmp26 -= 1.0;
      }
      
      if(fabs(tmp25) < 1e-10)
        tmp24 = pow(tmp22, tmp26);
      else
      {
        tmp28 = modf(1.0/tmp23, &tmp27);
        if(tmp28 > 0.5)
        {
          tmp28 -= 1.0;
          tmp27 += 1.0;
        }
        else if(tmp28 < -0.5)
        {
          tmp28 += 1.0;
          tmp27 -= 1.0;
        }
        if(fabs(tmp28) < 1e-10 && ((unsigned long)tmp27 & 1))
        {
          tmp24 = -pow(-tmp22, tmp25)*pow(tmp22, tmp26);
        }
        else
        {
          throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp22, tmp23);
        }
      }
    }
    else
    {
      tmp24 = pow(tmp22, tmp23);
    }
    if(isnan(tmp24) || isinf(tmp24))
    {
      throwStreamPrint(threadData, "%s:%d: Invalid root: (%g)^(%g)", __FILE__, __LINE__, tmp22, tmp23);
    }
    _yPP_d = ((_n) * (_n - 1.0)) * (tmp24);

    tmp29 = _delta;
    if (tmp29 == 0) {throwStreamPrint(threadData, "Division by zero %s", "yP_d / delta");}
    tmp30 = _delta2;
    if (tmp30 == 0) {throwStreamPrint(threadData, "Division by zero %s", "(yP_d / delta - yPP_d) / delta2");}
    tmp31 = 8.0;
    if (tmp31 == 0) {throwStreamPrint(threadData, "Division by zero %s", "(yP_d / delta - yPP_d) / delta2 / 8.0");}
    _a1 = (-((((_yP_d) / tmp29 - _yPP_d) / tmp30) / tmp31));

    tmp32 = 2.0;
    if (tmp32 == 0) {throwStreamPrint(threadData, "Division by zero %s", "(yPP_d - 12.0 * a1 * delta2) / 2.0");}
    _a3 = (_yPP_d - (((12.0) * (_a1)) * (_delta2))) / tmp32;

    _a5 = _y_d - ((_delta2) * (_a3 + (_delta2) * (_a1)));

    _y = _a5 + (_x2) * (_a3 + (_x2) * (_a1));

    {
      if(!(((_a5 > 0.0) && (0.0 < _n)) && (_n < 2.0)))
      {
        {
          FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Utilities/Math/Functions/regNonZeroPower.mo",32,4,32,102,0};
          omc_assert(threadData, info, MMC_STRINGDATA(_OMC_LIT0));
        }
      }
    }
  }
  _return: OMC_LABEL_UNUSED
  return _y;
}
modelica_metatype boxptr_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData_t *threadData, modelica_metatype _x, modelica_metatype _n, modelica_metatype _delta)
{
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _y;
  modelica_metatype out_y;
  tmp1 = mmc_unbox_real(_x);
  tmp2 = mmc_unbox_real(_n);
  tmp3 = mmc_unbox_real(_delta);
  _y = omc_Buildings_Utilities_Math_Functions_regNonZeroPower(threadData, tmp1, tmp2, tmp3);
  out_y = mmc_mk_rcon(_y);
  return out_y;
}

DLLExport
void omc_Modelica_Fluid_Utilities_checkBoundary(threadData_t *threadData, modelica_string _mediumName, string_array _substanceNames, modelica_boolean _singleState, modelica_boolean _define_p, real_array _X_boundary, modelica_string _modelName)
{
  modelica_integer _nX;
  modelica_integer tmp1;
  modelica_string _X_str = NULL;
  static int tmp2 = 0;
  modelica_string tmp3;
  modelica_string tmp4;
  static int tmp5 = 0;
  modelica_integer tmp6;
  modelica_integer tmp7;
  modelica_integer tmp8;
  modelica_string tmp9;
  modelica_string tmp10;
  modelica_integer tmp11;
  modelica_integer tmp12;
  modelica_integer tmp13;
  modelica_string tmp14;
  modelica_metatype tmpMeta[8] __attribute__((unused)) = {0};
  _tailrecursive: OMC_LABEL_UNUSED
  tmp1 = size_of_dimension_base_array(_X_boundary, ((modelica_integer) 1));
  _nX = tmp1;
  // _X_str has no default value.
  {
    if(!((!_singleState) || (_singleState && _define_p)))
    {
      tmpMeta[0] = stringAppend(_OMC_LIT2,_modelName);
      tmpMeta[1] = stringAppend(tmpMeta[0],_OMC_LIT3);
      tmpMeta[2] = stringAppend(tmpMeta[1],_mediumName);
      tmpMeta[3] = stringAppend(tmpMeta[2],_OMC_LIT4);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Utilities.mo",18,5,23,3,0};
        omc_assert(threadData, info, MMC_STRINGDATA(tmpMeta[3]));
      }
    }
  }

  tmp6 = ((modelica_integer) 1); tmp7 = 1; tmp8 = _nX;
  if(!(((tmp7 > 0) && (tmp6 > tmp8)) || ((tmp7 < 0) && (tmp6 < tmp8))))
  {
    modelica_integer _i;
    for(_i = ((modelica_integer) 1); in_range_integer(_i, tmp6, tmp8); _i += tmp7)
    {
      {
        if(!(real_array_get1(_X_boundary, 1, _i) >= 0.0))
        {
          tmpMeta[0] = stringAppend(_OMC_LIT5,_mediumName);
          tmpMeta[1] = stringAppend(tmpMeta[0],_OMC_LIT6);
          tmpMeta[2] = stringAppend(tmpMeta[1],_modelName);
          tmpMeta[3] = stringAppend(tmpMeta[2],_OMC_LIT7);
          tmp3 = modelica_integer_to_modelica_string(_i, ((modelica_integer) 0), 1);
          tmpMeta[4] = stringAppend(tmpMeta[3],tmp3);
          tmpMeta[5] = stringAppend(tmpMeta[4],_OMC_LIT8);
          tmp4 = modelica_real_to_modelica_string(real_array_get1(_X_boundary, 1, _i), ((modelica_integer) 6), ((modelica_integer) 0), 1);
          tmpMeta[6] = stringAppend(tmpMeta[5],tmp4);
          tmpMeta[7] = stringAppend(tmpMeta[6],_OMC_LIT9);
          {
            FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Utilities.mo",26,7,33,3,0};
            omc_assert(threadData, info, MMC_STRINGDATA(tmpMeta[7]));
          }
        }
      }
    }
  }

  if(((_nX > ((modelica_integer) 0)) && (fabs(sum_real_array(_X_boundary) - 1.0) > 1e-10)))
  {
    _X_str = _OMC_LIT10;

    tmp11 = ((modelica_integer) 1); tmp12 = 1; tmp13 = _nX;
    if(!(((tmp12 > 0) && (tmp11 > tmp13)) || ((tmp12 < 0) && (tmp11 < tmp13))))
    {
      modelica_integer _i;
      for(_i = ((modelica_integer) 1); in_range_integer(_i, tmp11, tmp13); _i += tmp12)
      {
        tmpMeta[0] = stringAppend(_X_str,_OMC_LIT11);
        tmp9 = modelica_integer_to_modelica_string(_i, ((modelica_integer) 0), 1);
        tmpMeta[1] = stringAppend(tmpMeta[0],tmp9);
        tmpMeta[2] = stringAppend(tmpMeta[1],_OMC_LIT12);
        tmp10 = modelica_real_to_modelica_string(real_array_get1(_X_boundary, 1, _i), ((modelica_integer) 6), ((modelica_integer) 0), 1);
        tmpMeta[3] = stringAppend(tmpMeta[2],tmp10);
        tmpMeta[4] = stringAppend(tmpMeta[3],_OMC_LIT13);
        tmpMeta[5] = stringAppend(tmpMeta[4],string_array_get1(_substanceNames, 1, _i));
        tmpMeta[6] = stringAppend(tmpMeta[5],_OMC_LIT14);
        _X_str = tmpMeta[6];
      }
    }

    tmpMeta[0] = stringAppend(_OMC_LIT15,_mediumName);
    tmpMeta[1] = stringAppend(tmpMeta[0],_OMC_LIT6);
    tmpMeta[2] = stringAppend(tmpMeta[1],_modelName);
    tmpMeta[3] = stringAppend(tmpMeta[2],_OMC_LIT14);
    tmpMeta[4] = stringAppend(tmpMeta[3],_OMC_LIT16);
    tmp14 = modelica_real_to_modelica_string(sum_real_array(_X_boundary), ((modelica_integer) 6), ((modelica_integer) 0), 1);
    tmpMeta[5] = stringAppend(tmpMeta[4],tmp14);
    tmpMeta[6] = stringAppend(tmpMeta[5],_OMC_LIT17);
    tmpMeta[7] = stringAppend(tmpMeta[6],_X_str);
    omc_Modelica_Utilities_Streams_error(threadData, tmpMeta[7]);
  }
  _return: OMC_LABEL_UNUSED
  return;
}
void boxptr_Modelica_Fluid_Utilities_checkBoundary(threadData_t *threadData, modelica_metatype _mediumName, modelica_metatype _substanceNames, modelica_metatype _singleState, modelica_metatype _define_p, modelica_metatype _X_boundary, modelica_metatype _modelName)
{
  modelica_integer tmp1;
  modelica_integer tmp2;
  tmp1 = mmc_unbox_integer(_singleState);
  tmp2 = mmc_unbox_integer(_define_p);
  omc_Modelica_Fluid_Utilities_checkBoundary(threadData, _mediumName, *((base_array_t*)_substanceNames), tmp1, tmp2, *((base_array_t*)_X_boundary), _modelName);
  return;
}

void omc_Modelica_Utilities_Streams_error(threadData_t *threadData, modelica_string _string)
{
  ModelicaError(MMC_STRINGDATA(_string));
  return;
}

Radiator_Radiator_Medium_ThermodynamicState omc_Radiator_Radiator_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_Radiator_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_Radiator_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_Radiator_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
Radiator_Radiator_Medium_ThermodynamicState omc_Radiator_Radiator_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X)
{
  Radiator_Radiator_Medium_ThermodynamicState _state;
  Radiator_Radiator_Medium_ThermodynamicState tmp1;
  Radiator_Radiator_Medium_ThermodynamicState tmp2;
  _tailrecursive: OMC_LABEL_UNUSED
  Radiator_Radiator_Medium_ThermodynamicState_construct(threadData, _state); // _state has no default value.
  tmp2._p = _p;
  tmp2._T = _T;
  tmp1 = tmp2;
  Radiator_Radiator_Medium_ThermodynamicState_copy(tmp1, _state);;
  _return: OMC_LABEL_UNUSED
  return _state;
}
modelica_metatype boxptr_Radiator_Radiator_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  Radiator_Radiator_Medium_ThermodynamicState _state;
  modelica_metatype out_state;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_T);
  _state = omc_Radiator_Radiator_Medium_setState__pTX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  tmpMeta[0] = mmc_mk_rcon(_state._p);
  tmpMeta[1] = mmc_mk_rcon(_state._T);
  out_state = mmc_mk_box3(3, &Radiator_Radiator_Medium_ThermodynamicState__desc, tmpMeta[0], tmpMeta[1]);
  return out_state;
}

DLLExport
modelica_real omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData_t *threadData, Radiator_Radiator_Medium_ThermodynamicState _state)
{
  modelica_real _cp;
  _tailrecursive: OMC_LABEL_UNUSED
  // _cp has no default value.
  _cp = 4184.0;
  _return: OMC_LABEL_UNUSED
  return _cp;
}
modelica_metatype boxptr_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_Radiator_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _cp;
  modelica_metatype out_cp;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _cp = omc_Radiator_Radiator_Medium_specificHeatCapacityCp(threadData, tmp1);
  out_cp = mmc_mk_rcon(_cp);
  return out_cp;
}

Radiator_Radiator_res_Medium_ThermodynamicState omc_Radiator_Radiator_res_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_Radiator_res_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_Radiator_res_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_Radiator_res_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
modelica_real omc_Radiator_Radiator_res_Medium_dynamicViscosity(threadData_t *threadData, Radiator_Radiator_res_Medium_ThermodynamicState _state)
{
  modelica_real _eta;
  _tailrecursive: OMC_LABEL_UNUSED
  // _eta has no default value.
  _eta = 0.001;
  _return: OMC_LABEL_UNUSED
  return _eta;
}
modelica_metatype boxptr_Radiator_Radiator_res_Medium_dynamicViscosity(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_Radiator_res_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _eta;
  modelica_metatype out_eta;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _eta = omc_Radiator_Radiator_res_Medium_dynamicViscosity(threadData, tmp1);
  out_eta = mmc_mk_rcon(_eta);
  return out_eta;
}

Radiator_Radiator_vol_Medium_ThermodynamicState omc_Radiator_Radiator_vol_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_Radiator_vol_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_Radiator_vol_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_Radiator_vol_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
modelica_real omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData_t *threadData, modelica_real _p, modelica_real _h, real_array _X)
{
  modelica_real _T;
  modelica_real tmp1;
  _tailrecursive: OMC_LABEL_UNUSED
  // _T has no default value.
  tmp1 = 4184.0;
  if (tmp1 == 0) {throwStreamPrint(threadData, "Division by zero %s", "h / 4184.0");}
  _T = 273.15 + (_h) / tmp1;
  _return: OMC_LABEL_UNUSED
  return _T;
}
modelica_metatype boxptr_Radiator_Radiator_vol_Medium_temperature__phX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _h, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real _T;
  modelica_metatype out_T;
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_h);
  _T = omc_Radiator_Radiator_vol_Medium_temperature__phX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  out_T = mmc_mk_rcon(_T);
  return out_T;
}

Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState omc_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_density(threadData_t *threadData, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state)
{
  modelica_real _d;
  _tailrecursive: OMC_LABEL_UNUSED
  // _d has no default value.
  _d = 995.586;
  _return: OMC_LABEL_UNUSED
  return _d;
}
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_density(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _d;
  modelica_metatype out_d;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _d = omc_Radiator_Radiator_vol_dynBal_Medium_density(threadData, tmp1);
  out_d = mmc_mk_rcon(_d);
  return out_d;
}

DLLExport
Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X)
{
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state;
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState tmp1;
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState tmp2;
  _tailrecursive: OMC_LABEL_UNUSED
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_construct(threadData, _state); // _state has no default value.
  tmp2._p = _p;
  tmp2._T = _T;
  tmp1 = tmp2;
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState_copy(tmp1, _state);;
  _return: OMC_LABEL_UNUSED
  return _state;
}
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state;
  modelica_metatype out_state;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_T);
  _state = omc_Radiator_Radiator_vol_dynBal_Medium_setState__pTX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  tmpMeta[0] = mmc_mk_rcon(_state._p);
  tmpMeta[1] = mmc_mk_rcon(_state._T);
  out_state = mmc_mk_box3(3, &Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState__desc, tmpMeta[0], tmpMeta[1]);
  return out_state;
}

DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X)
{
  modelica_real _h;
  _tailrecursive: OMC_LABEL_UNUSED
  // _h has no default value.
  _h = (4184.0) * (_T - 273.15);
  _return: OMC_LABEL_UNUSED
  return _h;
}
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_real _h;
  modelica_metatype out_h;
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_T);
  _h = omc_Radiator_Radiator_vol_dynBal_Medium_specificEnthalpy__pTX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  out_h = mmc_mk_rcon(_h);
  return out_h;
}

DLLExport
modelica_real omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData_t *threadData, Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState _state)
{
  modelica_real _u;
  _tailrecursive: OMC_LABEL_UNUSED
  // _u has no default value.
  _u = (4184.0) * (_state._T - 273.15);
  _return: OMC_LABEL_UNUSED
  return _u;
}
modelica_metatype boxptr_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_Radiator_vol_dynBal_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _u;
  modelica_metatype out_u;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _u = omc_Radiator_Radiator_vol_dynBal_Medium_specificInternalEnergy(threadData, tmp1);
  out_u = mmc_mk_rcon(_u);
  return out_u;
}

Radiator_flow__sink_Medium_ThermodynamicState omc_Radiator_flow__sink_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_flow__sink_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_flow__sink_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_flow__sink_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
Radiator_flow__sink_Medium_ThermodynamicState omc_Radiator_flow__sink_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X)
{
  Radiator_flow__sink_Medium_ThermodynamicState _state;
  Radiator_flow__sink_Medium_ThermodynamicState tmp1;
  Radiator_flow__sink_Medium_ThermodynamicState tmp2;
  _tailrecursive: OMC_LABEL_UNUSED
  Radiator_flow__sink_Medium_ThermodynamicState_construct(threadData, _state); // _state has no default value.
  tmp2._p = _p;
  tmp2._T = _T;
  tmp1 = tmp2;
  Radiator_flow__sink_Medium_ThermodynamicState_copy(tmp1, _state);;
  _return: OMC_LABEL_UNUSED
  return _state;
}
modelica_metatype boxptr_Radiator_flow__sink_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  Radiator_flow__sink_Medium_ThermodynamicState _state;
  modelica_metatype out_state;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_T);
  _state = omc_Radiator_flow__sink_Medium_setState__pTX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  tmpMeta[0] = mmc_mk_rcon(_state._p);
  tmpMeta[1] = mmc_mk_rcon(_state._T);
  out_state = mmc_mk_box3(3, &Radiator_flow__sink_Medium_ThermodynamicState__desc, tmpMeta[0], tmpMeta[1]);
  return out_state;
}

DLLExport
modelica_real omc_Radiator_flow__sink_Medium_specificEnthalpy(threadData_t *threadData, Radiator_flow__sink_Medium_ThermodynamicState _state)
{
  modelica_real _h;
  _tailrecursive: OMC_LABEL_UNUSED
  // _h has no default value.
  _h = (4184.0) * (_state._T - 273.15);
  _return: OMC_LABEL_UNUSED
  return _h;
}
modelica_metatype boxptr_Radiator_flow__sink_Medium_specificEnthalpy(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_flow__sink_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _h;
  modelica_metatype out_h;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _h = omc_Radiator_flow__sink_Medium_specificEnthalpy(threadData, tmp1);
  out_h = mmc_mk_rcon(_h);
  return out_h;
}

Radiator_flow__source_Medium_ThermodynamicState omc_Radiator_flow__source_Medium_ThermodynamicState(threadData_t *threadData, modelica_real omc_p, modelica_real omc_T)
{
  Radiator_flow__source_Medium_ThermodynamicState tmp1;
  tmp1._p = omc_p;
  tmp1._T = omc_T;
  return tmp1;
}

modelica_metatype boxptr_Radiator_flow__source_Medium_ThermodynamicState(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T)
{
  return mmc_mk_box3(3, &Radiator_flow__source_Medium_ThermodynamicState__desc, _p, _T);
}

DLLExport
Radiator_flow__source_Medium_ThermodynamicState omc_Radiator_flow__source_Medium_setState__pTX(threadData_t *threadData, modelica_real _p, modelica_real _T, real_array _X)
{
  Radiator_flow__source_Medium_ThermodynamicState _state;
  Radiator_flow__source_Medium_ThermodynamicState tmp1;
  Radiator_flow__source_Medium_ThermodynamicState tmp2;
  _tailrecursive: OMC_LABEL_UNUSED
  Radiator_flow__source_Medium_ThermodynamicState_construct(threadData, _state); // _state has no default value.
  tmp2._p = _p;
  tmp2._T = _T;
  tmp1 = tmp2;
  Radiator_flow__source_Medium_ThermodynamicState_copy(tmp1, _state);;
  _return: OMC_LABEL_UNUSED
  return _state;
}
modelica_metatype boxptr_Radiator_flow__source_Medium_setState__pTX(threadData_t *threadData, modelica_metatype _p, modelica_metatype _T, modelica_metatype _X)
{
  modelica_real tmp1;
  modelica_real tmp2;
  Radiator_flow__source_Medium_ThermodynamicState _state;
  modelica_metatype out_state;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmp1 = mmc_unbox_real(_p);
  tmp2 = mmc_unbox_real(_T);
  _state = omc_Radiator_flow__source_Medium_setState__pTX(threadData, tmp1, tmp2, *((base_array_t*)_X));
  tmpMeta[0] = mmc_mk_rcon(_state._p);
  tmpMeta[1] = mmc_mk_rcon(_state._T);
  out_state = mmc_mk_box3(3, &Radiator_flow__source_Medium_ThermodynamicState__desc, tmpMeta[0], tmpMeta[1]);
  return out_state;
}

DLLExport
modelica_real omc_Radiator_flow__source_Medium_specificEnthalpy(threadData_t *threadData, Radiator_flow__source_Medium_ThermodynamicState _state)
{
  modelica_real _h;
  _tailrecursive: OMC_LABEL_UNUSED
  // _h has no default value.
  _h = (4184.0) * (_state._T - 273.15);
  _return: OMC_LABEL_UNUSED
  return _h;
}
modelica_metatype boxptr_Radiator_flow__source_Medium_specificEnthalpy(threadData_t *threadData, modelica_metatype _state)
{
  Radiator_flow__source_Medium_ThermodynamicState tmp1;
  modelica_real tmp2;
  modelica_real tmp3;
  modelica_real _h;
  modelica_metatype out_h;
  modelica_metatype tmpMeta[2] __attribute__((unused)) = {0};
  tmpMeta[0] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 2)));
  tmp2 = mmc_unbox_real(tmpMeta[0]);
  tmp1._p = tmp2;
  tmpMeta[1] = (MMC_FETCH(MMC_OFFSET(MMC_UNTAGPTR(_state), 3)));
  tmp3 = mmc_unbox_real(tmpMeta[1]);
  tmp1._T = tmp3;
  _h = omc_Radiator_flow__source_Medium_specificEnthalpy(threadData, tmp1);
  out_h = mmc_mk_rcon(_h);
  return out_h;
}

#ifdef __cplusplus
}
#endif
