/* Asserts */
#include "Radiator_model.h"
#if defined(__cplusplus)
extern "C" {
#endif


/*
equation index: 1107
type: ALGORITHM

  assert(flow_source.ports[1].h_outflow >= -10000000000.0 and flow_source.ports[1].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= flow_source.ports[1].h_outflow <= 10000000000.0, has value: " + String(flow_source.ports[1].h_outflow, "g"));
*/
void Radiator_eqFunction_1107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1107};
  modelica_boolean tmp0;
  modelica_boolean tmp1;
  static const MMC_DEFSTRINGLIT(tmp2,117,"Variable violating min/max constraint: -10000000000.0 <= flow_source.ports[1].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp3;
  static int tmp4 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp4)
  {
    tmp0 = GreaterEq(data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */,-10000000000.0);
    tmp1 = LessEq(data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */,10000000000.0);
    if(!(tmp0 && tmp1))
    {
      tmp3 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[147] /* flow_source.ports[1].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp2),tmp3);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_source.ports[1].h_outflow >= -10000000000.0 and flow_source.ports[1].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp4 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1108
type: ALGORITHM

  assert(T_z_source.T >= 0.0, "Variable violating min constraint: 0.0 <= T_z_source.T, has value: " + String(T_z_source.T, "g"));
*/
void Radiator_eqFunction_1108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1108};
  modelica_boolean tmp5;
  static const MMC_DEFSTRINGLIT(tmp6,67,"Variable violating min constraint: 0.0 <= T_z_source.T, has value: ");
  modelica_string tmp7;
  static int tmp8 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp8)
  {
    tmp5 = GreaterEq(data->localData[0]->realVars[138] /* T_z_source.T variable */,0.0);
    if(!tmp5)
    {
      tmp7 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[138] /* T_z_source.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp6),tmp7);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/HeatTransfer/Sources/PrescribedTemperature.mo",6,3,7,43,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nT_z_source.T >= 0.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp8 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1109
type: ALGORITHM

  assert(Radiator.port_a.h_outflow >= -10000000000.0 and Radiator.port_a.h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= Radiator.port_a.h_outflow <= 10000000000.0, has value: " + String(Radiator.port_a.h_outflow, "g"));
*/
void Radiator_eqFunction_1109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1109};
  modelica_boolean tmp9;
  modelica_boolean tmp10;
  static const MMC_DEFSTRINGLIT(tmp11,112,"Variable violating min/max constraint: -10000000000.0 <= Radiator.port_a.h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp12;
  static int tmp13 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp13)
  {
    tmp9 = GreaterEq(data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */,-10000000000.0);
    tmp10 = LessEq(data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */,10000000000.0);
    if(!(tmp9 && tmp10))
    {
      tmp12 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[41] /* Radiator.port_a.h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp11),tmp12);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.port_a.h_outflow >= -10000000000.0 and Radiator.port_a.h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp13 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1110
type: ALGORITHM

  assert(Radiator.sta_a.T >= 1.0 and Radiator.sta_a.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.sta_a.T <= 10000.0, has value: " + String(Radiator.sta_a.T, "g"));
*/
void Radiator_eqFunction_1110(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1110};
  modelica_boolean tmp14;
  modelica_boolean tmp15;
  static const MMC_DEFSTRINGLIT(tmp16,86,"Variable violating min/max constraint: 1.0 <= Radiator.sta_a.T <= 10000.0, has value: ");
  modelica_string tmp17;
  static int tmp18 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp18)
  {
    tmp14 = GreaterEq(data->localData[0]->realVars[53] /* Radiator.sta_a.T variable */,1.0);
    tmp15 = LessEq(data->localData[0]->realVars[53] /* Radiator.sta_a.T variable */,10000.0);
    if(!(tmp14 && tmp15))
    {
      tmp17 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[53] /* Radiator.sta_a.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp16),tmp17);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.sta_a.T >= 1.0 and Radiator.sta_a.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp18 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1111
type: ALGORITHM

  assert(Radiator.sta_b.T >= 1.0 and Radiator.sta_b.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.sta_b.T <= 10000.0, has value: " + String(Radiator.sta_b.T, "g"));
*/
void Radiator_eqFunction_1111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1111};
  modelica_boolean tmp19;
  modelica_boolean tmp20;
  static const MMC_DEFSTRINGLIT(tmp21,86,"Variable violating min/max constraint: 1.0 <= Radiator.sta_b.T <= 10000.0, has value: ");
  modelica_string tmp22;
  static int tmp23 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp23)
  {
    tmp19 = GreaterEq(data->localData[0]->realVars[54] /* Radiator.sta_b.T variable */,1.0);
    tmp20 = LessEq(data->localData[0]->realVars[54] /* Radiator.sta_b.T variable */,10000.0);
    if(!(tmp19 && tmp20))
    {
      tmp22 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[54] /* Radiator.sta_b.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp21),tmp22);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Media/package.mo",5870,7,5870,44,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.sta_b.T >= 1.0 and Radiator.sta_b.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp23 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1112
type: ALGORITHM

  assert(Radiator.vol[1].T >= 1.0 and Radiator.vol[1].T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].T <= 10000.0, has value: " + String(Radiator.vol[1].T, "g"));
*/
void Radiator_eqFunction_1112(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1112};
  modelica_boolean tmp24;
  modelica_boolean tmp25;
  static const MMC_DEFSTRINGLIT(tmp26,87,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].T <= 10000.0, has value: ");
  modelica_string tmp27;
  static int tmp28 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp28)
  {
    tmp24 = GreaterEq(data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */,1.0);
    tmp25 = LessEq(data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */,10000.0);
    if(!(tmp24 && tmp25))
    {
      tmp27 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[55] /* Radiator.vol[1].T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp26),tmp27);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",37,3,41,31,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].T >= 1.0 and Radiator.vol[1].T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp28 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1113
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.medium.T >= 1.0 and Radiator.vol[1].dynBal.medium.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.medium.T <= 10000.0, has value: " + String(Radiator.vol[1].dynBal.medium.T, "g"));
*/
void Radiator_eqFunction_1113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1113};
  modelica_boolean tmp29;
  modelica_boolean tmp30;
  static const MMC_DEFSTRINGLIT(tmp31,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[1].dynBal.medium.T <= 10000.0, has value: ");
  modelica_string tmp32;
  static int tmp33 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp33)
  {
    tmp29 = GreaterEq(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */,1.0);
    tmp30 = LessEq(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */,10000.0);
    if(!(tmp29 && tmp30))
    {
      tmp32 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[90] /* Radiator.vol[1].dynBal.medium.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp31),tmp32);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",22,5,24,30,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.medium.T >= 1.0 and Radiator.vol[1].dynBal.medium.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp33 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1114
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[1].dynBal.ports_H_flow[1] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[1].dynBal.ports_H_flow[1] <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.ports_H_flow[1], "g"));
*/
void Radiator_eqFunction_1114(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1114};
  modelica_boolean tmp34;
  modelica_boolean tmp35;
  static const MMC_DEFSTRINGLIT(tmp36,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[1].dynBal.ports_H_flow[1] <= 100000000.0, has value: ");
  modelica_string tmp37;
  static int tmp38 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp38)
  {
    tmp34 = GreaterEq(data->localData[0]->realVars[115] /* Radiator.vol[1].dynBal.ports_H_flow[1] variable */,-100000000.0);
    tmp35 = LessEq(data->localData[0]->realVars[115] /* Radiator.vol[1].dynBal.ports_H_flow[1] variable */,100000000.0);
    if(!(tmp34 && tmp35))
    {
      tmp37 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[115] /* Radiator.vol[1].dynBal.ports_H_flow[1] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp36),tmp37);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[1].dynBal.ports_H_flow[1] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp38 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1115
type: ALGORITHM

  assert(Radiator.vol[1].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[1].dynBal.ports_H_flow[2] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[1].dynBal.ports_H_flow[2] <= 100000000.0, has value: " + String(Radiator.vol[1].dynBal.ports_H_flow[2], "g"));
*/
void Radiator_eqFunction_1115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1115};
  modelica_boolean tmp39;
  modelica_boolean tmp40;
  static const MMC_DEFSTRINGLIT(tmp41,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[1].dynBal.ports_H_flow[2] <= 100000000.0, has value: ");
  modelica_string tmp42;
  static int tmp43 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp43)
  {
    tmp39 = GreaterEq(data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */,-100000000.0);
    tmp40 = LessEq(data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */,100000000.0);
    if(!(tmp39 && tmp40))
    {
      tmp42 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[116] /* Radiator.vol[1].dynBal.ports_H_flow[2] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp41),tmp42);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[1].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[1].dynBal.ports_H_flow[2] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp43 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1116
type: ALGORITHM

  assert(Radiator.vol[2].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[2].ports[2].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[2].ports[2].h_outflow <= 10000000000.0, has value: " + String(Radiator.vol[2].ports[2].h_outflow, "g"));
*/
void Radiator_eqFunction_1116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1116};
  modelica_boolean tmp44;
  modelica_boolean tmp45;
  static const MMC_DEFSTRINGLIT(tmp46,121,"Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[2].ports[2].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp47;
  static int tmp48 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp48)
  {
    tmp44 = GreaterEq(data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */,-10000000000.0);
    tmp45 = LessEq(data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */,10000000000.0);
    if(!(tmp44 && tmp45))
    {
      tmp47 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[130] /* Radiator.vol[2].ports[2].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp46),tmp47);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[2].ports[2].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp48 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1117
type: ALGORITHM

  assert(Radiator.vol[2].T >= 1.0 and Radiator.vol[2].T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].T <= 10000.0, has value: " + String(Radiator.vol[2].T, "g"));
*/
void Radiator_eqFunction_1117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1117};
  modelica_boolean tmp49;
  modelica_boolean tmp50;
  static const MMC_DEFSTRINGLIT(tmp51,87,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].T <= 10000.0, has value: ");
  modelica_string tmp52;
  static int tmp53 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp53)
  {
    tmp49 = GreaterEq(data->localData[0]->realVars[56] /* Radiator.vol[2].T variable */,1.0);
    tmp50 = LessEq(data->localData[0]->realVars[56] /* Radiator.vol[2].T variable */,10000.0);
    if(!(tmp49 && tmp50))
    {
      tmp52 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[56] /* Radiator.vol[2].T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp51),tmp52);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",37,3,41,31,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].T >= 1.0 and Radiator.vol[2].T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp53 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1118
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.medium.T >= 1.0 and Radiator.vol[2].dynBal.medium.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.medium.T <= 10000.0, has value: " + String(Radiator.vol[2].dynBal.medium.T, "g"));
*/
void Radiator_eqFunction_1118(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1118};
  modelica_boolean tmp54;
  modelica_boolean tmp55;
  static const MMC_DEFSTRINGLIT(tmp56,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[2].dynBal.medium.T <= 10000.0, has value: ");
  modelica_string tmp57;
  static int tmp58 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp58)
  {
    tmp54 = GreaterEq(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */,1.0);
    tmp55 = LessEq(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */,10000.0);
    if(!(tmp54 && tmp55))
    {
      tmp57 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[91] /* Radiator.vol[2].dynBal.medium.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp56),tmp57);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",22,5,24,30,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.medium.T >= 1.0 and Radiator.vol[2].dynBal.medium.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp58 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1119
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[2].dynBal.ports_H_flow[1] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[2].dynBal.ports_H_flow[1] <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.ports_H_flow[1], "g"));
*/
void Radiator_eqFunction_1119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1119};
  modelica_boolean tmp59;
  modelica_boolean tmp60;
  static const MMC_DEFSTRINGLIT(tmp61,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[2].dynBal.ports_H_flow[1] <= 100000000.0, has value: ");
  modelica_string tmp62;
  static int tmp63 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp63)
  {
    tmp59 = GreaterEq(data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */,-100000000.0);
    tmp60 = LessEq(data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */,100000000.0);
    if(!(tmp59 && tmp60))
    {
      tmp62 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[117] /* Radiator.vol[2].dynBal.ports_H_flow[1] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp61),tmp62);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[2].dynBal.ports_H_flow[1] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp63 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1120
type: ALGORITHM

  assert(Radiator.vol[2].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[2].dynBal.ports_H_flow[2] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[2].dynBal.ports_H_flow[2] <= 100000000.0, has value: " + String(Radiator.vol[2].dynBal.ports_H_flow[2], "g"));
*/
void Radiator_eqFunction_1120(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1120};
  modelica_boolean tmp64;
  modelica_boolean tmp65;
  static const MMC_DEFSTRINGLIT(tmp66,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[2].dynBal.ports_H_flow[2] <= 100000000.0, has value: ");
  modelica_string tmp67;
  static int tmp68 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp68)
  {
    tmp64 = GreaterEq(data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */,-100000000.0);
    tmp65 = LessEq(data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */,100000000.0);
    if(!(tmp64 && tmp65))
    {
      tmp67 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[118] /* Radiator.vol[2].dynBal.ports_H_flow[2] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp66),tmp67);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[2].dynBal.ports_H_flow[2] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp68 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1121
type: ALGORITHM

  assert(Radiator.vol[3].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[3].ports[2].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[3].ports[2].h_outflow <= 10000000000.0, has value: " + String(Radiator.vol[3].ports[2].h_outflow, "g"));
*/
void Radiator_eqFunction_1121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1121};
  modelica_boolean tmp69;
  modelica_boolean tmp70;
  static const MMC_DEFSTRINGLIT(tmp71,121,"Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[3].ports[2].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp72;
  static int tmp73 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp73)
  {
    tmp69 = GreaterEq(data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */,-10000000000.0);
    tmp70 = LessEq(data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */,10000000000.0);
    if(!(tmp69 && tmp70))
    {
      tmp72 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[131] /* Radiator.vol[3].ports[2].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp71),tmp72);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[3].ports[2].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp73 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1122
type: ALGORITHM

  assert(Radiator.vol[3].T >= 1.0 and Radiator.vol[3].T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].T <= 10000.0, has value: " + String(Radiator.vol[3].T, "g"));
*/
void Radiator_eqFunction_1122(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1122};
  modelica_boolean tmp74;
  modelica_boolean tmp75;
  static const MMC_DEFSTRINGLIT(tmp76,87,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].T <= 10000.0, has value: ");
  modelica_string tmp77;
  static int tmp78 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp78)
  {
    tmp74 = GreaterEq(data->localData[0]->realVars[57] /* Radiator.vol[3].T variable */,1.0);
    tmp75 = LessEq(data->localData[0]->realVars[57] /* Radiator.vol[3].T variable */,10000.0);
    if(!(tmp74 && tmp75))
    {
      tmp77 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[57] /* Radiator.vol[3].T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp76),tmp77);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",37,3,41,31,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].T >= 1.0 and Radiator.vol[3].T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp78 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1123
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.medium.T >= 1.0 and Radiator.vol[3].dynBal.medium.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.medium.T <= 10000.0, has value: " + String(Radiator.vol[3].dynBal.medium.T, "g"));
*/
void Radiator_eqFunction_1123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1123};
  modelica_boolean tmp79;
  modelica_boolean tmp80;
  static const MMC_DEFSTRINGLIT(tmp81,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[3].dynBal.medium.T <= 10000.0, has value: ");
  modelica_string tmp82;
  static int tmp83 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp83)
  {
    tmp79 = GreaterEq(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */,1.0);
    tmp80 = LessEq(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */,10000.0);
    if(!(tmp79 && tmp80))
    {
      tmp82 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[92] /* Radiator.vol[3].dynBal.medium.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp81),tmp82);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",22,5,24,30,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.medium.T >= 1.0 and Radiator.vol[3].dynBal.medium.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp83 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1124
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[3].dynBal.ports_H_flow[1] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[3].dynBal.ports_H_flow[1] <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.ports_H_flow[1], "g"));
*/
void Radiator_eqFunction_1124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1124};
  modelica_boolean tmp84;
  modelica_boolean tmp85;
  static const MMC_DEFSTRINGLIT(tmp86,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[3].dynBal.ports_H_flow[1] <= 100000000.0, has value: ");
  modelica_string tmp87;
  static int tmp88 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp88)
  {
    tmp84 = GreaterEq(data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */,-100000000.0);
    tmp85 = LessEq(data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */,100000000.0);
    if(!(tmp84 && tmp85))
    {
      tmp87 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[119] /* Radiator.vol[3].dynBal.ports_H_flow[1] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp86),tmp87);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[3].dynBal.ports_H_flow[1] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp88 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1125
type: ALGORITHM

  assert(Radiator.vol[3].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[3].dynBal.ports_H_flow[2] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[3].dynBal.ports_H_flow[2] <= 100000000.0, has value: " + String(Radiator.vol[3].dynBal.ports_H_flow[2], "g"));
*/
void Radiator_eqFunction_1125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1125};
  modelica_boolean tmp89;
  modelica_boolean tmp90;
  static const MMC_DEFSTRINGLIT(tmp91,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[3].dynBal.ports_H_flow[2] <= 100000000.0, has value: ");
  modelica_string tmp92;
  static int tmp93 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp93)
  {
    tmp89 = GreaterEq(data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */,-100000000.0);
    tmp90 = LessEq(data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */,100000000.0);
    if(!(tmp89 && tmp90))
    {
      tmp92 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[120] /* Radiator.vol[3].dynBal.ports_H_flow[2] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp91),tmp92);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[3].dynBal.ports_H_flow[2] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp93 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1126
type: ALGORITHM

  assert(Radiator.vol[4].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[4].ports[2].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[4].ports[2].h_outflow <= 10000000000.0, has value: " + String(Radiator.vol[4].ports[2].h_outflow, "g"));
*/
void Radiator_eqFunction_1126(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1126};
  modelica_boolean tmp94;
  modelica_boolean tmp95;
  static const MMC_DEFSTRINGLIT(tmp96,121,"Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[4].ports[2].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp97;
  static int tmp98 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp98)
  {
    tmp94 = GreaterEq(data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */,-10000000000.0);
    tmp95 = LessEq(data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */,10000000000.0);
    if(!(tmp94 && tmp95))
    {
      tmp97 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[132] /* Radiator.vol[4].ports[2].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp96),tmp97);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[4].ports[2].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp98 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1127
type: ALGORITHM

  assert(Radiator.vol[4].T >= 1.0 and Radiator.vol[4].T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].T <= 10000.0, has value: " + String(Radiator.vol[4].T, "g"));
*/
void Radiator_eqFunction_1127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1127};
  modelica_boolean tmp99;
  modelica_boolean tmp100;
  static const MMC_DEFSTRINGLIT(tmp101,87,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].T <= 10000.0, has value: ");
  modelica_string tmp102;
  static int tmp103 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp103)
  {
    tmp99 = GreaterEq(data->localData[0]->realVars[58] /* Radiator.vol[4].T variable */,1.0);
    tmp100 = LessEq(data->localData[0]->realVars[58] /* Radiator.vol[4].T variable */,10000.0);
    if(!(tmp99 && tmp100))
    {
      tmp102 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[58] /* Radiator.vol[4].T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp101),tmp102);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",37,3,41,31,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].T >= 1.0 and Radiator.vol[4].T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp103 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1128
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.medium.T >= 1.0 and Radiator.vol[4].dynBal.medium.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.medium.T <= 10000.0, has value: " + String(Radiator.vol[4].dynBal.medium.T, "g"));
*/
void Radiator_eqFunction_1128(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1128};
  modelica_boolean tmp104;
  modelica_boolean tmp105;
  static const MMC_DEFSTRINGLIT(tmp106,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[4].dynBal.medium.T <= 10000.0, has value: ");
  modelica_string tmp107;
  static int tmp108 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp108)
  {
    tmp104 = GreaterEq(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */,1.0);
    tmp105 = LessEq(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */,10000.0);
    if(!(tmp104 && tmp105))
    {
      tmp107 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[93] /* Radiator.vol[4].dynBal.medium.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp106),tmp107);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",22,5,24,30,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.medium.T >= 1.0 and Radiator.vol[4].dynBal.medium.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp108 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1129
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[4].dynBal.ports_H_flow[1] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[4].dynBal.ports_H_flow[1] <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.ports_H_flow[1], "g"));
*/
void Radiator_eqFunction_1129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1129};
  modelica_boolean tmp109;
  modelica_boolean tmp110;
  static const MMC_DEFSTRINGLIT(tmp111,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[4].dynBal.ports_H_flow[1] <= 100000000.0, has value: ");
  modelica_string tmp112;
  static int tmp113 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp113)
  {
    tmp109 = GreaterEq(data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */,-100000000.0);
    tmp110 = LessEq(data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */,100000000.0);
    if(!(tmp109 && tmp110))
    {
      tmp112 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[121] /* Radiator.vol[4].dynBal.ports_H_flow[1] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp111),tmp112);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[4].dynBal.ports_H_flow[1] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp113 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1130
type: ALGORITHM

  assert(Radiator.vol[4].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[4].dynBal.ports_H_flow[2] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[4].dynBal.ports_H_flow[2] <= 100000000.0, has value: " + String(Radiator.vol[4].dynBal.ports_H_flow[2], "g"));
*/
void Radiator_eqFunction_1130(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1130};
  modelica_boolean tmp114;
  modelica_boolean tmp115;
  static const MMC_DEFSTRINGLIT(tmp116,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[4].dynBal.ports_H_flow[2] <= 100000000.0, has value: ");
  modelica_string tmp117;
  static int tmp118 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp118)
  {
    tmp114 = GreaterEq(data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */,-100000000.0);
    tmp115 = LessEq(data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */,100000000.0);
    if(!(tmp114 && tmp115))
    {
      tmp117 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[122] /* Radiator.vol[4].dynBal.ports_H_flow[2] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp116),tmp117);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[4].dynBal.ports_H_flow[2] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp118 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1131
type: ALGORITHM

  assert(Radiator.vol[5].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[5].ports[2].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[5].ports[2].h_outflow <= 10000000000.0, has value: " + String(Radiator.vol[5].ports[2].h_outflow, "g"));
*/
void Radiator_eqFunction_1131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1131};
  modelica_boolean tmp119;
  modelica_boolean tmp120;
  static const MMC_DEFSTRINGLIT(tmp121,121,"Variable violating min/max constraint: -10000000000.0 <= Radiator.vol[5].ports[2].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp122;
  static int tmp123 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp123)
  {
    tmp119 = GreaterEq(data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */,-10000000000.0);
    tmp120 = LessEq(data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */,10000000000.0);
    if(!(tmp119 && tmp120))
    {
      tmp122 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[133] /* Radiator.vol[5].ports[2].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp121),tmp122);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].ports[2].h_outflow >= -10000000000.0 and Radiator.vol[5].ports[2].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp123 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1132
type: ALGORITHM

  assert(Radiator.vol[5].T >= 1.0 and Radiator.vol[5].T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].T <= 10000.0, has value: " + String(Radiator.vol[5].T, "g"));
*/
void Radiator_eqFunction_1132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1132};
  modelica_boolean tmp124;
  modelica_boolean tmp125;
  static const MMC_DEFSTRINGLIT(tmp126,87,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].T <= 10000.0, has value: ");
  modelica_string tmp127;
  static int tmp128 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp128)
  {
    tmp124 = GreaterEq(data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */,1.0);
    tmp125 = LessEq(data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */,10000.0);
    if(!(tmp124 && tmp125))
    {
      tmp127 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[59] /* Radiator.vol[5].T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp126),tmp127);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/MixingVolumes/BaseClasses/PartialMixingVolume.mo",37,3,41,31,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].T >= 1.0 and Radiator.vol[5].T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp128 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1133
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.medium.T >= 1.0 and Radiator.vol[5].dynBal.medium.T <= 10000.0, "Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.medium.T <= 10000.0, has value: " + String(Radiator.vol[5].dynBal.medium.T, "g"));
*/
void Radiator_eqFunction_1133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1133};
  modelica_boolean tmp129;
  modelica_boolean tmp130;
  static const MMC_DEFSTRINGLIT(tmp131,101,"Variable violating min/max constraint: 1.0 <= Radiator.vol[5].dynBal.medium.T <= 10000.0, has value: ");
  modelica_string tmp132;
  static int tmp133 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp133)
  {
    tmp129 = GreaterEq(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */,1.0);
    tmp130 = LessEq(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */,10000.0);
    if(!(tmp129 && tmp130))
    {
      tmp132 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[94] /* Radiator.vol[5].dynBal.medium.T variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp131),tmp132);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Media/Water.mo",22,5,24,30,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.medium.T >= 1.0 and Radiator.vol[5].dynBal.medium.T <= 10000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp133 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1134
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[5].dynBal.ports_H_flow[1] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[5].dynBal.ports_H_flow[1] <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.ports_H_flow[1], "g"));
*/
void Radiator_eqFunction_1134(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1134};
  modelica_boolean tmp134;
  modelica_boolean tmp135;
  static const MMC_DEFSTRINGLIT(tmp136,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[5].dynBal.ports_H_flow[1] <= 100000000.0, has value: ");
  modelica_string tmp137;
  static int tmp138 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp138)
  {
    tmp134 = GreaterEq(data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */,-100000000.0);
    tmp135 = LessEq(data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */,100000000.0);
    if(!(tmp134 && tmp135))
    {
      tmp137 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[123] /* Radiator.vol[5].dynBal.ports_H_flow[1] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp136),tmp137);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.ports_H_flow[1] >= -100000000.0 and Radiator.vol[5].dynBal.ports_H_flow[1] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp138 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1135
type: ALGORITHM

  assert(Radiator.vol[5].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[5].dynBal.ports_H_flow[2] <= 100000000.0, "Variable violating min/max constraint: -100000000.0 <= Radiator.vol[5].dynBal.ports_H_flow[2] <= 100000000.0, has value: " + String(Radiator.vol[5].dynBal.ports_H_flow[2], "g"));
*/
void Radiator_eqFunction_1135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1135};
  modelica_boolean tmp139;
  modelica_boolean tmp140;
  static const MMC_DEFSTRINGLIT(tmp141,121,"Variable violating min/max constraint: -100000000.0 <= Radiator.vol[5].dynBal.ports_H_flow[2] <= 100000000.0, has value: ");
  modelica_string tmp142;
  static int tmp143 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp143)
  {
    tmp139 = GreaterEq(data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */,-100000000.0);
    tmp140 = LessEq(data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */,100000000.0);
    if(!(tmp139 && tmp140))
    {
      tmp142 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[124] /* Radiator.vol[5].dynBal.ports_H_flow[2] variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp141),tmp142);
      {
        FILE_INFO info = {"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/BuildingsLibrary/Buildings 9.1.0/Fluid/Interfaces/ConservationEquation.mo",125,3,125,47,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].dynBal.ports_H_flow[2] >= -100000000.0 and Radiator.vol[5].dynBal.ports_H_flow[2] <= 100000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp143 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1136
type: ALGORITHM

  assert(flow_sink.ports[2].h_outflow >= -10000000000.0 and flow_sink.ports[2].h_outflow <= 10000000000.0, "Variable violating min/max constraint: -10000000000.0 <= flow_sink.ports[2].h_outflow <= 10000000000.0, has value: " + String(flow_sink.ports[2].h_outflow, "g"));
*/
void Radiator_eqFunction_1136(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1136};
  modelica_boolean tmp144;
  modelica_boolean tmp145;
  static const MMC_DEFSTRINGLIT(tmp146,115,"Variable violating min/max constraint: -10000000000.0 <= flow_sink.ports[2].h_outflow <= 10000000000.0, has value: ");
  modelica_string tmp147;
  static int tmp148 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp148)
  {
    tmp144 = GreaterEq(data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */,-10000000000.0);
    tmp145 = LessEq(data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */,10000000000.0);
    if(!(tmp144 && tmp145))
    {
      tmp147 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[142] /* flow_sink.ports[2].h_outflow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp146),tmp147);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",16,5,17,84,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.ports[2].h_outflow >= -10000000000.0 and flow_sink.ports[2].h_outflow <= 10000000000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp148 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1106
type: ALGORITHM

  assert(Radiator.vol[2].ports[1].m_flow >= -100000.0 and Radiator.vol[2].ports[1].m_flow <= 100000.0, "Variable violating min/max constraint: -100000.0 <= Radiator.vol[2].ports[1].m_flow <= 100000.0, has value: " + String(Radiator.vol[2].ports[1].m_flow, "g"));
*/
void Radiator_eqFunction_1106(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1106};
  modelica_boolean tmp149;
  modelica_boolean tmp150;
  static const MMC_DEFSTRINGLIT(tmp151,108,"Variable violating min/max constraint: -100000.0 <= Radiator.vol[2].ports[1].m_flow <= 100000.0, has value: ");
  modelica_string tmp152;
  static int tmp153 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp153)
  {
    tmp149 = GreaterEq(data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */,-100000.0);
    tmp150 = LessEq(data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */,100000.0);
    if(!(tmp149 && tmp150))
    {
      tmp152 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[134] /* Radiator.vol[2].ports[1].m_flow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp151),tmp152);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",13,5,14,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[2].ports[1].m_flow >= -100000.0 and Radiator.vol[2].ports[1].m_flow <= 100000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp153 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1105
type: ALGORITHM

  assert(Radiator.vol[3].ports[1].m_flow >= -100000.0 and Radiator.vol[3].ports[1].m_flow <= 100000.0, "Variable violating min/max constraint: -100000.0 <= Radiator.vol[3].ports[1].m_flow <= 100000.0, has value: " + String(Radiator.vol[3].ports[1].m_flow, "g"));
*/
void Radiator_eqFunction_1105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1105};
  modelica_boolean tmp154;
  modelica_boolean tmp155;
  static const MMC_DEFSTRINGLIT(tmp156,108,"Variable violating min/max constraint: -100000.0 <= Radiator.vol[3].ports[1].m_flow <= 100000.0, has value: ");
  modelica_string tmp157;
  static int tmp158 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp158)
  {
    tmp154 = GreaterEq(data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */,-100000.0);
    tmp155 = LessEq(data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */,100000.0);
    if(!(tmp154 && tmp155))
    {
      tmp157 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[135] /* Radiator.vol[3].ports[1].m_flow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp156),tmp157);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",13,5,14,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[3].ports[1].m_flow >= -100000.0 and Radiator.vol[3].ports[1].m_flow <= 100000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp158 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1104
type: ALGORITHM

  assert(Radiator.vol[4].ports[1].m_flow >= -100000.0 and Radiator.vol[4].ports[1].m_flow <= 100000.0, "Variable violating min/max constraint: -100000.0 <= Radiator.vol[4].ports[1].m_flow <= 100000.0, has value: " + String(Radiator.vol[4].ports[1].m_flow, "g"));
*/
void Radiator_eqFunction_1104(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1104};
  modelica_boolean tmp159;
  modelica_boolean tmp160;
  static const MMC_DEFSTRINGLIT(tmp161,108,"Variable violating min/max constraint: -100000.0 <= Radiator.vol[4].ports[1].m_flow <= 100000.0, has value: ");
  modelica_string tmp162;
  static int tmp163 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp163)
  {
    tmp159 = GreaterEq(data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */,-100000.0);
    tmp160 = LessEq(data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */,100000.0);
    if(!(tmp159 && tmp160))
    {
      tmp162 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[136] /* Radiator.vol[4].ports[1].m_flow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp161),tmp162);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",13,5,14,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[4].ports[1].m_flow >= -100000.0 and Radiator.vol[4].ports[1].m_flow <= 100000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp163 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1103
type: ALGORITHM

  assert(Radiator.vol[5].ports[1].m_flow >= -100000.0 and Radiator.vol[5].ports[1].m_flow <= 100000.0, "Variable violating min/max constraint: -100000.0 <= Radiator.vol[5].ports[1].m_flow <= 100000.0, has value: " + String(Radiator.vol[5].ports[1].m_flow, "g"));
*/
void Radiator_eqFunction_1103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1103};
  modelica_boolean tmp164;
  modelica_boolean tmp165;
  static const MMC_DEFSTRINGLIT(tmp166,108,"Variable violating min/max constraint: -100000.0 <= Radiator.vol[5].ports[1].m_flow <= 100000.0, has value: ");
  modelica_string tmp167;
  static int tmp168 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp168)
  {
    tmp164 = GreaterEq(data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */,-100000.0);
    tmp165 = LessEq(data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */,100000.0);
    if(!(tmp164 && tmp165))
    {
      tmp167 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[137] /* Radiator.vol[5].ports[1].m_flow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp166),tmp167);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",13,5,14,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nRadiator.vol[5].ports[1].m_flow >= -100000.0 and Radiator.vol[5].ports[1].m_flow <= 100000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp168 = 1;
    }
  }
  TRACE_POP
}

/*
equation index: 1102
type: ALGORITHM

  assert(flow_sink.ports[1].m_flow >= -100000.0 and flow_sink.ports[1].m_flow <= 100000.0, "Variable violating min/max constraint: -100000.0 <= flow_sink.ports[1].m_flow <= 100000.0, has value: " + String(flow_sink.ports[1].m_flow, "g"));
*/
void Radiator_eqFunction_1102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1102};
  modelica_boolean tmp169;
  modelica_boolean tmp170;
  static const MMC_DEFSTRINGLIT(tmp171,102,"Variable violating min/max constraint: -100000.0 <= flow_sink.ports[1].m_flow <= 100000.0, has value: ");
  modelica_string tmp172;
  static int tmp173 = 0;
  modelica_metatype tmpMeta[1] __attribute__((unused)) = {0};
  if(!tmp173)
  {
    tmp169 = GreaterEq(data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */,-100000.0);
    tmp170 = LessEq(data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */,100000.0);
    if(!(tmp169 && tmp170))
    {
      tmp172 = modelica_real_to_modelica_string_format(data->localData[0]->realVars[143] /* flow_sink.ports[1].m_flow variable */, (modelica_string) mmc_strings_len1[103]);
      tmpMeta[0] = stringAppend(MMC_REFSTRINGLIT(tmp171),tmp172);
      {
        FILE_INFO info = {"C:/Program Files/OpenModelica1.18.0-64bit/lib/omlibrary/Modelica 4.0.0/Fluid/Interfaces.mo",13,5,14,68,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f\nflow_sink.ports[1].m_flow >= -100000.0 and flow_sink.ports[1].m_flow <= 100000.0", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        omc_assert_warning_withEquationIndexes(info, equationIndexes, MMC_STRINGDATA(tmpMeta[0]));
      }
      tmp173 = 1;
    }
  }
  TRACE_POP
}
/* function to check assert after a step is done */
OMC_DISABLE_OPT
int Radiator_checkForAsserts(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  Radiator_eqFunction_1107(data, threadData);

  Radiator_eqFunction_1108(data, threadData);

  Radiator_eqFunction_1109(data, threadData);

  Radiator_eqFunction_1110(data, threadData);

  Radiator_eqFunction_1111(data, threadData);

  Radiator_eqFunction_1112(data, threadData);

  Radiator_eqFunction_1113(data, threadData);

  Radiator_eqFunction_1114(data, threadData);

  Radiator_eqFunction_1115(data, threadData);

  Radiator_eqFunction_1116(data, threadData);

  Radiator_eqFunction_1117(data, threadData);

  Radiator_eqFunction_1118(data, threadData);

  Radiator_eqFunction_1119(data, threadData);

  Radiator_eqFunction_1120(data, threadData);

  Radiator_eqFunction_1121(data, threadData);

  Radiator_eqFunction_1122(data, threadData);

  Radiator_eqFunction_1123(data, threadData);

  Radiator_eqFunction_1124(data, threadData);

  Radiator_eqFunction_1125(data, threadData);

  Radiator_eqFunction_1126(data, threadData);

  Radiator_eqFunction_1127(data, threadData);

  Radiator_eqFunction_1128(data, threadData);

  Radiator_eqFunction_1129(data, threadData);

  Radiator_eqFunction_1130(data, threadData);

  Radiator_eqFunction_1131(data, threadData);

  Radiator_eqFunction_1132(data, threadData);

  Radiator_eqFunction_1133(data, threadData);

  Radiator_eqFunction_1134(data, threadData);

  Radiator_eqFunction_1135(data, threadData);

  Radiator_eqFunction_1136(data, threadData);

  Radiator_eqFunction_1106(data, threadData);

  Radiator_eqFunction_1105(data, threadData);

  Radiator_eqFunction_1104(data, threadData);

  Radiator_eqFunction_1103(data, threadData);

  Radiator_eqFunction_1102(data, threadData);
  
  TRACE_POP
  return 0;
}

#if defined(__cplusplus)
}
#endif

