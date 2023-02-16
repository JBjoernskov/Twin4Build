/* Linearization */
#include "Radiator_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
const char *Radiator_linear_model_frame()
{
  return "model linearized_model \"Radiator\" \n  parameter Integer n = 6 \"number of states\";\n  parameter Integer m = 3 \"number of inputs\";\n  parameter Integer p = 5 \"number of outputs\";\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] =\n\t[%s];\n\n"
  "  parameter Real D[p, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "\n"
  "  Real 'x_Energy' = x[1];\n""  Real 'x_Radiator.vol[1].dynBal.U' = x[2];\n""  Real 'x_Radiator.vol[2].dynBal.U' = x[3];\n""  Real 'x_Radiator.vol[3].dynBal.U' = x[4];\n""  Real 'x_Radiator.vol[4].dynBal.U' = x[5];\n""  Real 'x_Radiator.vol[5].dynBal.U' = x[6];\n"
  "  Real 'u_indoorTemperature' = u[1];\n""  Real 'u_supplyWaterTemperature' = u[2];\n""  Real 'u_waterFlowRate' = u[3];\n"
  "  Real 'y_Energy' = y[1];\n""  Real 'y_Power' = y[2];\n""  Real 'y_Q_con' = y[3];\n""  Real 'y_Q_rad' = y[4];\n""  Real 'y_outletWaterTemperature' = y[5];\n"
  "equation\n  der(x) = A * x + B * u;\n  y = C * x + D * u;\nend linearized_model;\n";
}
const char *Radiator_linear_model_datarecovery_frame()
{
  return "model linearized_model \"Radiator\" \n parameter Integer n = 6 \"number of states\";\n  parameter Integer m = 3 \"number of inputs\";\n  parameter Integer p = 5 \"number of outputs\";\n  parameter Integer nz = 140 \"data recovery variables\";\n"
  "  parameter Real x0[6] = %s;\n"
  "  parameter Real u0[3] = %s;\n"
  "  parameter Real z0[140] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] =\n\t[%s];\n\n"
  "  parameter Real D[p, m] =\n\t[%s];\n\n"
  "  parameter Real Cz[nz, n] =\n\t[%s];\n\n"
  "  parameter Real Dz[nz, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "  output Real z[nz];\n"
  "\n"
  "  Real 'x_Energy' = x[1];\n""  Real 'x_Radiator.vol[1].dynBal.U' = x[2];\n""  Real 'x_Radiator.vol[2].dynBal.U' = x[3];\n""  Real 'x_Radiator.vol[3].dynBal.U' = x[4];\n""  Real 'x_Radiator.vol[4].dynBal.U' = x[5];\n""  Real 'x_Radiator.vol[5].dynBal.U' = x[6];\n"
  "  Real 'u_indoorTemperature' = u[1];\n""  Real 'u_supplyWaterTemperature' = u[2];\n""  Real 'u_waterFlowRate' = u[3];\n"
  "  Real 'y_Energy' = y[1];\n""  Real 'y_Power' = y[2];\n""  Real 'y_Q_con' = y[3];\n""  Real 'y_Q_rad' = y[4];\n""  Real 'y_outletWaterTemperature' = y[5];\n"
  "  Real 'z_$cse1.T' = z[1];\n""  Real 'z_$cse1.p' = z[2];\n""  Real 'z_$cse10' = z[3];\n""  Real 'z_$cse11' = z[4];\n""  Real 'z_$cse12.T' = z[5];\n""  Real 'z_$cse12.p' = z[6];\n""  Real 'z_$cse2' = z[7];\n""  Real 'z_$cse3' = z[8];\n""  Real 'z_$cse4' = z[9];\n""  Real 'z_$cse5' = z[10];\n""  Real 'z_$cse6' = z[11];\n""  Real 'z_$cse7' = z[12];\n""  Real 'z_$cse8' = z[13];\n""  Real 'z_$cse9' = z[14];\n""  Real 'z_Power' = z[15];\n""  Real 'z_Q_con' = z[16];\n""  Real 'z_Q_rad' = z[17];\n""  Real 'z_Radiator.Q_flow' = z[18];\n""  Real 'z_Radiator.dTCon[1]' = z[19];\n""  Real 'z_Radiator.dTCon[2]' = z[20];\n""  Real 'z_Radiator.dTCon[3]' = z[21];\n""  Real 'z_Radiator.dTCon[4]' = z[22];\n""  Real 'z_Radiator.dTCon[5]' = z[23];\n""  Real 'z_Radiator.dTRad[1]' = z[24];\n""  Real 'z_Radiator.dTRad[2]' = z[25];\n""  Real 'z_Radiator.dTRad[3]' = z[26];\n""  Real 'z_Radiator.dTRad[4]' = z[27];\n""  Real 'z_Radiator.dTRad[5]' = z[28];\n""  Real 'z_Radiator.dp' = z[29];\n""  Real 'z_Radiator.port_a.h_outflow' = z[30];\n""  Real 'z_Radiator.preCon[1].Q_flow' = z[31];\n""  Real 'z_Radiator.preCon[2].Q_flow' = z[32];\n""  Real 'z_Radiator.preCon[3].Q_flow' = z[33];\n""  Real 'z_Radiator.preCon[4].Q_flow' = z[34];\n""  Real 'z_Radiator.preCon[5].Q_flow' = z[35];\n""  Real 'z_Radiator.preRad[1].Q_flow' = z[36];\n""  Real 'z_Radiator.preRad[2].Q_flow' = z[37];\n""  Real 'z_Radiator.preRad[3].Q_flow' = z[38];\n""  Real 'z_Radiator.preRad[4].Q_flow' = z[39];\n""  Real 'z_Radiator.preRad[5].Q_flow' = z[40];\n""  Real 'z_Radiator.res.dp' = z[41];\n""  Real 'z_Radiator.sta_a.T' = z[42];\n""  Real 'z_Radiator.sta_b.T' = z[43];\n""  Real 'z_Radiator.vol[1].T' = z[44];\n""  Real 'z_Radiator.vol[2].T' = z[45];\n""  Real 'z_Radiator.vol[3].T' = z[46];\n""  Real 'z_Radiator.vol[4].T' = z[47];\n""  Real 'z_Radiator.vol[5].T' = z[48];\n""  Real 'z_Radiator.vol[1].dynBal.Hb_flow' = z[49];\n""  Real 'z_Radiator.vol[2].dynBal.Hb_flow' = z[50];\n""  Real 'z_Radiator.vol[3].dynBal.Hb_flow' = z[51];\n""  Real 'z_Radiator.vol[4].dynBal.Hb_flow' = z[52];\n""  Real 'z_Radiator.vol[5].dynBal.Hb_flow' = z[53];\n""  Real 'z_Radiator.vol[1].dynBal.m' = z[54];\n""  Real 'z_Radiator.vol[2].dynBal.m' = z[55];\n""  Real 'z_Radiator.vol[3].dynBal.m' = z[56];\n""  Real 'z_Radiator.vol[4].dynBal.m' = z[57];\n""  Real 'z_Radiator.vol[5].dynBal.m' = z[58];\n""  Real 'z_Radiator.vol[1].dynBal.mWat_flow_internal' = z[59];\n""  Real 'z_Radiator.vol[2].dynBal.mWat_flow_internal' = z[60];\n""  Real 'z_Radiator.vol[3].dynBal.mWat_flow_internal' = z[61];\n""  Real 'z_Radiator.vol[4].dynBal.mWat_flow_internal' = z[62];\n""  Real 'z_Radiator.vol[5].dynBal.mWat_flow_internal' = z[63];\n""  Real 'z_Radiator.vol[1].dynBal.mb_flow' = z[64];\n""  Real 'z_Radiator.vol[2].dynBal.mb_flow' = z[65];\n""  Real 'z_Radiator.vol[3].dynBal.mb_flow' = z[66];\n""  Real 'z_Radiator.vol[4].dynBal.mb_flow' = z[67];\n""  Real 'z_Radiator.vol[5].dynBal.mb_flow' = z[68];\n""  Real 'z_Radiator.vol[1].dynBal.medium.MM' = z[69];\n""  Real 'z_Radiator.vol[2].dynBal.medium.MM' = z[70];\n""  Real 'z_Radiator.vol[3].dynBal.medium.MM' = z[71];\n""  Real 'z_Radiator.vol[4].dynBal.medium.MM' = z[72];\n""  Real 'z_Radiator.vol[5].dynBal.medium.MM' = z[73];\n""  Real 'z_Radiator.vol[1].dynBal.medium.R_s' = z[74];\n""  Real 'z_Radiator.vol[2].dynBal.medium.R_s' = z[75];\n""  Real 'z_Radiator.vol[3].dynBal.medium.R_s' = z[76];\n""  Real 'z_Radiator.vol[4].dynBal.medium.R_s' = z[77];\n""  Real 'z_Radiator.vol[5].dynBal.medium.R_s' = z[78];\n""  Real 'z_Radiator.vol[1].dynBal.medium.T' = z[79];\n""  Real 'z_Radiator.vol[2].dynBal.medium.T' = z[80];\n""  Real 'z_Radiator.vol[3].dynBal.medium.T' = z[81];\n""  Real 'z_Radiator.vol[4].dynBal.medium.T' = z[82];\n""  Real 'z_Radiator.vol[5].dynBal.medium.T' = z[83];\n""  Real 'z_Radiator.vol[1].dynBal.medium.T_degC' = z[84];\n""  Real 'z_Radiator.vol[2].dynBal.medium.T_degC' = z[85];\n""  Real 'z_Radiator.vol[3].dynBal.medium.T_degC' = z[86];\n""  Real 'z_Radiator.vol[4].dynBal.medium.T_degC' = z[87];\n""  Real 'z_Radiator.vol[5].dynBal.medium.T_degC' = z[88];\n""  Real 'z_Radiator.vol[1].dynBal.medium.X[1]' = z[89];\n""  Real 'z_Radiator.vol[2].dynBal.medium.X[1]' = z[90];\n""  Real 'z_Radiator.vol[3].dynBal.medium.X[1]' = z[91];\n""  Real 'z_Radiator.vol[4].dynBal.medium.X[1]' = z[92];\n""  Real 'z_Radiator.vol[5].dynBal.medium.X[1]' = z[93];\n""  Real 'z_Radiator.vol[1].dynBal.medium.d' = z[94];\n""  Real 'z_Radiator.vol[2].dynBal.medium.d' = z[95];\n""  Real 'z_Radiator.vol[3].dynBal.medium.d' = z[96];\n""  Real 'z_Radiator.vol[4].dynBal.medium.d' = z[97];\n""  Real 'z_Radiator.vol[5].dynBal.medium.d' = z[98];\n""  Real 'z_Radiator.vol[1].dynBal.medium.p_bar' = z[99];\n""  Real 'z_Radiator.vol[2].dynBal.medium.p_bar' = z[100];\n""  Real 'z_Radiator.vol[3].dynBal.medium.p_bar' = z[101];\n""  Real 'z_Radiator.vol[4].dynBal.medium.p_bar' = z[102];\n""  Real 'z_Radiator.vol[5].dynBal.medium.p_bar' = z[103];\n""  Real 'z_Radiator.vol[1].dynBal.ports_H_flow[1]' = z[104];\n""  Real 'z_Radiator.vol[1].dynBal.ports_H_flow[2]' = z[105];\n""  Real 'z_Radiator.vol[2].dynBal.ports_H_flow[1]' = z[106];\n""  Real 'z_Radiator.vol[2].dynBal.ports_H_flow[2]' = z[107];\n""  Real 'z_Radiator.vol[3].dynBal.ports_H_flow[1]' = z[108];\n""  Real 'z_Radiator.vol[3].dynBal.ports_H_flow[2]' = z[109];\n""  Real 'z_Radiator.vol[4].dynBal.ports_H_flow[1]' = z[110];\n""  Real 'z_Radiator.vol[4].dynBal.ports_H_flow[2]' = z[111];\n""  Real 'z_Radiator.vol[5].dynBal.ports_H_flow[1]' = z[112];\n""  Real 'z_Radiator.vol[5].dynBal.ports_H_flow[2]' = z[113];\n""  Real 'z_Radiator.vol[1].heatPort.Q_flow' = z[114];\n""  Real 'z_Radiator.vol[2].heatPort.Q_flow' = z[115];\n""  Real 'z_Radiator.vol[3].heatPort.Q_flow' = z[116];\n""  Real 'z_Radiator.vol[4].heatPort.Q_flow' = z[117];\n""  Real 'z_Radiator.vol[5].heatPort.Q_flow' = z[118];\n""  Real 'z_Radiator.vol[2].ports[2].h_outflow' = z[119];\n""  Real 'z_Radiator.vol[3].ports[2].h_outflow' = z[120];\n""  Real 'z_Radiator.vol[4].ports[2].h_outflow' = z[121];\n""  Real 'z_Radiator.vol[5].ports[2].h_outflow' = z[122];\n""  Real 'z_Radiator.vol[2].ports[1].m_flow' = z[123];\n""  Real 'z_Radiator.vol[3].ports[1].m_flow' = z[124];\n""  Real 'z_Radiator.vol[4].ports[1].m_flow' = z[125];\n""  Real 'z_Radiator.vol[5].ports[1].m_flow' = z[126];\n""  Real 'z_T_z_source.T' = z[127];\n""  Real 'z_T_z_source.port.Q_flow' = z[128];\n""  Real 'z_flow_sink.X_in_internal[1]' = z[129];\n""  Real 'z_flow_sink.p_in_internal' = z[130];\n""  Real 'z_flow_sink.ports[2].h_outflow' = z[131];\n""  Real 'z_flow_sink.ports[1].m_flow' = z[132];\n""  Real 'z_flow_sink.ports[2].m_flow' = z[133];\n""  Real 'z_flow_source.T_in' = z[134];\n""  Real 'z_flow_source.X_in_internal[1]' = z[135];\n""  Real 'z_flow_source.ports[1].h_outflow' = z[136];\n""  Real 'z_indoorTemperature' = z[137];\n""  Real 'z_outletWaterTemperature' = z[138];\n""  Real 'z_supplyWaterTemperature' = z[139];\n""  Real 'z_waterFlowRate' = z[140];\n"
  "equation\n  der(x) = A * x + B * u;\n  y = C * x + D * u;\n  z = Cz * x + Dz * u;\nend linearized_model;\n";
}
#if defined(__cplusplus)
}
#endif

