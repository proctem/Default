import pandas as pd
import numpy as np

##################################################################PROCESS MODEL BEGINS##############################################################################

def ChemProcess_Model(data):

  EcNatGas = 53.6

  ngCcontnt = 50.3


  hEFF = 0.80
  eEFF = 0.50


  construction_prd = 3
  operating_prd = 27
  project_life = construction_prd + operating_prd

  util_fac = np.zeros(project_life)
  util_fac[construction_prd] = 0.70
  util_fac[(construction_prd+1)] = 0.80
  util_fac[(construction_prd+2):] = 0.95

  prodQ = util_fac * data['Cap']

  feedQ = prodQ / data['Yld']

  fuelgas = data['feedEcontnt'] * (1 - data['Yld']) * feedQ   

  Rheat = data['Heat_req'] * (prodQ / hEFF)

  dHF = Rheat - fuelgas
  netHeat = np.maximum(0, dHF)          

  Relec = data['Elect_req'] * (prodQ / eEFF)

  #ghg_dir = Rheat * data['feedCcontnt']       
  ghg_dir = (fuelgas * data['feedCcontnt']) + (dHF * ngCcontnt / 1000)

  ghg_ind = Relec * ngCcontnt / 1000  


  return prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind

##################################################################PROCESS MODEL ENDS##############################################################################


#####################################################MICROECONOMIC MODEL BEGINS##################################################################################

def MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value):

  prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
  elEFF = 0.90


  Infl = 0.02  
  RR = 0.035  
  IRR = 0.10  


  shrDebt = 0.60
  shrEquity = 1 - shrDebt
  wacc = (shrDebt * RR) + (shrEquity * IRR)


  construction_prd = 3
  operating_prd = 27
  project_life = construction_prd + operating_prd

  baseYear = data['Base_Yr']
  Year = list(range(baseYear, baseYear + project_life))


  yr1_capex = 0.20
  yr2_capex = 0.50
  yr3_capex = 0.30

  OwnerCost = 0.10



  corpTAX = np.zeros(project_life)
  corpTAX[:] = data['corpTAX']


  corpTAX[:construction_prd] = 0


  credit = 0.10


  feedprice = [0] * project_life
  fuelprice = [0] * project_life
  elecprice = [0] * project_life

  ######NEW START####################
  capex = [0] * project_life
  opex = [0] * project_life
  capexContrN = [0] * project_life
  opexContrN = [0] * project_life
  feedContrN = [0] * project_life
  utilContrN = [0] * project_life
  bankContrN = [0] * project_life
  taxContrN = [0] * project_life
  ContrDenom = [0] * project_life
  ######NEW END#####################


  if opex_mode == "Inflated":

    for i in range(project_life):
        feedprice[i] = data["Feed_Price"] * ((1 + Infl) ** i)
        fuelprice[i] = data["Fuel_Price"] * ((1 + Infl) ** i)
        elecprice[i] = data["Elect_Price"] * ((1 + Infl) ** i)
  else:

    for i in range(project_life):
        feedprice[i] = data["Feed_Price"]
        fuelprice[i] = data["Fuel_Price"]
        elecprice[i] = data["Elect_Price"]




  feedcst = feedQ * feedprice
  fuelcst = netHeat * fuelprice
  eleccst = elEFF * Relec * elecprice


  CarbonTAX = [data["CO2price"]] * project_life


  if carbon_value == "Yes":
    CO2cst = CarbonTAX * ghg_dir
  else:
    CO2cst = [0] * project_life


  
  Yrly_invsmt = [0] * project_life

  
  ######NEW START####################
  capex[0] = yr1_capex * data["CAPEX"]
  capex[1] = yr2_capex * data["CAPEX"]
  capex[2] = yr3_capex * data["CAPEX"]
  opex[construction_prd:] = data["OPEX"] + feedcst[construction_prd:] + fuelcst[construction_prd:] + eleccst[construction_prd:] + CO2cst[construction_prd:]
  ########NEW END####################

  Yrly_invsmt[0] = yr1_capex * data["CAPEX"]
  Yrly_invsmt[1] = yr2_capex * data["CAPEX"]
  Yrly_invsmt[2] = yr3_capex * data["CAPEX"]
  Yrly_invsmt[construction_prd:] = data["OPEX"] + feedcst[construction_prd:] + fuelcst[construction_prd:] + eleccst[construction_prd:] + CO2cst[construction_prd:]

  
  bank_chrg = [0] * project_life

  if fund_mode == "Debt":    #----------------------------------------------------DEBT----------------------------------
    for i in range(project_life):
        if i <= (construction_prd + 1):
            bank_chrg[i] = RR * sum(Yrly_invsmt[:i+1])
        else:
            bank_chrg[i] = RR * sum(Yrly_invsmt[:construction_prd+1])

    
    deprCAPEX = (1-OwnerCost)*sum(Yrly_invsmt[:construction_prd])
    
    cshflw = [0] * project_life 
    dctftr = [0] * project_life  
    #----------------------------------------------------------------------------Green field
    if plant_mode == "Green":
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + IRR) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + IRR) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_invsmt
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      for i in range(construction_prd + 1, project_life):
          if sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]) < 0:
              bank_chrg[i] = RR * abs(sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]))
          else:
              bank_chrg[i] = 0

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0  
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life  

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
          else:
              if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                  tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * (corpTAX[i])
                  depr_asst += (deprCAPEX - depr_asst)

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              else:
                  tax_pybl[i] = NetRevn[i] * (corpTAX[i])

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  ######NEW END###################


    #----------------------------------------------------------------------------Brown field
    else:
      bank_chrg = [0] * project_life
      Yrly_invsmt[:construction_prd] = [0] * construction_prd
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + IRR) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + IRR) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_invsmt
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      for i in range(construction_prd + 1, project_life):
          if sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]) < 0:
              bank_chrg[i] = RR * abs(sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]))
          else:
              bank_chrg[i] = 0

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0  
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life  

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
          else:
              tax_pybl[i] = NetRevn[i] * (corpTAX[i])

              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  ######NEW END###################


  elif fund_mode == "Equity":   #-----------------------------------------------EQUITY-------------------------------
    bank_chrg = [0] * project_life

    
    deprCAPEX = (1-OwnerCost)*sum(Yrly_invsmt[:construction_prd])
    
    cshflw = [0] * project_life 
    dctftr = [0] * project_life  
    #----------------------------------------------------------------------------Green field
    if plant_mode == "Green":
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + IRR) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + IRR) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_cost
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0 
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life 

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
          else:
              if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                  tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * (corpTAX[i])
                  depr_asst += (deprCAPEX - depr_asst)

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              else:
                  tax_pybl[i] = NetRevn[i] * (corpTAX[i])

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
                  dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  ######NEW END###################



    #----------------------------------------------------------------------------Brown field
    else:
      bank_chrg = [0] * project_life
      Yrly_invsmt[:construction_prd] = [0] * construction_prd
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + IRR) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + IRR) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + IRR) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_cost
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0  
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life  

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + IRR) ** i)
          else:
              tax_pybl[i] = NetRevn[i] * (corpTAX[i])

              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + IRR) ** i)
              dctftr[i] = prodQ[i] / ((1 + IRR) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + IRR) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + IRR) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  ######NEW END###################


  else:     #fund_mode is Mixed     ----------------------------------------------MIXED---------------------------------
    for i in range(project_life):
        if i <= (construction_prd + 1):
            bank_chrg[i] = RR * sum(shrDebt * Yrly_invsmt[:i+1])
        else:
            bank_chrg[i] = RR * sum(shrDebt * Yrly_invsmt[:construction_prd+1])

    
    deprCAPEX = (1-OwnerCost)*sum(Yrly_invsmt[:construction_prd])
    
    cshflw = [0] * project_life  
    dctftr = [0] * project_life  
    #----------------------------------------------------------------------------Green field
    if plant_mode == "Green":
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + wacc) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + wacc) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + wacc) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + wacc) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_invsmt
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      for i in range(construction_prd + 1, project_life):
          if sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]) < 0:
              bank_chrg[i] = RR * abs(sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]))
          else:
              bank_chrg[i] = 0

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0  
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life  

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
              dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
          else:
              if depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) < deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                  dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) > deprCAPEX:
                  tax_pybl[i] = (NetRevn[i] + depr_asst - deprCAPEX) * (corpTAX[i])
                  depr_asst += (deprCAPEX - depr_asst)

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
                  dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + wacc) ** i)
              elif depr_asst < deprCAPEX and (NetRevn[i] + depr_asst) == deprCAPEX:
                  tax_pybl[i] = 0
                  depr_asst += NetRevn[i]

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
                  dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
              else:
                  tax_pybl[i] = NetRevn[i] * (corpTAX[i])

                  cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
                  dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

                  dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
                  cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + wacc) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)      
  ######NEW END###################



    #----------------------------------------------------------------------------Brown field
    else:
      bank_chrg = [0] * project_life
      Yrly_invsmt[:construction_prd] = [0] * construction_prd
      Yrly_cost = [sum(x) for x in zip(Yrly_invsmt, bank_chrg)]

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + wacc) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i]))) / ((1 + wacc) ** i)
      Pstar = sum(cshflw) / sum(dctftr)
      Rstar = Pstar * prodQ

      for i in range(len(Year)):
        cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) * (1 - (corpTAX[i])) / ((1 + wacc) ** i)
        dctftr[i] = (prodQ[i] * (1 - (corpTAX[i])) * ((1 + Infl) ** i)) / ((1 + wacc) ** i)
      Pstaro = sum(cshflw) / sum(dctftr)
      Pstark = [0] * project_life
      for i in range(project_life):
        Pstark[i] = Pstaro * ((1 + Infl) ** i)
      Rstark = [Pstark[i] * prodQ[i] for i in range(project_life)]

      
      #NetRevn = Rstark - Yrly_invsmt
      NetRevn = [r - y for r, y in zip(Rstark, Yrly_cost)]

      
      for i in range(construction_prd + 1, project_life):
          if sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]) < 0:
              bank_chrg[i] = RR * abs(sum(NetRevn[:i]) - sum(bank_chrg[:i - 1]))
          else:
              bank_chrg[i] = 0

      
      TIC = data['CAPEX'] + sum(bank_chrg)

      
      tax_pybl = [0] * project_life  
      depr_asst = 0  
      cshflw2 = [0] * project_life  
      dctftr2 = [0] * project_life  

      for i in range(len(Year)):
          if NetRevn[i] <= 0:
              tax_pybl[i] = 0
              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
              dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i]) / ((1 + wacc) ** i)
          else:
              tax_pybl[i] = NetRevn[i] * (corpTAX[i])

              cshflw[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i]) / ((1 + wacc) ** i)
              dctftr[i] = prodQ[i] / ((1 + wacc) ** i)

              dctftr2[i] = prodQ[i] * ((1 + Infl) ** i) / ((1 + wacc) ** i)
              cshflw2[i] = (Yrly_invsmt[i] + bank_chrg[i] + tax_pybl[i] * (1 - credit)) / ((1 + wacc) ** i)

      Ps = sum(cshflw) / sum(dctftr)
      Pso = sum(cshflw) / sum(dctftr2)
      Pc = sum(cshflw2) / sum(dctftr)
      Pco = sum(cshflw2) / sum(dctftr2)

  ######NEW START###################
      for i in range(len(Year)):
        ContrDenom[i] = prodQ[i] / ((1 + IRR) ** i)
        capexContrN[i] = (capex[i]) / ((1 + IRR) ** i)
        opexContrN[i] = (opex[i]) / ((1 + IRR) ** i)
        feedContrN[i] = (feedcst[i]) / ((1 + IRR) ** i)
        utilContrN[i] = (eleccst[i] + fuelcst[i]) / ((1 + IRR) ** i)
        bankContrN[i] = (bank_chrg[i]) / ((1 + IRR) ** i)
        taxContrN[i] = (tax_pybl[i]) / ((1 + IRR) ** i)
      capexContr = sum(capexContrN) / sum(ContrDenom)
      opexContr = sum(opexContrN) / sum(ContrDenom)
      feedContr = sum(feedContrN) / sum(ContrDenom)
      utilContr = sum(utilContrN) / sum(ContrDenom)
      bankContr = sum(bankContrN) / sum(ContrDenom)
      taxContr = sum(taxContrN) / sum(ContrDenom)
      otherContr = Ps - (capexContr + opexContr + feedContr + utilContr + bankContr + taxContr)
  ######NEW END###################


  return Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, construction_prd, Yrly_invsmt, bank_chrg, NetRevn, tax_pybl

#####################################################MICROECONOMIC MODEL ENDS##################################################################################


############################################################MACROECONOMIC MODEL BEGINS############################################################################

def MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value):


  PRIcoef = 0.3
  CONcoef = 0.7

  prodQ, _, _, _, _, _, _ = ChemProcess_Model(data)
  Ps, _, _, _, _, _, _, _, _, _, _, _, _, Year, project_life, construction_prd, Yrly_invsmt, bank_chrg, _, _ = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)

  pri_invsmt = [0] * project_life
  con_invsmt = [0] * project_life
  bank_invsmt = [0] * project_life

  pri_invsmt[:construction_prd] = [PRIcoef * Yrly_invsmt[i] for i in range(construction_prd)]
  # pri_invsmt[construction_prd:] = Yrly_invsmt[construction_prd:]        
  pri_invsmt[construction_prd:] = [data["OPEX"]] * len(pri_invsmt[construction_prd:])         
  con_invsmt[:construction_prd] = [CONcoef * Yrly_invsmt[i] for i in range(construction_prd)]
  bank_invsmt = bank_chrg


 
  # Chemicals and Chemical Products [C20] Multipliers
  output_PRI = multiplier[(multiplier['Country'] == location) &
                          (multiplier['Multiplier Type'] == "Output Multiplier") &
                          (multiplier['Sector'] == (location + "_" + "C20"))]

  pay_PRI = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Compensation (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "C20"))]

  job_PRI = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Employment Elasticity (Jobs per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "C20"))]

  tax_PRI = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Tax Revenue Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "C20"))]

  gdp_PRI = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Value-Added Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "C20"))]


  # Engineering Construction [F] Multipliers
  output_CON = multiplier[(multiplier['Country'] == location) &
                          (multiplier['Multiplier Type'] == "Output Multiplier") &
                          (multiplier['Sector'] == (location + "_" + "F"))]

  pay_CON = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Compensation (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "F"))]

  job_CON = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Employment Elasticity (Jobs per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "F"))]

  tax_CON = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Tax Revenue Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "F"))]

  gdp_CON = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Value-Added Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "F"))]


  # Financial and Insurance Services [K] Multipliers
  output_BAN = multiplier[(multiplier['Country'] == location) &
                          (multiplier['Multiplier Type'] == "Output Multiplier") &
                          (multiplier['Sector'] == (location + "_" + "K"))]

  pay_BAN = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Compensation (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "K"))]

  job_BAN = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Employment Elasticity (Jobs per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "K"))]

  tax_BAN = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Tax Revenue Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "K"))]

  gdp_BAN = multiplier[(multiplier['Country'] == location) &
                       (multiplier['Multiplier Type'] == "Value-Added Share (USD per million USD output)") &
                       (multiplier['Sector'] == (location + "_" + "K"))]




  pri_invsmt = pd.Series(pri_invsmt)
  con_invsmt = pd.Series(con_invsmt)
  bank_invsmt = pd.Series(bank_invsmt)

  ####################### GDP Impacts BEGIN #####################
  GDP_dirPRI = gdp_PRI['Direct Impact'].values[0] * pri_invsmt
  GDP_dirCON = gdp_CON['Direct Impact'].values[0] * con_invsmt
  GDP_dirBAN = gdp_BAN['Direct Impact'].values[0] * bank_invsmt

  GDP_indPRI = gdp_PRI['Indirect Impact'].values[0] * pri_invsmt
  GDP_indCON = gdp_CON['Indirect Impact'].values[0] * con_invsmt
  GDP_indBAN = gdp_BAN['Indirect Impact'].values[0] * bank_invsmt

  GDP_totPRI = gdp_PRI['Total Impact'].values[0] * pri_invsmt
  GDP_totCON = gdp_CON['Total Impact'].values[0] * con_invsmt
  GDP_totBAN = gdp_BAN['Total Impact'].values[0] * bank_invsmt

  GDP_dir = GDP_dirPRI + GDP_dirCON + GDP_dirBAN
  GDP_ind = GDP_indPRI + GDP_indCON + GDP_indBAN
  GDP_tot = GDP_totPRI + GDP_totCON + GDP_totBAN

  ####################### GDP Impacts END #######################


  ####################### Job Impacts BEGIN #####################
  JOB_dirPRI = job_PRI['Direct Impact'].values[0] * pri_invsmt
  JOB_dirCON = job_CON['Direct Impact'].values[0] * con_invsmt
  JOB_dirBAN = job_BAN['Direct Impact'].values[0] * bank_invsmt

  JOB_indPRI = job_PRI['Indirect Impact'].values[0] * pri_invsmt
  JOB_indCON = job_CON['Indirect Impact'].values[0] * con_invsmt
  JOB_indBAN = job_BAN['Indirect Impact'].values[0] * bank_invsmt

  JOB_totPRI = job_PRI['Total Impact'].values[0] * pri_invsmt
  JOB_totCON = job_CON['Total Impact'].values[0] * con_invsmt
  JOB_totBAN = job_BAN['Total Impact'].values[0] * bank_invsmt

  JOB_dir = JOB_dirPRI + JOB_dirCON + JOB_dirBAN
  JOB_ind = JOB_indPRI + JOB_indCON + JOB_indBAN
  JOB_tot = JOB_totPRI + JOB_totCON + JOB_totBAN

  ####################### Job Impacts END #######################


  ####################### Wages & Salaries Impacts BEGIN #####################
  PAY_dirPRI = pay_PRI['Direct Impact'].values[0] * pri_invsmt
  PAY_dirCON = pay_CON['Direct Impact'].values[0] * con_invsmt
  PAY_dirBAN = pay_BAN['Direct Impact'].values[0] * bank_invsmt

  PAY_indPRI = pay_PRI['Indirect Impact'].values[0] * pri_invsmt
  PAY_indCON = pay_CON['Indirect Impact'].values[0] * con_invsmt
  PAY_indBAN = pay_BAN['Indirect Impact'].values[0] * bank_invsmt

  PAY_totPRI = pay_PRI['Total Impact'].values[0] * pri_invsmt
  PAY_totCON = pay_CON['Total Impact'].values[0] * con_invsmt
  PAY_totBAN = pay_BAN['Total Impact'].values[0] * bank_invsmt

  PAY_dir = PAY_dirPRI + PAY_dirCON + PAY_dirBAN
  PAY_ind = PAY_indPRI + PAY_indCON + PAY_indBAN
  PAY_tot = PAY_totPRI + PAY_totCON + PAY_totBAN

  ####################### Wages & Salaries Impacts END #######################


  ####################### Taxation Impacts (Potential Tax Revenues) BEGIN ################
  TAX_dir = [0] * project_life
  TAX_ind = [0] * project_life
  TAX_tot = [0] * project_life

  for i in range(construction_prd, project_life):
      TAX_dir[i] = tax_PRI['Direct Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))
      TAX_ind[i] = tax_PRI['Indirect Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))
      TAX_tot[i] = tax_PRI['Total Impact'].values[0] * np.array(Yrly_invsmt[i] + (Ps * prodQ[i]))


  return GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI
  ####################### Taxation Impacts END ##################

############################################################# MACROECONOMIC MODEL ENDS ############################################################


############################################################# ANALYTICS MODEL BEGINS ############################################################

def Analytics_Model2(multiplier, project_data, location, product, plant_mode, fund_mode, opex_mode, carbon_value, plant_size, plant_effy):

  # Filtering data to choose country in which chemical plant is located and the type of product from the plant
  dt = project_data[(project_data['Country'] == location) & (project_data['Main_Prod'] == product) & (project_data['Plant_Size'] == plant_size) & (project_data['Plant_Effy'] == plant_effy)]


  Infl = 0.02  

  tempNUM = 1000000
  results=[]
  for index, data in dt.iterrows():

    prodQ, feedQ, Rheat, netHeat, Relec, ghg_dir, ghg_ind = ChemProcess_Model(data)
    Ps, Pso, Pc, Pco, capexContr, opexContr, feedContr, utilContr, bankContr, taxContr, otherContr, cshflw, cshflw2, Year, project_life, construction_prd, Yrly_invsmt, bank_chrg, NetRevn, tax_pybl = MicroEconomic_Model(data, plant_mode, fund_mode, opex_mode, carbon_value)
    GDP_dir, GDP_ind, GDP_tot, JOB_dir, JOB_ind, JOB_tot, PAY_dir, PAY_ind, PAY_tot, TAX_dir, TAX_ind, TAX_tot, GDP_totPRI, JOB_totPRI, PAY_totPRI, GDP_dirPRI, JOB_dirPRI, PAY_dirPRI = MacroEconomic_Model(multiplier, data, location, plant_mode, fund_mode, opex_mode, carbon_value)

    Yrly_cost = np.array(Yrly_invsmt) + np.array(bank_chrg)

    Ps = [Ps] * project_life
    Pc = [Pc] * project_life
    Psk = [0] * project_life
    Pck = [0] * project_life

    for i in range(project_life):
      Psk[i] = Pso * ((1 + Infl) ** i)
      Pck[i] = Pco * ((1 + Infl) ** i)


    Rs = [Ps[i] * prodQ[i] for i in range(project_life)]
    NRs = [Rs[i] - Yrly_cost[i] for i in range(project_life)]


    Rsk = Psk * prodQ
    NRsk = Rsk - Yrly_cost

    ccflows = np.cumsum(NRs)
    ccflowsk = np.cumsum(NRsk)

    cost_modes = ["Supply Cost", "Cash Cost"]
    if plant_mode == "Green":
      cost_mode = cost_modes[0]
    else:
      cost_mode = cost_modes[1]


    pri_bothJOB = [0] * project_life
    pri_directJOB = [0] * project_life
    pri_indirectJOB = [0] * project_life

    All_directJOB = [0] * project_life
    All_indirectJOB = [0] * project_life
    All_bothJOB = [0] * project_life

    pri_bothGDP = GDP_totPRI
    pri_directGDP = GDP_dirPRI
    pri_indirectGDP = GDP_totPRI - GDP_dirPRI
    All_bothGDP = GDP_tot
    All_directGDP =  GDP_dir
    All_indirectGDP = GDP_tot - GDP_dir

    pri_bothTAX = TAX_tot
    pri_directTAX = TAX_dir
    pri_indirectTAX = TAX_ind

    pri_bothPAY = PAY_totPRI
    pri_directPAY = PAY_dirPRI
    pri_indirectPAY = PAY_totPRI - PAY_dirPRI
    All_bothPAY = PAY_tot
    All_directPAY = PAY_dir
    All_indirectPAY = PAY_tot - PAY_dir



    pri_bothJOB[construction_prd:] = JOB_totPRI[construction_prd:]
    pri_directJOB[construction_prd:] = JOB_dirPRI[construction_prd:]
    pri_indirectJOB[construction_prd:] = JOB_totPRI[construction_prd:]  - JOB_dirPRI[construction_prd:]

    pri_bothJOB[:construction_prd] = JOB_totPRI[:construction_prd]
    pri_directJOB[:construction_prd] = JOB_dirPRI[:construction_prd]
    pri_indirectJOB[:construction_prd] = JOB_totPRI[:construction_prd]  - JOB_dirPRI[:construction_prd]



    All_bothJOB[construction_prd:] = JOB_tot[construction_prd:]
    All_directJOB[construction_prd:] = JOB_dir[construction_prd:]
    All_indirectJOB[construction_prd:] = JOB_tot[construction_prd:]  - JOB_dir[construction_prd:]

    All_bothJOB[:construction_prd] = JOB_tot[:construction_prd]
    All_directJOB[:construction_prd] = JOB_dir[:construction_prd]
    All_indirectJOB[:construction_prd] = JOB_tot[:construction_prd]  - JOB_dir[:construction_prd]



    result = pd.DataFrame({
        'Year': Year,
        'Process Technology': [data['ProcTech']] * project_life,
        'Plant Size': [data['Plant_Size']] * project_life,
        'Plant Efficiency': [data['Plant_Effy']] * project_life,
        'Feedstock Input (TPA)': feedQ,
        'Product Output (TPA)': prodQ,
        'Direct GHG Emissions (TPA)': ghg_dir,
        'Cost Mode': [cost_mode]  * project_life,
        'Real cumCash Flow': ccflows,
        'Nominal cumCash Flow': ccflowsk,
        'Constant$ Breakeven Price': Ps,
        'Capex portion': [capexContr] * project_life,
        'Opex portion': [opexContr] * project_life,
        'Feed portion': [feedContr] * project_life,
        'Util portion': [utilContr] * project_life,
        'Bank portion': [bankContr] * project_life,
        'Tax portion': [taxContr] * project_life,
        'Other portion': [otherContr] * project_life,
        'Current$ Breakeven Price': Psk,
        'Constant$ SC wCredit': Pc,
        'Current$ SC wCredit': Pck,
        'Project Finance': [fund_mode] * project_life,
        'Carbon Valued': [carbon_value] * project_life,
        'Feedstock Price ($/t)': [data['Feed_Price']] * project_life,
        'pri_directGDP': np.array(pri_directGDP)/tempNUM,
        'pri_bothGDP': np.array(pri_bothGDP)/tempNUM,
        'All_directGDP': np.array(All_directGDP)/tempNUM,
        'All_bothGDP': np.array(All_bothGDP)/tempNUM,
        'pri_directPAY': np.array(pri_directPAY)/tempNUM,
        'pri_bothPAY': np.array(pri_bothPAY)/tempNUM,
        'All_directPAY': np.array(All_directPAY)/tempNUM,
        'All_bothPAY': np.array(All_bothPAY)/tempNUM,
        'pri_directJOB': np.array(pri_directJOB)/tempNUM,
        'pri_bothJOB': np.array(pri_bothJOB)/tempNUM,
        'All_directJOB': np.array(All_directJOB)/tempNUM,
        'All_bothJOB': np.array(All_bothJOB)/tempNUM,
        'pri_directTAX': np.array(pri_directTAX)/tempNUM,
        'pri_bothTAX': np.array(pri_bothTAX)/tempNUM
    })
    results.append(result)


  results = pd.concat(results, ignore_index=True)



  return results

############################################################# ANALYTICS MODEL ENDS ############################################################

