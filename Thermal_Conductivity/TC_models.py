import os
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit
import xlrd
import xlwt
import pandas as pd
import tkinter as tk
from tkinter import IntVar
import csv
from scipy.constants import Boltzmann as k_B
from scipy.constants import Avogadro as Avog
import inspect

def KTM(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, C_p_mix=0, alpha=0, expon=0):
    print(f"[KTM] Compounds: {compound_input}")

    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    alpha_i_m_df=df['alpha_i_m (K^-1)']
    C_i_0_df=df['C_i_0 (m/s)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    rho_i_m_df=df['rho_m (g/m^3)']

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    alpha_i_m=np.zeros(len(compound_input))
    C_i_0=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    rho_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        alpha_i_m[i]=alpha_i_m_df[indices[i]]
        C_i_0[i]=C_i_0_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        rho_i_m[i]=rho_i_m_df[indices[i]]*0.001 # Convert to kg/m^3

    #Calculate Gruneisen parameter
    gamma_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        gamma_i_m[i]=((M_i[i]*alpha_i_m[i]*C_i_0[i]**2)/C_i_p[i])

    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)


    #calculate constant volume heat capacities at melting temp
    C_i_v=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        C_i_v[i]=C_i_p[i]/(1+alpha_i_m[i]*gamma_i_m[i]*T_i_m[i])

    
    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    n_i_c=np.zeros(len(compound_input))
    n_i_a=np.zeros(len(compound_input))
    comps = 0

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
            if comps == 0:
                n_i_c[i] = n_i[i]
                if "NO3" in Compound[i]:
                    n_i_c[i] = n_i_c[i] +1
                    n_i_a[i] = 3
                comps = 1
            else:
                if "NO3" in Compound[i]:
                    continue
                n_i_a[i] = n_i[i] - n_i_c[i]
        comps = 0

    # print("KT, number of ions: ",n_i)
    # print("KT, number of cations: ",n_i_c)
    # print("KT, number of anions: ",n_i_a)
        
    # Calculate compound psi term
    psi_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        psi_i[i] = 1 + n_i_c[i]/n_i_a[i]

    # Calculate compound number density
    n_dens_i = np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        n_dens_i[i] = Avog *n_i[i]/V_i_m[i]

    #Calculate compound thermal conductivity(T) based on compound minimum thermal conductivites
    lambda_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            lambda_i[i][j] = (1 + n_i_c[i]/n_i_a[i]) * k_B * n_dens_i[i]**(2/3) * C_i_0[i] * (1 - alpha_i_m[i]*(gamma_i_m[i] + 1/3)*(T[j] - T_i_m[i]))

    #Calculate volume fractions
    phi_i_m=np.zeros(len(compound_input))
    denom=0
    for i in range(len(compound_input)):
        phi_i_m[i]=V_i_m[i]*mol_fracs[i]
        denom=denom+V_i_m[i]*mol_fracs[i]
    phi_i_m=phi_i_m/denom  

    #Calculate weight fractions
    kappa_i_m=np.zeros(len(compound_input))
    denom=0
    for i in range(len(compound_input)):
        kappa_i_m[i]=M_i[i]*mol_fracs[i]
        denom=denom+M_i[i]*mol_fracs[i]
    kappa_i_m=kappa_i_m/denom 

    #Calculate taus for compounds
    Tau_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Tau_i_m[i] = ( C_i_v[i]*C_i_0[i] ) / n_i[i]


    #Calculate mixture speed of sound
    C_0_mix_m=0
    for i in range(len(compound_input)):
        C_0_mix_m=C_0_mix_m+phi_i_m[i]**2/(kappa_i_m[i]*C_i_0[i]**2)
    C_0_mix_m=1/np.sqrt(C_0_mix_m)

    #Calculate mixture specific heat
    #Calculate average number of ions
    C_p_mix=0
    n_mix=0
    if C_p_mix==0:
        for i in range(len(compound_input)):
            C_p_mix=C_p_mix+mol_fracs[i]*C_i_p[i]
            n_mix=n_mix+mol_fracs[i]*n_i[i]
    else: 
        C_p_mix=C_p_mix
        for i in range(len(compound_input)):
            n_mix=n_mix+mol_fracs[i]*n_i[i]
 

    #Calculate average molar volume
    V_mix_m=0
    if V_m==0:
        for i in range(len(compound_input)):
            V_mix_m=V_mix_m+mol_fracs[i]*M_i[i]*(1/rho_i_m[i])
    else: V_mix_m=V_m

        
    #Calculate mixture number density
    n_dens_mix = Avog * n_mix/V_mix_m


    #Calculate mixture thermal expansion
    alpha_mix_m=0
    if alpha==0:
        for i in range(len(compound_input)):
            alpha_mix_m=alpha_mix_m+phi_i_m[i]*alpha_i_m[i]
    else:alpha_mix_m=alpha
    

    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix=M_mix+mol_fracs[i]*M_i[i]
        

    #Calculate mixture gamma
    gamma_mix_m=M_mix*alpha_mix_m*C_0_mix_m**2*(1/C_p_mix)
    

    #Calculate kinetic thermal conductivity
    lambda_k = np.zeros(len(T))
    for j in range(len(T)):
        # if Compound == ['NaCl','UCl3'] and mol_fracs == [0.63,0.37]:
        #     lambda_k[j] = 1.240955586 * k_B * n_dens_mix**(2/3) * C_0_mix_m * (1 - alpha_mix_m*(gamma_mix_m + 1/3)*(T[j] - T_melt))
        # elif Compound == ['NaCl','UCl3'] and mol_fracs == [0.658,0.342]:
        #     lambda_k[j] = 1.258369925 * k_B * n_dens_mix**(2/3) * C_0_mix_m * (1 - alpha_mix_m*(gamma_mix_m + 1/3)*(T[j] - T_melt))
        # else:
        lambda_k[j] = (1 + np.sum(n_i_c)/np.sum(n_i_a)) * k_B * n_dens_mix**(2/3) * C_0_mix_m * (1 - alpha_mix_m*(gamma_mix_m + 1/3)*(T[j] - T_melt))
    
    
    #Calculate ideal thermal conductivity
    lambda_ideal = np.zeros(len(T))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            lambda_ideal[j] += lambda_i[i][j]*mol_fracs[i]


    # Calculate delta of mix
    delta = np.zeros(len(T))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            # if Compound == ['NaCl','UCl3'] and mol_fracs == [0.63,0.37]:
            #     delta[j] += (0.193646716)/M_mix * lambda_i[i][j]/lambda_ideal[j] * mol_fracs[i] * (1 - M_i[i]/M_mix)**2
            # elif Compound == ['NaCl','UCl3'] and mol_fracs == [0.658,0.342]:
            #     delta[j] += (0.171475397)/M_mix * lambda_i[i][j]/lambda_ideal[j] * mol_fracs[i] * (1 - M_i[i]/M_mix)**2
            # else:
            delta[j] += lambda_i[i][j]/lambda_ideal[j] * mol_fracs[i] * (1 - M_i[i]/M_mix)**2

    lambda_mix_T = lambda_k * (1 - delta)
    return(T,lambda_mix_T)

def KTM_Mix(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, C_p_mix=0, alpha=0, expon=0):
    
    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    alpha_i_m_df=df['alpha_i_m (K^-1)']
    C_i_0_df=df['C_i_0 (m/s)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    rho_i_m_df=df['rho_m (g/m^3)']

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    alpha_i_m=np.zeros(len(compound_input))
    C_i_0=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    rho_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        alpha_i_m[i]=alpha_i_m_df[indices[i]]
        C_i_0[i]=C_i_0_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        rho_i_m[i]=rho_i_m_df[indices[i]]*0.001 # Convert to kg/m^3

    print(f"[KTM_Mix] Compounds: {compound_input}")
    specific_heat_mix = C_p_mix
    density_mix = [0,0]
    sound_velocity_mix = [0,0]    
    molar_volume_mix = [0,0] 
    
    # Obtain measurement data for function
    func_name = inspect.currentframe().f_code.co_name
    r_ac_mix, density_mix, sound_velocity_mix, specific_heat_mix = prop_lookup(func_name,Compound,mol_fracs)

    #Calculate Gruneisen parameter
    gamma_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        gamma_i_m[i]=((M_i[i]*alpha_i_m[i]*C_i_0[i]**2)/C_i_p[i])

    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)


    #calculate constant volume heat capacities at melting temp
    C_i_v=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        C_i_v[i]=C_i_p[i]/(1+alpha_i_m[i]*gamma_i_m[i]*T_i_m[i])

    
    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    n_i_c=np.zeros(len(compound_input))
    n_i_a=np.zeros(len(compound_input))
    comps = 0

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
            if comps == 0:
                n_i_c[i] = n_i[i]
                if "NO3" in Compound[i]:
                    n_i_c[i] = n_i_c[i] +1
                    n_i_a[i] = 3
                comps = 1
            else:
                if "NO3" in Compound[i]:
                    continue
                n_i_a[i] = n_i[i] - n_i_c[i]
        comps = 0

    # print("KT, number of ions: ",n_i)
    # print("KT, number of cations: ",n_i_c)
    # print("KT, number of anions: ",n_i_a)
        
    # Calculate compound psi term
    psi_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        psi_i[i] = 1 + n_i_c[i]/n_i_a[i]

    # Calculate compound number density
    n_dens_i = np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        n_dens_i[i] = Avog *n_i[i]/V_i_m[i]

    #Calculate compound thermal conductivity(T) based on compound minimum thermal conductivites
    lambda_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            lambda_i[i][j] = (1 + n_i_c[i]/n_i_a[i]) * k_B * n_dens_i[i]**(2/3) * C_i_0[i] * (1 - alpha_i_m[i]*(gamma_i_m[i] + 1/3)*(T[j] - T_i_m[i]))

    #Calculate volume fractions
    phi_i_m=np.zeros(len(compound_input))
    denom=0
    for i in range(len(compound_input)):
        phi_i_m[i]=V_i_m[i]*mol_fracs[i]
        denom=denom+V_i_m[i]*mol_fracs[i]
    phi_i_m=phi_i_m/denom  

    #Calculate weight fractions
    kappa_i_m=np.zeros(len(compound_input))
    denom=0
    for i in range(len(compound_input)):
        kappa_i_m[i]=M_i[i]*mol_fracs[i]
        denom=denom+M_i[i]*mol_fracs[i]
    kappa_i_m=kappa_i_m/denom 

    #Calculate taus for compounds
    Tau_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Tau_i_m[i] = ( C_i_v[i]*C_i_0[i] ) / n_i[i]


    #Calculate mixture speed of sound
    C_0_mix_m=0
    C_0_mix_est = 0
    C_0_mix_data = 0 
    for i in range(len(compound_input)):
        C_0_mix_est += phi_i_m[i]**2/(kappa_i_m[i]*C_i_0[i]**2)
    C_0_mix_data = sound_velocity_mix[1]*T_melt+sound_velocity_mix[0]
    C_0_mix_est=1/np.sqrt(C_0_mix_est)
    if sound_velocity_mix == [0,0]:
        C_0_mix_m = C_0_mix_est
    else:
        C_0_mix_m = C_0_mix_data

    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix=M_mix+mol_fracs[i]*M_i[i]

    # Calculate average density, compare to measured
    rho_mix_m=0
    rho_mix_est = 0
    rho_mix_data = 0
    for i in range(len(compound_input)):
        rho_mix_est += mol_fracs[i]*M_i[i]*(1/rho_i_m[i])
    rho_mix_data = M_mix/(density_mix[0] + density_mix[1]*T_melt)
    if density_mix == [0,0]:
        rho_mix_m = rho_mix_est
    else:
        rho_mix_m = rho_mix_data

        rho_mix_est_avg = np.average(rho_mix_est)
        rho_mix_data_avg = np.average(rho_mix_data)
        error_density = 100 * (rho_mix_data_avg-rho_mix_est_avg)/rho_mix_data_avg
        # Density comparison logged but not printed for cleanliness


    #Calculate average molar volume
    V_mix_m=0
    V_mix_est = 0
    V_mix_data = 0
    V_mix_data_Vm = 0
    for i in range(len(compound_input)):    
        V_mix_est += mol_fracs[i]*M_i[i]*(1/rho_i_m[i])
    V_mix_data = M_mix/(density_mix[0] + density_mix[1]*T_melt)
    V_mix_data_Vm = molar_volume_mix[1]*T_melt+molar_volume_mix[0]
    if molar_volume_mix == [0,0]:
        if density_mix == [0,0]:
            V_mix_m = V_mix_est
        else:
            V_mix_m = V_mix_data
    else:
        V_mix_m = V_mix_data_Vm

        # V_mix_est_avg = np.average(V_mix_est)
        # V_mix_data_avg = np.average(V_mix_data)
        # error_densityVm = 100 * (V_mix_data_avg-V_mix_est_avg)/V_mix_data_avg
        # error_Vm = 100 * (V_mix_data_Vm-V_mix_est_avg)/V_mix_data_Vm
        #print("")
        #print("Molar volume - From estimated density: ", V_mix_est_avg)
        #print("Molar volume - From mix data density: ", V_mix_data_avg)
        #print("Molar volume - Mix data Molar Volume: ", V_mix_data_Vm)
        #print("Molar volume % Difference, Density Data/Estimation: ", error_densityVm, " %")
        #print("Molar volume % Difference, Molar Volume Data/Estimation: ", error_Vm, " %")
    
    #Calculate mixture specific heat
    #Calculate average number of ions
    C_p_mix=0
    C_p_mix_est = 0
    C_p_mix_data = 0
    n_mix=0
    for i in range(len(compound_input)):
        C_p_mix_est += C_p_mix+mol_fracs[i]*C_i_p[i]
        n_mix = n_mix + mol_fracs[i]*n_i[i]
    C_p_mix_data = specific_heat_mix
    if specific_heat_mix[0] == 0:
        C_p_mix = C_p_mix_est
    else:
        if specific_heat_mix[2] == 'm':
            C_p_mix = C_p_mix_data[1]*T_melt+C_p_mix_data[0]
        elif specific_heat_mix[2] == 'g':   # Converts units from J/g/K to J/mol/K
            C_p_mix = (C_p_mix_data[1]*T_melt+C_p_mix_data[0])*1000*(M_mix)

        # Specific heat comparison logged but not printed for cleanliness
        
    #Calculate mixture number density
    n_dens_mix = Avog * n_mix/V_mix_m


    #Calculate mixture thermal expansion
    alpha_mix_m=0
    if alpha==0:
        for i in range(len(compound_input)):
            alpha_mix_m=alpha_mix_m+phi_i_m[i]*alpha_i_m[i]
    else:alpha_mix_m=alpha
        

    #Calculate mixture gamma
    gamma_mix_m=M_mix*alpha_mix_m*C_0_mix_m**2*(1/C_p_mix)
    

    #Calculate kinetic thermal conductivity
    lambda_k = np.zeros(len(T))
    for j in range(len(T)):
        lambda_k[j] = (1 + np.sum(n_i_c)/np.sum(n_i_a)) * k_B * n_dens_mix**(2/3) * C_0_mix_m * (1 - alpha_mix_m*(gamma_mix_m + 1/3)*(T[j] - T_melt))
    
    
    #Calculate ideal thermal conductivity
    lambda_ideal = np.zeros(len(T))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            lambda_ideal[j] += lambda_i[i][j]*mol_fracs[i]


    # Calculate delta of mix
    delta = np.zeros(len(T))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            delta[j] += lambda_i[i][j]/lambda_ideal[j] * mol_fracs[i] * (1 - M_i[i]/M_mix)**2
    lambda_mix_T = lambda_k * (1 - delta)
    return(T,lambda_mix_T)

def PGM(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, sound_velocity_mix=0, C_p_mix=0, alpha=0, expon=0):
    print(f"[PGM] Compounds: {compound_input}")
    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    C_i_p_sp_df=df['C_i_p_sp (J/g/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    C_0_T_A_df=df['SoS(T)_A (A+B*T)']
    C_0_T_B_df=df['SoS(T)_B']
    rho_T_A_df=df['A (Density (g/cm3):   A - BT(K))']
    rho_T_B_df=df['B (Density (g/cm3):   A - BT(K))'] 

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    C_i_p_sp=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    C_0_T_A=np.zeros(len(compound_input))
    C_0_T_B=np.zeros(len(compound_input))
    rho_T_A=np.zeros(len(compound_input))
    rho_T_B=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        C_i_p_sp[i]=C_i_p_sp_df[indices[i]]/0.001   # Convert to J/kg/K
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        C_0_T_A[i]=C_0_T_A_df[indices[i]]
        C_0_T_B[i]=C_0_T_B_df[indices[i]]
        rho_T_A[i]=rho_T_A_df[indices[i]]/0.001
        rho_T_B[i]=rho_T_B_df[indices[i]]/0.001

    # print('')
    # print('####### Phonon Gas Model ############################')

    specific_heat_mix = C_p_mix


    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
    
    #print("PGM, number of ions: ",n_i)


    # if Compound == ['NaCl','UCl3'] and mol_fracs == [0.63,0.37]:
    #     density_mix = [4.22*100**3/1000,-0.00113*100**3/1000]   # Desyatnik, 1975
    # elif Compound == ['LiF','BeF2']:
    #     density_mix = [2518.15,-0.424]  # Vidrio, 2022, FLiBe 33.59mol%BeF2
    #     sound_velocity_mix = [4272.309646,-1.101929884] # 66-34% Cantor, 1968
    #     specific_heat_mix = 2414.7*0.033  # 67-33% Sohal, 2010  
    #     #density_mix = [2413,-0.488]  # Janz, 1974, FLiBe 33mol%BeF2
    #     #2110 * (1.885 + 2.762*mol_fracs[1] + mol_fracs[1]**2) / (1.773 + 2.663*mol_fracs[0] )
    # elif Compound == ['LiF','NaF','KF']:
    #     density_mix = [2729.3,-0.73]  # 46.5-11.5-42%, Vriesema [1979], Ingersoll et al. [2007], and Williams et al. [2006]
    #     sound_velocity_mix = [3241.15,-1.20]    # 46.5-11.5-42% Robertson, 2022
    #     specific_heat_mix = 1882.8*0.0413  # 46.5-11.5-42% Sohal, 2010
    # else:
    density_mix = [0,0]
    sound_velocity_mix = [0,0]    
    specific_heat_mix = [0,0]  

    #Find the temperature dependent density from data
    rho_i = np.zeros((len(Compound),len(T)))
    for i in range(len(Compound)):
        
        for j in range(len(T)):
            rho_i[i][j] = rho_T_A[i] + rho_T_B[i]*T[j] 

    
    # Calculate the temp-dependent sound velocity of compounds
    C_0_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_i[i][j] = C_0_T_B[i]*T[j]+C_0_T_A[i]


    # Calculate molar volume of compounds from temp-dependent density
    V_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_i[i][j] = M_i[i] / rho_i[i][j]

    
    # Calculate volume fractions
    phi_i=np.zeros((len(Compound),len(T)))
    for j in range(len(T)):
        denom_phi = 0
        for i in range(len(compound_input)):
            phi_i[i][j] = V_i[i][j]*mol_fracs[i]
            denom_phi += V_i[i][j]*mol_fracs[i]
        for i in range(len(compound_input)):
            phi_i[i][j]=phi_i[i][j]/denom_phi

    # Calculate mass (fractions
    kappa_i=np.zeros(len(Compound))
    denom_kappa = 0
    for i in range(len(compound_input)):
        kappa_i[i] = M_i[i]*mol_fracs[i]
        denom_kappa += M_i[i]*mol_fracs[i]
    kappa_i=kappa_i/denom_kappa           


    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix=M_mix+mol_fracs[i]*M_i[i]


    # Calculate the average temp-dependent molar volume of mixture
    V_mix = np.zeros(len(T))      
    V_mix_est = np.zeros(len(T)) 
    V_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_mix_est[j] += mol_fracs[i]*V_i[i][j]
            if density_mix == [0,0]:
                pass
            else:
                V_mix_data[j] = M_mix/(density_mix[0] + density_mix[1]*T[j])
    if density_mix == [0,0]:
        V_mix = V_mix_est
    else:
        V_mix = V_mix_data

        V_mix_est_avg = np.average(V_mix_est)
        V_mix_data_avg = np.average(V_mix_data)
        error = 100 * (V_mix_data_avg-V_mix_est_avg)/V_mix_data_avg
        #print("Molar volume from estimated mix density: ", V_mix_est_avg)
        #print("Molar volume from data mix density: ", V_mix_data_avg)
        #print("Molar volume % Difference: ", error, " %")


    #Calculate the temp-dependent density of mixture
    rho_mix = np.zeros(len(T))
    rho_mix_est = np.zeros(len(T)) 
    rho_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            rho_mix_est[j] += rho_i[i][j]*mol_fracs[i]   #M_mix/V_mix[j]
            rho_mix_data[j] = density_mix[0] + density_mix[1]*T[j]   # If mixture temp-dependent density data exists
    if density_mix == [0,0]:
        rho_mix = rho_mix_est
    else:
        rho_mix = rho_mix_data

        rho_mix_est_avg = np.average(rho_mix_est)
        rho_mix_data_avg = np.average(rho_mix_data)
        error = 100 * (rho_mix_data_avg-rho_mix_est_avg)/rho_mix_data_avg



    # Calculate the temp-dependent sound velocity of mixture
    C_0_mix = np.zeros(len(T))
    C_0_mix_est = np.zeros(len(T)) 
    C_0_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_mix_est[j] += phi_i[i][j]**2 / (kappa_i[i] * C_0_i[i][j]**2)
            C_0_mix_data[j] = sound_velocity_mix[1]*T[j]+sound_velocity_mix[0]    # If mixture temp-dependent sound velocity data exists
    for j in range(len(T)):
        C_0_mix_est[j] = 1 / np.sqrt(C_0_mix_est[j])
    if sound_velocity_mix == [0,0]:
        C_0_mix = C_0_mix_est
    else:
        C_0_mix = C_0_mix_data




    # Calculate mixture specific heat & # of ions
    C_p_mix=0
    C_p_mix_est = 0
    C_p_mix_data = 0
    n_mix=0
    for i in range(len(compound_input)):
        C_p_mix_est += C_p_mix+mol_fracs[i]*C_i_p[i]
        C_p_mix_data=specific_heat_mix
        n_mix=n_mix+mol_fracs[i]*n_i[i]
    if specific_heat_mix[0] == 0:
        C_p_mix = C_p_mix_est
    else:
        C_p_mix = C_p_mix_data

    # Calculate mixture radial distance
    r_ac_mix = 0
    for i in range(len(compound_input)):
        r_ac_mix += mol_fracs[i]*(r_a[i]+r_c[i])

    print(f"  → mean_free_path (sum of ionic radii): {r_ac_mix*1e10:.4f} Å")

    #Calculate compound thermal conductivity
    lambda_i_m=np.zeros(len(T))
    lambda_i_mg=np.zeros(len(T))
    lambda_i_mb=np.zeros(len(T))
    sound_w_time = 0
    for j in range(len(T)):
        lambda_i_m[j] = 1/3 * C_i_p_sp[i] * rho_mix[j] * C_0_mix[j] * r_ac_mix    # Verified with Zhao's results, uses Zhao's data and specific heat capacity
        #lambda_i_mg[j] = 1/3 * C_p_mix * 1/M_mix * rho_mix[j] * C_0_mix[j] * r_ac_mix  # Verified with Zhao's results, uses Zhao's data and but MSTDB heat capacity
        lambda_i_mb[j] = 1/3 * C_p_mix * 1/V_mix[j] * C_0_mix[j] * r_ac_mix  # Uses molar volume at melting point only 
    

    # if sound_w_time == 1:
    #     print("Calculated with temp-dependent sound velocity data.")
    # else:
    #     print("No temp-dependent sound velocity data available. Calculated with melting temp sound velocity only.")
    

    nan_check = np.isnan(lambda_i_m)
    contains_nan = nan_check.any()
    if contains_nan or lambda_i_m[0] == 0:
        return(T,lambda_i_mb)
    else:
        return(T,lambda_i_m)

def PGM_Mix(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, sound_velocity_mix=0, C_p_mix=0, alpha=0, expon=0):
    print(f"[PGM_Mix] Compounds: {compound_input}")
    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    C_i_p_sp_df=df['C_i_p_sp (J/g/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    C_0_T_A_df=df['SoS(T)_A (A+B*T)']
    C_0_T_B_df=df['SoS(T)_B']
    rho_T_A_df=df['A (Density (g/cm3):   A - BT(K))']
    rho_T_B_df=df['B (Density (g/cm3):   A - BT(K))'] 

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    C_i_p_sp=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    C_0_T_A=np.zeros(len(compound_input))
    C_0_T_B=np.zeros(len(compound_input))
    rho_T_A=np.zeros(len(compound_input))
    rho_T_B=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        C_i_p_sp[i]=C_i_p_sp_df[indices[i]]/0.001   # Convert to J/kg/K
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        C_0_T_A[i]=C_0_T_A_df[indices[i]]
        C_0_T_B[i]=C_0_T_B_df[indices[i]]
        rho_T_A[i]=rho_T_A_df[indices[i]]/0.001
        rho_T_B[i]=rho_T_B_df[indices[i]]/0.001

    # print('')
    # print('################ Phonon Gas Model, Mix Data ############################')

    specific_heat_mix = C_p_mix
    density_mix = [0,0]
    sound_velocity_mix = [0,0]    
    molar_volume_mix = [0,0] 
    
    # Obtain measurement data for function
    func_name = inspect.currentframe().f_code.co_name
    r_ac_mix, density_mix, sound_velocity_mix, specific_heat_mix = prop_lookup(func_name,Compound,mol_fracs)


    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
    
    #print("PGM, number of ions: ",n_i)


    #Find the temperature dependent density from data
    rho_i = np.zeros((len(Compound),len(T)))
    for i in range(len(Compound)):
        
        for j in range(len(T)):
            rho_i[i][j] = rho_T_A[i] + rho_T_B[i]*T[j] 

    
    # Calculate the temp-dependent sound velocity of compounds
    C_0_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_i[i][j] = C_0_T_B[i]*T[j]+C_0_T_A[i]


    # Calculate molar volume of compounds from temp-dependent density
    V_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_i[i][j] = M_i[i] / rho_i[i][j]

    
    # Calculate volume fractions
    phi_i=np.zeros((len(Compound),len(T)))
    for j in range(len(T)):
        denom_phi = 0
        for i in range(len(compound_input)):
            phi_i[i][j] = V_i[i][j]*mol_fracs[i]
            denom_phi += V_i[i][j]*mol_fracs[i]
        for i in range(len(compound_input)):
            phi_i[i][j]=phi_i[i][j]/denom_phi

    # Calculate mass (fractions
    kappa_i=np.zeros(len(Compound))
    denom_kappa = 0
    for i in range(len(compound_input)):
        kappa_i[i] = M_i[i]*mol_fracs[i]
        denom_kappa += M_i[i]*mol_fracs[i]
    kappa_i=kappa_i/denom_kappa           


    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix += mol_fracs[i]*M_i[i]


    # Calculate the average temp-dependent molar volume of mixture
    V_mix = np.zeros(len(T))      
    V_mix_est = np.zeros(len(T)) 
    V_mix_data = np.zeros(len(T)) 
    V_mix_data_Vm = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_mix_est[j] += mol_fracs[i]*V_i[i][j]
            V_mix_data[j] = M_mix/(density_mix[0] + density_mix[1]*T[j])
            V_mix_data_Vm[j] = molar_volume_mix[0] + molar_volume_mix[1]*T[j]
    if molar_volume_mix == [0,0]:
        if density_mix == [0,0]:
            V_mix = V_mix_est
        else:
            V_mix = V_mix_data
    else:
        V_mix = V_mix_data_Vm


        V_slope_est,V_int_est = np.polyfit(T,V_mix_est,1)
        V_data = molar_volume_mix[1]*T_melt+molar_volume_mix[0]
        V_est = V_slope_est*T_melt+V_int_est
        
        error_C0_slope = 100 * (molar_volume_mix[1]-V_slope_est)/molar_volume_mix[1]
        error_C0_melt = 100 * (V_data-V_est)/V_data
        # error_C0_slope = 100 * (sound_velocity_mix[1]-V_slope_est)/sound_velocity_mix[1]
        # error_C0_int = 100 * (sound_velocity_mix[0]-V_int_est)/sound_velocity_mix[0]
        #print("Molar volume Melt Temp- Estimated:", V_est)
        #print("Molar volume Melt Temp- Data:", V_data)
        #print("Molar volume Slope % Difference: ", error_C0_slope, " %")
        #print("Molar volume Melt Temp % Difference: ", error_C0_melt, " %")
        # print("Sound Velocity Intercept % Difference: ", error_C0_int, " %")
        # error_V_slope = 100 * (molar_volume_mix[1]-V_slope_est)/molar_volume_mix[1]
        # error_V_int = 100 * (molar_volume_mix[0]-V_int_est)/molar_volume_mix[0]
        # print("Molar Volume Slope % Difference: ", error_V_slope, " %")
        # print("Molar Volume Intercept % Difference: ", error_V_int, " %")

        # V_mix_est_avg = np.average(V_mix_est)
        # V_mix_data_avg = np.average(V_mix_data)
        # V_mix_data_Vm_avg = np.average(V_mix_data_Vm)
        # error_densityVm = 100 * (V_mix_data_avg-V_mix_est_avg)/V_mix_data_avg
        # error_Vm = 100 * (V_mix_data_Vm_avg-V_mix_est_avg)/V_mix_data_Vm_avg
        # #print("")
        # print("Molar Volume - From estimated density: ", V_mix_est_avg)
        # print("Molar volume - From mix data density: ", V_mix_data_avg)
        # print("Molar volume - Mix data Molar Volume: ", V_mix_data_Vm_avg)
        # print("Molar Volume % Difference, Density Data/Estimation: ", error_densityVm, " %")
        # print("Molar Volume % Difference, Molar Volume Data/Estimation: ", error_Vm, " %")


    #Calculate the temp-dependent density of mixture
    rho_mix = np.zeros(len(T))
    rho_mix_est = np.zeros(len(T)) 
    rho_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            rho_mix_est[j] += rho_i[i][j]*mol_fracs[i]   #M_mix/V_mix[j]
            rho_mix_data[j] = density_mix[0] + density_mix[1]*T[j]   # If mixture temp-dependent density data exists
    if density_mix == [0,0]:
        rho_mix = rho_mix_est
    else:
        rho_mix = rho_mix_data

        rho_slope_est,rho_int_est = np.polyfit(T,rho_mix_est,1)
        error_rho_slope = 100 * (density_mix[1]-rho_slope_est)/density_mix[1]
        error_rho_int = 100 * (density_mix[0]-rho_int_est)/density_mix[0]
        # print("Density Slope % Difference: ", error_rho_slope, " %")
        # print("Density Intercept % Difference: ", error_rho_int, " %")
        # rho_mix_est_avg = np.average(rho_mix_est)
        # rho_mix_data_avg = np.average(rho_mix_data)
        # error = 100 * (rho_mix_data_avg-rho_mix_est_avg)/rho_mix_data_avg
        # print("Estimated mix density: ", rho_mix_est_avg)
        # print("Data mix density: ", rho_mix_data_avg)
        # print("Density % Difference: ", error, " %")


    # Calculate the temp-dependent sound velocity of mixture
    C_0_mix = np.zeros(len(T))
    C_0_mix_est = np.zeros(len(T)) 
    C_0_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_mix_est[j] += phi_i[i][j]**2 / (kappa_i[i] * C_0_i[i][j]**2)
            C_0_mix_data[j] = sound_velocity_mix[1]*T[j]+sound_velocity_mix[0]    # If mixture temp-dependent sound velocity data exists
    for j in range(len(T)):
        C_0_mix_est[j] = 1 / np.sqrt(C_0_mix_est[j])
    if sound_velocity_mix == [0,0]:
        C_0_mix = C_0_mix_est
    else:
        C_0_mix = C_0_mix_data

        C0_slope_est,C0_int_est = np.polyfit(T,C_0_mix_est,1)
        vsound_data = sound_velocity_mix[1]*T_melt+sound_velocity_mix[0]
        vsound_est = C0_slope_est*T_melt+C0_int_est
        
        error_C0_slope = 100 * (sound_velocity_mix[1]-C0_slope_est)/sound_velocity_mix[1]
        error_C0_melt = 100 * (vsound_data-vsound_est)/vsound_data
        # error_C0_slope = 100 * (sound_velocity_mix[1]-C0_slope_est)/sound_velocity_mix[1]
        # error_C0_int = 100 * (sound_velocity_mix[0]-C0_int_est)/sound_velocity_mix[0]
        # print("Sound Velocity Slope % Difference: ", error_C0_slope, " %")
        # print("Sound Velocity Melt Temp % Difference: ", error_C0_melt, " %")
        # print("Sound Velocity Intercept % Difference: ", error_C0_int, " %")

        # C_0_mix_est_avg = np.average(C_0_mix_est)
        # C_0_mix_data_avg = np.average(C_0_mix_data)
        # error = 100 * (C_0_mix_data_avg-C_0_mix_est_avg)/C_0_mix_data_avg
        # print("Estimated mix sound velocity: ", C_0_mix_est_avg)
        # print("Data mix sound velocity: ", C_0_mix_data_avg)
        # print("sound velocity % Difference: ", error, " %")


    # Calculate mixture specific heat & # of ions
    C_p_mix=0
    C_p_mix_est = 0
    C_p_mix_data = 0
    n_mix=0
    for i in range(len(compound_input)):
        C_p_mix_est += C_p_mix+mol_fracs[i]*C_i_p[i]
        C_p_mix_data=specific_heat_mix
        n_mix=n_mix+mol_fracs[i]*n_i[i]
    if specific_heat_mix[0] == 0:
        C_p_mix = C_p_mix_est
    else:
        if specific_heat_mix[2] == 'm':
            C_p_mix = C_p_mix_data[1]*T_melt+C_p_mix_data[0]
        elif specific_heat_mix[2] == 'g':   # Converts units from J/g/K to J/mol/K
            C_p_mix = (C_p_mix_data[1]*T_melt+C_p_mix_data[0])*1000*(M_mix)

    # Calculate mixture radial distance
    r_ac_mix = 0
    for i in range(len(compound_input)):
        r_ac_mix += mol_fracs[i]*(r_a[i]+r_c[i])

    print(f"  → mean_free_path (sum of ionic radii): {r_ac_mix*1e10:.4f} Å")


    #Calculate compound thermal conductivity
    lambda_i_m=np.zeros(len(T))
    lambda_i_mg=np.zeros(len(T))
    lambda_i_mb=np.zeros(len(T))
    sound_w_time = 0
    for j in range(len(T)):
        lambda_i_m[j] = 1/3 * C_i_p_sp[i] * rho_mix[j] * C_0_mix[j] * r_ac_mix    # Verified with Zhao's results, uses Zhao's data and specific heat capacity
        #lambda_i_mg[j] = 1/3 * C_p_mix * 1/M_mix * rho_mix[j] * C_0_mix[j] * r_ac_mix  # Verified with Zhao's results, uses Zhao's data and but MSTDB heat capacity
        lambda_i_mb[j] = 1/3 * C_p_mix * 1/V_mix[j] * C_0_mix[j] * r_ac_mix  # Uses molar volume at melting point only 
    

    # if sound_w_time == 1:
    #     print("Calculated with temp-dependent sound velocity data.")
    # else:
    #     print("No temp-dependent sound velocity data available. Calculated with melting temp sound velocity only.")

    nan_check = np.isnan(lambda_i_m)
    contains_nan = nan_check.any()
    if contains_nan or lambda_i_m[0] == 0:
        return(T,lambda_i_mb)
    else:
        return(T,lambda_i_m)
    
def SCM(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, sound_velocity_mix=0, C_p_mix=0, alpha=0, expon=0):
    print(f"[SCM] Compounds: {compound_input}")

    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    C_i_p_sp_df=df['C_i_p_sp (J/g/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    C_0_T_A_df=df['SoS(T)_A (A+B*T)']
    C_0_T_B_df=df['SoS(T)_B']
    rho_T_A_df=df['A (Density (g/cm3):   A - BT(K))']
    rho_T_B_df=df['B (Density (g/cm3):   A - BT(K))'] 

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    C_i_p_sp=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    C_0_T_A=np.zeros(len(compound_input))
    C_0_T_B=np.zeros(len(compound_input))
    rho_T_A=np.zeros(len(compound_input))
    rho_T_B=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        C_i_p_sp[i]=C_i_p_sp_df[indices[i]]/0.001   # Convert to J/kg/K
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        C_0_T_A[i]=C_0_T_A_df[indices[i]]
        C_0_T_B[i]=C_0_T_B_df[indices[i]]
        rho_T_A[i]=rho_T_A_df[indices[i]]/0.001
        rho_T_B[i]=rho_T_B_df[indices[i]]/0.001

    # print('')
    # print('####### Phonon Gas Model ############################')

    specific_heat_mix = C_p_mix


    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
    
    #print("PGM, number of ions: ",n_i)
    density_mix = [0,0]
    sound_velocity_mix = [0,0]
    specific_heat_mix = [0,0]

    # Obtain measurement data for function
    func_name = inspect.currentframe().f_code.co_name
    r_ac_mix, density_mix, sound_velocity_mix, specific_heat_mix = prop_lookup(func_name,Compound,mol_fracs)

    # Load SCL data and determine mean free path source
    scl_data = load_scl_data()
    r_ac_mix_scl, pair_scls = get_weighted_scl(compound_input, mol_fracs, scl_data)

    # If SCL data found, prefer it; otherwise try prop_lookup or use sum radii
    mean_free_path_source = None
    if r_ac_mix_scl is not None:
        r_ac_mix = r_ac_mix_scl
        mean_free_path_source = 'SCL_results.csv'
    else:
        r_ac_mix_prop, _, _, _ = prop_lookup(func_name, compound_input, mol_fracs)
        if r_ac_mix_prop and r_ac_mix_prop != 0:
            r_ac_mix = r_ac_mix_prop
            mean_free_path_source = 'prop_lookup'
        else:
            # fallback: average sum of radii
            r_sum = 0
            for i in range(len(compound_input)):
                r_sum += mol_fracs[i] * (r_a[i] + r_c[i])
            r_ac_mix = r_sum
            mean_free_path_source = 'sum_radii'

    # Concise data collection summary
    try:
        compound_count = len(Compound_df)
    except Exception:
        compound_count = 'NA'
    try:
        mstdb_count = len(MSTDB_df) if MSTDB_df is not None else 'NA'
    except Exception:
        mstdb_count = 'NA'
    try:
        scl_count = len(scl_data) if scl_data is not None else 0
    except Exception:
        scl_count = 0
    print(f"Data summary: compounds={compound_count}, MSTDB={mstdb_count}, SCL_entries={scl_count}")
    print(f"Mean free path: source={mean_free_path_source}, value={r_ac_mix*1e10:.3f} Å")

    

    #Find the temperature dependent density from data
    rho_i = np.zeros((len(Compound),len(T)))
    for i in range(len(Compound)):
        
        for j in range(len(T)):
            rho_i[i][j] = rho_T_A[i] + rho_T_B[i]*T[j] 

    
    # Calculate the temp-dependent sound velocity of compounds
    C_0_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_i[i][j] = C_0_T_B[i]*T[j]+C_0_T_A[i]


    # Calculate molar volume of compounds from temp-dependent density
    V_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_i[i][j] = M_i[i] / rho_i[i][j]

    
    # Calculate volume fractions
    phi_i=np.zeros((len(Compound),len(T)))
    for j in range(len(T)):
        denom_phi = 0
        for i in range(len(compound_input)):
            phi_i[i][j] = V_i[i][j]*mol_fracs[i]
            denom_phi += V_i[i][j]*mol_fracs[i]
        for i in range(len(compound_input)):
            phi_i[i][j]=phi_i[i][j]/denom_phi

    # Calculate mass (fractions
    kappa_i=np.zeros(len(Compound))
    denom_kappa = 0
    for i in range(len(compound_input)):
        kappa_i[i] = M_i[i]*mol_fracs[i]
        denom_kappa += M_i[i]*mol_fracs[i]
    kappa_i=kappa_i/denom_kappa           


    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix=M_mix+mol_fracs[i]*M_i[i]


    # Calculate the average temp-dependent molar volume of mixture
    V_mix = np.zeros(len(T))      
    V_mix_est = np.zeros(len(T)) 
    V_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_mix_est[j] += mol_fracs[i]*V_i[i][j]
            if density_mix == [0,0]:
                pass
            else:
                V_mix_data[j] = M_mix/(density_mix[0] + density_mix[1]*T[j])
    if density_mix == [0,0]:
        V_mix = V_mix_est
    else:
        V_mix = V_mix_data

        V_mix_est_avg = np.average(V_mix_est)
        V_mix_data_avg = np.average(V_mix_data)
        error = 100 * (V_mix_data_avg-V_mix_est_avg)/V_mix_data_avg
        #print("Molar volume from estimated mix density: ", V_mix_est_avg)
        #print("Molar volume from data mix density: ", V_mix_data_avg)
        #print("Molar volume % Difference: ", error, " %")


    #Calculate the temp-dependent density of mixture
    rho_mix = np.zeros(len(T))
    rho_mix_est = np.zeros(len(T)) 
    rho_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            rho_mix_est[j] += rho_i[i][j]*mol_fracs[i]   #M_mix/V_mix[j]
            rho_mix_data[j] = density_mix[0] + density_mix[1]*T[j]   # If mixture temp-dependent density data exists
    if density_mix == [0,0]:
        rho_mix = rho_mix_est
    else:
        rho_mix = rho_mix_data

        rho_mix_est_avg = np.average(rho_mix_est)
        rho_mix_data_avg = np.average(rho_mix_data)
        error = 100 * (rho_mix_data_avg-rho_mix_est_avg)/rho_mix_data_avg
        #print("Estimated mix density: ", rho_mix_est_avg)
        #print("Data mix density: ", rho_mix_data_avg)
        # print("Density % Difference: ", error, " %")


    # Calculate the temp-dependent sound velocity of mixture
    C_0_mix = np.zeros(len(T))
    C_0_mix_est = np.zeros(len(T)) 
    C_0_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_mix_est[j] += phi_i[i][j]**2 / (kappa_i[i] * C_0_i[i][j]**2)
            C_0_mix_data[j] = sound_velocity_mix[1]*T[j]+sound_velocity_mix[0]    # If mixture temp-dependent sound velocity data exists
    for j in range(len(T)):
        C_0_mix_est[j] = 1 / np.sqrt(C_0_mix_est[j])
    if sound_velocity_mix == [0,0]:
        C_0_mix = C_0_mix_est
    else:
        C_0_mix = C_0_mix_data

        C_0_mix_est_avg = np.average(C_0_mix_est)
        C_0_mix_data_avg = np.average(C_0_mix_data)
        error = 100 * (C_0_mix_data_avg-C_0_mix_est_avg)/C_0_mix_data_avg
        #print("Estimated mix sound velocity: ", C_0_mix_est_avg)
        #print("Data mix sound velocity: ", C_0_mix_data_avg)
        # print("sound velocity % Difference: ", error, " %")


    # Calculate mixture specific heat & # of ions
    C_p_mix=0
    C_p_mix_est = 0
    C_p_mix_data = 0
    n_mix=0
    for i in range(len(compound_input)):
        C_p_mix_est += C_p_mix+mol_fracs[i]*C_i_p[i]
        C_p_mix_data=specific_heat_mix
        n_mix=n_mix+mol_fracs[i]*n_i[i]
    if specific_heat_mix[0] == 0:
        C_p_mix = C_p_mix_est
    else:
        C_p_mix = C_p_mix_data

        C_p_mix_est_avg = np.average(C_p_mix_est)
        C_p_mix_data_avg = np.average(C_p_mix_data)
        error = 100 * (C_p_mix_data_avg-C_p_mix_est_avg)/C_p_mix_data_avg
        #print("Estimated mix specific heat: ", C_p_mix_est_avg)
        #print("Data mix specific heat: ", C_p_mix_data_avg)
        # print("specific heat % Difference: ", error, " %")

    # print("Calculated mixture specific heat (melting point): ",C_p_mix/V_mix[0])
    # print("Calculated mixtur sound velocity (melting point): ",C_0_mix[0])
    # print("Calculated average mean free path: ",r_ac_mix)

    # Calculate temperature-dependent slopes for specific heat and sound velocity
    # First, ensure C_p_mix is an array with the same length as T
    if not hasattr(C_p_mix, '__len__') or len(C_p_mix) == 1:
        # If C_p_mix is a scalar or single value, create an array with that value
        C_p_mix_array = np.full_like(T, C_p_mix[0] if hasattr(C_p_mix, '__len__') else C_p_mix)
    else:
        C_p_mix_array = C_p_mix
    
    # Calculate specific heat per volume for all temperatures
    C_p_vol = np.zeros_like(T)
    for j in range(len(T)):
        if V_mix[j] > 0:
            C_p_vol[j] = C_p_mix_array[j] / V_mix[j]  # J/m³/K
    
    # Use linear regression to find the slope of C_p_vol vs T
    if len(T) > 1:
        specific_heat_slope = np.polyfit(T - T[0], C_p_vol, 1)[0]  # dC_p/dT in J/m³/K²
        sound_velocity_slope = np.polyfit(T - T[0], C_0_mix, 1)[0]  # dC_0/dT in m/s/K
    else:
        specific_heat_slope = 0
        sound_velocity_slope = 0
    
    # Calculate thermal conductivity for both methods
    lambda_i_m = np.zeros(len(T))
    lambda_i_mb = np.zeros(len(T))
    
    # k as product of sums
    for j in range(len(T)):
        # Method 1: Using specific heat capacity from data
        lambda_i_m[j] = 1/3 * C_i_p_sp[0] * rho_mix[j] * C_0_mix[j] * r_ac_mix
        
        # Method 2: Using molar heat capacity and molar volume
        if V_mix[j] > 0:
            lambda_i_mb[j] = 1/3 * C_p_vol[j] * C_0_mix[j] * r_ac_mix

    # # k as sum of products
    
    # # Calculate partial specific heat for each compound
    # C_p_part = [np.zeros(len(T)) for _ in range(len(Compound))]
    # for i in range(len(Compound)):
    #     # Calculate bond strength factor from PDF peaks
    #     f_K = pair_scls[i][1] - pair_scls[i][0]
    #     for j in range(len(T)):
    #         C_p_part[i][j] += mol_fracs[i]*C_p_vol[j]

    # # Calculate partial sound velocity
    # v_part = np.zeros(len(T))
    # for i in range(len(Compound)):
    #     for j in range(len(T)):
    #         v_part[j] += mol_fracs[i]*C_0_mix[j]

    # for j in range(len(T)):
    #     lambda_i_s[j] = 1/3 * (C_i_p_sp[0] * rho_mix[j] + C_p_vol[j]) * C_0_mix[j] * r_ac_mix
    
    # Check for NaN values or zero thermal conductivity
    nan_check = np.isnan(lambda_i_m)
    contains_nan = nan_check.any()
    
    if contains_nan or lambda_i_m[0] == 0:
        cp_vol_at_melt = C_p_vol[0] if len(C_p_vol) > 0 else 0
        # print(f"k: {lambda_i_mb[0]:.3f} W/m/K")
        # print(f"cp: {cp_vol_at_melt:.2f} J/m³/K")
        # print(f"vs: {C_0_mix[0]:.2f} m/s")
        
        return (T, {
            'thermal_conductivity': lambda_i_mb,
            'specific_heat_m': cp_vol_at_melt,  # J/m³/K at melting point
            'specific_heat_prime': specific_heat_slope,  # dC_p/dT in J/m³/K²
            'sound_velocity_m': C_0_mix[0],  # m/s at melting point
            'sound_velocity_prime': sound_velocity_slope  # dC_0/dT in m/s/K
        })
    else:
        # print(f"k: {lambda_i_m[0]:.3f} W/m/K")
        # print(f"cp: {C_i_p_sp[0] * rho_mix[0]:.2f} J/m³/K")
        # print(f"vs: {C_0_mix[0]:.2f} m/s")
        # print(f"lambda: {r_ac_mix*1e10:.3f} Å")
        
        return (T, {
            'thermal_conductivity': lambda_i_m,
            'specific_heat_m': C_i_p_sp[0] * rho_mix[0],  # J/m³/K at melting point
            'specific_heat_prime': specific_heat_slope,  # dC_p/dT in J/m³/K²
            'sound_velocity_m': C_0_mix[0],  # m/s at melting point
            'sound_velocity_prime': sound_velocity_slope  # dC_0/dT in m/s/K
        })

def SCM_Mix(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, sound_velocity_mix=0, C_p_mix=0, alpha=0, expon=0):
    print(f"[SCM_Mix] Compounds: {compound_input}")
    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    C_i_p_sp_df=df['C_i_p_sp (J/g/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    C_0_T_A_df=df['SoS(T)_A (A+B*T)']
    C_0_T_B_df=df['SoS(T)_B']
    rho_T_A_df=df['A (Density (g/cm3):   A - BT(K))']
    rho_T_B_df=df['B (Density (g/cm3):   A - BT(K))'] 

    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    C_i_p_sp=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    C_0_T_A=np.zeros(len(compound_input))
    C_0_T_B=np.zeros(len(compound_input))
    rho_T_A=np.zeros(len(compound_input))
    rho_T_B=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        C_i_p_sp[i]=C_i_p_sp_df[indices[i]]/0.001   # Convert to J/kg/K
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        C_0_T_A[i]=C_0_T_A_df[indices[i]]
        C_0_T_B[i]=C_0_T_B_df[indices[i]]
        rho_T_A[i]=rho_T_A_df[indices[i]]/0.001
        rho_T_B[i]=rho_T_B_df[indices[i]]/0.001

    # print('')
    # print('####### Phonon Gas Model ############################')

    specific_heat_mix = C_p_mix


    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
    
    #print("PGM, number of ions: ",n_i)
    density_mix = [0,0]
    sound_velocity_mix = [0,0]
    specific_heat_mix = [0,0]

    # Obtain measurement data for function
    func_name = inspect.currentframe().f_code.co_name
    r_ac_mix, density_mix, sound_velocity_mix, specific_heat_mix = prop_lookup(func_name,Compound,mol_fracs)

    # Load SCL data and determine mean free path source for SCM_Mix
    scl_data = load_scl_data()
    r_ac_mix_scl, pair_scls = get_weighted_scl(compound_input, mol_fracs, scl_data)

    mean_free_path_source_mix = None
    if r_ac_mix_scl is not None:
        r_ac_mix = r_ac_mix_scl
        mean_free_path_source_mix = 'SCL_results.csv'
    else:
        r_ac_mix_prop, _, _, _ = prop_lookup(func_name, compound_input, mol_fracs)
        if r_ac_mix_prop and r_ac_mix_prop != 0:
            r_ac_mix = r_ac_mix_prop
            mean_free_path_source_mix = 'prop_lookup'
        else:
            r_sum = 0
            for i in range(len(compound_input)):
                r_sum += mol_fracs[i] * (r_a[i] + r_c[i])
            r_ac_mix = r_sum
            mean_free_path_source_mix = 'sum_radii'

    # Concise data collection summary for SCM_Mix
    try:
        compound_count = len(Compound_df)
    except Exception:
        compound_count = 'NA'
    try:
        mstdb_count = len(MSTDB_df) if MSTDB_df is not None else 'NA'
    except Exception:
        mstdb_count = 'NA'
    try:
        scl_count = len(scl_data) if scl_data is not None else 0
    except Exception:
        scl_count = 0
    print(f"Data summary: compounds={compound_count}, MSTDB={mstdb_count}, SCL_entries={scl_count}")
    print(f"Mean free path: source={mean_free_path_source_mix}, value={r_ac_mix*1e10:.3f} Å")

    #Find the temperature dependent density from data
    rho_i = np.zeros((len(Compound),len(T)))
    for i in range(len(Compound)):
        
        for j in range(len(T)):
            rho_i[i][j] = rho_T_A[i] + rho_T_B[i]*T[j] 

    
    # Calculate the temp-dependent sound velocity of compounds
    C_0_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_i[i][j] = C_0_T_B[i]*T[j]+C_0_T_A[i]


    # Calculate molar volume of compounds from temp-dependent density
    V_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_i[i][j] = M_i[i] / rho_i[i][j]

    
    # Calculate volume fractions
    phi_i=np.zeros((len(Compound),len(T)))
    for j in range(len(T)):
        denom_phi = 0
        for i in range(len(compound_input)):
            phi_i[i][j] = V_i[i][j]*mol_fracs[i]
            denom_phi += V_i[i][j]*mol_fracs[i]
        for i in range(len(compound_input)):
            phi_i[i][j]=phi_i[i][j]/denom_phi

    # Calculate mass (fractions
    kappa_i=np.zeros(len(Compound))
    denom_kappa = 0
    for i in range(len(compound_input)):
        kappa_i[i] = M_i[i]*mol_fracs[i]
        denom_kappa += M_i[i]*mol_fracs[i]
    kappa_i=kappa_i/denom_kappa           


    #Calculate mixture molecular weight
    M_mix=0
    for i in range(len(compound_input)):
        M_mix=M_mix+mol_fracs[i]*M_i[i]


    # Calculate the average temp-dependent molar volume of mixture
    V_mix = np.zeros(len(T))      
    V_mix_est = np.zeros(len(T)) 
    V_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            V_mix_est[j] += mol_fracs[i]*V_i[i][j]
            if density_mix == [0,0]:
                pass
            else:
                V_mix_data[j] = M_mix/(density_mix[0] + density_mix[1]*T[j])
    if density_mix == [0,0]:
        V_mix = V_mix_est
    else:
        V_mix = V_mix_data

        V_mix_est_avg = np.average(V_mix_est)
        V_mix_data_avg = np.average(V_mix_data)
        error = 100 * (V_mix_data_avg-V_mix_est_avg)/V_mix_data_avg
        #print("Molar volume from estimated mix density: ", V_mix_est_avg)
        #print("Molar volume from data mix density: ", V_mix_data_avg)
        #print("Molar volume % Difference: ", error, " %")


    #Calculate the temp-dependent density of mixture
    rho_mix = np.zeros(len(T))
    rho_mix_est = np.zeros(len(T)) 
    rho_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            rho_mix_est[j] += rho_i[i][j]*mol_fracs[i]   #M_mix/V_mix[j]
            rho_mix_data[j] = density_mix[0] + density_mix[1]*T[j]   # If mixture temp-dependent density data exists
    if density_mix == [0,0]:
        rho_mix = rho_mix_est
    else:
        rho_mix = rho_mix_data

        rho_mix_est_avg = np.average(rho_mix_est)
        rho_mix_data_avg = np.average(rho_mix_data)
        error = 100 * (rho_mix_data_avg-rho_mix_est_avg)/rho_mix_data_avg
        #print("Estimated mix density: ", rho_mix_est_avg)
        #print("Data mix density: ", rho_mix_data_avg)
        # print("Density % Difference: ", error, " %")


    # Calculate the temp-dependent sound velocity of mixture
    C_0_mix = np.zeros(len(T))
    C_0_mix_est = np.zeros(len(T)) 
    C_0_mix_data = np.zeros(len(T)) 
    for i in range(len(compound_input)):
        for j in range(len(T)):
            C_0_mix_est[j] += phi_i[i][j]**2 / (kappa_i[i] * C_0_i[i][j]**2)
            C_0_mix_data[j] = sound_velocity_mix[1]*T[j]+sound_velocity_mix[0]    # If mixture temp-dependent sound velocity data exists
    for j in range(len(T)):
        C_0_mix_est[j] = 1 / np.sqrt(C_0_mix_est[j])
    if sound_velocity_mix == [0,0]:
        C_0_mix = C_0_mix_est
    else:
        C_0_mix = C_0_mix_data

        C_0_mix_est_avg = np.average(C_0_mix_est)
        C_0_mix_data_avg = np.average(C_0_mix_data)
        error = 100 * (C_0_mix_data_avg-C_0_mix_est_avg)/C_0_mix_data_avg
        #print("Estimated mix sound velocity: ", C_0_mix_est_avg)
        #print("Data mix sound velocity: ", C_0_mix_data_avg)
        # print("sound velocity % Difference: ", error, " %")


    # Calculate mixture specific heat & # of ions
    C_p_mix=0
    C_p_mix_est = 0
    C_p_mix_data = 0
    n_mix=0
    for i in range(len(compound_input)):
        C_p_mix_est += C_p_mix+mol_fracs[i]*C_i_p[i]
        C_p_mix_data=specific_heat_mix
        n_mix=n_mix+mol_fracs[i]*n_i[i]
    if specific_heat_mix[0] == 0:
        C_p_mix = C_p_mix_est
    else:
        if specific_heat_mix[2] == 'm':
            C_p_mix = C_p_mix_data[1]*T_melt+C_p_mix_data[0]
        elif specific_heat_mix[2] == 'g':   # Converts units from J/g/K to J/mol/K
            C_p_mix = (C_p_mix_data[1]*T_melt+C_p_mix_data[0])*1000*(M_mix)

        C_p_mix_est_avg = np.average(C_p_mix_est)
        C_p_mix_data_avg = np.average(C_p_mix)
        error = 100 * (C_p_mix_data_avg-C_p_mix_est_avg)/C_p_mix_data_avg
        #print("Estimated mix specific heat: ", C_p_mix_est_avg)
        #print("Data mix specific heat: ", C_p_mix_data_avg)
        # print("specific heat % Difference: ", error, " %")
    # print("Calculated mixture specific heat (melting point): ",C_p_mix/V_mix[0])
    # print("Calculated mixtur sound velocity (melting point): ",C_0_mix[0])
    # print("Calculated average mean free path: ",r_ac_mix)

    # Calculate temperature-dependent slopes for specific heat and sound velocity
    # Use linear regression to find the slope of C_p_mix vs T
    if len(T) > 1:
        # Calculate specific heat per volume for all temperatures
        C_p_vol = np.zeros(len(T))
        for j in range(len(T)):
            C_p_vol[j] = C_p_mix / V_mix[j]  # Convert to J/m³/K
        
        # Calculate specific heat slope (dC_p/dT) in J/m³/K²
        specific_heat_slope = np.polyfit(T - T[0], C_p_vol, 1)[0]
        
        # Calculate sound velocity slope (dC_0/dT) in m/s/K
        sound_velocity_slope = np.polyfit(T - T[0], C_0_mix, 1)[0]
    else:
        specific_heat_slope = 0
        sound_velocity_slope = 0

    # Calculate compound thermal conductivity
    lambda_i_m=np.zeros(len(T))
    lambda_i_mg=np.zeros(len(T))
    lambda_i_mb=np.zeros(len(T))
    sound_w_time = 0
    for j in range(len(T)):
        lambda_i_m[j] = 1/3 * C_i_p_sp[i] * rho_mix[j] * C_0_mix[j] * r_ac_mix    # Verified with Zhao's results, uses Zhao's data and specific heat capacity
        #lambda_i_mg[j] = 1/3 * C_p_mix * 1/M_mix * rho_mix[j] * C_0_mix[j] * r_ac_mix  # Verified with Zhao's results, uses Zhao's data and but MSTDB heat capacity
        lambda_i_mb[j] = 1/3 * C_p_mix * 1/V_mix[j] * C_0_mix[j] * r_ac_mix  # Uses molar volume at melting point only 
    

    # if sound_w_time == 1:
    #     print("Calculated with temp-dependent sound velocity data.")
    # else:
    #     print("No temp-dependent sound velocity data available. Calculated with melting temp sound velocity only.")
    

    nan_check = np.isnan(lambda_i_m)
    contains_nan = nan_check.any()
    if contains_nan or lambda_i_m[0] == 0:
        # print("k: ",lambda_i_mb[0])
        # # print("PGM-PDF_Avg (1/3*Cp*vs): ",lambda_i_mb[0]/r_ac_mix)
        # # print("PGM-PDF_Avg (1/3*Cp*MFP): ",lambda_i_mb[0]/C_0_mix[0])
        # # print("PGM-PDF_Avg (1/3*vs*MFP): ",lambda_i_mb[0]/(C_p_mix/V_mix[0]))
        # # print("lambda_BC: ",lambda_i_mb[0]/(1/3*C_p_mix*C_0_mix[0]))
        print("l_sc: ",r_ac_mix)
        # print("cp: ",C_p_mix/V_mix[0])
        # print("vs: ",C_0_mix[0])
        return (T, {
            'thermal_conductivity': lambda_i_mb,
            'specific_heat_m': C_p_mix/V_mix[0],
            'specific_heat_prime': specific_heat_slope,  # dC_p/dT in J/m³/K²
            'sound_velocity_m': C_0_mix[0],
            'sound_velocity_prime': sound_velocity_slope  # dC_0/dT in m/s/K
        })
    else:
        # print("k: ",lambda_i_m[0])
        # # print("PGM-PDF_Avg (1/3*Cp*vs): ",lambda_i_mb[0]/r_ac_mix)
        # # print("PGM-PDF_Avg (1/3*Cp*MFP): ",lambda_i_mb[0]/C_0_mix[0])
        # # print("PGM-PDF_Avg (1/3*vs*MFP): ",lambda_i_mb[0]/C_i_p_sp[0])
        # # print("lambda_BC: ",lambda_i_m[0]/(1/3*C_p_mix*C_0_mix[0]))
        # # print("lambda_SCL: ",r_ac_mix)
        # print("cp: ",C_i_p_sp[0]* rho_mix[0])
        # print("vs: ",C_0_mix[0])
        print("l_sc: ", r_ac_mix)
        return (T, {
            'thermal_conductivity': lambda_i_m,
            'specific_heat_m': C_i_p_sp[0] * rho_mix[0],
            'specific_heat_prime': specific_heat_slope,  # dC_p/dT in J/m³/K²
            'sound_velocity_m': C_0_mix[0],
            'sound_velocity_prime': sound_velocity_slope  # dC_0/dT in m/s/K
        })

def Ideal(df,MSTDB_df,SCL_PDF_df,compound_input,mol_fracs,Temp_Range, V_m=0, density_mix=0, C_p_mix=0, alpha=0, expon=0):
    print("################ Ideal Mixing ############################")
    print(compound_input,mol_fracs)
    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']
    M_i_df=df['M_i (g/mol)']
    T_i_m_df=df['T_i_m (K)']
    V_i_m_df=df['V_i_m (m^3/mol)']
    alpha_i_m_df=df['alpha_i_m (K^-1)']
    C_i_0_df=df['C_i_0 (m/s)']
    C_i_p_df=df['C_i_p (J/mol/K)']
    r_c_df=df['r_c (m)']
    r_a_df=df['r_a (m)']
    rho_i_m_df=df['rho_m (g/m^3)']
    # TC_SCM	Ref	TC_exp
    TC_SCM_df=df['TC_SCM']
    TC_exp_df=df['TC_exp']
    # TC_exp Ref	TC_SCM Ref
    # TC_exp_ref_df=df['TC_exp Ref']


    #initialize arrays of indices of compounds in the dataframe, and boolean values of whether they are in the dataframe
    indices=np.zeros(len(compound_input))
    indices_exist=np.full(len(compound_input),False)


    #find indices in dataframe and store index values
    for i in range(len(compound_input)):
        for j in range(len(Compound_df)):
            if compound_input[i]==Compound_df[j]:
                indices[i]=j
                indices_exist[i]=True

    #check to make sure all the input compounds actually exist. Quit program if not
    for i in range(len(compound_input)):
        problem=False
        if indices_exist[i]==False:
            print('Warning: ' + str(compound_input[i]) + ' is not included in the compound data spreadsheet' )
            problem=True
        if problem==True:
            print('Force quit.')
            quit()

    #Generate arrays of all parameters that already exist in the dataframe, but in the order of the input compound array
    Compound=[]
    for i in range(len(compound_input)):
        Compound.append('')
    M_i=np.zeros(len(compound_input))
    T_i_m=np.zeros(len(compound_input))
    V_i_m=np.zeros(len(compound_input))
    alpha_i_m=np.zeros(len(compound_input))
    C_i_0=np.zeros(len(compound_input))
    C_i_p=np.zeros(len(compound_input))
    r_c=np.zeros(len(compound_input))
    r_a=np.zeros(len(compound_input))
    rho_i_m=np.zeros(len(compound_input))
    TC_SCM=np.zeros(len(compound_input))
    TC_exp=np.zeros(len(compound_input))
    # TC_exp_ref=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        Compound[i]=Compound_df[indices[i]]
        M_i[i]=M_i_df[indices[i]]*0.001         # Convert to kg/mol
        T_i_m[i]=T_i_m_df[indices[i]]
        V_i_m[i]=V_i_m_df[indices[i]]
        alpha_i_m[i]=alpha_i_m_df[indices[i]]
        C_i_0[i]=C_i_0_df[indices[i]]
        C_i_p[i]=C_i_p_df[indices[i]]
        r_c[i]=r_c_df[indices[i]]
        r_a[i]=r_a_df[indices[i]]
        rho_i_m[i]=rho_i_m_df[indices[i]]*0.001 # Convert to kg/m^3
        TC_SCM[i]=TC_SCM_df[indices[i]]
        TC_exp[i]=TC_exp_df[indices[i]]
        # TC_exp_ref[i]=TC_exp_ref_df[indices[i]]

    #Calculate Gruneisen parameter
    gamma_i_m=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        gamma_i_m[i]=((M_i[i]*alpha_i_m[i]*C_i_0[i]**2)/C_i_p[i])

    #initialize a temperature range
    T_melt = Temp_Range[0]
    T=np.linspace(T_melt,Temp_Range[1],num=100)


    #calculate constant volume heat capacities at melting temp
    C_i_v=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        C_i_v[i]=C_i_p[i]/(1+alpha_i_m[i]*gamma_i_m[i]*T_i_m[i])

    
    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    n_i_c=np.zeros(len(compound_input))
    n_i_a=np.zeros(len(compound_input))
    comps = 0

    #Calculate number of ions per compound
    n_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        for c in Compound[i]:
            if c.isupper():
                n_i[i] = n_i[i] + 1
            elif c.isnumeric():
                c = int(c)
                n_i[i] = n_i[i] + (c-1)
            if comps == 0:
                n_i_c[i] = n_i[i]
                if "NO3" in Compound[i]:
                    n_i_c[i] = n_i_c[i] +1
                    n_i_a[i] = 3
                comps = 1
            else:
                if "NO3" in Compound[i]:
                    continue
                n_i_a[i] = n_i[i] - n_i_c[i]
        comps = 0
        
    # Calculate compound psi term
    psi_i=np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        psi_i[i] = 1 + n_i_c[i]/n_i_a[i]

    # Calculate compound number density
    n_dens_i = np.zeros(len(compound_input))
    for i in range(len(compound_input)):
        n_dens_i[i] = Avog *n_i[i]/V_i_m[i]

    #Calculate compound thermal conductivity(T) based on compound minimum thermal conductivites
    lambda_i = np.zeros((len(Compound),len(T)))
    for i in range(len(compound_input)):
        if TC_exp[i] != 0:
            print("Using experimental data from " + Compound[i])
            for j in range(len(T)):
                lambda_i[i][j] = TC_exp[i]* (1 - alpha_i_m[i]*(gamma_i_m[i] + 1/3)*(T[j] - T_i_m[i]))
        elif TC_SCM[i] != 0:
            print("Using SCM data from " + Compound[i])
            for j in range(len(T)):
                lambda_i[i][j] = TC_SCM[i]
        else:
            print("Using Gheribi model for " + Compound[i])
            for j in range(len(T)):
                lambda_i[i][j] = (1 + n_i_c[i]/n_i_a[i]) * k_B * n_dens_i[i]**(2/3) * C_i_0[i] * (1 - alpha_i_m[i]*(gamma_i_m[i] + 1/3)*(T[j] - T_i_m[i]))
 
    #Calculate ideal thermal conductivity
    lambda_mix_T = np.zeros(len(T))
    for i in range(len(compound_input)):
        for j in range(len(T)):
            lambda_mix_T[j] += lambda_i[i][j]*mol_fracs[i]

    print("Ideal Mixing: ",lambda_mix_T[0])
    return(T,lambda_mix_T)

def load_scl_data():
    """Load and process SCL data from SCL_results.csv"""
    scl_data = []
    try:
        # Try several sensible locations for SCL_results.csv:
        candidate_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SCL_results.csv'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Structural_Coherence_Length', 'SCL_results.csv'),
            'SCL_results.csv'
        ]
        scl_df = None
        used_path = None
        for p in candidate_paths:
            try:
                if os.path.exists(p):
                    scl_df = pd.read_csv(p, encoding='latin-1')
                    used_path = p
                    break
            except Exception:
                continue
        if scl_df is None:
            raise FileNotFoundError('SCL_results.csv not found in expected locations')

        # Process each row in the SCL results
        for _, row in scl_df.iterrows():
            comp_str = row['Composition']
            components, fractions = parse_composition(comp_str)

            # Store the composition data with pairs
            entry = {
                'components': components,
                'fractions': fractions,
                'avg_scl': float(row['Average SCL (A)']),
                'pairs': {},
                'pair_f_K_pk_data': {},
                'pair_f_K_min_data': {}
            }

            # Store pair data
            for i in range(1, 7):  # Check for up to 6 pairs
                pair_col = f'Pair {i} Label'
                scl_col = f'Pair {i} SCL_i (A)'
                pair_f_K_pk_col = f'Pair {i} Peak Y'
                pair_f_K_min_col = f'Pair {i} Min Y'
                if pair_col in row and pd.notna(row[pair_col]) and scl_col in row and pd.notna(row[scl_col]):
                    entry['pairs'][row[pair_col]] = float(row[scl_col])
                    entry['pair_f_K_pk_data'][row[pair_col]] = float(row[pair_f_K_pk_col])
                    entry['pair_f_K_min_data'][row[pair_col]] = float(row[pair_f_K_min_col])
            scl_data.append(entry)
    except FileNotFoundError:
        print("SCL_results.csv not found. Using average SCL values.")
    return scl_data

def get_weighted_scl(compound_input, mol_fracs, scl_data):                    
    """
    Calculate weighted SCL for the given composition.
    
    Args:
        compound_input: List of compound names (e.g., ['LiF', 'NaF', 'KF'])
        mol_fracs: List of mole fractions corresponding to compound_input
        scl_data: List of SCL data entries from load_scl_data()
    """
    # Normalize input mole fractions to sum to 1
    total = sum(mol_fracs)
    if abs(total - 1.0) > 1e-6:
        mol_fracs = [f/total for f in mol_fracs]
    
    # Create a list of (compound, fraction) pairs and sort them by compound name for consistency
    input_pairs = sorted(zip(compound_input, mol_fracs), key=lambda x: x[0])
    input_compounds = [x[0] for x in input_pairs]
    input_fracs = [x[1] for x in input_pairs]

    # Find matching composition in SCL data (allowing for different order and small fraction differences)
    matched_entry = None
    for entry in scl_data:
        entry_pairs = sorted(zip(entry['components'], entry['fractions']), key=lambda x: x[0])
        entry_compounds = [x[0] for x in entry_pairs]
        entry_fracs = [x[1] for x in entry_pairs]

        if set(entry_compounds) == set(input_compounds):
            # Reorder the entry's fractions to match the input order
            reordered_fracs = []
            for comp in input_compounds:
                idx = entry_compounds.index(comp)
                reordered_fracs.append(entry_fracs[idx])

            if all(abs(f1 - f2) < 0.02 for f1, f2 in zip(input_fracs, reordered_fracs)):
                matched_entry = entry
                # Update the entry's components and fractions to match the input order
                matched_entry['components'] = input_compounds
                matched_entry['fractions'] = input_fracs
                break
    
    if not matched_entry:
        print(f"No SCL data found for composition: {compound_input} with fractions {mol_fracs}")
        # Print first few SCL entries for debugging
        for i, e in enumerate(scl_data[:5]):
            try:
                print(f"  SCL entry[{i}]: components={e['components']}, fractions={e['fractions']}, pairs={list(e['pairs'].keys())}")
            except Exception:
                pass
        return None, None
    
    # If no pair data is available, return the average SCL
    if not matched_entry['pairs']:
        print(f"Using average SCL for composition: {compound_input}")
        return matched_entry['avg_scl'] * 1e-10, None  # Convert from A to m

    # Build a normalized map of pair labels from the SCL entry: remove hyphens and digits
    def _normalize_label(s):
        return ''.join([c for c in s if not c.isdigit() and c != '-'])

    pair_map = { _normalize_label(k): v for k, v in matched_entry['pairs'].items() }

    # For each compound, normalize and look up its pair SCL (e.g., 'KCl' -> match 'K-Cl' -> 'KCl')
    weighted_sum = 0.0
    total_weight = 0.0
    pair_scls = {}
    for comp, frac in zip(input_compounds, input_fracs):
        comp_norm = ''.join([c for c in comp if not c.isdigit()])
        # Try direct match
        if comp_norm in pair_map:
            scl_i = pair_map[comp_norm]
            pair_scls[comp_norm] = scl_i
            weighted_sum += scl_i * frac
            total_weight += frac
        else:
            # Try to find any pair that contains the normalized comp as substring
            found = False
            for k, v in pair_map.items():
                if comp_norm in k:
                    pair_scls[k] = v
                    weighted_sum += v * frac
                    total_weight += frac
                    found = True
                    break
            if not found:
                # No per-compound pair found; warn and continue
                print(f"Warning: no pair SCL found in SCL entry for component '{comp}' (normalized '{comp_norm}')")

    if total_weight > 0:
        avg_scl = (weighted_sum / total_weight) * 1e-10  # Convert from A to m
        print(f"Using weighted average SCL of {avg_scl*1e10:.3f} A for composition: {compound_input}")
        print(f"Mole fractions: {mol_fracs}")
        print(f"Individual pair SCLs: {pair_scls}")
        return avg_scl, pair_scls

    # Fall back to average SCL if no pairs were found
    print(f"Using average SCL (fallback) for composition: {compound_input}")
    return matched_entry['avg_scl'] * 1e-10, None  # Convert from A to m

def parse_composition(comp_str):
    """Parse composition string like '0.465LiF-0.115NaF-0.42KF' into components and fractions."""
    components = []
    fractions = []
    
    # Split by '-' and process each component
    for part in comp_str.split('-'):
        # Find the index where the compound name starts
        comp_start = 0
        for i, c in enumerate(part):
            if c.isalpha():
                comp_start = i
                break
        
        # Extract fraction and compound
        frac = float(part[:comp_start])
        comp = part[comp_start:]
        
        components.append(comp)
        fractions.append(frac)
    
    # Normalize fractions to sum to 1 (in case of rounding errors)
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:  # Only normalize if there's a significant difference
        fractions = [f/total for f in fractions]
    
    return components, fractions

def compositions_match(comp1, fracs1, comp2, fracs2, tol=0.01):
    """Check if two compositions match (allowing for different order and small fraction differences)."""
    if set(comp1) != set(comp2):
        return False
    
    # Create a mapping of compound to fraction for both compositions
    comp_frac1 = {c: f for c, f in zip(comp1, fracs1)}
    comp_frac2 = {c: f for c, f in zip(comp2, fracs2)}
    
    # Check all fractions are within tolerance
    for comp in comp_frac1:
        if abs(comp_frac1[comp] - comp_frac2[comp]) > tol:
            return False
    
    return True

def density(df,sheet_name):

    #Define smaller dataframes of each variable contained in the spreadsheet
    Compound_df=df['Compound']

    #Define smaller dataframes of each variable contained in the spreadsheet
    rho_T_A_df=df['A (Density (g/cm3):   A - BT(K))']
    rho_T_B_df=df['B (Density (g/cm3):   A - BT(K))'] 

    #find indices in dataframe and store index values
    for j in range(len(Compound_df)):
        if sheet_name==Compound_df[j]:
            index=j

    density_A=rho_T_A_df[index]*100**3/1000
    density_B=rho_T_B_df[index]*100**3/1000

    if pd.isna(density_A):
        # Read the Excel file
        df_density = pd.read_excel('Density.xlsx', sheet_name=sheet_name)
                
        # Extract the column and calculate the average
        density_column = df_density['A (Density (g/cm3):   A - BT(K))']
        density_A = density_column.mean()*100**3/1000

        # Extract the column and calculate the average
        density_column = df_density['B (Density (g/cm3):   A - BT(K))']
        density_B = density_column.mean()*100**3/1000
    else:
        pass
    
    return density_A, density_B


def get_cell_value(csv_file, target_row_header, target_column_header):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if row['Formula'] == target_row_header:
                return row[target_column_header]
    
    # If the target row or column header is not found, return None or handle it as needed
    return None

def prop_lookup(method, Compound, mol_fracs):
    r_ac_mix = 0
    density_mix = [0,0]
    sound_velocity_mix = [0,0]
    specific_heat_mix = [0,0]
    
    # Function to extract base compound name (removing source information in parentheses)
    def get_base_compound(compound_list):
        return [c.split(' (')[0] if isinstance(c, str) and '(' in c else c for c in compound_list]
    
    # Handle both string and list inputs for Compound
    if isinstance(Compound, list):
        base_compound = get_base_compound(Compound)
        compound_key = Compound  # Keep original for exact matching
    else:
        base_compound = Compound.split(' (')[0] if '(' in Compound else Compound
      
    if 'SCM' in method:
        # Check for exact match with source first (e.g., 'CaCl2 (Bu)')
        if isinstance(Compound, list) and len(Compound) == 1 and isinstance(Compound[0], str):
            if 'CaCl2 (Bu)' in Compound[0]:
                r_ac_mix = 4.8E-10  # Example value for Bu data, replace with actual value
            elif 'CaCl2 (McGreevey)' in Compound[0] or 'CaCl2' in Compound[0]:
                r_ac_mix = 5.033965984936765E-10  # Original McGreevy value
        # Fall back to base compound matching if no source-specific match found
        if r_ac_mix == 0:
            if base_compound == ['LiCl'] or (isinstance(Compound, str) and 'LiCl' in Compound):
                r_ac_mix = 4.118098594123714E-10    # 100% Walz, 878 K, 2019
            elif base_compound == ['NaCl'] or (isinstance(Compound, str) and 'NaCl' in Compound):
                r_ac_mix = 4.599211434277691E-10    # 100% Walz, 1074.15 K, 2019
            elif base_compound == ['KCl'] or (isinstance(Compound, str) and 'KCl' in Compound):
                r_ac_mix = 5.061494421764213E-10    # 100% Walz, 1043 K, 2019
            elif base_compound == ['LiF'] or (isinstance(Compound, str) and 'LiF' in Compound):
                r_ac_mix = 3.44935E-10    # 100% Walz, 1121 K, 2019
            elif base_compound == ['NaF'] or (isinstance(Compound, str) and 'NaF' in Compound):
                r_ac_mix = 3.7969068224247864E-10   # 100% Walz, 1266.15 K, 2019
            elif base_compound == ['KF'] or (isinstance(Compound, str) and 'KF' in Compound):
                r_ac_mix = 4.254924144410876E-10    # 100% Walz, 1131.15 K, 2019
            elif base_compound == ['RbF'] or (isinstance(Compound, str) and 'RbF' in Compound):
                r_ac_mix = 3.143356787E-10          # 100% Walz, 1068.15 K, 2019
            elif base_compound == ['CsF'] or (isinstance(Compound, str) and 'CsF' in Compound):
                r_ac_mix = 3.392004301E-10          # 100% Walz, 955.15 K, 2019
            elif base_compound == ['MgCl2'] or (isinstance(Compound, str) and 'MgCl2' in Compound):
                r_ac_mix = 4.7852054075804915E-10   # 100% McGreevy, 998 K, 1987
            elif base_compound == ['CaCl2'] or (isinstance(Compound, str) and 'CaCl2' in Compound):
                r_ac_mix = 5.033965984936765E-10    # 100% McGreevy, 1093 K, 1987
            elif base_compound == ['SrCl2'] or (isinstance(Compound, str) and 'SrCl2' in Compound):
                r_ac_mix = 5.10679E-10              # 100% McGreevy, 1198 K, 1987
            elif base_compound == ['NaNO3'] or (isinstance(Compound, str) and 'NaNO3' in Compound):
                r_ac_mix = 2.921000675E-10          # 100% 
        elif sorted(Compound) == sorted(['NaCl','UCl3']): # and mol_fracs == [0.63,0.37]:
            r_ac_mix = 2.89258E-10 #3.00685145E-10  # 64-36% Andersson, 2022
        elif Compound == ['LiF','BeF2'] and mol_fracs == [0.5,0.5]:
            r_ac_mix = 1.9572e-10  # 50-50% Sun, 2024
        elif Compound == ['LiF','BeF2'] and mol_fracs == [0.66,0.34]:
            r_ac_mix = 2.080697E-10 # 2.090906E-10 #1.7449105E-10  first peak   # #1.660250E-10     # 50-50% Sun, 2024
        elif Compound == ['LiF','NaF']:
            r_ac_mix = 2.28638E-10  # 60-40% Grizzi, 2024
        elif Compound == ['LiF','NaF','KF']:
            r_ac_mix = 2.32906E-10 # 2.618998138E-10  # 46.5-11.5-42% Frandsen, 2020
        elif Compound == ['NaF','KF','MgF2']:
            r_ac_mix = 2.56459E-10  # 34.5-59-6.5%, Rudenko, 2024
        elif Compound == ['MgCl2','NaCl','KCl']:
            r_ac_mix = 2.77031E-10  # 20.47-41.3-38.23%, Jiang, 2024
        # elif sorted(Compound) == sorted(['LiF','KF','UF4']):
        #     r_ac_mix = 2.733382581251662e-10  # 0.2727LiF-0.1818NaF-0.091UF4 Grizzi, 2024
        elif Compound == ['LiF','NaF','UF4']:
            r_ac_mix = 2.19714e-10  # 54.54LiF-336.36NaF-9.1UF4 Grizzi, 2024
        elif Compound == ['NaCl','KCl','ZnCl']:
            r_ac_mix = 2.65503e-10  # 0.22NaCl-0.393KCl-0.387ZnCl2",'Xi, 2024; 1073K
        else:
            r_ac_mix = 0

    # Properties of mixtures (if known)
            # density_mix = [A,B]           <-- kg/m^3; A: Intercept, B: Slope
            # sound_velocity_mix = [A,B]    <-- m/s;    A: Intercept, B: Slope
            # specific_heat_mix = [A,B,u]   <-- J/molK; A: Intercept, B: Slope, u: Units ('g'=J/g-K, 'm'=J/mol-K)
    if 'Mix' in method:
        # Use base_compound for mixture comparisons
        if Compound == ['NaCl','UCl3'] or Compound == ['UCl3','NaCl']:
            specific_heat_mix = [0.59,0,'g']    # Rose, 2023
            density_mix = [3856.705588,-0.830163884]   # Desyatnik, 1975; Agca, 2022        density_mix = [4220,-0.103]  # Parker 2022 [110], 0.667-0.333   
        if Compound == ['NaF','UF4'] or Compound == ['UF4','NaF']:
            density_mix = [4780,-0.82]   # Blanke, 1958 (MSTDB-TP, 0.76NaF)
        if Compound == ['NaF','KF','UF4'] or Compound == ['UF4','NaF','KF'] or Compound == ['KF','NaF','UF4'] or Compound == ['UF4','KF','NaF']:
            density_mix = [5129.1,-0.12]   # Fache 2023 [177] (MSTDB-TP, 0.556-0.187-0.257)
        if Compound == ['KCl','UCl3'] or Compound == ['UCl3','KCl']:
            if mol_fracs == [0.85,0.15] or mol_fracs == [0.15,0.85]:
                specific_heat_mix = [0.776451613,0,'g']     # Kim, 2023
                density_mix = [2302.446,0]                  # Bratescu, 2023
            if mol_fracs == [0.75,0.25] or mol_fracs == [0.25,0.75]:
                specific_heat_mix = [0.630783358,0,'g']    # Kim, 2023
                density_mix = [2772.128,0]   # Bratescu, 2023
            if mol_fracs == [0.65,0.35] or mol_fracs == [0.35,0.65]:
                specific_heat_mix = [0.527826334,0,'g']    # Kim, 2023
                density_mix = [3236.471,0]   # Bratescu, 2023     
            if mol_fracs == [0.5,0.5]:
                specific_heat_mix = [0.461451613,0,'g']    # Kim, 2023
                density_mix = [3918.574,0]   # Bratescu, 2023   
        elif Compound == ['LiF','BeF2'] or Compound == ['BeF2','LiF']:
            if mol_fracs == [0.66,0.34] or mol_fracs == [0.34,0.66]:
                density_mix = [2410,-0.488]  # Cantor 1973 [29], FLiBe 33.59mol%BeF2
                sound_velocity_mix = [4272.309646,-1.101929884] # 66-34% Cantor, 1968
                specific_heat_mix = [79.9,0,'m']    # Rosenthal 1969 [122]
                # specific_heat_mix = [2.12735,0,'g']  # 67-33% Avg from Rosenthal, 1969 and Lichtenstein, 2020 
                # All very different
                # specific_heat_mix = [79.9,0,'m']  # 67-33% Rosenthal, 1969 (Same as Sohal in J/g-K)
                # specific_heat_mix = [1.84,0,'g']  # 67-33% Lichtenstein, 2020 
                # specific_heat_mix = [2.4147,0,'g']  # 67-33% Sohal, 2010 <-- Not from reliable measurements
                #density_mix = [2413,-0.488]  # Janz, 1974, FLiBe 33mol%BeF2
                #2110 * (1.885 + 2.762*mol_fracs[1] + mol_fracs[1]**2) / (1.773 + 2.663*mol_fracs[0] )
            elif mol_fracs == [0.5,0.5]:
                density_mix = [2350,-0.424]  # Cantor 1973 [29], FLiBe 50mol%BeF2
                # sound_velocity_mix = [4272.309646,-1.101929884] # 66-34% Cantor, 1968
        elif Compound == ['LiF','NaF'] or Compound == ['NaF','LiF']:
            density_mix = [2530,-0.555]     #Janz 1974 [69]
            sound_velocity_mix = [3244,-0.787]    # 63LiF-37%NaF Minchenko, 1985
            specific_heat_mix = [125,-0.0666,'m']   # Powers 1963 [114]
        elif Compound == ['KCl','NaCl'] or Compound == ['NaCl','KCl']:
            density_mix = [2130,-0.568]     # Van Artsdalen 1955 [144]
        elif Compound == ['KCl','LiCl'] or Compound == ['LiCl','KCl']:
            # if mol_fracs == [0.582,0.418] or mol_fracs == [0.418,0.582]:
            specific_heat_mix = [70.9,0,'m']    # Redkin 2017 [118] 
            density_mix = [2030,-0.528]     # Van Artsdalen 1955 [144]
            # if mol_fracs == [0.637,0.363] or mol_fracs == [0.363,0.637]:
            #     specific_heat_mix = [70.9,0,'m']    # Redkin 2017 [118] 
            #     density_mix = [2130,-0.568]     # Van Artsdalen 1 55 [144]
        elif Compound == ['LiF','NaF','KF'] or Compound == ['KF','LiF','NaF'] or Compound == ['NaF','KF','LiF']:
            density_mix = [2680,-0.268]  # Gallagher 2021 [157]
            sound_velocity_mix = [3295.78,-1.20]    # 46.5-11.5-42% Robertson, 2022     #[3241.15,-1.20]    # 46.5-11.5-42% Robertson, 2022
            specific_heat_mix = [40.3,0.0439,'m']  # Rogers 1982 [121]
            # density_mix = [2729.3,-0.73]  # 46.5-11.5-42%, Vriesema [1979], Ingersoll et al. [2007], and Williams et al. [2006]
            # sound_velocity_mix = [3295.78,-1.20]    # 46.5-11.5-42% Robertson, 2022     #[3241.15,-1.20]    # 46.5-11.5-42% Robertson, 2022
            # specific_heat_mix = [1.90557,0,'g']  # 46.5-11.5-42% Sohal, 2010        #[1882.8*0.0413,0,'m']  # 46.5-11.5-42% Sohal, 2010
            # # molar_volume_mix = [1.34991e-5,7.55e-9] # Kubikova,2013 
        elif Compound == ['NaCl','CaCl2'] or Compound == ['CaCl2','NaCl']:  #and mol_fracs == [0.4903,0.5097]:
            density_mix = [2284.532 ,-0.406]  # 49.03-50.97%, Wei, 2022 (NaCa1) FIXED
            specific_heat_mix = [2.039484384,-0.00114002,'g']  # 49.03-50.97%, Wei, 2022 (NaCa1) GOOD
        elif Compound == ['KCl','CaCl2'] or Compound == ['CaCl2','KCl']:  #and mol_fracs == [0.4903,0.5097]:
            density_mix = [2274.784,-0.492]  # Wei, 2022 (KaCa3, Janz, 1978), 0.282CaCl2-0.718KCl
        elif Compound == ['NaCl','CaCl2','MgCl2'] or Compound == ['NaCl','MgCl2','CaCl2'] or Compound == ['MgCl2','CaCl2','NaCl'] or Compound == ['MgCl2','NaCl','CaCl2'] or Compound == ['CaCl2','MgCl2','NaCl'] or Compound == ['CaCl2','NaCl','MgCl2']:#and mol_fracs == [0.535,0.15,0.315]:
            density_mix = [1966.924,-0.215]  # 53.5-15-31.5%, Wei, 2022 (NaCaMg1) FIXED
            specific_heat_mix = [0.729590059, 0.000437771,'g']  # 53.5-15-31.5%, Wei, 2022 (NaCaMg1) FIXED


        elif Compound == ['NaCl','CaCl2','KCl2'] or Compound == ['KCl2','NaCl','CaCl2'] or Compound == ['CaCl2','KCl2','NaCl']:#and mol_fracs == [0.417,0.525,0.058]:
            density_mix = [2206.804,-0.451]  # 41.7-52.5-5.8%, Wei, 2022 (NaCaK) GOOD
            specific_heat_mix = [0.999094437,3.66312E-06,'g']  # 41.7-52.5-5.8%, Wei, 2022 (NaCaK)
        elif Compound == ['NaF','KF','MgF2'] or Compound == ['NaF','MgF2','KF'] or Compound == ['MgF2','KF','NaF'] or Compound == ['MgF2','NaF','KF'] or Compound == ['KF','MgF2','NaF'] or Compound == ['KF','NaF','MgF2']:
            density_mix = [2730,-0.658]  # Solano 2021 [134]
            specific_heat_mix = [1.35,0,'g']#[74,0,'m']  # 34.5-59-6.5%, Rudenko, 2024 GOOD
        elif Compound == ['MgCl2','NaCl','KCl'] or Compound == ['MgCl2','KCl','NaCl'] or Compound == ['KCl','MgCl2','NaCl'] or Compound == ['KCl','NaCl','MgCl2'] or Compound == ['NaCl','MgCl2','KCl'] or Compound == ['NaCl','KCl','MgCl2']:
            # if mol_fracs == [0.38,0.21,0.41] or mol_fracs == [0.38,0.41,0.21] or mol_fracs == [0.41,0.38,0.21]:
            # Doesn't match out PDF
            density_mix = [2111,-0.564]  # 45.98-15.11-38.91%, Wang, 2021
            specific_heat_mix = [1.437955,-0.0005,'g']  # 45.98-15.11-38.91%, Wang, 2021 FIXED
            
        if sound_velocity_mix == [0,0]:
            if base_compound == ['LiCl'] or (isinstance(Compound, str) and 'LiCl' in Compound):
                sound_velocity_mix = [3500, 0] 
            elif base_compound == ['NaCl'] or (isinstance(Compound, str) and 'NaCl' in Compound):
                sound_velocity_mix = [2720, 0]
            elif base_compound == ['KCl'] or (isinstance(Compound, str) and 'KCl' in Compound):
                sound_velocity_mix = [2360, 0]
            elif base_compound == ['LiF'] or (isinstance(Compound, str) and 'LiF' in Compound):
                sound_velocity_mix = [3500, 0]
            elif base_compound == ['NaF'] or (isinstance(Compound, str) and 'NaF' in Compound):
                sound_velocity_mix = [2720, 0]
            elif base_compound == ['KF'] or (isinstance(Compound, str) and 'KF' in Compound):
                sound_velocity_mix = [2360, 0]
            elif base_compound == ['MgCl2'] or (isinstance(Compound, str) and 'MgCl2' in Compound):
                sound_velocity_mix = [3400, 0]
            elif base_compound == ['CaCl2'] or (isinstance(Compound, str) and 'CaCl2' in Compound):
                sound_velocity_mix = [2200, 0]
    else:
        density_mix = [0,0]
        sound_velocity_mix = [0,0]
        specific_heat_mix = [0,0]

    return r_ac_mix, density_mix, sound_velocity_mix, specific_heat_mix

def functionlibrary():
    functions = {
    'KTM' : KTM,
    'KTM, Mix Data' : KTM_Mix,
    'PGM' : PGM,
    'PGM, Mix Data' : PGM_Mix,
    'SCM' : SCM,
    'SCM, Mix Data' : SCM_Mix,
    'Ideal' : Ideal,
    }
    return functions

    