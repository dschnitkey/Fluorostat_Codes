#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:03:50 2019

@author: diane
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:26:13 2019

@author: diane
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mt

mt.rcParams.update({'font.size': 18})
plt.tight_layout()

os.chdir('/Users/diane/Lab Work/Fluorostat')
#Load the specra of each fluorophore
YFP = pd.read_csv("YFP.csv")
CFP = pd.read_csv("CFP.csv")
#mTurq = pd.read_csv("mTurq.csv")

#Load in the different filters
path = '/Users/diane/Lab Work/Fluorostat/'
yfp_em = pd.read_csv(path + "yfp_emission", sep= "\t")
yfp_ex = pd.read_csv(path + "yfp_excitation", sep= "\t")
cfp_em = pd.read_csv(path + "cfp_emission", sep= "\t")
cfp_ex = pd.read_csv(path + "cfp_excitation", sep= "\t")

#Load in the Dichroics
ex_dichroic = pd.read_csv(path + "DMLP490R_emissiondichroic", delim_whitespace = True)
em_dichroic = pd.read_csv(path + "DMLP505R_excitationdichroic", delim_whitespace = True)

#Load in the different lasers
CFP_LED =pd.read_csv(path + "M430L4_violet", sep= "\t")
#YFP_LED_1 = pd.read_csv(path + "M490L4_blue", sep= "\t")
#YFP_LED_2 = pd.read_csv(path + "M505L3_cyan", sep= "\t")
YFP_LED_3 = pd.read_csv(path + "M505L4_cyan", sep= "\t")


#%% 
#Convert all the various x steps to 1nm steps so I can mutiply them by each other, you only need this function
#if your spectra do not have the same step sizes for wavelength and you want to directly compare or multiply
#them. This function takes a spectrum and converts to having one entry for every nm. It finds the first wavelength 
#that is greater than or equal to an integer value, and assigns that as your % transmission given a nm range

#inputs:
##inputspectra - the pandas array that has a table of wavelengths and transmission, can have 2 or 3 columns
#wavlengthlabel - a string containing the header for the wavelength column of the array
#emexlabel - a list of strings containing the headers for the emission/transmission/reflection data
#numcols - number of columns in the pandas array
#upper_cap - the highest wavelength you are interested in seeing
#lower_cap - the lowest wavelength

#returns:
# a new pandas array that takes integer steps through your wavelength range.

def int_wavelengths(inputspectra, wavelengthlabel, emexlabel, numcols, upper_cap = 700, lower_cap = 400):
    steps = (upper_cap-lower_cap + 1)
    wavelengths = np.linspace(lower_cap, upper_cap, steps )
    x = 0
    nm = np.zeros([steps,numcols])
    #print(emexlabel[0])
    #print(inputspectra[emexlabel[0]])
    for i in range(len(inputspectra)):
        lower_cap  = wavelengths[x]
        #print(str(inputspectra[wavelengthlabel][i]) + "\t" + str(lower_cap) + "\t" + str(inputspectra[emexlabel[0]][i]))

        if inputspectra[wavelengthlabel][i]>= lower_cap:
            #print(i, inputspectra[emexlabel[0]][i], x, lower_cap)
            if numcols == 3:
                nm[x] = [lower_cap, inputspectra[emexlabel[0]][i], inputspectra[emexlabel[1]][i]]
            elif numcols == 2:
                #print([lower_cap, inputspectra[emexlabel[0]][x]])
                nm[x] = [lower_cap, inputspectra[emexlabel[0]][i]]
            x +=1 
        if (inputspectra[wavelengthlabel][i] > upper_cap) or (x > steps -1):
            break
    nm = pd.DataFrame(nm, columns = [wavelengthlabel]+ emexlabel)
    return(nm)
#%%
#convert the spectra into integers for wavelength so they can be multiplied.
YFP_nm = int_wavelengths(YFP, "Wavelength", ["Excitation", "Emission"], 3)
CFP_nm = int_wavelengths(CFP, "Wavelength", ["Excitation", "Emission"], 3)
yfp_em_nm = int_wavelengths(yfp_em, "Wavelength (nm)", ["% Transmission"], 2)
yfp_ex_nm = int_wavelengths(yfp_ex, "Wavelength (nm)", ["% Transmission"], 2)
cfp_em_nm = int_wavelengths(cfp_em, "Wavelength (nm)", ["% Transmission"], 2)
cfp_ex_nm = int_wavelengths(cfp_ex, "Wavelength (nm)", ["% Transmission"], 2)
ex_dichroic_nm = int_wavelengths(ex_dichroic, "Wavelength(nm)", ["%_Transmission", "%_Reflectance"], 3)
em_dichroic_nm = int_wavelengths(em_dichroic, "Wavelength(nm)", ["%_Transmission", "%_Reflectance"], 3)
CFP_LED_nm = int_wavelengths(CFP_LED, "Wavelength (nm)", ['Normalized Intensity'], 2)
YFP_LED_nm = int_wavelengths(YFP_LED_3, "Wavelength (nm)", ['Normalized Intensity'], 2)

#%%
#Let's do a step by step trip through the optics

#start with individual excitation LEDs

#CFP
plt.figure()
plt.fill_between(CFP_LED_nm["Wavelength (nm)"], CFP_LED_nm["Normalized Intensity"]*100,alpha = .5, color = [0,1,1])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Cyan LED Spectrum")
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "Cyan_LED_Spectrum.png")
#YFP
plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"], (YFP_LED_nm["Normalized Intensity"]*100), alpha = .5, color = [1,1,0])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Yellow LED Spectrum")
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "Yellow_LED_Spectrum.png")

#Waveform that gets through the excitation

#CFP
plt.figure()
plt.fill_between(CFP_LED_nm["Wavelength (nm)"], CFP_LED_nm["Normalized Intensity"]*100,alpha = .5, color = [0,1,1])
plt.plot(cfp_ex_nm["Wavelength (nm)"], cfp_ex_nm["% Transmission"], color = "c", lw = 3)
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("CFP Excitation Filter")
plt.legend(["Filter Pass","LED Spectrum"], prop={'size': 14})
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "CFP_Excitation_Filter.png")

#YFP
plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"], (YFP_LED_nm["Normalized Intensity"]*100), alpha = .5, color = [1,1,0])
plt.plot(yfp_ex_nm["Wavelength (nm)"], yfp_ex_nm["% Transmission"],  color = "y",lw= 3)
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("YFP Excitation Filter")
plt.legend(["Filter Pass", "LED Spectrum"], prop={'size': 14})
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "YFP_Excitation_Filter.png")
#%%
#Lights that hit the Dichroic

#Multiply transmission through the filter by the LED emission to get the spectra that hits the dichroic
incom_di_CFP = CFP_LED_nm["Normalized Intensity"] * cfp_ex_nm["% Transmission"]
incom_di_YFP = YFP_LED_nm["Normalized Intensity"] * yfp_ex_nm["% Transmission"]
#plot
plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],incom_di_CFP, alpha = .5, color = [0,1,1])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],incom_di_YFP,  alpha = .5, color = [1,1,0])
plt.plot(ex_dichroic_nm["Wavelength(nm)"], ex_dichroic_nm["%_Transmission"],lw = 3, color = [.8,.5,0])
plt.plot(ex_dichroic_nm["Wavelength(nm)"], ex_dichroic_nm["%_Reflectance"],lw = 3, color = [0,.5,.8])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Light Incident on Ex Dichroic")
plt.legend(["Trans","Refl", "C LED", "Y LED"], prop={'size': 14})
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "Light_Incident_on_Ex_Dichroic.png")

#Light that gets sent to the sample
reflected_to_sample = incom_di_CFP * ex_dichroic_nm["%_Reflectance"]/100
transmitted_to_sample = incom_di_YFP * ex_dichroic_nm["%_Transmission"]/100
#plot
plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],reflected_to_sample, alpha = .5, color = [0,1,1])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],transmitted_to_sample,  alpha = .5, color = [1,1,0])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Spectra Reaching Sample")
plt.legend(["C LED", "Y LED"], prop={'size': 14})
plt.xlim([400,700])
plt.ylim([0,100])
plt.tight_layout()
plt.savefig(path + "Spectra_Reaching_Sample.png")
#%%
#Are these sections specific? will they only excite the right color?
#Light that gets sent to the sample
plt.figure()
reflected_to_sample = incom_di_CFP * ex_dichroic_nm["%_Reflectance"]/100
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],reflected_to_sample, alpha = .5, color = [0,1,1])

transmitted_to_sample = incom_di_YFP * ex_dichroic_nm["%_Transmission"]/100
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],transmitted_to_sample,  alpha = .5, color = [1,1,0])

plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--", lw = 4)
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c--", lw = 4)
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.title("LEDs are Specific \n to Correct Excitation",  fontsize = 16)
plt.legend(["YFP Ex", "CFP Ex", "Cyan LED", "Yellow LED"], prop={'size': 14})
plt.xlim([400,700])
plt.ylim([0,100])
plt.tight_layout()
plt.savefig(path + "Specific_Excitation.png")

#%%
#After hitting the sample, we will get emitted light from CFP and YFP, as well as Some excitation lights
#Light let in by emission filters
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],reflected_to_sample, alpha = .5, color = [0,1,1])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],transmitted_to_sample,  alpha = .5, color = [1,1,0])
plt.plot(em_dichroic["Wavelength(nm)"], em_dichroic["%_Transmission"],"y", lw = 4)
plt.plot(em_dichroic["Wavelength(nm)"], em_dichroic["%_Reflectance"],"c", lw = 4)
#plt.fill_between(yfp_em["Wavelength (nm)"], yfp_em["% Transmission"], color =[.8,.8,0], alpha = .5)
#plt.fill_between(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], color = [0,.8,.8], alpha = .5)
plt.fill_between(YFP["Wavelength"],YFP["Emission"], alpha = .3, color = [.8,.8,0])
plt.fill_between(CFP["Wavelength"],CFP["Emission"],  alpha = .3,color = [0,.8,.8])

plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.legend(["Trans","Refl", "C LED", "Y LED","YFP Em","CFP Em"], prop={'size': 14})
plt.title("All Spectra On 2nd Dichroic")
plt.xlim([400,700])
plt.ylim([0,100])
plt.tight_layout()
plt.savefig(path + "All_Spectra_Coming_From_Sample.png")
#%%
#Light that gets reflected towards CFP PMT
YFP_LED_reflected = transmitted_to_sample * em_dichroic_nm["%_Reflectance"]/100
CFP_LED_reflected = reflected_to_sample * em_dichroic_nm["%_Reflectance"]/100
YFP_Em_reflected = YFP_nm["Emission"] * em_dichroic_nm["%_Reflectance"]/100
CFP_Em_reflected = CFP_nm["Emission"] * em_dichroic_nm["%_Reflectance"]/100

plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_LED_reflected, alpha = .5, color = [1,1,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],CFP_LED_reflected,  alpha = .5, color = [0,1,1])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_Em_reflected,  alpha = .3, color = [.8,.8,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],CFP_Em_reflected,  alpha = .3,color = [0,.8,.8])
plt.plot(cfp_em_nm['Wavelength (nm)'],cfp_em_nm['% Transmission'], lw = 3)
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.legend(["CFP Em Filter", "Y LED", "C LED","YFP Em","CFP Em"], prop={'size': 14})
plt.title("Reflected Off 2nd Dichroic")
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "Ref_2_Dichroic.png")
#%%
#Light that gets transmitted towards YFP PMT
YFP_LED_trans = transmitted_to_sample * em_dichroic_nm["%_Transmission"]/100
YFP_Em_trans = YFP_nm["Emission"] * em_dichroic_nm["%_Transmission"]/100
CFP_Em_trans = CFP_nm["Emission"] * em_dichroic_nm["%_Transmission"]/100

plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_LED_trans, alpha = .5, color = [1,1,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_Em_trans, alpha = .3, color = [.8,.8,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],CFP_Em_trans, alpha = .3,color = [0,.8,.8])
plt.plot(yfp_em_nm['Wavelength (nm)'],yfp_em_nm['% Transmission'], lw = 3, color = "y")
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Light Transmitted Through Dichroic", fontsize = 20)
plt.legend(["YFP Em Filter", "Y LED","YFP Em","CFP Em"], prop={'size': 14})
plt.xlim([400,700])
plt.tight_layout()
plt.savefig(path + "Trans_2_Dichroic.png")
#%%
#Final lights that hit the PMTs

CFP_YFPPMT = CFP_Em_trans * yfp_em_nm['% Transmission']/100
YFP_YFPPMT = YFP_Em_trans * yfp_em_nm['% Transmission'] /100

plt.figure()

plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_YFPPMT, alpha = .3, color = [.8,.8,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],CFP_YFPPMT, alpha = .3,color = [0,.8,.8])
plt.legend(["YFP Em","CFP Em"], prop={'size': 14})
plt.xlabel("wavelength(nm)")
plt.title("Light Arriving at Yellow PMT")
plt.ylabel("% Transmission/Reflection")
plt.tight_layout()
plt.savefig(path + "Hits_Yellow_PMT.png")
#%%
YFPLED_CFPPMT = YFP_LED_reflected * cfp_em_nm['% Transmission']/100
YFP_CFPPMT = YFP_Em_reflected * cfp_em_nm['% Transmission']/100
CFP_CFPPMT = CFP_Em_reflected * cfp_em_nm['% Transmission']/100

plt.figure()
#plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFPLED_CFPPMT, alpha = .5, color = [1,1,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],YFP_CFPPMT,  alpha = .3, color = [.8,.8,0])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],CFP_CFPPMT,  alpha = .3,color = [0,.8,.8])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.legend([ "Y LED","YFP Em","CFP Em"], prop={'size': 14})
plt.title("Light Arriving at Cyan PMT")
plt.tight_layout()
plt.savefig(path + "Hits_Cyan_PMT.png")