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

os.chdir('/Users/diane/Lab Work/Fluorostat')
#Load the specra of each fluorophore
YFP = pd.read_csv("YFP.csv")
CFP = pd.read_csv("CFP.csv")
#mTurq = pd.read_csv("mTurq.csv")

#Plot the two spectra
plt.subplot(3,1,1)
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(YFP["Wavelength"],YFP["Emission"], "y")
plt.legend(["YFP Excitation", "YFP Emission"])
plt.xlim([300,700])
plt.title("Spectra")
plt.subplot(3,1,2)
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c--")
plt.plot(CFP["Wavelength"],CFP["Emission"], "c")
plt.legend(["CFP Excitation", "CFP Emission"])
plt.xlim([300,700])
plt.subplot(3,1,3)
plt.plot(mTurq["wavelength"],mTurq["mTurquoise ab"], "c--")
plt.plot(mTurq["wavelength"],mTurq["mTurquoise em"], "c")
plt.legend(["mTurq Excitation", "mTurq Emission"])
plt.xlim([300,700])

#%%

#Load in the different filters
path = '/Users/diane/Lab Work/Fluorostat/'
yfp_em = pd.read_csv(path + "yfp_emission", sep= "\t")
yfp_ex = pd.read_csv(path + "yfp_excitation", sep= "\t")
cfp_em = pd.read_csv(path + "cfp_emission", sep= "\t")
cfp_ex = pd.read_csv(path + "cfp_excitation", sep= "\t")
#GFP exitation filter doesn't have downloadable CSV has a mean of 525 with 18nm spread on either side
gfp_em = np.zeros([2, len(YFP["Wavelength"])])
gfp_em[0,:] = YFP["Wavelength"]
gfp_em_min = 525 - 9
gfp_em_max = 525 + 9
min_mask = gfp_em[0,:] > gfp_em_min
max_mask = gfp_em[0,:] < gfp_em_max
mask = min_mask & max_mask
gfp_em[1,:] = mask*100
gfp_ex = pd.read_csv(path + "gfp_excitation", sep= "\t")


#%%
plt.fill_between(YFP["Wavelength"],YFP["Emission"], color ="y", alpha = .5)
plt.fill_between(CFP["Wavelength"],CFP["Emission"], color = "c", alpha = .5)
plt.legend(["YFP Emission","CFP Emission"])
plt.xlim([450,700])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.title("YFP and CFP Emission Overlap")
plt.savefig(path + "YFPCFP_Emission.png")

#%%
#Load in the Dichroics
ex_dichroic = pd.read_csv(path + "DMLP490R_emissiondichroic", delim_whitespace = True)
em_dichroic = pd.read_csv(path + "DMLP505R_excitationdichroic", delim_whitespace = True)



#%%
plt.figure()

mt.rcParams.update({'font.size': 18})

#mt.rc('legend', labelsize=12) 
#plot each on 
#plt.subplot(2,1,1)
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"], "g")
plt.legend(["YFP Excitation","YFP Ex Filter"], loc = 2, prop={'size': 16})
plt.title("YFP Excitation")
plt.xlim([400,550])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
#plt.tight_layout()
plt.savefig(path + "YFP_Excitation.png")

#%%
plt.figure()
plt.plot(YFP["Wavelength"],YFP["Emission"], "y")
plt.plot(yfp_em["Wavelength (nm)"], yfp_em["% Transmission"], "g")
plt.legend([ "YFP Emission", "YFP Em Filter"],loc = 1, prop={'size': 16})
plt.title("YFP Emission")
plt.xlim([450,700])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.savefig(path + "YFP_Emission.png")
#%%
'''
plt.figure()
#plt.subplot(2,1,2)
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(YFP["Wavelength"],YFP["Emission"], "y")
plt.plot(gfp_em[0,:], gfp_em[1,:], "y")
plt.plot(gfp_ex["Wavelength (nm)"], gfp_ex["% Transmission"], "g")
plt.legend(["YFP Excitation", "YFP Emission", "GFP Em Filter", "GFP Ex Filter"])

plt.xlim([200,700])

'''

plt.figure()

#plt.subplot(2,1,1)
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c--")
plt.plot(cfp_ex["Wavelength (nm)"], cfp_ex["% Transmission"], "b")
plt.title("CFP Excitation")
plt.legend(["CFP Excitation", "CFP Ex Filter"],loc = 2, prop={'size': 16})
plt.xlim([300,500])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.savefig(path + "CFP_Excitation.png")
#%%

plt.figure()
plt.plot(CFP["Wavelength"],CFP["Emission"], "c")
plt.plot(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], "b")
plt.title("CFP Emission")
plt.legend(["CFP Emission", "CFP Em Filter"], loc = 1, prop={'size': 16})
plt.xlim([400,700])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.savefig(path + "CFP_Emission.png")
'''
plt.figure()
#plt.subplot(2,1,2)
plt.plot(mTurq["wavelength"],mTurq["mTurquoise ab"]* 100, "c--") 
plt.plot(mTurq["wavelength"],mTurq["mTurquoise em"]* 100, "c")
plt.plot(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], "c")
plt.plot(cfp_ex["Wavelength (nm)"], cfp_ex["% Transmission"], "b")
plt.legend(["mTurq Excitation", "mTurq Emission", "CFP Em Filter", "CFP Ex Filter"])
plt.xlim([200,700])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
'''

#%%
#plot them all together
plt.figure()
plt.title("Filter Passes")
plt.plot(yfp_em["Wavelength (nm)"], yfp_em["% Transmission"], "y")
plt.plot(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], "c")
plt.plot(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"], "g")
plt.plot(cfp_ex["Wavelength (nm)"], cfp_ex["% Transmission"], "b")
plt.xlim([200,700])
plt.legend(["YFP Emission Filter","CFP Emission Filter","YFP Excitation Filter","CFP Excitation Filter"])
#%%

#Load in the different lasers
CFP_LED =pd.read_csv(path + "M430L4_violet", sep= "\t")
#YFP_LED_1 = pd.read_csv(path + "M490L4_blue", sep= "\t")
#YFP_LED_2 = pd.read_csv(path + "M505L3_cyan", sep= "\t")
YFP_LED_3 = pd.read_csv(path + "M505L4_cyan", sep= "\t")

plt.figure()
#plot each on 
'''
plt.figure()
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"], "g")
plt.plot(YFP_LED_1["Wavelength (nm)"], YFP_LED_1["Normalized Intensity"]*100,"b")
plt.plot
plt.title("YFP LED Spectrum")
plt.xlim([200,700])

plt.figure()
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"], "g")
plt.plot(YFP_LED_2["Wavelength (nm)"], YFP_LED_2["Normalized Intensity"]*100,"b")
plt.plot
plt.title("YFP LED Spectrum")
plt.xlim([200,700])
'''
#plt.subplot(2,1,1)
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y--")
plt.plot(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"], "g")
#plt.plot(yfp_em["Wavelength (nm)"], yfp_em["% Transmission"], "y")
plt.plot(YFP_LED_3["Wavelength (nm)"], YFP_LED_3["Normalized Intensity"]*100,"m")
plt.plot
plt.legend(["YFP Ex", "Ex Filter Pass", "LED Spectrum"], loc = 1, prop={'size': 14})
plt.title("YFP Excitation vs LED spectra")
plt.xlim([400,700])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.savefig(path + "YFP_LED.png")

plt.figure()
#plt.subplot(2,1,2)
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c--")
plt.plot(cfp_ex["Wavelength (nm)"], cfp_ex["% Transmission"], "g")
#plt.plot(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], "b")
plt.plot(CFP_LED["Wavelength (nm)"], CFP_LED["Normalized Intensity"]*100,"m")
plt.legend(["CFP Ex", "Ex Filter Pass", "LED Spectrum"], loc = 1, prop={'size': 14})
plt.title("CFP Excitation vs LED spectra")
plt.xlim([350,600])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission")
plt.savefig(path + "CFP_LED.png")
#%%
#Dichroic Locations

#excitation Dichroic, wants to separate excitation wavelengths
plt.title("Excitations with 490 nm Dichroic")
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c")
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y")
plt.plot(ex_dichroic["Wavelength(nm)"], ex_dichroic["%_Transmission"],"--")
plt.plot(ex_dichroic["Wavelength(nm)"], ex_dichroic["%_Reflectance"],"--")
plt.legend(["CFP Ex", "YFP Ex", "DC Trans", "DC Reflect"], loc = 1, prop={'size': 14})
plt.xlim([350,650])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.savefig(path + "Dichroic_490.png")

#emission Dichroic, separate emission wavelenghts
plt.figure()
plt.title("Emission with 505 nm Dichroic")
plt.plot(YFP["Wavelength"],YFP["Emission"], "y")
plt.plot(CFP["Wavelength"],CFP["Emission"], "c")
plt.plot(em_dichroic["Wavelength(nm)"], em_dichroic["%_Transmission"],"--")
plt.plot(em_dichroic["Wavelength(nm)"], em_dichroic["%_Reflectance"],"--")
plt.legend(["CFP Ex", "YFP Ex", "DC Trans", "DC Reflect"], loc = 1, prop={'size': 14})
plt.xlim([350,750])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.savefig(path + "Dichroic_505.png")

#%%
#Plot all my Filters and see if they overlap 
#Excitations

plt.fill_between(ex_dichroic["Wavelength(nm)"], ex_dichroic["%_Transmission"],alpha = .2, color = [1,.5,0])
plt.fill_between(ex_dichroic["Wavelength(nm)"], ex_dichroic["%_Reflectance"],alpha = .2, color = [0,.5,1])
plt.fill_between(yfp_ex["Wavelength (nm)"], yfp_ex["% Transmission"],  color = "y", alpha = .5)
plt.fill_between(cfp_ex["Wavelength (nm)"], cfp_ex["% Transmission"], color = "c", alpha = .5)
plt.plot(CFP["Wavelength"],CFP["Excitation"], "c", lw = 4, color = [0,.5,.5])
plt.plot(YFP["Wavelength"],YFP["Excitation"], "y", lw = 4, color = [.5,.5,0])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.xlim([380,700])

#Emissions
plt.figure()
plt.plot(YFP["Wavelength"],YFP["Emission"], lw = 4, color = [.5,.5,0])
plt.plot(CFP["Wavelength"],CFP["Emission"], lw = 4,color = [0,.5,.5])
plt.fill_between(em_dichroic["Wavelength(nm)"], em_dichroic["%_Transmission"], alpha = .2,color = [1,.5,0])
plt.fill_between(em_dichroic["Wavelength(nm)"], em_dichroic["%_Reflectance"], alpha = .2, color = [0,.5,1])
plt.fill_between(yfp_em["Wavelength (nm)"], yfp_em["% Transmission"], color ="y", alpha = .5)
plt.fill_between(cfp_em["Wavelength (nm)"], cfp_em["% Transmission"], color = "c", alpha = .5)
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.xlim([400,700])

#%% 
#Convert all the various x steps to 1nm steps so I can mutiply them by each other

def int_wavelengths(inputspectra, wavelengthlabel, emexdata, numcols, upper_cap = 700, lower_cap = 400):
    steps = (upper_cap-lower_cap + 1)
    wavelengths = np.linspace(lower_cap, upper_cap, steps )
    x = 0
    nm = np.zeros([steps,numcols])
    #print(emexdata[0])
    #print(inputspectra[emexdata[0]])
    for i in range(len(inputspectra)):
        lower_cap  = wavelengths[x]
        #print(str(inputspectra[wavelengthlabel][i]) + "\t" + str(lower_cap) + "\t" + str(inputspectra[emexdata[0]][i]))

        if inputspectra[wavelengthlabel][i]>= lower_cap:
            #print(i, inputspectra[emexdata[0]][i], x, lower_cap)
            if numcols == 3:
                nm[x] = [lower_cap, inputspectra[emexdata[0]][i], inputspectra[emexdata[1]][i]]
            elif numcols == 2:
                #print([lower_cap, inputspectra[emexdata[0]][x]])
                nm[x] = [lower_cap, inputspectra[emexdata[0]][i]]
            x +=1 
        if (inputspectra[wavelengthlabel][i] > upper_cap) or (x > steps -1):
            break
    nm = pd.DataFrame(nm, columns = [wavelengthlabel]+ emexdata)
    return(nm)
#%%
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
plt.savefig(path + "Cyan_LED_Spectrum.png")
#YFP
plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"], (YFP_LED_nm["Normalized Intensity"]*100), alpha = .5, color = [1,1,0])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Yellow LED Spectrum")
plt.xlim([400,700])
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
plt.savefig(path + "YFP_Excitation_Filter.png")
#%%
#Lights that hit the Dichroic

incom_di_CFP = CFP_LED_nm["Normalized Intensity"] * cfp_ex_nm["% Transmission"]
incom_di_YFP = YFP_LED_nm["Normalized Intensity"] * yfp_ex_nm["% Transmission"]

plt.figure()
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],incom_di_CFP, alpha = .5, color = [0,1,1])
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],incom_di_YFP,  alpha = .5, color = [1,1,0])
plt.plot(ex_dichroic_nm["Wavelength(nm)"], ex_dichroic_nm["%_Transmission"],lw = 3, color = [.8,.5,0])
plt.plot(ex_dichroic_nm["Wavelength(nm)"], ex_dichroic_nm["%_Reflectance"],lw = 3, color = [0,.5,.8])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Light Incident on Ex Dichroic")
plt.legend(["Trans","Refl", "C LED", "Y LED"], prop={'size': 14})
plt.xlim([380,700])
plt.savefig(path + "Light_Incident_on_Ex_Dichroic.png")

#Light that gets sent to the sample
plt.figure()
reflected_to_sample = incom_di_CFP * ex_dichroic_nm["%_Reflectance"]/100
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],reflected_to_sample, alpha = .5, color = [0,1,1])

transmitted_to_sample = incom_di_YFP * ex_dichroic_nm["%_Transmission"]/100
plt.fill_between(YFP_LED_nm["Wavelength (nm)"],transmitted_to_sample,  alpha = .5, color = [1,1,0])
plt.xlabel("wavelength(nm)")
plt.ylabel("% Transmission/Reflection")
plt.title("Spectra Reaching Sample")
plt.legend(["C LED", "Y LED"], prop={'size': 14})
plt.xlim([380,700])
plt.ylim([0,100])
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
plt.xlim([380,700])
plt.ylim([0,100])
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
plt.xlim([380,700])
plt.ylim([0,100])
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
plt.xlim([380,700])
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
#plt.xlim([380,700])
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
plt.savefig(path + "Hits_Cyan_PMT.png")