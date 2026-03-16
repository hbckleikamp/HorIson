# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:43:47 2025

@author: hkleikamp
"""
#%%
import os
import sys
import warnings
from inspect import getsourcefile
from pathlib import Path

import pandas as pd
import numpy as np
import math

from itertools import combinations_with_replacement
from collections import Counter
from scipy.stats import multinomial
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks,peak_widths,argrelmin,argrelmax#,argrelextrema
from scipy.sparse import coo_matrix
from scipy.fft import next_fast_len #or just nearest power of 2?
# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

#%%

# general arguments
composition=str(Path(basedir,"CartMFP_test_mass_CASMI2022.tsv" )) # file,folder or composition string
isotope_table=str(Path(basedir, "isotope_table.tsv"))             # table containing isotope metadata in .tsv format
Output_folder=str(Path(basedir, "HorIson_Output"))                # table containing isotope metadata in .tsv format

Output_file=""

method="multi" #Options: fft, fft_hires, multi
normalize=False #Options: False, sum, max, mono
min_intensity=1e-6
isotope_range=[-2,6] #low, high

#### method specific arguments

#used in fft_hires and convolve functions
pick_peaks=True 
peak_fwhm=0.01      
add_mono=True        #add monoisotopic mass back
add_borders=False    #compute peak borders
add_area=False       #calculate peak area
correct_charge=True  #divide by charge to m/z output

#fft specific
packing=True
mass_shift=True
batch_size=1e4  #batch size used in fft hires


#multinomial specific
prune=1e-6      #pruning during combinations
min_chance=1e-4 #pruning after  combinations
convolve="fast" #False (no convolution) "fast" (output peaks) or "full" (output complete)
convolve_batch=1e4
Precomputed=[] #
verbose=True #print messages or not?


#%%


if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse

    parser = argparse.ArgumentParser(
                        prog='HorIson',
                        description='isotope simulation tool, see: https://github.com/hbckleikamp/HorIson')
    
    #filepaths
    parser.add_argument("-i", "--composition",       required = False, default=str(Path(basedir, "CartMFP_test_mass_CASMI2022.tsv")),
                        help="Required: input composition string, or tabular file with elemental compositions, or folder with files")

    parser.add_argument("--isotope_table",       required = False, default=str(Path(basedir, "isotope_table.tsv")),
                        help="Optional: path to isotope table in .tsv format")
    
    parser.add_argument("-o", "--Output_folder",       required = False, default=str(Path(basedir, "HorIson_Output")) ,
                        help="Optional: Output folder path")
    
    parser.add_argument("--Output_file",       required = False, default="", help="Optional: Output file name")
    
    
    parser.add_argument("--method",       required = False, default="fft",
                        help="Optional: isotopic modelling method")
    
    parser.add_argument("-n,--normalize",       required = False, default=False,
                        help="Optional: normalize the output (Options: don't normalize (False), normalize to total ('sum'), normalize to most abundant isotope ('max'), normalize to monoisotopic peak (mono)")
    
    parser.add_argument("--min_intensity",       required = False, default=1e-6,
                        help="Only retain Fourier columns that have one abundance over x" )
    
    parser.add_argument("--batch_size",       required = False, default=1e4,type=int,
                        help="FFt hires only: perform fourier transform in batches of x" )
    parser.add_argument("-fwhm, --peak_fwhm",       required = False, default=0.01,
                        help="Optional: Peak full width at half maxmium. Used in fft_hires and convolve functions. can be supplied as single value, or as a list of values corresponding to different masses." )
    
    parser.add_argument("--packing",       required = False, default=True,
                        help="FFt hires only: pack ffts for faster computation" )
    parser.add_argument("--bpick_peaks",       required = False,  default=True,
                        help="FFt hires only: retain peaks only" )
    parser.add_argument("--add_mono",       required = False, default=True,
                        help="Add back monoisotopic mass" )
    parser.add_argument("--correct_charge",       required = False,  default=True,
                        help="Correct with charge to m/z" )
    


    
    #multinomial specific arguments
    #parser.add_argument('--convolve', action='store_true',help="Multinomial only: convolve multinomial output with gaussian")
    parser.add_argument('--convolve', default="fast",help="Multinomial only: convolve multinomial output with gaussian")
    parser.add_argument("--convolve_batch",       required = False, default=1e4,type=int,
                        help="Multinomial only: perform gaussian convolution in batches of x" )
    
    parser.add_argument("--isotope_range",       required = False, default=[-2,6],
                        help="Multinomial only: isotopes computed in multinomial" )
    parser.add_argument("--prune",       required = False, default=1e6,
                        help="Multinomial only: pruning combinations below x chance of occurring (after during element combination)" )
    parser.add_argument("--min_chance",       required = False, default=1e6,
                        help="Multinomial only: pruning combinations below x chance of occurring (after element combination)" )
    
    
    
    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    print("")
    print(args) 
    print("")
    locals().update(args)

#fix arguments
if type(isotope_range)==str: isotope_range=isotope_range.strip("[]").split(",")
isotope_range=np.arange(int(isotope_range[0]),int(isotope_range[1])+1)

if "," in composition: composition=[i.strip() for i in composition.split(",")]
else: composition=[composition]


# if not isinstance(peak_fwhm, Iterable): peak_fwhm=[peak_fwhm]
# elif "," in peak_fwhm: peak_fwhm=[i.strip() for i in peak_fwhm.split(",")]

#%% Utility functions

def is_float(x):
    try:
        float(x)
        return True
    except: 
        return False

#parse form
def parse_form(form): #chemical formula parser
    e,c,comps="","",[]
    for i in form:
        if i.isupper(): #new entry   
            if e: 
                if not c: c="1"
                comps.append([e,c])
            e,c=i,""         
        elif i.islower(): e+=i
        elif i.isdigit(): c+=i
    if e: 
        if not c: c="1"
        comps.append([e,c])
    
    cdf=pd.DataFrame(comps,columns=["elements","counts"]).set_index("elements").T.astype(int)
    
    #add charge
    cdf["+"]=form.count("+")
    cdf["-"]=form.count("-")
    return cdf

def getMz(form): #this could be vectorized for speed up in the future
    cdf=parse_form(form)
    return (cdf.values*mono_elmass.loc[cdf.columns].values).sum() / cdf[["+","-"]].sum(axis=1)

# read input table (dynamic delimiter detection)
def read_table(tabfile, *,
               Keyword=[], #rewrite multiple keywords
               ):

    if type(Keyword)==str: Keyword=[i.strip() for i in Keyword.split(",")]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        try:
            tab = pd.read_excel(tabfile, engine='openpyxl')
        except:
            with open(tabfile, "r") as f:
                tab = pd.DataFrame(f.read().splitlines())

        # dynamic delimiter detection: if file delimiter is different, split using different delimiters until the desired column name is found
        if len(Keyword):
            if not tab.columns.isin(Keyword).any():
                delims = [i[0] for i in Counter(
                    [i for i in str(tab.iloc[0]) if not i.isalnum()]).most_common()]
                for delim in delims:
                    if delim == " ":
                        delim = "\s"
                    try:
                        tab = pd.read_csv(tabfile, sep=delim)
                        if tab.columns.isin(Keyword).any():
                            return True,tab
                    except:
                        pass

            return False,tab

    return True,tab

#vectorized find nearest mass
#https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
def find_closest(A, target): #returns index of closest array of A within target
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def read_formulas(form,charge=None,elements=None): #either a single string or a DataFrame with element columns

    if type(form)==str: form=parse_form(form)  #single string
    elif  not (isinstance(form, pd.DataFrame) or isinstance(form, pd.Series)): #iterable
        form=pd.concat([parse_form(f) for f in form]).reset_index(drop=True).fillna(0)  
        form=form.reset_index(drop=True)
    #DataFrame with columns (recommended input)
    elif isinstance(form, pd.DataFrame):        
        if type(elements)==type(None):
            elements=form.columns[form.columns.isin(mono_elmass.index)].tolist()
        form=form[elements]
    
    
    elements=form.columns.tolist()
    if type(charge)!=type(None):
        if "-" in elements: form["-"]*=charge
        if "+" in elements: form["+"]*=charge
        
    if "-" in elements or "+" in elements:
        z=np.zeros((len(form),1))
        if "-" in elements: z+=form["-"].values #form[:,np.argwhere(np.array(elements)=="-")[0]]
        if "+" in elements: z+=form["+"].values #form[:,np.argwhere(np.array(elements)=="+")[0]]
        charge=z.flatten().astype(int)        


    mono_mass=(form*mono_elmass.loc[elements].values).sum(axis=1).values
    # rel_mass=tables.set_index("symbol").loc[elements,["Relative Atomic Mass",'Isotopic  Composition']].prod(axis=1)
    # avg_mass=(form[elements]*rel_mass.groupby(rel_mass.index,sort=False).sum().values.flatten()).sum(axis=1).values
    
    #pop "+" ,"-" 
    if "-" in elements: form.pop("-")
    if "+" in elements: form.pop("+")
    elements=form.columns.tolist()
    
    form=form.values.astype(int)
    
    return form,elements,charge,mono_mass#,rel_mass,avg_mass

def read_resolution(fwhms,avg_mass): #input is either a file or numeric value

    if isinstance(fwhms,pd.DataFrame):
        if len(fwhms)==len(avg_mass): fwhms=fwhms["fwhm"].values
        else: fwhms=np.interp(avg_mass,fwhms["mz"],fwhms["fwhm"])
        return fwhms

    if isinstance(fwhms,np.ndarray) or type(fwhms)==type(list()):
        
        if len(fwhms)==len(avg_mass): 
            return np.array(fwhms)
        
        
        else:
            print("length of supplied fhwms is not equal to length of masses!")
            if len(fwhms)==1:
                print("using fwhm "+str(peak_fwhms[0])+" for all masses!")
                return np.repeat(np.array(fwhms),len(avg_mass))   
            else:
                print("using base default fwhm: 0.001")
                return np.repeat(np.array([0.001]),len(avg_mass))        
                

    if is_float(fwhms): 
        return np.repeat(np.array([fwhms]),len(avg_mass))   

    else:
        if os.path.exists(fwhms):
            print("FWHM filepath does not exist!")
            check,fwhms=read_table(fwhms,Keyword="fwhm")
            if not check: 
                print("incorrect fwhm file format! (Table requires columns 'mz','fwhm')  Skipping file: "+c)
           
                    
            print("using base default fwhm: 0.001")
            return np.repeat(np.array([0.001]),len(avg_mass))            
 

       


       
#%% isotope functions


def fft_lowres(form,
               
               bins=False,
               return_borders=False,
               
               
               #general arguments
               min_intensity=min_intensity,
               isotope_range=isotope_range,
               normalize=normalize,
               batch_size=batch_size,
               elements=[],
               mass_calc=True,
               add_mono=True
               ): 
    #%%
   
    
    #read input compositions
    if not len(elements): form,elements,charge,mono_mass=read_formulas(form) #,rel_mass,avg_mass
    else:
        mono_mass=(form*mono_elmass.loc[elements].values).sum(axis=1).values
        t=tables.set_index("symbol").loc[elements]
        elements=list(set(elements)-set(("+","-"))-set(t[t['Isotopic  Composition']==1].index))      #remove isotope-less elements
        form=form[elements].values.astype(int)



    #shift isotope table
    t=tables.set_index("symbol").loc[elements]
    t["delta shifted"]=t["delta neutrons"].groupby(t.index,sort=False).transform("min").values
    t["delta neutrons"]-=t["delta shifted"]
    t["delta_mass"]-=t["delta_mass"].groupby(t.index,sort=False).transform("min").values
    el_shift=t.loc[t["Standard Isotope"],"delta shifted"][elements]
  
    
    if return_borders: print("Low resolution fft prediction")
    l_iso=t.pivot(columns="delta neutrons",values='Isotopic  Composition').fillna(0)
    l_iso[list(set(np.arange(l_iso.columns.min(),l_iso.columns.max()+1))-set(l_iso.columns))]=0
    l_iso=l_iso.loc[elements,:]

    #esimating optimal grid size: 5+mu+6*s
    if not bins:
        p,dn=t['Isotopic  Composition'],t["delta neutrons"] #t["delta_mass"]
        
        mu=(dn*p).groupby(t.index).sum()[elements]
        var=(p * dn**2).groupby(p.index).sum()
        mutot=(form*mu.values.reshape(1,-1)).sum(axis=1)
        sig=np.sqrt((form*var[elements].values.reshape(1,-1)).sum(axis=1))
        
        # mu=(dn*p).groupby(t.index,sort=False).sum()[elements]
        # dnmu=dn-mu
        # var=(dnmu**2*p.values).groupby(dnmu.index).sum()[elements]
        # mutot,sig=form@mu,np.sqrt(form@var)
        
        maxp=np.array([next_fast_len(i) for i in np.ceil(mutot+5+sig*6).astype(int)])#+5 #add static constant
        bins=maxp.max()
        print("bins used: "+str(bins))
        #future: digitize with precomputed nfls? groupby maxp?

    pad=bins-len(l_iso.columns)
    
   
    
    #pad and fft shift negative ions
    l_iso[l_iso.columns.max()+np.arange(1,pad+1)]=0
    l_iso=l_iso.T.sort_index().T
    F=np.fft.fft(l_iso,axis=1)
    
    if mass_calc:
        e_iso=t.pivot(columns="delta neutrons",values='delta_mass').fillna(0).loc[elements,:]
        e_iso[list(set(np.arange(bins))-set(e_iso.columns))]=0 #pad 
        G = np.fft.fft(l_iso*e_iso, axis=1)
        ratio = G / (F + 1e-30)                   

    bdfs,iso_masses=[],[]
    maxns,maxps=[],[]
    forms=np.array_split(form.astype(int),np.arange(0,len(form),int(batch_size))[1:])
    for batch,counts in enumerate(forms):
        shifts=(counts*el_shift.values.reshape(1,-1)).sum(axis=1) #negative isotope shift
        print("batch: "+str(batch))

        

        #fft of abundances  (log trick instead of power)
        logF = np.log(F + 1e-30)  # avoid log(0)
        total_logF = counts @ logF # Matrix multiply
        T = np.exp(total_logF) # Back to linear space
        baseline = pd.DataFrame(np.real(np.fft.ifft(T, axis=1)))   # Inverse FFT for all formulas at once
        bcols=baseline.columns 
        
        
        if mass_calc:  #mass fft 
            weighted_sum = counts @ ratio             # (F, K)
            numerator_fft = T * weighted_sum
            numerator = np.fft.ifft(numerator_fft, axis=1).real
            iso_mass = pd.DataFrame(np.where(baseline > min_intensity, numerator / baseline, 0.0),columns=baseline.columns)

        if np.any(el_shift<0): #correct negative isotope shifts
            min_col,max_col = shifts.max(), baseline.shape[1]
            bcols = np.arange(min_col, max_col + 1)
            target_idx = np.arange(max_col) + shifts[:, None] - min_col
            
            out = np.full((len(baseline), len(bcols)), np.nan)
            out[np.arange(len(baseline))[:, None], target_idx] = baseline
            baseline=pd.DataFrame(out,columns=bcols).fillna(0)
            
            if mass_calc:
                out[np.arange(len(baseline))[:, None], target_idx] = iso_mass
                iso_mass=pd.DataFrame(out,columns=bcols).fillna(0)
 
        if return_borders: # or not len(isotope_range):
            q=(baseline<min_intensity).values
            q[:,-1]=True
            qmaxn,qmaxp=np.argmax(q[:,::-1],axis=1)+shifts,np.argmax(q,axis=1)
            maxn,maxp=qmaxn.max(),qmaxp.max()
            baseline=baseline[np.arange(maxn,maxp+1)]
            
        if return_borders:
            maxns.append(qmaxn)
            maxps.append(qmaxp)
            continue
        #%%
        if len(isotope_range): #only keep selected isotopes in output
            baseline=baseline.loc[:,baseline.columns[baseline.columns.isin(isotope_range)]]
            
        baseline.index=baseline.index+batch*batch_size
        
        bdfs.append(baseline)
        if mass_calc: iso_masses.append(iso_mass[baseline.columns])
    

    if return_borders: return np.hstack(maxns),np.hstack(maxps) #min(maxns),max(maxps)

    bdfs=pd.concat(bdfs).fillna(0)
    
    #normalize
    if normalize=="sum": bdfs=bdfs.divide(bdfs.sum(axis=1),axis=0)  #to total
    if normalize=="max": bdfs=bdfs.divide(bdfs.max(axis=1),axis=0)  #to highest
    if normalize=="mono": bdfs=bdfs.divide(bdfs[0],axis=0)          #to mono

    q=[True]
    if min_intensity>0: 
        q=bdfs>min_intensity
    
        
        
    if mass_calc:      
        iso_masses=pd.concat(iso_masses).fillna(0)
        #combine outputs

        
        
        if add_mono: iso_masses+=mono_mass.reshape(-1,1)
        
        if min_intensity>0: 
            ii=iso_masses.values[q].round(0).astype(int)
 
            return pd.DataFrame(np.vstack([bdfs.index[np.argwhere(q)[:,0]], 
                              iso_masses.values[q],
                              bdfs.values[q],
                              ii]).T,
                                columns=["ix","mass","abundance","isotope"])
     
       
        return bdfs,iso_masses
    
    
    
    return bdfs


#%%

#FFT with resolution
def fft_highres(form,
                
                
                #general arguments
                peak_fwhm=peak_fwhm,
                min_intensity=min_intensity,
                isotope_range=isotope_range,
                normalize=normalize,
                batch_size=batch_size,
                pick_peaks=pick_peaks,
                elements=[],
                charge=None, #unused
                packing=packing,
                #mass_shift=mass_shift,
                correct_charge=correct_charge,
                add_mono=add_mono,
                divisor=4,
                ): 

    #%%
  
    import time
    s=time.time()
    
    ### parse charge
    if isinstance(form,pd.DataFrame):
        if type(charge)==type(None):
            if "charge" in form.columns.str.lower():
                print("charge detected, dividing output masses by charge")
                charge=form.iloc[:,np.argwhere(form.columns.str.lower()=="charge")[0]].values

    if type(charge)==type(None): charge=1
    
        
    if type(charge)!=type(None):    
        charge=np.array(charge).reshape(-1,1).flatten()
        if (len(charge)==1) & (type(form)!=str): charge=np.repeat(charge,len(form))

    
    #read input compositions
    if not len(elements): form,elements,charge,mono_mass=read_formulas(form) #,rel_mass,avg_mass
    else:
        mono_mass=(form[elements]*mono_elmass.loc[elements].values).sum(axis=1).values
        t=tables.set_index("symbol").loc[elements]
        elements=list(set(elements)-set(("+","-"))-set(t[t['Isotopic  Composition']==1].index))      #remove isotope-less elements
        form=form[elements].astype(int)
        
    peak_fwhm=read_resolution(peak_fwhm,mono_mass) #avg_mass)
    peak_fwhm*=charge #properly handle charge
    peak_fwhm/=divisor 
    ufwhms=np.unique(peak_fwhm)
    minfwhm=ufwhms[0]

    #determine min number of bins (based on fast fft)
    form=pd.DataFrame(form,columns=elements) #bit redundant Datafarme and array conversions?
    print("")
    

    amaxn,amaxp=fft_lowres(form, elements=elements,
                         return_borders=True,min_intensity=min_intensity,batch_size=batch_size*10,mass_calc=False) 


    #%%

    form[["maxn","maxp"]]=np.vstack([amaxn,amaxp]).T
    form=form.reset_index(drop=True)
    # form[["maxn","maxp"]]=[0,14] #test
    form.loc[form.loc[:,"maxn"]>0,"maxn"]=0 #limit maxn to maximum of 0 
    form.loc[form.loc[:,"maxp"]<0,"maxp"]=0 #limit maxp to minimum of 0 
    
    
    np.clip(form["maxn"],None,0)
    
    imass=tables.set_index("symbol").loc[elements]
    ns=imass[~imass['Standard Isotope']] #find max packing window
    
    multi_fwhm=False #flag
    if len(ufwhms)==1: #only one width

        maxn,maxp=min(amaxn),max(amaxp)    
        iso_bins=np.arange(maxn,maxp+minfwhm,minfwhm)
        imass=imass[(imass.delta_mass.values>=maxn) & (imass.delta_mass.values>=maxn)] #clip to maxn,maxp
        
        bi=find_closest(iso_bins,imass.delta_mass.values) #iso bin
        imass["col_ix"]=bi
        fimass=imass.merge(pd.Series(np.arange(len(elements)),index=elements,name="row_ix"),how="left",right_index=True,left_index=True)

        #construct base array
        mi_space=np.zeros([len(elements),len(iso_bins)])
        mi_space[fimass["row_ix"].values,fimass["col_ix"].values]=fimass['Isotopic  Composition']
        m_iso=pd.DataFrame(mi_space,index=elements,columns=iso_bins)

    
    else:
        multi_fwhm=True
        form["fwhm"]=peak_fwhm
  

    bdfs=[]
    for n,g in form.groupby(["maxn","maxp"]):
        maxn,maxp=n
        #clip with highest element
        
        ge=np.array(elements)[np.any(g[elements]>0,axis=0)]
        
          
        #### FFT grid construction ####
        
        #find min fwhm
        if multi_fwhm:
            minfwhm=g.fwhm.min() 
            fiso_bins=np.arange(maxn,maxp+minfwhm,minfwhm)
        else:
            bl,br=np.searchsorted(iso_bins,[maxn,maxp])
            fiso_bins=iso_bins[bl:br+1]

        
        max_spread=abs(maxp//ns["delta neutrons"]*(ns["delta_mass"]-ns["delta neutrons"])).max()/minfwhm
        msi=int(np.ceil(max_spread))

        if packing:
            be=find_closest(fiso_bins,np.arange(maxn,maxp+1)) #cut away from the centers
            mids=(be[:-1]+np.diff(be)/2).astype(int)
            w=int((np.diff(be).mean()-max_spread*2)/2) 
            fiso_bins=fiso_bins[~np.isin(np.arange(len(fiso_bins)),(mids.reshape(-1,1)+np.arange(-w,w+1)).flatten())]
        
        if multi_fwhm:
            fimass=imass[(imass.delta_mass.values>=maxn) & (imass.delta_mass.values>=maxn)] #clip to maxn,maxp
            bi=find_closest(iso_bins,fimass.delta_mass.values) #iso bin
            fimass["col_ix"]=bi
            fimass=fimass.merge(pd.Series(np.arange(len(ge)),index=elements,name="row_ix"),how="left",right_index=True,left_index=True)

            #construct base array
            mi_space=np.zeros([len(elements),len(iso_bins)])
            mi_space[fimass["row_ix"].values,fimass["col_ix"].values]=fimass['Isotopic  Composition']
            m_iso=pd.DataFrame(mi_space,index=elements,columns=iso_bins)
        
        #only keep elements in mat
        b_iso=m_iso.loc[ge] 

        
        mat=b_iso[fiso_bins].values
        
        #fft padding
        padl,padr=np.arange(maxn-msi,maxn)*minfwhm,maxp+np.arange(1,msi+1)*minfwhm
        totlen=len(fiso_bins)+len(padl)+len(padr)
        padr2=next_fast_len(totlen)-totlen
        if padr2: padr2=padr[-1]+np.arange(1,padr2+1)*minfwhm
        else: padr2=[]
        mat=np.hstack([np.zeros((len(mat),len(padl))),mat,np.zeros((len(mat),len(padr))),np.zeros((len(mat),len(padr2)))])
        fiso_bins=np.hstack([padl,fiso_bins,padr,padr2])

        
        #zero shift
        z=fiso_bins<0
        mat=np.hstack([mat[:,~z],mat[:,z]])
        fiso_bins=np.hstack([fiso_bins[~z],fiso_bins[z]])
        
        vffts=np.fft.fft(mat,axis=1)
        
        # if mass_shift:
        #     bimass=fimass.loc[ge]
        #     bimass["row_ix"]=np.unique(bimass.row_ix,return_inverse=True)[1]
        #     mmat=np.zeros(mat.shape)
        #     cols=find_closest(fiso_bins,bimass.delta_mass.values)
        #     md=bimass['delta_mass']-fiso_bins[cols]
        #     mmat[bimass.row_ix.values,cols]=md
        #     G = np.fft.fft(mmat*mat, axis=1)
        #     ratio = G / (vffts + 1e-30)         
        
        #### FFT convolution  ####

        # Take log once
        log_E = np.log(vffts + 1e-30)  
        log_spectra_fft = g[ge].values @ log_E           # Matrix multiply
        spectra_fft = np.exp(log_spectra_fft)                  # Back to linear space
        baseline = np.real(np.fft.ifft(spectra_fft, axis=1))   # Inverse FFT for all formulas at once
  


        # if mass_shift:  #mass fft 
        #     weighted_sum = g[ge].values @ ratio             # (F, K)
        #     numerator_fft = spectra_fft * weighted_sum
        #     numerator = np.fft.ifft(numerator_fft, axis=1).real
        #     # iso_mass = pd.DataFrame(np.where(baseline > min_intensity, numerator / baseline, 0.0),columns=fiso_bins,index=g.index)


        #     #fix divide by zero warning
        #     iso_mass = pd.DataFrame( np.divide( numerator, baseline, 
        #                                        out=np.zeros_like(numerator, dtype=float), where=baseline > min_intensity), 
        #                             columns=fiso_bins,index=g.index )

  
        #sort_columns
        bdf=pd.DataFrame(baseline,columns=fiso_bins)
        bdf=bdf.T.sort_index().T
        bcols=bdf.columns
        
      
        #### Gaussian convolution  ####
     
        if (divisor>1) or multi_fwhm: 
         
            if not multi_fwhm: 
                sig=divisor/2.355
                bdf=gaussian_filter1d(bdf,sigma=sig,mode="constant",cval=0.0)
                bdf*=sig*np.sqrt(2*np.pi)
            else:  
                sigs=divisor*g.fwhm.values/minfwhm/2.355
                bdf=[gaussian_filter1d(bdf[ix],sigma=sigs[ix],mode="constant",cval=0.0) for ix,i in enumerate(peak_fwhm)]
                bdf*=sigs.reshape(-1,1)*np.sqrt(2*np.pi)
            
            bdf=pd.DataFrame(bdf,columns=bcols,index=g.index)
            
      
    
        if len(isotope_range): #only keep selected isotopes in output
            q=np.round(bdf.columns,0).astype(int).isin(isotope_range)
            bdf=bdf.iloc[:,q]

            
        if normalize=="sum":  bdf=bdf.divide(bdf.sum(axis=1),axis=0)
        if normalize=="max":  bdf=bdf.divide(bdf.max(axis=1),axis=0)
        if normalize=="mono": bdf=bdf.divide(bdf.loc[:,0],axis=0)
        bdf[bdf<min_intensity]=0
            

        if pick_peaks:
            x,y=argrelmax(bdf.values,axis=1,mode="wrap") #?
            d=np.vstack([g.index[x],bdf.columns[y],bdf.values[x,y]]).T
  
        else: 
            keep=np.argwhere(bdf>min_intensity)
            abundance=bdf.values[keep[:,0],keep[:,1]]
            mass=bdf.columns[keep[:,1]]
            d=np.vstack([keep[:,0],mass,abundance]).T
        
        
        # if mass_shift:
        #     iso_mass=iso_mass[bcols]
        #     rows = iso_mass.index.get_indexer(d[:,0])
        #     cols = find_closest(bcols,d[:,1])
     
        #     d[:,1]-=iso_mass.values[rows,cols]
            
        bdfs.append(d)
        

    bdfs=pd.DataFrame(np.vstack(bdfs),columns=["ix","iso_mass","abundance"]).sort_values(by=["ix","iso_mass"])
  
    if add_mono: bdfs["iso_mass"]+=mono_mass[bdfs.ix.astype(int)]
    if correct_charge: bdfs["iso_mass"]/=charge[bdfs.ix.astype(int)] #divide or multiply
    #%%
    return bdfs.sort_index()


#Precomputed

def Precompute_multi(form,elements=None,isotope_range=isotope_range,prune=prune,min_chance=min_chance):

#%%
    print("Multinomial prediction")
    
    if type(elements)==type(None): elements=form.columns[form.columns.isin(tables.symbol)].tolist()
    if isinstance(form,pd.DataFrame): form=form[elements]
    
    
    #####  1.  Composition space   #####
    edf=pd.DataFrame(np.vstack([form.min(axis=0),form.max(axis=0)]).T,index=elements)
    edf.columns=["low","high"]
    edf["arr"]=[np.arange(i[0],i[1]+1) for i in edf.values]
    lim=edf[["low","high"]].reset_index(names="symbol").drop_duplicates().set_index("symbol").astype(int)
    
    #Isotope combinations
    ni=tables[tables.symbol.isin(edf.index)].reset_index() #get all isotopes
    ni=ni[~ni["Standard Isotope"]].reset_index() #no 0
    ni.index+=1 


    #### No pruning during combinations
    if not prune:
        kdf=[]
        for i in np.arange(0,isotope_range.max()+1):
            kdf.append(pd.DataFrame(list(combinations_with_replacement(ni.index.tolist(), i))))
        kdf=pd.concat(kdf).reset_index(drop=True).fillna(0)
        c=kdf.apply(Counter,axis=1)
        kdf=pd.DataFrame(c.values.tolist()).fillna(0)
        
        kdf.columns=kdf.columns.astype(int)
        if 0 in kdf.columns: kdf.pop(0) #no 0



    #### Pruning during combinations
    if prune or not len(isotope_range):
       
        #initialize
        cs=ni[['Isotopic  Composition']].values       
        el_ix=ni[['Isotopic  Composition']].values.T 
        els=ni["symbol"]
        ixs=[np.arange(len(ni)).reshape(-1,1)]

        ecounts=[]
        while True:
            v=cs.reshape(-1,1)*el_ix
            
            #filter on minimum chance
            aq=np.argwhere(v>prune)
            if not len(aq): break
            
            #unique combinations
            i=np.hstack([ixs[-1][aq[:,0]],aq[:,1].reshape(-1,1)])
            idf=pd.DataFrame(pd.DataFrame(i).apply(Counter,axis=1).values.tolist()).fillna(0).astype(int).T.sort_index()
            idf=idf.T.drop_duplicates().T
            
            ue=np.unique(idf.index)
            el_ix=ni.loc[ue+1,'Isotopic  Composition'].values.reshape(1,-1) #update el_ix
            els=ni.loc[ue+1,"symbol"].values                                #update els
            
            #filter on isotope range
            if (len(isotope_range)) & (ni["delta neutrons"].min()>0):  
                q_isotope_range=(idf*ni["delta neutrons"].values[[ue]].T).sum(axis=0).isin(isotope_range)
                idf=idf.T[q_isotope_range].T
            
            #filter on element counts
            idf.index=els
            idf=idf.groupby(idf.index).sum() 
            ilim=lim.loc[idf.index,"high"].values
            qs=np.vstack([i<=ilim[ix] for ix,i in enumerate(idf.values)]).all(axis=0)

            qx=idf.columns[qs].values
            aqq=aq[qx]
        
            cs=v[aqq[:,0],aqq[:,1]] #update cs
            print(len(qx))
            if not len(cs): break
            ixs.append(i[qx])
            
            
        #convert to kdf format.
        kdf=[]
        for i in ixs:
            c=pd.DataFrame(i).apply(Counter,axis=1)
            kdf.append(pd.DataFrame(c.values.tolist()).fillna(0))
            
        kdf=pd.concat(kdf).fillna(0).astype(int)
        kdf.index=np.arange(1,len(kdf)+1)
        kdf.loc[0,:]=0
        kdf.columns=ni.index
        kdf=kdf.sort_index()

    kdfc=kdf.columns
    kdf=kdf[kdf.columns.sort_values()]
    isos=ni.loc[kdfc,'isotope_symbol'].tolist()

    kdf["delta_mass"]=(kdf*ni.loc[kdfc,"delta_mass"]).sum(axis=1)
    kdf["delta_neutrons"]=(kdf*ni.loc[kdfc,"delta neutrons"]).sum(axis=1).astype(int)
    if len(isotope_range): kdf=kdf[kdf.delta_neutrons.isin(isotope_range)]
        
    kdf.columns=isos+["delta_mass","delta_neutrons"]
    kdf=kdf.sort_values(by="delta_mass")
    kdf_mass=kdf.delta_mass.values

    #parse iso_string
    iso_string=kdf.iloc[:,:-2].astype(int).values.astype(str)
    q=iso_string=="0"
    iso_string=iso_string+kdf.columns[:-2].values+","
    iso_string[q]=""
    
    #return iso_string #test
    iso_string=pd.Series(iso_string.sum(axis=1)).astype(str).str.strip(",").values
    kdf["iso_string"]=iso_string
    
    ###### 2. Precompute multinomials #######
    
    #% Split kdf into groups that contain different elements
    earrs,uis=[],[]
    inc=0
    for test,e in enumerate(edf.index):
        eix=np.argwhere(kdf.columns.str.startswith(e+"."))[:,0]
    
        u,ui=np.unique(kdf.iloc[:,eix].astype(int).values,return_inverse=True,axis=0)
        ll,ul=edf.loc[e,"low"],edf.loc[e,"high"]
        ecounts=np.arange(ll,ul+1)
        nv=ni.loc[eix+1,'Isotopic  Composition'].values
        nv=[1-nv.sum()]+nv.tolist()
        earrs.append(np.vstack([np.array([multinomial(i,nv).pmf([i-sum(r)]+r.tolist()) for i in ecounts]) for r in u]).T)
        uis.append(ui+inc)
        inc+=len(u)

    uis=np.vstack(uis).T #uis stores precomputed probability distributions for different element counts for elements that have isotopes
    uif=uis.flatten()

    #mapping element counts to earrs
    m_ecounts=[]
    for ie in edf.arr.values:
        ie=ie.astype(int)
        zm=np.zeros(ie.max()+1,dtype=np.uint16)
        zm[ie]=np.arange(len(ie))
        m_ecounts.append(zm)

    return earrs,m_ecounts,uif,uis,kdf_mass,kdf

#%%


#%%
def multi_conv(form,
               
               #general arguments
               prune=prune,
               min_chance=min_chance,
               isotope_range=isotope_range,
               
               peak_fwhm=peak_fwhm,
               convolve=convolve,
               convolve_batch=convolve_batch,
               #add convolve batch option
       
               normalize=normalize,
               elements=None,
               charge=None,
               Precomputed=Precomputed,
               verbose=verbose,

               correct_charge=correct_charge,
               add_mono=add_mono
               #min_chance=1e-6,prune=1e-6, convolve=False,isotope_range=np.arange(-2,7)
               
               ): 

    #%%

    import time
    s=time.time()

    ## parse charge
    if type(charge)==type(None):
        if isinstance(form,pd.DataFrame): #attempt to read from dataframe
            if "charge" in form.columns.str.lower():
                print("charge detected, dividing output masses by charge")
                cc=np.argwhere(form.columns.str.lower()=="charge")[0][0]
                charge=form.iloc[:,cc].astype(int).values
        else: charge=1
            
            
    
    ## parse formula   
    if type(elements)==type(None):
        
        if type(form)==str: form=parse_form(form).reset_index(drop=True)  #single string
        elif  not (isinstance(form, pd.DataFrame) or isinstance(form, pd.Series)): #iterable
            form=pd.concat([parse_form(f) for f in form]).reset_index(drop=True).fillna(0)          
        if isinstance(form, pd.DataFrame): #DataFrame with columns (recommended input)        
            elements=form.columns[form.columns.isin(mono_elmass.index)].tolist()
    else:
        mono_mass=(form[elements]*mono_elmass.loc[elements].values).sum(axis=1).values

    
    ## parse charge
    if type(charge)!=type(None):
        charge=np.array(charge).reshape(-1,1).flatten()
        if (len(charge)==1) & (type(form)!=str): charge=np.repeat(charge,len(form))
        if "-" in elements:
            if np.all(form["-"]==1): form["-"]*=charge
        
        if "+" in elements:
            if np.all(form["+"]==1): form["+"]*=charge

    
    ## calculate mass
    mono_mass=(form*mono_elmass.loc[elements].values).sum(axis=1).values
    
    ## detect index:
    xx=np.arange(len(form))
    if (isinstance(form, pd.DataFrame) or isinstance(form, pd.Series)): xx=form.index

    t=tables.set_index("symbol").loc[elements]
    elements=list(set(elements)-set(("+","-"))-set(t[t['Isotopic  Composition']==1].index))      #remove isotope-less elements
    form=form[elements].values.astype(int)
    
    ## parse charge
    if "-" in elements or "+" in elements:
        z=np.zeros((len(form),1))
        if "-" in elements: z+=form[:,np.argwhere(np.array(elements)=="-")[0]]
        if "+" in elements: z+=form[:,np.argwhere(np.array(elements)=="+")[0]]
        charge=z.flatten().astype(int)        

    
    if type(charge)!=type(None): mono_mass/=charge
    peak_fwhm=read_resolution(peak_fwhm,mono_mass)    

    print("Parsing time: "+str(time.time()-s))   


    s=time.time()

    if len(Precomputed): earrs,m_ecounts,uif,uis,kdf_mass,kdf=Precomputed
    else:                earrs,m_ecounts,uif,uis,kdf_mass,kdf=Precompute_multi(form,
                                                                               elements=elements,
                                                                               isotope_range=isotope_range,
                                                                               prune=prune,
                                                                               min_chance=min_chance)
    print("Pecomputation time: "+str(time.time()-s))   

    s=time.time()
    
    #### 3. Predict #####
    res_ixs=[]

    res_ix=np.arange(len(form))
    
    
    #Fast unique elements
    present=form>0
    bits = (1 << np.arange(form.shape[1], dtype=np.uint64))
    ecform = (present * bits).sum(axis=1)

    def masks_to_bool(masks, n_elements):
        bits = (1 << np.arange(n_elements, dtype=np.uint64))
        return (masks[:, None] & bits) != 0

    ur,uri=np.unique(ecform,return_inverse=True)
    ur=masks_to_bool(ur,form.shape[1])
    #%%
    multi_preds=[]
    
    import time
    t1,t2,t3,t4=0,0,0,0
    tot=time.time()
    
    for ixr,r in enumerate(ur): #loop over combinations of elements
        #if verbose: print(ixr)
        
        #s=time.time()
        qf=np.argwhere(uri==ixr)[:,0] 
        bf=form[qf]
   
        #chances is the concatenated multinomial table                
        chances=np.hstack([earrs[eix][m_ecounts[eix][bf[:,eix]]]  for eix,e in enumerate(earrs) if len(earrs[eix])])
        chances[chances<min_chance]=0 #"big matrix" of chances for those element combinations
 
        #column combinations that should be computed  (uis links to rows of kdf, columns to elements (edf))
        q=np.all(np.in1d(uif,np.argwhere(chances.sum(axis=0)>0)[:,0]).reshape(-1,uis.shape[1]),axis=1)
        fuis=uis[q] #remove combinations that are zero in batch
        
        ux = np.empty((len(bf), len(fuis)), dtype=chances.dtype) #Pre-allocate output array (faster?)
        for ci, u in enumerate(fuis):
            ux[:, ci] = chances[:,u].prod(axis=1)


        #mask redundant rows from product
        unique_cols, inverse = np.unique(fuis, return_inverse=True)
        inverse = inverse.reshape(fuis.shape)
        present_unique = chances[:, unique_cols] > 0  # shape = (R_chances, n_unique)
        vals_rows = present_unique[:, inverse]      # shape = (R_chances, R_fuis, k)
        mask = np.all(vals_rows, axis=2)            # shape = (R_chances, R_fuis)
        
        
        # #combine chances
        ux = np.empty((len(bf), len(fuis)), dtype=chances.dtype) #Pre-allocate output array (faster?)

        for ci, u in enumerate(fuis):
            rows = np.flatnonzero(mask[:, ci])          # indices of valid rows
            if len(rows) == 0:
                continue
            # only index the valid rows
            ux[rows, ci] = chances[rows[:, None], u].prod(axis=1)
                    
      
        uxq=ux>min_chance
        qq=np.argwhere(uxq)
        
        multi_preds.append(np.vstack([res_ix[qf][qq[:,0]],kdf_mass[q][qq[:,1]],ux[uxq],kdf.index[q][qq[:,1]]]).T)
        
        
    print(time.time()-tot)

#%%
    print("Prediction time: "+str(time.time()-s))
        
    multi_df=pd.DataFrame(np.vstack(multi_preds),columns=["index","mass","abundance","isotope"]).set_index("index")
    multi_df.index=multi_df.index.astype(int)
    multi_df=multi_df.sort_index()
    multi_df["isotope"]=kdf.loc[multi_df["isotope"].astype(int),"iso_string"].values

    if normalize=="mono": multi_df["abundance"]/=multi_df.loc[multi_df.mass==0,"abundance"].loc[multi_df.index]
    
    s=time.time()
    if convolve=="weighted_sum"  : #fastest 
        multi_df["isotope"]=multi_df["mass"].round(0).astype(int)
        multi_df["wa"]=multi_df["mass"]*multi_df["abundance"]
        ws=multi_df.reset_index().groupby(["index","isotope"])[["wa","abundance"]].sum().reset_index()
        ws["mass"]=ws["wa"]/ws["abundance"]
        multi_df=ws.set_index("index")[["mass","abundance","isotope"]]
             
    if convolve=="fast": multi_df=convolve_fast(multi_df,peak_fwhm,mono_mass,convolve_batch=convolve_batch,verbose=verbose,charge=charge)
    if convolve=="full": multi_df=convolve_full(multi_df,peak_fwhm,mono_mass,convolve_batch=convolve_batch,verbose=verbose,charge=charge)

    print("Convolution time: "+str(time.time()-s))
    
    if not convolve or convolve=="weighted_sum":
        
        #add isotope
        if type(charge)!=type(None): 
            multi_df["charge"]=charge[multi_df.index.values]
            #comb["isotope"]*=charge[comb.index] 
        else:
            multi_df["charge"]=1

        if correct_charge: multi_df.mass/=multi_df.charge
        if add_mono:       multi_df.mass+=mono_mass[multi_df.index.values]    
            
    if normalize=="sum": multi_df["abundance"]=multi_df["abundance"]/multi_df.groupby(multi_df.index)["abundance"].transform("sum")
    if normalize=="max": multi_df["abundance"]=multi_df["abundance"]/multi_df.groupby(multi_df.index)["abundance"].transform("max")

    multi_df.index=xx[multi_df.index]
    #%%
    return multi_df     


#%%

def convolve_full(multi_df,peak_fwhm,mono_mass,divisor=10,convolve_batch=convolve_batch,verbose=verbose,charge=None):
    
    #%%
    
    
    #add min abundance
    
    multi_df["isotope"]=multi_df.mass.round(0).astype(int)
    multi_df["zmass"]=multi_df["mass"]-multi_df["isotope"]
    multi_df["peak_fwhm"]=peak_fwhm[multi_df.index]
    
    ui=np.unique(multi_df.index)
    if convolve_batch: batch_groups=[ui[i:i+int(convolve_batch)] for i in range(0,len(ui),int(convolve_batch))]
    else: batch_groups=[ui]
    
    #if w> points_r- points_l (weighted mean)
    #if points  >2 (convolve on grid)
    #place outer most points on outside
    res=[]
    for batch,b in enumerate(batch_groups):
        if verbose: print("batch: "+str(batch))

        #read group
        g=multi_df.loc[b,:]        
         
        gx=np.vstack([g.index,g.isotope.values]).T
        gs=np.hstack([0,np.argwhere(~(gx[1:]==gx[:-1]).all(axis=1))[:,0]+1,len(gx)])
        g["row"]=np.repeat(np.arange(len(gs)-1),np.diff(gs))
        g["minmass"]=g.groupby("row")["zmass"].transform("min")
        g["cmass"]=g["zmass"]-g["minmass"]
        mdelt=g.cmass.max()
        
        fwhms=g.groupby(g.index)["peak_fwhm"].nth(0)
        ufwhms=np.unique(fwhms)
        minsigma=ufwhms[0]

        #constructing gaussian space
        l,u=-g.peak_fwhm.min(),mdelt+g.peak_fwhm.max()
        x=np.linspace(l,u,(int(np.round((u-l)/minsigma))*divisor))
        ycors=find_closest(x,g.cmass) 

        xcors=g.row
        zmat=coo_matrix((g.abundance, (xcors, ycors))).toarray()

        #convolve with gaussian (Gaussian FWHM=2*sqrt(2ln(2)) =~ 2.355*s ) (so sigma is fwhm/2.355)
        if len(ufwhms)==1: 
            sig=divisor/2.355
            gmat=gaussian_filter1d(zmat,sigma=sig,mode="constant",cval=0.0)*sig*np.sqrt(2*np.pi)
        else:               
            
            sigs=divisor*ufwhms/minsigma/2.355
            gmat=[gaussian_filter1d(zmat[ix],sigma=sigs[ix],mode="constant",cval=0.0)*sigs[ix]*np.sqrt(2*np.pi) #* divisor*2.355*i/minsigma  
                                  for ix,i in enumerate(fwhms)]
        
        keep=np.argwhere(gmat>min_intensity)
        i=gmat[gmat>min_intensity] #abundance
        m=g.groupby("row")[["isotope","minmass"]].nth(0).values.sum(axis=1)[keep[:,0]]+x[keep[:,1]]             #mass
        ix=g.groupby("row").nth(0).index[keep[:,0]]
        
        res.append(np.vstack([ix,m,i]).T)

    res=pd.DataFrame(np.vstack(res),columns=["ix","mass","abundance"])
    res["ix"]=res["ix"].astype(int)
    res["isotope"]=res["mass"].round(0).astype(int)
    
    #add isotope
    if type(charge)!=type(None): 
        res["charge"]=charge[res["ix"].values]
        #comb["isotope"]*=charge[comb.index] 
    else:
        res["charge"]=1
    
    if correct_charge: res.mass/=res.charge
    if add_mono:       res.mass+=mono_mass[res["ix"].values]

    
    res=res.set_index("ix")
    
    #%%
    return res

#%% Draft code: 2nd derivative peak picking of gmat
 # #% pick peaks
 # d=np.diff(gmat,axis=1)
 # dd=np.diff(d,axis=1)     #2nd derivative
 # ndd=np.abs(dd)/np.abs(dd).max(axis=1).reshape(-1,1) #normalized change
 # tol=0.05
 
 # #find peaks
 # px,py=argrelmin(dd,axis=1) #row coordinate, column coordinate
 # py+=1 
 # q=ndd[px,py]>tol #remove nonsense peaks
 # px,py=px[q],py[q]
 
 # #compute base data
 # m=x[py-1]+zms[px]          #mass
 # a=gmat[px,py]              #abundances
 # i=gu[px] #g.index[gui].values[px] #index
 # v=pd.DataFrame(np.vstack([m,a,i]).T,columns=["mass","abundance","gx"])
 
 # if add_borders:

 #     #find valleys
 #     minx,miny=argrelmax(dd,axis=1)
 #     miny+=1
 #     q=ndd[minx,miny]>tol #remove nonsense peaks
 #     minx,miny=minx[q],miny[q]
     
     
 #     pdf=pd.DataFrame(np.vstack([np.vstack([px,py,np.ones(len(px),np.int32)]).T,
 #                                 np.vstack([minx,miny,np.zeros(len(minx),np.int32)]).T]),columns=["row","col","p"]).sort_values(by=["row","col"])
 #     pr=np.argwhere(pdf.p==1)[:,0]
 #     pb=pd.DataFrame(np.hstack([pdf[pdf.p==1],np.vstack([pdf.iloc[pr-1].col,pdf.iloc[pr+1].col]).T]),columns=["row","col","p","l","r"])
 #     p_end=np.argwhere((pb.l.values[1:]!=pb.r.values[:-1]) | (pb.row.values[1:]!=pb.row.values[:-1]))[:,0]
 #     el,er=np.hstack([0,p_end+1]),np.hstack([p_end,len(pb)-1])

 #     #peak borders
 #     ext=np.ceil(con.peak_fwhm/minfwhm*divisor*((3-2.355))).values
 #     pb.loc[el,"l"]-=ext[pb.iloc[el]["row"].values]
 #     pb.loc[er,"r"]+=ext[pb.iloc[el]["row"].values]

     
 #     bc=zms[pb.row].reshape(-1,1)+x[pb[["l","r"]]]                                     #borders of con
 #     v[["mass_l","mass_r"]]=bc
     
 #     if add_area:
 #         ac=np.array([(gmat[g.row,g.l:g.r+1]*(x[1]-x[0])).sum() for n,g in pb.iterrows()]) #area of con
 #         v["area"]=ac
 
 # res.append(v)

def convolve_fast(multi_df,peak_fwhm,mono_mass,divisor=10,convolve_batch=convolve_batch,verbose=verbose,charge=None,add_borders=add_borders,add_area=add_area):
#%%



    
    if add_area: add_borders=True
    
    if verbose:
        print("Convolving multinomial with gaussian")
        if convolve_batch:
            print("Convolve batch size: "+str(int(convolve_batch)))
    

    comb=[]
    mix=multi_df.index.values
    gs=np.hstack([ 0,np.argwhere(mix[:-1]!=mix[1:])[:,0]+1,len(mix)])
    exw=np.repeat(peak_fwhm,np.diff(gs))
    multi_df["peak_fwhm"]=exw
    
    qw=np.hstack([True,abs(multi_df.mass.values[1:]-multi_df.mass.values[:-1])>exw[:-1]*2,True])
    qw[gs]=True #where new group == True
    
    ds=np.diff(np.argwhere(qw)[:,0])
    multi_df["gx"]=    np.repeat(np.arange(len(ds)),ds)
    rix=multi_df.gx.drop_duplicates().reset_index().set_index("gx") #swapped index

    nc=np.argwhere(ds==1)[:,0] #if points ==1 (dont convolve)
    q=np.in1d(multi_df.gx,nc)
    nocon,con=multi_df[q],multi_df[~q]
    

    if len(nocon):
        
        v=nocon[["mass","abundance","gx"]]
        
        if add_borders:
            w=2.55 * nocon.peak_fwhm /2 #borders of nocon full width vs fwhm: 6*sigma /2
            v["mass_l"]=v.mass-w
            v["mass_r"]=v.mass+w
            
        if add_area:     v["area"]=nocon.abundance*1.0645*nocon.peak_fwhm 

        comb.append(v)
    


    if len(con):
        #find groups where peak_fwhm > conr-conl
        cong=np.hstack([0,np.argwhere(con.gx.values[:-1]!=con.gx.values[1:])[:,0]+1,len(con)]) #find groups
        conl,conr=con.mass.iloc[cong[:-1]],con.mass.iloc[cong[1:]-1]                           #compute borders
        con.loc[:,"zm"]=np.repeat(conl,np.diff(cong))                                          #add zero mass
        
        pp=con.sort_values(by="peak_fwhm").set_index("gx")
        pp["mass"]-=pp["zm"]
        
        ui=np.unique(pp.index)
        if convolve_batch: batch_groups=[ui[i:i+int(convolve_batch)] for i in range(0,len(ui),int(convolve_batch))]
        else: batch_groups=[ui]
        
        #%%
        res=[]
        for batch,b in enumerate(batch_groups):
            if verbose: print("batch: "+str(batch))

            #read group
            g=pp.loc[b,:]  
            cm=g.mass #centered mass
             
            fwhms=g.groupby(g.index)["peak_fwhm"].nth(0)
            borders=g.groupby(g.index).agg({"mass":["min","max"]}).values
            ufwhms=np.unique(fwhms)
            minfwhm=ufwhms[0]
            bw=minfwhm/divisor
                
            pad=2
            if add_borders: pad=fwhms/bw*4
            lb,rb=(borders[:,0]-pad*bw).min(), (borders[:,1]+pad*bw).max()  
            x=np.arange(lb,rb+bw,bw)

            #constructing gaussian space
            ycors=find_closest(x,cm.values) 
            gu,gui=np.unique(g.index,return_index=True) 
            zms=g.iloc[gui].zm.values #does this consider the bin shifting (no!)?
            xcors=find_closest(gu,g.index) 
            zmat=coo_matrix((g.abundance, (xcors, ycors))).toarray()
            
            #convolve with gaussian (Gaussian FWHM=2*sqrt(2ln(2)) =~ 2.355*s ) (so sigma is fwhm/2.355)
            sigs=divisor*fwhms.values/minfwhm/2.355
            if len(ufwhms)==1: 
                sig=divisor/2.355
                gmat=gaussian_filter1d(zmat,sigma=sig,mode="constant",cval=0.0)#*sig*np.sqrt(2*np.pi)
            else:               
                gmat=[gaussian_filter1d(zmat[ix],sigma=sigs[ix],mode="constant",cval=0.0)#*sigs[ix]*np.sqrt(2*np.pi) #* divisor*2.355*i/minsigma  
                                      for ix,i in enumerate(fwhms)]
                
            gmat*=sigs.reshape(-1,1)*np.sqrt(2*np.pi)


            ### argelex peak picking
            px,py=argrelmax(gmat,axis=1) #peaks
            
            m=x[py-1]+zms[px]          #mass
            a=gmat[px,py]              #abundances
            i=gu[px] #g.index[gui].values[px] #index
            v=pd.DataFrame(np.vstack([m,a,i]).T,columns=["mass","abundance","gx"])
            
            if add_borders:
                vx,vy=argrelmin(gmat,axis=1) #valleys
     
                
                #add zeros and maxs
                vv=np.ones(len(vx),np.int32)
                vv=np.hstack([vv,[1,0]*len(gmat)])
                
                vx=np.hstack([vx,np.repeat(np.arange(len(gmat)),2)])
                vy=np.hstack([vy,[0,zmat.shape[1]-1]*len(gmat)]) #append

                #merge vals
                ps=pd.DataFrame(np.vstack([np.vstack([px,py,np.zeros(len(px),np.int32),
                                                      np.ones(len(px),np.int32)]).T,
                                           
                                           np.vstack([vx,vy,vv,
                                                      np.zeros(len(vx),np.int32) ]).T,]),              
                                
                                columns=["row","col","v","p"])
                ps=ps.sort_values(by=["row","col"]).reset_index(drop=True)
                ps["g"]=ps["row"]+ps["v"].cumsum()
                

                #lowest and highest per group
                bs=ps[ps.p==1].reset_index(names=["x"]).groupby(["row","g"],sort=False).agg({"col":["min","max"],"x":["min","max"]})
                bs.columns=bs.columns.droplevel()
                bs.columns=["min_c","max_c","min_x","max_x"]
                bs=bs.reset_index()
                w=(sigs[bs.row]*3).round(0)
                
                #peak borders
                bs["bl"]=np.vstack([ps.loc[bs["min_x"]-1,"col"].values,bs["min_c"]-w]).astype(int).max(axis=0)
                bs["br"]=np.vstack([ps.loc[bs["max_x"]+1,"col"].values,bs["max_c"]+w]).astype(int).min(axis=0)
            
                bc=zms[bs.row].reshape(-1,1)+x[bs[["bl","br"]]]                                     #borders of con
                v[["mass_l","mass_r"]]=bc #error more  borders than peaks!
                
                if add_area:
                    ac=np.array([(gmat[g.row,g.bl:g.br+1]*(x[1]-x[0])).sum() for n,g in bs.iterrows()]) #area of con
                    v["area"]=ac

            res.append(v)
            
        if len(res):
            comb.append(pd.concat(res))

    #merge results
    comb=pd.concat(comb)
    comb=comb.sort_values(by="gx")
    comb.index=rix.loc[comb.gx].values.flatten()
    
    #add isotope
    comb["isotope"]=comb.mass
    if type(charge)!=type(None): 
        comb["charge"]=charge[comb.index] 
        #comb["isotope"]*=charge[comb.index] 
    else:
        comb["charge"]=1
    comb["isotope"]=comb["isotope"].round(0).astype(int)    
    
    if correct_charge: comb["mass"]/=comb["charge"]
    if add_mono:       
        comb["mass"]+=mono_mass[comb.index]
        if add_borders:
            comb[["mass_l","mass_r"]]+=mono_mass[comb.index].reshape(-1,1)
        
    comb=comb.sort_values(by=["gx","mass"])
    comb.pop("gx")
  #%%
    return comb

#%% Get elemental metadata


if not os.path.exists(isotope_table):
    
    #%%
    print("isotope table not found: parsing from NIST website!")

    url="https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
    tables = pd.read_html(url)[0]
    
    
    #remove fully nan rows and columns
    tables=tables[~tables.isnull().all(axis=1)]
    tables=tables[tables.columns[~tables.isnull().all(axis=0)]]
    
    
    tables.columns=["atomic_number","symbol","mass_number",'Relative Atomic Mass',
           'Isotopic  Composition', 'Standard Atomic Weight', 'Notes']
    tables=tables[["atomic_number","symbol","mass_number",'Relative Atomic Mass',
           'Isotopic  Composition', 'Standard Atomic Weight']]
    
    tables.loc[tables["atomic_number"]==1,"symbol"]="H" #remove deuterium and tritium trivial names
    tables=tables[tables['Isotopic  Composition'].notnull()].reset_index(drop=True)
    
    
    #floatify mass and composition
    for i in ['Relative Atomic Mass',
           'Isotopic  Composition', 'Standard Atomic Weight']:
        tables[i]=tables[i].str.replace(" ","").str.replace("(","").str.replace(")","").str.replace(u"\xa0",u"")
    tables[['Relative Atomic Mass', 'Isotopic  Composition']]=tables[['Relative Atomic Mass', 'Isotopic  Composition']].astype(float) 
    tables['Standard Atomic Weight']=tables['Standard Atomic Weight'].str.strip("[]").str.split(",")

    tables=pd.read_csv(isotope_table,sep="\t")
        
    #set Standard isotope
    tables["Standard Isotope"]=False
    
    #mono isotopic as standard
    mdf=tables.sort_values(by=["symbol",'Isotopic  Composition'],ascending=False).groupby("symbol",sort=False).nth(0)[['symbol','Relative Atomic Mass']]
    tables.loc[mdf.index,"Standard Isotope"]=True
    
    # #lowest mass as standard
    # tables.loc[tables.groupby("symbol").nth(0).index,"Standard Isotope"]=True
    
    tables["mass_number"]=tables["mass_number"].astype(int)
    tables["isotope_symbol"]=tables["symbol"]+"."+tables["mass_number"].astype(str)
    tables["delta_mass"]=tables["Relative Atomic Mass"]-tables.loc[tables["Standard Isotope"],["symbol","Relative Atomic Mass"]].set_index("symbol").loc[tables["symbol"]].values.flatten()
    tables["delta neutrons"]=np.round( tables["delta_mass"],0).astype(int)
    
    #normalise isotopic composition
    tables['Isotopic  Composition']=tables['Isotopic  Composition']/tables.groupby("symbol")['Isotopic  Composition'].transform('sum')

    #tables.to_csv("shifted_test.csv",sep="\t")

    tables.to_csv(isotope_table,sep="\t")

else:
    
    tables=pd.read_csv(isotope_table,sep="\t")
    

#add electron gain/loss
emass = 0.000548579909  # electron mass
t=tables.iloc[:2].copy()
t[:]=0
t[["symbol","Standard Isotope","Isotopic  Composition","Relative Atomic Mass"]]=np.array([["+","-"],[True,True],[1,1],[-emass,+emass]]).T
tables=pd.concat([tables,t]).reset_index(drop=True)
tables[["Isotopic  Composition","Relative Atomic Mass"]]=tables[["Isotopic  Composition","Relative Atomic Mass"]].astype(float)
tables["Standard Isotope"]=tables["Standard Isotope"].astype(bool)

mono_elmass=tables[tables["Standard Isotope"]].set_index("symbol")["Relative Atomic Mass"]
all_elements=list(set(tables.symbol))

#construct mdf


#%% Load input data

if __name__=="__main__":
    
    #load composition
    if "." not in composition[0]: composition=[composition]  
    elif os.path.isdir(composition[0]): 
        composition=[str(Path(composition,i)) for i in os.listdir(composition[0])]
        composition.sort()

    if is_float(peak_fwhm): peak_fwhms=[float(peak_fwhm)]
    elif os.path.isdir(peak_fwhm): 
        [str(Path(peak_fwhm,i)) for i in os.listdir(peak_fwhm[0])]
        peak_fwhms.sort()
    
    if len(peak_fwhms)==1:
        peak_fwhms=peak_fwhms*len(composition)
        
    if not os.path.exists(Output_folder): os.makedirs(Output_folder)
    
    for cix,c in enumerate(composition):
    
        Outpath=str(Path(Output_folder,Output_file))
        
        #read composition input
        if type(c)==list: 
            fdf=pd.concat([parse_form(i) for i in c]).reset_index(drop=True).fillna(0)  
            if not len(Output_file): Outpath=str(Path(Output_folder,method+"_isosim.tsv"))
        else:             
            if not len(Output_file): Outpath=str(Path(Output_folder,Path(c).stem+"_"+method+"_isosim.tsv"))
            check,fdf=read_table(c,Keyword="Composition")
            if check: fdf=pd.concat([parse_form(i) for i in fdf["Composition"].values]).reset_index(drop=True).fillna(0)   
            else:
                check,fdf=read_table(c,Keyword=all_elements)
                if not check: 
                    print("incorrect composition file format! Skipping file: "+c)
                    continue
        
    
        elements=fdf.columns[fdf.columns.isin(tables.symbol)]
        form=fdf[elements]
        res=fft_lowres (form)
        if method=="fft":         res=fft_lowres (form)
        if method=="fft_hires":   res=fft_highres(form,peak_fwhm=peak_fwhms[cix])
        if method=="multi":       res=multi_conv( form,peak_fwhm=peak_fwhms[cix])
        
        print("")
        print("Writing output: "+str(Outpath))
        res.to_csv(Outpath,sep="\t")
    
    
    
    








#%%




