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
from collections.abc import Iterable
from scipy.stats import multinomial
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks,peak_widths,argrelmax

# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

#%%

composition=str(Path(basedir,"CartMFP_test_mass_CASMI2022.tsv" )) #file,folder or composition string
isotope_table=str(Path(basedir, "isotope_table.tsv"))             # table containing isotope metadata in .tsv format
Output_folder=str(Path(basedir, "HorIson_Output"))             # table containing isotope metadata in .tsv format
Output_file=""

method="multi" #Options: fft, fft_hires, multi
normalize=False #Options: False, sum, max
min_intensity=1e-6
isotope_range=[-2,6] #low, high

#method specific arguments
batch_size=1e4  #batch size used in fft hires
pick_peaks=True #fft_hires

peak_fwhm=0.01  #used in fft_hires and convolve functions
prune=1e-6      #multinomial specific: pruning during combinations
min_chance=1e-4 #multinomial specific: pruning after  combinations
convolve=True



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
                        help="Optional: normalize the output (Options: don't normalize (False), normalize to total ('sum'), normalize to most abundant isotope ('max')")
    
    parser.add_argument("--min_intensity",       required = False, default=1e-6,
                        help="Only retain Fourier columns that have one abundance over x" )
    
    parser.add_argument("--batch_size",       required = False, default=1e4,
                        help="FFt hires only: perform fourier transform in batches of x" )
    parser.add_argument("-fwhm, --peak_fwhm",       required = False, default=0.01,
                        help="Optional: Peak full width at half maxmium. Used in fft_hires and convolve functions. can be supplied as single value, or as a list of values corresponding to different masses." )
    
    #multinomial specific arguments
    parser.add_argument('--convolve', action='store_true',help="Multinomial only: convolve multinomial output with gaussian")
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

    
#%% To Solve
#fft hires has a weird +bin on mono, needs peak picking?
#multi conv isotope range does not work 


#%% Utility functions

def is_float(x):
    try:
        float(x)
        return True
    except: 
        return False

        
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
    return cdf

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



def read_formulas(form): #either a single string or a DataFrame with element columns
    if type(form)==str: form=parse_form(form)  #single string
    
    
    elif  not (isinstance(form, pd.DataFrame) or isinstance(form, pd.Series)): #DataFrame with columns (recommended input)
        form=pd.concat([parse_form(f) for f in form]).reset_index(drop=True).fillna(0)  
        
    elements=form.columns
    form=form.values.astype(int)
    mono_mass=(form*tables[tables["Standard Isotope"]].set_index("symbol").loc[elements,"Relative Atomic Mass"].values).sum(axis=1)
    rel_mass=tables.set_index("symbol").loc[elements,["Relative Atomic Mass",'Isotopic  Composition']].prod(axis=1)
    avg_mass=(form*rel_mass.groupby(rel_mass.index,sort=False).sum().values.flatten()).sum(axis=1)
    return form,elements,mono_mass,rel_mass,avg_mass

def read_resolution(fwhms,avg_mass): #input is either a file or numeric value


    if isinstance(fwhms,pd.DataFrame):
        if len(fwhms)==len(avg_mass): fwhms=fwhms["fwhm"].values
        else: fwhms=np.interp(avg_mass,fwhms["mz"],fwhms["fwhm"])
        return fwhms

    if isinstance(fwhms,np.ndarray) or type(fwhms)==list():
        
        if len(fwhms)==len(avg_mass): 
            return fwhms
        else:
            print("length of supplied fhwms is not equal to length of masses!")

    if not is_float(fwhms): 
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
               ): 
    
    #read input compositions
    if not len(elements): form,elements,mono_mass,rel_mass,avg_mass=read_formulas(form)
    if return_borders: print("Low resolution fft prediction")
    
    l_iso=tables.set_index("symbol").loc[elements].pivot(columns="delta neutrons",values='Isotopic  Composition').fillna(0)
    l_iso[list(set(np.arange(l_iso.columns.min(),l_iso.columns.max()+1))-set(l_iso.columns))]=0
    l_iso=l_iso.loc[elements,:]
    
    #determine padding
    if not bins:
        N=max(2*abs(mono_mass-avg_mass).max(),50)
        bins=2**math.ceil(math.log2(N))
        print("number of bins: " +str(bins))
    pad=bins-len(l_iso.columns)
    
    #pad and fft shift negative ions
    l_iso[l_iso.columns.max()+np.arange(1,pad+1)]=0
    l_iso=l_iso.T.sort_index().T
    l_iso=l_iso[l_iso.columns[l_iso.columns>=0].tolist()+l_iso.columns[l_iso.columns<0].tolist()]
    
    #fft 
    bdfs=[]
    maxns,maxps=[],[]
    forms=np.array_split(form,np.arange(0,len(form),int(batch_size))[1:])
    for batch,form_batch in enumerate(forms):
        print("batch: "+str(batch))
        
        one=np.ones([len(form),len(l_iso.columns)])*complex(1, 0) 
        for e in range(len(elements)):
            one*=np.fft.fft(l_iso.iloc[e,:])**form[:,e].reshape(-1,1)
        baseline=np.fft.ifft(one).real
            
        q=baseline<min_intensity
        maxn,maxp=-np.argmax(q[:,::-1],axis=1).max(),np.argmax(q,axis=1).max()
        if maxn<0: bdf=pd.DataFrame(np.hstack([baseline[:,maxn:],baseline[:,:maxp]]),columns=np.arange(maxn,maxp))
        else:      bdf=pd.DataFrame(baseline[:,:maxp],columns=np.arange(maxn,maxp)) 
        
        if len(isotope_range): #only keep selected isotopes in output
            bdf=bdf.iloc[:,np.round(bdf.columns,0).astype(int).isin(isotope_range)]
        
        bdfs.append(bdf)
        maxns.append(maxn)
        maxps.append(maxp)
    
    bdfs=pd.concat(bdfs).fillna(0)
    if normalize=="sum": bdfs=bdfs.divide(bdfs.sum(axis=1),axis=0)
    if normalize=="max": bdfs=bdfs.divide(bdfs.max(axis=1),axis=0)
#%%
    if return_borders: return min(maxns),max(maxps)
    return bdf

#FFT with resolution
def fft_highres(form,
                
                extend=0.5,
                dummy=1000,
                
                #general arguments
                peak_fwhm=peak_fwhm,
                min_intensity=min_intensity,
                isotope_range=isotope_range,
                normalize=normalize,
                batch_size=batch_size,
                pick_peaks=pick_peaks,
                elements=[]
                
                ): 

    #%%
    #test
    form="Ni3S3-"
    elements=[]
    extend=0.5
    dummy=1000
    pick_peaks=True
    
    #read input compositions
    if not len(elements): form,elements,mono_mass,rel_mass,avg_mass=read_formulas(form)
    peak_fwhm=read_resolution(peak_fwhm,avg_mass)
    peak_fwhm/=1.25 #(prevents round-off issues from digitize)
    

    #determine min number of bins (based on fast fft)
    print("")
    maxn,maxp=fft_lowres(pd.DataFrame(form,columns=elements), 
                         return_borders=True,min_intensity=1e-6) 
    #iso_bins=np.linspace(maxn-extend,maxp+extend,int(np.round((maxp-maxn)/peak_fwhm.min(),0)))
    
    iso_bins=np.hstack([np.linspace(maxn-extend,0,int(np.round(maxn/peak_fwhm.min()))),
                        np.linspace(0,maxp+extend,int(np.round(maxp/peak_fwhm.min())))])
    #make sure isobins has zero
    
    N=len(iso_bins)
    bins=2**math.ceil(math.log2(N))
    pad=bins-N
    
    #digitize to mass resolution
    imass=tables.set_index("symbol").loc[elements]
    imass["col_ix"]=np.digitize(imass.delta_mass.values,iso_bins)   
    imass=imass.merge(pd.Series(np.arange(len(elements)),index=elements,name="row_ix"),how="left",right_index=True,left_index=True)
    
    #pad and fft shift negative ions
    mi_space=np.zeros([len(elements),bins])
    mi_space[imass["row_ix"].values,imass["col_ix"].values]=imass['Isotopic  Composition']
    m_iso=pd.DataFrame(mi_space,index=elements)
    m_iso.columns=np.hstack([iso_bins,dummy+np.arange(pad)])
    m_iso=m_iso[m_iso.columns[m_iso.columns>=0].tolist()+m_iso.columns[m_iso.columns<0].tolist()] #zero shift
    zval=m_iso.columns[np.argmax(m_iso.sum())]
    
    print("")
    print("High resolution fft prediction")
    print("(Warning, hi-res fft only accepts a single fwhm value for now )")
    
    #batched high-res prediction
    bdfs=[]
    forms=np.array_split(form,np.arange(0,len(form),int(batch_size))[1:])
    for batch,form_batch in enumerate(forms):
        print("batch: "+str(batch))
        
        one=np.ones([len(form_batch),bins])*complex(1, 0) 
        for e in range(len(elements)):
            one*=np.fft.fft(m_iso.iloc[e,:])**form_batch[:,e].reshape(-1,1)
        baseline=np.fft.ifft(one).real
        
        #only keep above treshold
        bdf=pd.DataFrame(baseline,columns=m_iso.columns)
        bdf.columns=bdf.columns-zval #correct for binshift
        
        if len(isotope_range): #only keep selected isotopes in output
            bdf=bdf.iloc[:,np.round(bdf.columns,0).astype(int).isin(isotope_range)]

        if normalize=="sum": bdf=bdf.divide(bdf.sum(axis=1),axis=0)
        if normalize=="max": bdf=bdf.divide(bdf.max(axis=1),axis=0)
        
        bdf[bdf<min_intensity]=0
        if pick_peaks:
            x,y=argrelmax(bdf.values,axis=1,mode="wrap")
            bdfs.append(np.vstack([x,bdf.columns[y],bdf.values[x,y]]).T)
            #would be more accurate to do integral under peak, but this would be slow
            
        else: bdfs.append(bdf.iloc[:,np.argwhere(np.any(bdf>0,axis=0))[:,0]])
        

    if pick_peaks:
        bdfs=pd.DataFrame(np.vstack(bdfs),columns=["ix","iso_mass","abundance"])
    else:
        bdfs=pd.concat(bdfs).fillna(0).T.sort_index().T.reset_index(drop=True)
        bdfs.columns=np.round(bdfs.columns,4)
    
    
    #%%

    
    return bdfs

def multi_conv(form,
               
               #general arguments
               prune=prune,
               min_chance=min_chance,
               peak_fwhm=peak_fwhm,
               convolve=convolve,
               isotope_range=isotope_range,
               normalize=normalize,
               elements=[]
               
               
               ): #min_chance=1e-6,prune=1e-6, convolve=False,isotope_range=np.arange(-2,7)
#%%

    # #test
    # form=mfp[elements]
    # elements=[]
    # peak_fwhm=peak_fwhm
   #%%
    if not len(elements): form,elements,mono_mass,rel_mass,avg_mass=read_formulas(form)
    peak_fwhm=read_resolution(peak_fwhm,avg_mass)

    print("Multinomial prediction")
    
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
    if prune:
       
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
    kdf=kdf[kdf.delta_neutrons.isin(isotope_range)]
    kdf.columns=isos+["delta_mass","delta_neutrons"]
    kdf=kdf.sort_values(by="delta_mass")
    kdf_mass=kdf.delta_mass.values

    #parse iso_string
    iso_string=kdf.iloc[:,:-2].astype(int).values.astype(str)
    q=iso_string=="0"
    iso_string=iso_string+kdf.columns[:-2].values+","
    iso_string[q]=""
    iso_string=pd.Series(iso_string.sum(axis=1)).str.strip(",").values
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
        zm=np.zeros(ie.max()+1,dtype=np.uint16)
        zm[ie]=np.arange(len(ie))
        m_ecounts.append(zm)

    #put multi-index on kdf? or just map form to kdf 
    f2kdf=[]
    for ix,e in enumerate(edf.index):
        eix=np.argwhere(kdf.columns.str.startswith(e+"."))[:,0]
        if len(eix):
            f2kdf.append(eix)
            
            
    #### 3. Predict #####
    res_ixs=[]
    multi_preds=[]
    res_ix=np.arange(len(form))

    ur,uri=np.unique(form.astype(bool),axis=0,return_inverse=True) #unique combinations of elements in formulas
    for ixr,r in enumerate(ur):
        print(ixr)
        
        qf=np.argwhere(uri==ixr)[:,0]
        bf=form[qf]
        
   
        chances=np.hstack([earrs[eix][m_ecounts[eix][bf[:,eix]]]  for eix,e in enumerate(earrs) if len(earrs[eix])])
        chances[chances<min_chance]=0
        q=np.all(np.in1d(uif,np.argwhere(chances.sum(axis=0)>0)[:,0]).reshape(-1,uis.shape[1]),axis=1)
        fuis=uis[q] #remove combinations that are zero in batch    
        ux=np.vstack([chances[:,u].prod(axis=1) for ui,u in enumerate(fuis) if len(fuis)]).T #for each combination, calculate the chance (slow!)
        
        ux[ux<min_chance]=0
        qq=np.argwhere(ux>0)

        d=np.vstack([kdf_mass[q][qq[:,1]],ux[qq[:,0],qq[:,1]],kdf.index[q][qq[:,1]]]).T #add kdf indices
        d=np.array_split(d,np.argwhere(np.diff(qq[:,0])>0)[:,0]+1) #diff on qq[:,0] >0 for array_split

        res_ixs.extend(res_ix[qf])
        multi_preds.extend(d)

    multi_preds=[multi_preds[i] for i in np.argsort(res_ixs)] #resort according to index 

    multi_df=pd.DataFrame(np.vstack(multi_preds),columns=["mass","abundance","isotope"])
    multi_df.index=np.repeat(np.arange(len(form)),[len(i) for i in multi_preds])
    multi_df["isotope"]=kdf.loc[multi_df["isotope"].astype(int),"iso_string"].values

    

    if convolve: multi_df=convolve_gauss(multi_df,peak_fwhm,mono_mass)
    if normalize=="sum": multi_df["abundance"]/multi_df.groupby(multi_df.index)["abundance"].transform("sum")
    if normalize=="max": multi_df["abundance"]/multi_df.groupby(multi_df.index)["abundance"].transform("max")
  
    #%%
    return multi_df     




def convolve_gauss(multi_df,peak_fwhm,mono_mass,divisor=10):
    #%%

    print("Convolving multinomial with gaussian")

    m=multi_df.mass.values
    pred_ix=multi_df.index.values
    mi=np.round(m,0).astype(int)
    cm=m-mi

    fwhm=peak_fwhm
    if isinstance(peak_fwhm, Iterable): fwhm=min(peak_fwhm)
    l,u=cm.min()-fwhm*1.1,cm.max()+fwhm*1.1
    x=np.linspace(l,u,(int(np.round((u-l)/fwhm))*divisor))
    bw=np.diff(x)[0]

    ux=np.vstack([pred_ix,mi]).T
    nz=np.argwhere((np.diff(ux,axis=0)!=0).any(axis=1))[:,0]
    uixs=np.diff(np.hstack([-1,nz,len(m)-1]))
    
    zmat=np.zeros([len(uixs),len(x)]) #make sparse?
    zmat[np.repeat(np.arange(len(uixs)),uixs),find_closest(x,cm)]=multi_df["abundance"]

    indices=ux[np.hstack([nz,len(m)-1])]
    

    #convolve with gaussian
    if isinstance(peak_fwhm, Iterable): gmat=[gaussian_filter1d(zmat[ix],sigma=divisor/2.355*peak_fwhm[i]/fwhm)*divisor for ix,i in enumerate(pred_ix[np.hstack([nz,len(m)-1])])]
    else: gmat=gaussian_filter1d(zmat,sigma=divisor/2.355)*divisor #Gaussian FWHM=2*sqrt(2ln(2)) =~ 2.355*s   #https://en.wikipedia.org/wiki/Full_width_at_half_maximum



    #pick peaks
    convolved_peaks=[]
    for ix,i in enumerate(gmat):
        p=find_peaks(i)
        pw=peak_widths(i,p[0])
        convolved_peaks.append(np.vstack([x[p[0]],pw[0]*bw,i[p[0]]]).T)


    cpeak_df=pd.DataFrame(np.hstack([np.repeat(indices,[len(i) for i in convolved_peaks],axis=0),
                                     np.vstack(convolved_peaks)]),columns=["ix","isotope","mass","width","abundance"])

    cpeak_df["ix"]=cpeak_df["ix"].astype(int)
    cpeak_df["mass"]+=(mono_mass[cpeak_df["ix"]]+cpeak_df["isotope"])

    
#%%
    return cpeak_df.set_index("ix")

#%% Get elemental metadata


if not os.path.exists(isotope_table):
    
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
    
    
    #set Standard isotope
    mdf=tables.sort_values(by=["symbol",'Isotopic  Composition'],ascending=False).groupby("symbol",sort=False).nth(0)[['symbol','Relative Atomic Mass']]
    tables["Standard Isotope"]=False
    tables.loc[mdf.index,"Standard Isotope"]=True
    tables["mass_number"]=tables["mass_number"].astype(int)
    tables["isotope_symbol"]=tables["symbol"]+"."+tables["mass_number"].astype(str)
    tables["delta_mass"]=tables["Relative Atomic Mass"]-tables.loc[tables["Standard Isotope"],["symbol","Relative Atomic Mass"]].set_index("symbol").loc[tables["symbol"]].values.flatten()
    tables["delta neutrons"]=np.round( tables["delta_mass"],0).astype(int)
    
    #normalise isotopic composition
    tables['Isotopic  Composition']=tables['Isotopic  Composition']/tables.groupby("symbol")['Isotopic  Composition'].transform('sum')

    tables.to_csv(isotope_table,sep="\t")

else:
    
    tables=pd.read_csv(isotope_table,sep="\t")
    
all_elements=list(set(tables.symbol))




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
        
        if method=="fft":         res=fft_lowres (form)
        if method=="fft_hires":   res=fft_highres(form,peak_fwhm=peak_fwhms[cix])
        if method=="multi":       res=multi_conv(form,peak_fwhm=peak_fwhms[cix])
        
        print("")
        print("Writing output: "+str(Outpath))
        res.to_csv(Outpath,sep="\t")
    
    
    
    








#%%




