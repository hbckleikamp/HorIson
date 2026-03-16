# HorIson

HorIson is a tool for vectorized isotope simulation, that applies Fast-fourier transform (FFT) or multinomial products.
<br>

#### How does HorIson work

HorIson provides 3 algorithms for different types of istope simulation.
1. FFT_Lowres, which computes convolved coarse isotopes (1 per 1 Da) at high speed.
2. FFT_Highres, which computes fine isotopes and profile spectra at variable resolution.
3. Multi_conv: which computes separate isotope combinstions at high speed. <br> <br>

To enable vectorization for FFT-based algorithms, isotope masses are placed on a uniform grid.
For multinomial products, global isotope combinations and multnomial distribtuions are precomputed.

#### Usage:
HorIson can be imported as a module, or called from the command line.
As input for each algorithm, a failepath to a tabular can be supplied, or an element string, or a DataFrame or array can be used when importing as a module.
When no header information is present, the element order should be supplied with the element parameter.

# Arguments.

## General arguments

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|elements|[] | Specifies element column order if a headerless array is used as input |
|min_itensity|1e-6|Filters isotopes in output below a certain probability level|
|isotope_ranges|[-2,6]|Filters isotopes in output to a certain mass range|
|charge|1| charge of computed formulas, accepts either single value or array of different charges  |
|normalize|False|Normalizes probabilies either total ("sum"), largest isotope ("max") or monoisotopic peak ("mono") |
|batch_size|1e4|How many formulas to simulate at once|
|add_mono|True|Add monoisotopic mass back to simluated isotope masses|
|peak_fwhm|0.01|Peak FWHM in Da, used for generating profile data (FFT_Highres, Multi_conv), accepts either single value or array of different FWHMs |
|divisor|4|Subsampling rate for convolution (FFT_Highres, Multi_conv) | 


## FFT_Lowres specific arguments


|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|bins| False| Set a fixed maximum grid size |
|mass_calc| True| Calculate exact isotope masses|
|return_borders|False| provide grid-sizes as output|


## FFT_Highres specific arguments

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|packing| True|  Compress redundant grid sections |
|peak_picking|True| Output centroid data |

## Mutli_conv specific arguments

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|prune| 1e-6|  remove isotope combinations below a chance threshold or outside of isotope range |
|min_chance| 1e-4| remove isotope combinations after pruning |
|convolve|"fast"| Output raw combinations (False) or profile (peak) data ("full") or centroid data ("fast")
|add_borders|False|Compute peak borders|
|add_area| False| Compute peak area|
|Precomputed|[]|Re-use precomputed isotope combinations |




#### Licensing:

The pipeline is licensed with standard MIT-license. <br>
If you would like to use this pipeline in your research, please cite the following papers: 
      



