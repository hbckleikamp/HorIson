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

## 1. FFT_Lowres specific arguments

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|bins| | |

elements
min_intensity
isotope_range
normalize False
batch_size
add_mono


## 1. FFT_Lowres specific arguments




|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|bins| | |

mass_calc



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

Additional chemical constraints are provided by implementing some of Fiehn's 7 Golden rules, which filters unrealistic or impossible compositions.
This can drastically reduce the size of your composition space. These include:  Rule #2 – LEWIS and SENIOR check; Rule #4 – Hydrogen/Carbon element ratio check; Rule #5 heteroatom ratio check and Rule #6 – element probability check.

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-filt_7gr| True | Toggle global to apply or remove 7 golden rules filtering|
|-filt_LewisSenior| True | Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence) |
|-filt_ratios | "HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" | #Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage |
|-filt_NOPS| True    | #6 – element probability check. |

Additional arguments can be supplied to affect the performance and output paths: 

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-maxmem | 10e9 |  Amount of memory used in bytes |
|-mass_blowup | 100000 |blowup factor to convert float masses to integers|
|-write_mass  | True | construct a mass lookup table (faster MFP but larger database)|
|-Cartesian_output_folder | "Cart_Output" | Path to output folder |
|-Cartesian_output_file   |<depends on parameters> | Output database name |




#### Licensing

The pipeline is licensed with standard MIT-license. <br>
If you would like to use this pipeline in your research, please cite the following papers: 
      



