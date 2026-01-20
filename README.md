# Gruvimeter
This repository contains all code produced for the processing and analysis of data from the diamagnetic levitation based gravimeter of the Ulbricht group within the University of Southampton. Mainly written by myself Michael Hawker for my MSci. research placement.
My trace file loading and understanding of power spectral density production is derivative of Ashley Setter's Optoanalysis python package.

The uploaded code falls into four categories, the largest of which is the processing pipeline. Three pipeline chunks may be found in this repository, all of which were used to convert raw oscilloscope trace files into meaningful acceleration measurements. 

Chunk 1 handles the conversion of raw trace files into displacement (in metres) and creates intermediate save files in .mat file format. This includes quadrature creation from the two channels, angle extraction and conversion.
Chunk 2 gathers intermediate saves from chunk 1 by naming convention within a directory and downsamples them sequentially, allowing for the creation of massive data sets weeks long and saving them in .mat file format. 
Chunk 3 takes saves from chunk 2 and converts the large stitched data sets into acceleration by a flat transfer function determined by Hooke's Law and Newton's second law, some initial analysis is completed here just to ensure the data appears healthy.

The analysis code in its entirety may be found in the .py file "Post Processing Analysis Code with Residual Allans". This script loads the .dat files outputted by TSoft for the tidal model and observed gravimeter data, creates the mask effect spectral fingerpint 
filter from the models FFT and verifies the spectral fingerprint matches. It then produces the overlay plot for the raw, filtered and model traces with an alignment trace for the spring and neap tides created from a manually entered reference full moon from the Royal Museums Greenwich calendar. The stability and sensitivity of the gravimeter were analysed via Allan deviation plots, though these are limited since tidal effects can't be truly removed from a three dimensional sensor with a one dimensional model. 
Some redundant analyses have been left in since they may prove useful, notably the production of PSDs, autocorrelation and cross correlation.

A null hypothesis testing method for the spectral analysis of the tidal model component extraction can be found in "White noise ratio checker.py". This includes sanity checks against white, pink and brown noise to confirm that matching ratios are not produced by the filter. A fitted noise can also be produced from loaded data to ensure the match is not coincidental with some form of noise present in the data. 

Lastly "Timetrace visual check bulk.py" is just a timetrace script, it loads trace files from a directory and processes them sequentially, outputting plots for large scale visual inspection of data. This was used prior to processing to ensure only quality data showing no electrical glitches, power issues with the laser etc. were processed for the analysis of the gravimeter's functionality.
