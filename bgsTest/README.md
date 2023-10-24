This directory contains the code used to generate and run the SLiM code for simulating introgression in the presence of BGS for the IntroUNET manuscript. Briefly, this code generates a set of 100 SLiM scripts, each modeling a different portion of chr3L in release 6 of the *Drosophila melanogaster* assembly. CDS coordinates, recombination rates, and the DFE of coding mutations are obtained as specified in the manuscript (or see `extractSlimScriptForDmelAnnotDFE_randomWindow.py` for details).

This code was run using stdpopsim version 0.2.0 and SLiM version 4.0.1 (on Linux compute nodes).

To generate and run the slim scripts, simply run `python pipeline.py`. Next, to recover and print out the average recombination rate within the central 100 kb and across the entire 1 Mb window for each replicate, simply run `python parseRecRateFromSlimScripts.py`.
