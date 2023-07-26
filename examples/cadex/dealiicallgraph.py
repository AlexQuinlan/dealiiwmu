#!/usr/bin/python3

import argparse                                                                                                                                                                                            
import os                                                                                                                                                                                                  
import subprocess 

## Have a properly linked executable

## Make sure debug symbols are enabled

## Decide on callgrind, perf, gprof

verbose = 'true'
job = "cadex"
details = "false"  ## Not enabled yet in this script

## ?
FILTER="-v -e gfortran -e libc -e libm- -e unknown -e libgomp -e libpardiso600 -e ld-2.33.so -e 7fa17f -e libpthread -e libblas -e libz -e libgcc" #
FILTER="-v dontfilterme"

def gprof2dot(typestr):                                                                                                                                                                                    
    ## the -p path filter has been removed
    return("gprof2dot -n0 -e0 -f {} --skew 0.1 -p cadex ".format(typestr))

def createsvg(methodstr, gprofstr, details):                                                                                                                                                               
    if (verbose):                                                                                                                                                                                          
        print(methodstr)                                                                                                                                                                                   
        print(gprofstr)                                                                                                                                                                                    
    os.system(methodstr)                                                                                                                                                                                   
    os.system(gprofstr)                                                                                                                                                                                    
    if details:                                                                                                                                                                                            
        methodstr = methodstr.replace(".dii_",".all_").replace("-p ccx_2.19","")                                                                                                                           
        gprofstr = gprofstr.replace(".dii_",".all_").replace("-p ccx_2.19","")                                                                                                                             
        if (verbose):                                                                                                                                                                                      
            print(methodstr)                                                                                                                                                                               
            print(gprofstr)                                                                                                                                                                                
        os.system(methodstr)                                                                                                                                                                               
        os.system(gprofstr)

def gstring(typestr):                                                                                                                                                                                      
    gprofstr0 = "grep {} {}.dii_{}.dot | dot -Tsvg -o {}.dii_{}.svg ".format(FILTER,job,typestr,job,typestr)                                                                                               
    gprofstr1 = "&& ./create_html.py {}.dii_{}.svg".format(job,typestr)                                                                                                                                    
    return(gprofstr0+gprofstr1)  


#----------------------------------------------------------------------

## os.system("perf record -o cadex.perf -g ./cadex")  ## generates cadex.perf file
methodstr = "perf script -i {}.perf | c++filt | {} > {}.dii_perf.dot".format(job,gprof2dot("perf"),job)                                                                                            
gprofstr = gstring("perf")                                                                                                                                                                         
createsvg(methodstr, gprofstr, details)


## perf script -i cadex.perf | c++filt | gprof2dot -n0 -e0 -f perf --skew 0.1 -p cadex  > cadex.dii_perf.dot
