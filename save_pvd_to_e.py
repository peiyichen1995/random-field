# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *

in_file = 'C:\\Users\\peiyi\\Desktop\\moose file\\G2_Sample_99.vtk'
out_file = 'C:/Users/peiyi/Desktop/moose file/dududu.e'
field_name = 'gamma'
# create a new 'Legacy VTK Reader'
reader = LegacyVTKReader(registrationName='G2_Sample_99.vtk', FileNames=[in_file])



# save data
SaveData(out_file, proxy=reader, ChooseArraysToWrite=1,
    PointDataArrays=[field_name])
