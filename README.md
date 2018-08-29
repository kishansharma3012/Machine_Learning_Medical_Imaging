###PROJECT ABSTRACT:

In  this    project a   recent  method  for detection   and tracking    of  tools   in  minimally-invasive  interventions   should  be
implemented.    Based   on  convolutional   neural  networks,   the method  simultaneously  detects and estimates   the 
pose     of  the     main    tool    parts,  improving   the     current     state-of-the-art    on  the     EndoVis     challenge.  The     dataset
contains     a   sequence    of  videos  of  endoscopic  surgeries.  As  a   main    part    of  this    project,    different   network
architectures   should  be  used    to  implement   the method  before  comparing   the results for each    network.

###CONCLUSION:
- FCN :VGG didn’t work at all, Bias Problem ,ResNet got better results
Adjusting hyper-parameters might improve accuracy

- U-Net:  Performed better for 10 joints than 12 joints in experiment 1,Results for right clasper better than left clasper
Achieved the state-of-art results for some joints

- It is harder to train with FCN than it is to train with U-Net
