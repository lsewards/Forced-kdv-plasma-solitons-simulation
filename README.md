# One Dimensional Numerical Simulation of Solitons induced by charge space debris in plasma enviroment.
 
__written by Lewis Sewards as part of the EPSRC Vacation intership attended at the Univeristy of Liverpool__

## AI Declaration

In the development of this code, I used claude GenAI for Coding support and proofreading. However, I critically evaluated all outputs and remain responsible for the submitted work

## Introduction

This project aims to recreate the work of _Hartzell and Akin, 2019_ paper which investigated the velcoity dependance of 
solitons induced by charge space debris in a plasma environment. The simulation is implemented in Python and utilizes 
numerical methods to model the behavior of solitons under various conditions.

The simulation is based on the forced korteweg-de Vries (fKdV) equation, which is then numercically solved 
pseudo-spectral schemes, two in particular, the Fourier spectral method and the Chan & Kerkhoven (CK) and Fornberg & 
Whitham (FW). Both are implimented in the code and can be selected by the user. 


## Improvments and future work
- Currently to run the simulation, the user must manually change the parameters in the code. A future improvement would 
be to implement a user interface that allows for easy parameter selection and simulation control.
- The intialsation of the simulation can take place outside of the main loop, as currently the simulation checks every
time if its the first iteration which on long simulations can cause signfificantly slower run times.
- The code could be better generallised as at the minute it relies on specific sintax and variable names which makes it
difficult to adapt to other similar problems. A more generalised code would allow for easier adaptation to other problems 
in the future.

## References

The papers this work builds on are listed in [docs/REFERENCES.md](docs/REFERENCES.md).

## License

This project is released under the [MIT License](LICENSE). If you use this code in academic work, please cite this
repository.
