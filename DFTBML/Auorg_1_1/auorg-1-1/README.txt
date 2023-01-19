******************************************
Gold parametrization for organic molecules
******************************************

General information
===================

This set was designed to describe the optical excitations of thiolates on gold
nanoclusters. It is an extension of the mio-1-1 set with Au.

Note: This set assumes that you use shell resolved SCC in DFTB. In case you use
the DFTB+ code, you can achieve it by setting `OrbitalResolvedSCC = Yes`.


Maximal angular momenta
-----------------------
C: p
H: s
N: p
O: p
S: d
Au: d


Spin constants
--------------

Note, the calculation of the spin constants follows here for all elements the
convention as used for the Hubbard U values: For non-occupied atomic orbitals
(orbitals above HOMO) the corresponding value of the HOMO is used.

H:
     -0.07174
C:
     -0.03062     -0.02505
     -0.02505     -0.02265
N:
     -0.03318     -0.02755
     -0.02755     -0.02545
O:
     -0.03524     -0.02956
     -0.02956     -0.02785

S:
     -0.02137     -0.01699     -0.01699
     -0.01699     -0.01549     -0.01549
     -0.01699     -0.01549     -0.01549

Au:
     -0.01304     -0.01304   -0.00525
     -0.01304     -0.01304   -0.00525
     -0.00525     -0.00525   -0.01082



Relevant publications
=====================

[PRB98] M. Elstner, D. Porezag, G. Jungnickel, J. Elsner, M. Haugk,
Th. Frauenheim, S. Suhai, and G. Seifert, Phys. Rev. B 58, 7260 (1998).

[JMS01] T.A. Niehaus, M. Elstner, Th. Frauenheim, and S. Suhai,
J. Mol. Struct. (Theochem) 541, 185 (2001).

[JCC15] A. Fihey, C. Hettich, J. Touzeau, F. Maurel, A. Perrier, C. Köhler,
B. Aradi, and T. Frauenheim, J. Comp. Chem. 36, 2075 (2015).


Publications to be cited
========================

C,H,N,O - C,H,N,O: [PRB98]
C,H,N,O,S - S:	   [JMS01]
Au - C,H,N,O,S,Au: [JCC15]
