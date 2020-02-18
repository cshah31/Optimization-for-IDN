This reserach work is the part of the application RAFTES (Resilient ans Fault Tolerant Energy System), developed at University of Alaska fairbanks and Alaska Center for Energy and Power (http://acep.uaf.edu). This application will be integrated with an open source advanced distribution management system platform GridAPPS-D (https://gridapps-d.org) developed by Pacific Northwest National Laboratory.

# Optimization-for-Isolated Distribution Network Integaretd with Battery Energy Storage System.

Test Network:
The IEEE-123 bus test feeder used in this study consists of 91 load nodes and six load zones as shown in Figure 1. The nominal operating voltage of the test feeder is 4.16 kV. The optimization algorithm is implemented for load zone- 2 in the IEEE-123 test feeder shown in the Figure. Load zone-2 has 18 nodes and is isolated from the source node 150 and other load zones. Node 1 is a root/reference node, and it is equipped with a diesel generator with a quadratic fuel cost function. The branch nodes have a deterministic residential load profile and rooftop PV which is described in the next sub-section. The shapeable loads at each branch node are ten times the peak residential load and are intended to represent electric vehicles. The battery energy storage system is placed at each branch node with the rated power of 50% of the peak residential load and storage capacity of 5 hours.

Residential Load and Solar Data: 
The PECAN STREET project residential load data set from January 1, 2017 to June 30, 2017, is used in this research work [13]. Based on the results of the load profile aggregation (LPA) algorithm proposed in [14]. The capacity of the rooftop PV installed at each house on branch nodes is 6 kW. The hourly 2017 NSRDB solar data from January 1, 2017 to June 30, 2017, is used for the rooftop PV in this research work.

Battery Energy Storage Models:
The battery energy storage system is characterized by its rated power, rated energy, and charging/discharging cycle efficiencies [12]. The battery energy storage models that will be implemented in this paper are as follows:
1) Linear Model:
2) Nonlinear Model
3) Convex Nonlinear Model


References:
[1] S. H. Low, “Convex Relaxation of Optimal Power Flow – Part I: Formulation and Equivalence”, IEEE Transactions on Control of Network Systems, vol.1, no. 1, pp. 15-27, Mar. 2014.
[2] Y. Levron, J. M. Guerrero, and Y. Beck, “Optimal Power Flow in Microgrids with Energy Storage”, IEEE Transactions on Power Systems, vol. 28, no. 3, pp. 3226-3234, Aug. 2013.
[3] M. Sufyan, C. Tan, N. A. Rahim, S. R. S. Raihan, and M. A. Muhammad, “Dynamic Economic Dispatch of Isolated Microgrid with Energy Storage using MIQP”, in 2018 IEEE International Conference on Intelligent and Advanced System, Aug 2018.
[4] C. Wang, L. Fang, L. Pan, Y. Wang, “Optimal Dispatch of an Island MG Including a Multiple Energy Storage System”, in 2018 2nd IEEE Conference on Energy Internet and Energy System Integration.
[5] E. Munsing, J. Mather, and S. Moura, “Blockchains for Decentralized Optimization of Energy Resources in Microgrid Networks”, in 2017 IEEE Conference on Control Technology and Applications (CCTA).
[6] S. K. Jadhav, “Optimal Power Flow in Wind Farm Microgrid using Dynamic Programming”, in Proc. of 2018 IEEE International Conference on Emerging Trends and Innovations in Engineering and Technological Research, July 2018.
[7] M. A. Abdulgalil, A. M. Amin, M. Khalid, and M. AlMuhaini, “Optimal Sizing, Allocation, Dispatch and Power Flow of Energy Storage Systems Integrated with Distributed Generation Units and a Wind Farm”, in 2018 IEEE PES Asia-Pacific Power and Energy Engineering Conference.
[8] M. Nick, R. Cherkaoui, and M. Paolone, “Optimal Allocation of Dispersed Energy Storage Systems in Active Distribution Networks for Energy Balance and Grid Support”, IEEE Transactions on Power Systems, vol. 29, no. 5, pp. 2300-2310, Sep. 2014.
[9] A. Gabash and P. Li, “Active-Reactive Optimal Power Flow in Distribution Networks with Embedded Generation and Battery Storage”, IEEE Transaction on Power System, vol. 27, no. 4, pp. 2026- 2035, Nov. 2012.
[10] Q. Li, S. Yu, A. S. Al-Sumaiti, and K. Turitsyn, “Micro Water-Energy Nexus: Optimal Demand-Side Management and Quasi Convex Hull Relaxation”, IEEE Transactions on Control of Network Systems (Early Access), Dec. 2018.
[11] Q. Peng and S. H. Low, “Distributed Optimal Power Flow Algorithm for Radial Networks, I: Balanced Single Phase Case”, IEEE Transaction on Smart Grid, vol. 9, no. 1, pp. 111-121, Jan. 2018.
[12] C. Eyisi, A. S. Al-Sumaiti, K. Turitsyn, and Q. Li, “Mathematical Models for Optimization of Grid-Integrated Energy Storage Systems”, unpublished. Presented at IEEE/PES General Meeting 2019. [Online]. Available: https://arxiv.org/ftp/arxiv/papers/1901/1901.06374.pdf
[13] [Online]. Available: https://dataport.pecanstreet.org/idq
[14] J. Wang, X. Zhu, D. Lubkeman, N. Lu, N. Samaan, and B. Werts, “Load Aggregation Methods for Quasi-Static Power Flow Analysis on High PV Penetration Feeders”, in Proc. of 2018 IEEE/PES Transmission and Distribution Conference and Exposition (T&D), Apr. 2018.
[15] [Online]. Available: http://acep.uaf.edu/facilities/power-systemsintegration-lab.aspx
