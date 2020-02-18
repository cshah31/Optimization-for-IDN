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
