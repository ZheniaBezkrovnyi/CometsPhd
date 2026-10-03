#pragma once

namespace PhysicsConsts {
    constexpr double PI = 3.141592653589793238462643383279502884;
    constexpr double DEG_TO_RAD = PI / 180.0;
    constexpr double RAD_TO_DEG = 180.0 / PI;

    constexpr double SpeedOfLight = 299792458.0;      
    constexpr double kB = 1.380649e-23;               
    constexpr double PlanckH = 6.62607015e-34;   
    constexpr double Avogadro = 6.02214076e23;      

    constexpr double Sigma = 5.670374419184429e-8;   
    constexpr double G = 6.67430e-11;              
    constexpr double AtomicMassUnit = 1.66053906660e-27;  

    constexpr double M_H2O = 18.01528;          
    constexpr double m_gas_H2O = M_H2O * AtomicMassUnit; 
    constexpr double L_sub = 2.83e6;             

    constexpr double AU_METERS = 149597870700.0;        
    constexpr double GAUSS_K = 0.01720209895;            
    constexpr double GM_SUN = 1.32712440041279419e20;  
    constexpr double JD_J2000 = 2451545.0;                
    constexpr double OBLIQUITY_J2000_ARCSEC = 84381.448; 
    constexpr double SUN_V_MAG = -26.74;                 

    constexpr double SECONDS_PER_DAY = 86400.0;
    constexpr double SECONDS_PER_HOUR = 3600.0;

    constexpr double IncreaseAreaCometBy = 1e-7;
}