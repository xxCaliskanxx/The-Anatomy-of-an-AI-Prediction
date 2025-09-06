## Data Generation

The `drag_polar.csv` dataset was generated using the aerodynamic drag polar formula:

**Cd = CD0 + k × Cl²**

Where:  
- **CD0** = 0.02 (zero-lift drag coefficient)  
- **k** = 0.04 (lift-induced drag factor)  
- **Cl** values range from 0.20 to 1.50 in 0.05 increments  

A small random variation was added to Cd to simulate real-world measurement noise, making the data more realistic for model training and analysis.

